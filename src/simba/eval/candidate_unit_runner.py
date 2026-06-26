"""Run and review live candidate-unit compiler provider outputs."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import typing

from simba.eval import ambiguity_fail18, candidate_unit_ir, interpretation_runner

CANDIDATE_UNIT_PROMPT_VERSION = "candidate_unit_compiler_v1"
DEFAULT_QUALITY_REVIEW_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_post_infill_quality_review.json"
)
DEFAULT_GATE1_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_payloads.json"
)
DEFAULT_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_outputs.jsonl"
)
DEFAULT_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_report.json"
)

PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_INVALID_JSON = "invalid_json"
PARSE_STATUS_INVALID_SCHEMA = "invalid_schema"
PARSE_STATUS_EMPTY = "empty"

_ALLOWED_ANSWER_VARIABLES = {
    "entity",
    "action_obligation",
    "event",
    "semantic_type",
    "scalar_value",
    "duration",
}
_ALLOWED_POLICIES = {
    "canonical_entity",
    "action_obligation",
    "event_instance",
    "semantic_type",
    "scalar_value",
    "duration_sum",
}
_ALLOWED_AGGREGATIONS = {"count_distinct", "sum", "lookup"}
_ALLOWED_UNIT_STATUSES = {"included", "excluded", "merged"}


@dataclasses.dataclass(frozen=True)
class ProviderCandidateUnit:
    unit_id: str
    label: str
    status: str
    merge_target: str | None
    value: float | None
    unit: str | None
    evidence_session_ids: tuple[str, ...]
    evidence_spans: tuple[str, ...]
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "unit_id": self.unit_id,
            "label": self.label,
            "status": self.status,
            "merge_target": self.merge_target,
            "value": self.value,
            "unit": self.unit,
            "evidence_session_ids": list(self.evidence_session_ids),
            "evidence_spans": list(self.evidence_spans),
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class CandidateUnitParseResult:
    case_id: str
    parse_status: str
    answer_variable: str
    individuation_policy: str
    aggregation: str
    candidate_units: tuple[ProviderCandidateUnit, ...]
    facts: tuple[str, ...]
    query: str
    computed_answer: float | None
    rationale: str
    parse_errors: tuple[str, ...]

    def to_output_dict(
        self,
        *,
        provider: str,
        prompt_version: str,
        raw_output: str,
    ) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "provider": provider,
            "prompt_version": prompt_version,
            "raw_output": raw_output,
            "parse_status": self.parse_status,
            "answer_variable": self.answer_variable,
            "individuation_policy": self.individuation_policy,
            "aggregation": self.aggregation,
            "candidate_units": [unit.to_dict() for unit in self.candidate_units],
            "facts": list(self.facts),
            "query": self.query,
            "computed_answer": self.computed_answer,
            "rationale": self.rationale,
            "parse_errors": list(self.parse_errors),
        }


def build_fail18_candidate_unit_payload_artifact(
    *,
    quality_review_path: str | pathlib.Path = DEFAULT_QUALITY_REVIEW_PATH,
    gate1_payloads_path: str | pathlib.Path = DEFAULT_GATE1_PAYLOADS_PATH,
) -> dict[str, typing.Any]:
    """Build provider payloads for rows still blocked after interpretation review."""
    quality_review = _load_json(quality_review_path)
    gate1_payloads = _load_json(gate1_payloads_path)
    payload_by_id = {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in gate1_payloads.get("payloads", [])
        if isinstance(payload, dict)
    }
    blocked_cases = [
        case
        for case in quality_review.get("cases", [])
        if isinstance(case, dict)
        and (
            not bool(case.get("useful_for_compilation", False))
            or bool(case.get("quality_issues"))
        )
    ]
    compiler_contract = _default_compiler_contract()
    payloads = []
    skipped: list[dict[str, str]] = []
    for case in blocked_cases:
        case_id = str(case.get("case_id", ""))
        source_payload = payload_by_id.get(case_id)
        if source_payload is None:
            skipped.append({"case_id": case_id, "reason": "missing gate1 payload"})
            continue
        payloads.append(
            build_candidate_unit_payload(
                quality_case=case,
                source_payload=source_payload,
                compiler_contract=compiler_contract,
            )
        )
    return {
        "name": "fail18-candidate-unit-provider-payloads",
        "artifact_kind": "provider_payloads",
        "gate": "candidate_unit_compiler",
        "gate_status": "candidate_unit_payloads_only_not_run",
        "prompt_version": CANDIDATE_UNIT_PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_quality_review": str(quality_review_path),
        "source_payload_artifact": str(gate1_payloads_path),
        "total": len(payloads),
        "blocked_case_ids": [str(case.get("case_id", "")) for case in blocked_cases],
        "skipped_cases": skipped,
        "provider_visibility": {
            "gold_answer_visible": False,
            "gold_value_visible": False,
            "failure_mode_visible": False,
        },
        "payloads": payloads,
    }


def build_candidate_unit_payload(
    *,
    quality_case: dict[str, typing.Any],
    source_payload: dict[str, typing.Any],
    compiler_contract: tuple[str, ...],
) -> dict[str, typing.Any]:
    case = typing.cast("dict[str, typing.Any]", source_payload.get("case", {}))
    return {
        "task": (
            "Compile the question and evidence into candidate answer units. "
            "Do not produce final prose; emit auditable units first."
        ),
        "prompt_version": CANDIDATE_UNIT_PROMPT_VERSION,
        "compiler_contract": [
            *compiler_contract,
            "computed_answer must be a JSON number, not a quoted string.",
        ],
        "review_context": {
            "prior_quality_issues": list(quality_case.get("quality_issues", [])),
            "warning_issues": list(quality_case.get("warning_issues", [])),
            "observed_answer_shapes": list(
                quality_case.get("observed_answer_shapes", [])
            ),
            "instruction": (
                "The previous interpretation pass did not produce a usable "
                "gold-compatible reading. Enumerate every plausible answer unit "
                "with include, exclude, or merge status before computing."
            ),
        },
        "output_schema": {
            "case_id": str(quality_case.get("case_id", "")),
            "answer_variable": (
                "entity|action_obligation|event|semantic_type|scalar_value|duration"
            ),
            "individuation_policy": (
                "canonical_entity|action_obligation|event_instance|semantic_type|"
                "scalar_value|duration_sum"
            ),
            "aggregation": "count_distinct|sum|lookup",
            "candidate_units": [
                {
                    "unit_id": "stable string",
                    "label": "short answer-unit label",
                    "status": "included|excluded|merged",
                    "merge_target": "unit_id or null",
                    "value": "number for sum/lookup, otherwise null",
                    "unit": "unit label or null",
                    "evidence_session_ids": ["evidence_001"],
                    "evidence_spans": ["short exact evidence span"],
                    "reason_code": "stable snake_case string",
                    "reason": "short inclusion/exclusion/merge rationale",
                }
            ],
            "facts": ["prolog-like fact strings"],
            "query": "prolog-like answer(N) rule",
            "computed_answer": 0,
            "rationale": "one terse sentence",
        },
        "case": {
            "id": str(case.get("id", "")),
            "question": str(case.get("question", "")),
            "evidence_sessions": case.get("evidence_sessions", []),
        },
    }


def parse_candidate_unit_response(
    raw_output: str,
    *,
    expected_case_id: str | None = None,
) -> CandidateUnitParseResult:
    fallback_case_id = expected_case_id or ""
    if not raw_output.strip():
        return _invalid_result(
            case_id=fallback_case_id,
            status=PARSE_STATUS_EMPTY,
            errors=("empty provider output",),
        )
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return _invalid_result(
            case_id=fallback_case_id,
            status=PARSE_STATUS_INVALID_JSON,
            errors=(f"invalid JSON: {exc.msg} at char {exc.pos}",),
        )
    if not isinstance(decoded, dict):
        return _invalid_result(
            case_id=fallback_case_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=("root output must be a JSON object",),
        )
    return parse_candidate_unit_object(decoded, expected_case_id=expected_case_id)


def parse_candidate_unit_object(
    raw: dict[str, typing.Any],
    *,
    expected_case_id: str | None = None,
) -> CandidateUnitParseResult:
    fallback_case_id = expected_case_id or ""
    errors: list[str] = []
    case_id = _string_field(raw, "case_id", errors)
    if expected_case_id is not None and case_id and case_id != expected_case_id:
        errors.append(
            f"case_id {case_id!r} does not match expected {expected_case_id!r}"
        )
    answer_variable = _string_field(raw, "answer_variable", errors)
    if answer_variable and answer_variable not in _ALLOWED_ANSWER_VARIABLES:
        errors.append(f"unknown answer_variable {answer_variable!r}")
    individuation_policy = _string_field(raw, "individuation_policy", errors)
    if (
        individuation_policy
        and individuation_policy not in _ALLOWED_POLICIES
    ):
        errors.append(f"unknown individuation_policy {individuation_policy!r}")
    aggregation = _string_field(raw, "aggregation", errors)
    if aggregation and aggregation not in _ALLOWED_AGGREGATIONS:
        errors.append(f"unknown aggregation {aggregation!r}")
    computed_answer = _optional_number_field(raw, "computed_answer", errors)
    query = _string_field(raw, "query", errors)
    rationale = _string_field(raw, "rationale", errors)
    facts = _string_list_field(raw, "facts", errors, allow_empty=False)
    units = _candidate_units(raw.get("candidate_units"), errors)
    _validate_candidate_unit_graph(units, errors)
    if errors:
        return _invalid_result(
            case_id=case_id or fallback_case_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=tuple(errors),
        )
    return CandidateUnitParseResult(
        case_id=case_id,
        parse_status=PARSE_STATUS_PARSED,
        answer_variable=answer_variable,
        individuation_policy=individuation_policy,
        aggregation=aggregation,
        candidate_units=tuple(units),
        facts=tuple(facts),
        query=query,
        computed_answer=computed_answer,
        rationale=rationale,
        parse_errors=(),
    )


def build_provider_prompt(payload: dict[str, typing.Any]) -> str:
    return (
        "Complete this candidate-unit compiler task.\n"
        "Return exactly one strict JSON object matching output_schema. "
        "No markdown, prose, comments, or trailing text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def run_payloads(
    *,
    payload_artifact: dict[str, typing.Any],
    provider_command: str = interpretation_runner.DEFAULT_PROVIDER_COMMAND,
    timeout_seconds: int = interpretation_runner.DEFAULT_TIMEOUT_SECONDS,
    limit: int = 0,
) -> list[dict[str, typing.Any]]:
    payloads = list(payload_artifact.get("payloads", []))
    if limit > 0:
        payloads = payloads[:limit]
    rows: list[dict[str, typing.Any]] = []
    prompt_version = str(
        payload_artifact.get("prompt_version", CANDIDATE_UNIT_PROMPT_VERSION)
    )
    for payload in payloads:
        case_id = str(payload.get("case", {}).get("id", ""))
        provider_result = interpretation_runner.run_provider(
            command=provider_command,
            prompt=build_provider_prompt(payload),
            timeout_seconds=timeout_seconds,
        )
        parsed = parse_candidate_unit_response(
            provider_result.raw_output,
            expected_case_id=case_id,
        )
        row = parsed.to_output_dict(
            provider=provider_command,
            prompt_version=prompt_version,
            raw_output=provider_result.raw_output,
        )
        row.update(
            {
                "provider_exit_code": provider_result.exit_code,
                "provider_stderr": provider_result.stderr,
                "provider_timed_out": provider_result.timed_out,
                "latency_seconds": round(provider_result.latency_seconds, 3),
            }
        )
        if provider_result.exit_code != 0:
            row["parse_errors"] = [
                *row["parse_errors"],
                f"provider exited with code {provider_result.exit_code}",
            ]
        rows.append(row)
    return rows


def build_candidate_unit_report(
    *,
    rows: list[dict[str, typing.Any]],
    payload_artifact: dict[str, typing.Any],
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
) -> dict[str, typing.Any]:
    manifest_by_id = {
        str(row.get("question_id", "")): row
        for row in ambiguity_fail18.load_manifest(manifest_path)
    }
    expected_case_ids = _expected_case_ids(payload_artifact)
    observed_case_ids = [str(row.get("case_id", "")) for row in rows]
    observed_counts = collections.Counter(observed_case_ids)
    parsed_rows = [
        row
        for row in rows
        if row.get("parse_status") == PARSE_STATUS_PARSED
        and not _provider_failed(row)
    ]
    provider_failed_rows = [row for row in rows if _provider_failed(row)]
    case_reviews = [
        review_candidate_unit_row(row, manifest_by_id.get(str(row.get("case_id", ""))))
        for row in rows
    ]
    useful_rows = [
        case for case in case_reviews if case["useful_for_candidate_compilation"]
    ]
    rows_matching_gold = [
        case for case in case_reviews if case["recomputed_answer_matches_gold"]
    ]
    total_latency = sum(
        float(row.get("latency_seconds", 0.0) or 0.0) for row in rows
    )
    return {
        "name": "fail18-candidate-unit-provider-report",
        "artifact_kind": "candidate_unit_provider_report",
        "gate": "candidate_unit_compiler",
        "gate_status": "candidate_unit_provider_run_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "prompt_version": str(
            payload_artifact.get("prompt_version", CANDIDATE_UNIT_PROMPT_VERSION)
        ),
        "provider": str(rows[0].get("provider", "")) if rows else "",
        "source_payload_artifact": str(payloads_path),
        "source_outputs_artifact": str(outputs_path),
        "source_manifest": str(manifest_path),
        "rows_total": len(rows),
        "rows_expected": len(expected_case_ids),
        "rows_parsed": len(parsed_rows),
        "rows_failed_parse": len(rows) - len(parsed_rows),
        "rows_provider_succeeded": len(rows) - len(provider_failed_rows),
        "rows_provider_failed": len(provider_failed_rows),
        "rows_provider_timed_out": sum(
            1 for row in rows if bool(row.get("provider_timed_out", False))
        ),
        "missing_case_ids": sorted(set(expected_case_ids) - set(observed_case_ids)),
        "extra_case_ids": sorted(set(observed_case_ids) - set(expected_case_ids)),
        "duplicate_case_ids": sorted(
            case_id for case_id, count in observed_counts.items() if count > 1
        ),
        "average_candidate_units_per_row": round(
            sum(len(row.get("candidate_units", [])) for row in parsed_rows)
            / len(parsed_rows),
            3,
        )
        if parsed_rows
        else 0.0,
        "candidate_unit_review_summary": {
            "rows_total": len(case_reviews),
            "rows_useful_for_candidate_compilation": len(useful_rows),
            "rows_recomputed_answer_matches_gold": len(rows_matching_gold),
            "rows_recomputed_answer_misses_gold": (
                len(case_reviews) - len(rows_matching_gold)
            ),
            "issue_counts": _issue_counts(case_reviews),
        },
        "provider_cost_or_latency_if_available": {
            "total_latency_seconds": round(total_latency, 3),
            "average_latency_seconds": round(total_latency / len(rows), 3)
            if rows
            else 0.0,
        },
        "cases": case_reviews,
        "acceptance": {
            "outputs_cover_exactly_expected_cases": (
                sorted(observed_case_ids) == sorted(expected_case_ids)
                and len(observed_case_ids) == len(expected_case_ids)
            ),
            "provider_rows_succeeded": not provider_failed_rows,
            "parsed_rows_cover_expected_cases": (
                sorted(row["case_id"] for row in parsed_rows)
                == sorted(expected_case_ids)
            ),
            "raw_provider_output_retained": all("raw_output" in row for row in rows),
            "candidate_units_recomputed_before_gold_check": True,
        },
    }


def review_candidate_unit_row(
    row: dict[str, typing.Any],
    manifest_row: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    issues: list[str] = []
    warnings: list[str] = []
    if _provider_failed(row):
        issues.append("provider_failed")
    if row.get("parse_status") != PARSE_STATUS_PARSED:
        issues.append("parse_failed")
    candidate_units = [
        unit for unit in row.get("candidate_units", []) if isinstance(unit, dict)
    ]
    included_units = [
        unit for unit in candidate_units if str(unit.get("status")) == "included"
    ]
    if not candidate_units:
        issues.append("no_candidate_units")
    if not included_units:
        issues.append("no_included_candidate_units")
    recomputed_answer, recompute_issues = recompute_answer(row)
    issues.extend(recompute_issues)
    provider_answer = _optional_float(row.get("computed_answer"))
    provider_matches_recomputed = _numbers_match(provider_answer, recomputed_answer)
    if (
        row.get("parse_status") == PARSE_STATUS_PARSED
        and not provider_matches_recomputed
    ):
        issues.append("provider_computed_answer_mismatch")
    gold_value = (
        float(ambiguity_fail18.numeric_gold(manifest_row))
        if manifest_row is not None
        and ambiguity_fail18.numeric_gold(manifest_row) is not None
        else None
    )
    if gold_value is None:
        warnings.append("gold_value_unknown")
    recomputed_matches_gold = _numbers_match(recomputed_answer, gold_value)
    if gold_value is not None and not recomputed_matches_gold:
        issues.append("recomputed_answer_misses_gold")
    evidence_span_missing = [
        str(unit.get("unit_id", ""))
        for unit in candidate_units
        if not unit.get("evidence_spans")
    ]
    if evidence_span_missing:
        issues.append("candidate_unit_missing_evidence_span")
    useful = (
        row.get("parse_status") == PARSE_STATUS_PARSED
        and not _provider_failed(row)
        and provider_matches_recomputed
        and recomputed_matches_gold
        and not evidence_span_missing
    )
    return {
        "case_id": str(row.get("case_id", "")),
        "question": str(manifest_row.get("question", "")) if manifest_row else "",
        "failure_mode": str(manifest_row.get("failure_mode", ""))
        if manifest_row
        else "",
        "gold_value": gold_value,
        "answer_variable": row.get("answer_variable", ""),
        "individuation_policy": row.get("individuation_policy", ""),
        "aggregation": row.get("aggregation", ""),
        "candidate_unit_count": len(candidate_units),
        "included_candidate_unit_count": len(included_units),
        "excluded_candidate_unit_count": sum(
            1 for unit in candidate_units if str(unit.get("status")) == "excluded"
        ),
        "merged_candidate_unit_count": sum(
            1 for unit in candidate_units if str(unit.get("status")) == "merged"
        ),
        "provider_computed_answer": provider_answer,
        "recomputed_answer": recomputed_answer,
        "provider_answer_matches_recomputed": provider_matches_recomputed,
        "recomputed_answer_matches_gold": recomputed_matches_gold,
        "useful_for_candidate_compilation": useful,
        "quality_issues": sorted(set(issues)),
        "warning_issues": sorted(set(warnings)),
        "included_candidate_units": [
            {
                "unit_id": str(unit.get("unit_id", "")),
                "label": str(unit.get("label", "")),
                "value": unit.get("value"),
                "unit": unit.get("unit"),
                "reason_code": str(unit.get("reason_code", "")),
                "evidence_session_ids": list(unit.get("evidence_session_ids", [])),
            }
            for unit in included_units
        ],
    }


def recompute_answer(row: dict[str, typing.Any]) -> tuple[float | None, list[str]]:
    aggregation = str(row.get("aggregation", ""))
    candidate_units = [
        unit for unit in row.get("candidate_units", []) if isinstance(unit, dict)
    ]
    included_units = [
        unit for unit in candidate_units if str(unit.get("status")) == "included"
    ]
    if aggregation == "count_distinct":
        return float(len(included_units)), []
    if aggregation == "sum":
        values = [_optional_float(unit.get("value")) for unit in included_units]
        if any(value is None for value in values):
            return None, ["included_sum_unit_missing_numeric_value"]
        return float(sum(typing.cast("list[float]", values))), []
    if aggregation == "lookup":
        values = [_optional_float(unit.get("value")) for unit in included_units]
        numeric_values = [value for value in values if value is not None]
        if len(numeric_values) != 1:
            return None, ["lookup_requires_exactly_one_numeric_included_unit"]
        return float(numeric_values[0]), []
    return None, ["unsupported_aggregation"]


def _candidate_units(
    raw_units: typing.Any,
    errors: list[str],
) -> list[ProviderCandidateUnit]:
    if not isinstance(raw_units, list) or not raw_units:
        errors.append("candidate_units must be a non-empty list")
        return []
    units: list[ProviderCandidateUnit] = []
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            errors.append(f"candidate_units[{index}] must be a JSON object")
            continue
        unit_errors: list[str] = []
        unit_id = _string_field(raw, "unit_id", unit_errors)
        label = _string_field(raw, "label", unit_errors)
        status = _string_field(raw, "status", unit_errors)
        if status and status not in _ALLOWED_UNIT_STATUSES:
            unit_errors.append(f"unknown status {status!r}")
        merge_target = _nullable_string_field(raw, "merge_target", unit_errors)
        value = _nullable_number_field(raw, "value", unit_errors)
        unit = _nullable_string_field(raw, "unit", unit_errors)
        evidence_session_ids = _string_list_field(
            raw,
            "evidence_session_ids",
            unit_errors,
            allow_empty=False,
        )
        evidence_spans = _string_list_field(
            raw,
            "evidence_spans",
            unit_errors,
            allow_empty=False,
        )
        reason_code = _string_field(raw, "reason_code", unit_errors)
        reason = _string_field(raw, "reason", unit_errors)
        if unit_errors:
            errors.extend(
                f"candidate_units[{index}]: {error}" for error in unit_errors
            )
            continue
        units.append(
            ProviderCandidateUnit(
                unit_id=unit_id,
                label=label,
                status=status,
                merge_target=merge_target,
                value=value,
                unit=unit,
                evidence_session_ids=tuple(evidence_session_ids),
                evidence_spans=tuple(evidence_spans),
                reason_code=reason_code,
                reason=reason,
            )
        )
    return units


def _validate_candidate_unit_graph(
    units: list[ProviderCandidateUnit],
    errors: list[str],
) -> None:
    counts = collections.Counter(unit.unit_id for unit in units)
    duplicate_ids = sorted(unit_id for unit_id, count in counts.items() if count > 1)
    if duplicate_ids:
        errors.append("duplicate unit_id values: " + ", ".join(duplicate_ids))
    unit_ids = {unit.unit_id for unit in units}
    for unit in units:
        if unit.status == "merged":
            if not unit.merge_target:
                errors.append(f"unit {unit.unit_id!r} is merged but lacks merge_target")
            elif unit.merge_target == unit.unit_id:
                errors.append(f"unit {unit.unit_id!r} cannot merge into itself")
            elif unit.merge_target not in unit_ids:
                errors.append(
                    f"unit {unit.unit_id!r} merge_target "
                    f"{unit.merge_target!r} does not exist"
                )
        elif unit.merge_target is not None:
            errors.append(
                f"unit {unit.unit_id!r} has merge_target but status {unit.status!r}"
            )


def _invalid_result(
    *,
    case_id: str,
    status: str,
    errors: tuple[str, ...],
) -> CandidateUnitParseResult:
    return CandidateUnitParseResult(
        case_id=case_id,
        parse_status=status,
        answer_variable="",
        individuation_policy="",
        aggregation="",
        candidate_units=(),
        facts=(),
        query="",
        computed_answer=None,
        rationale="",
        parse_errors=errors,
    )


def _string_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")
        return ""
    return value.strip()


def _nullable_string_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string or null")
        return None
    return value.strip()


def _string_list_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
    *,
    allow_empty: bool,
) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return []
    items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(items) != len(value):
        errors.append(f"{key} must contain only non-empty strings")
    if not allow_empty and not items:
        errors.append(f"{key} must be non-empty")
    return items


def _optional_number_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> float | None:
    if key not in raw:
        errors.append(f"{key} must be a number")
        return None
    return _number(raw.get(key), key=key, errors=errors, allow_null=False)


def _nullable_number_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> float | None:
    return _number(raw.get(key), key=key, errors=errors, allow_null=True)


def _number(
    value: typing.Any,
    *,
    key: str,
    errors: list[str],
    allow_null: bool,
) -> float | None:
    if value is None and allow_null:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{key} must be a number" + (" or null" if allow_null else ""))
        return None
    return float(value)


def _optional_float(value: typing.Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _numbers_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) < 0.000001


def _provider_failed(row: dict[str, typing.Any]) -> bool:
    return (
        int(row.get("provider_exit_code", 0) or 0) != 0
        or bool(row.get("provider_timed_out", False))
    )


def _expected_case_ids(payload_artifact: dict[str, typing.Any]) -> list[str]:
    return [
        str(payload.get("case", {}).get("id", ""))
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    ]


def _issue_counts(cases: list[dict[str, typing.Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        for issue in case.get("quality_issues", []):
            counts[str(issue)] = counts.get(str(issue), 0) + 1
    return dict(sorted(counts.items()))


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def _default_compiler_contract() -> tuple[str, ...]:
    return candidate_unit_ir.load_fixture().compiler_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument(
        "--quality-review",
        type=pathlib.Path,
        default=DEFAULT_QUALITY_REVIEW_PATH,
    )
    parser.add_argument(
        "--gate1-payloads",
        type=pathlib.Path,
        default=DEFAULT_GATE1_PAYLOADS_PATH,
    )
    parser.add_argument("--payloads", type=pathlib.Path, default=DEFAULT_PAYLOADS_PATH)
    parser.add_argument("--outputs", type=pathlib.Path, default=DEFAULT_OUTPUTS_PATH)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--provider",
        default=interpretation_runner.DEFAULT_PROVIDER_COMMAND,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=interpretation_runner.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.build_payloads and not args.run_provider and not args.build_report:
        parser.print_help()
        return 2

    if args.build_payloads:
        artifact = build_fail18_candidate_unit_payload_artifact(
            quality_review_path=args.quality_review,
            gate1_payloads_path=args.gate1_payloads,
        )
        args.payloads.parent.mkdir(parents=True, exist_ok=True)
        args.payloads.write_text(
            f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        if not args.run_provider and not args.build_report:
            return 0

    payload_artifact = interpretation_runner.load_payload_artifact(args.payloads)
    if args.run_provider:
        rows = run_payloads(
            payload_artifact=payload_artifact,
            provider_command=args.provider,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
        interpretation_runner.write_jsonl(args.outputs, rows)
    else:
        rows = interpretation_runner.load_jsonl(args.outputs)

    if args.run_provider or args.build_report:
        report = build_candidate_unit_report(
            rows=rows,
            payload_artifact=payload_artifact,
            outputs_path=args.outputs,
            payloads_path=args.payloads,
            manifest_path=args.manifest,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

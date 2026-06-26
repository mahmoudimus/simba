"""Answer-unit witness runner for fail18 direct-answer baselines.

This is deliberately smaller than the candidate-unit compiler. The provider emits
the answer-bearing units it used, and the verifier checks only the load-bearing
parts: spans resolve and arithmetic matches the included units.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import math
import pathlib
import re
import typing

from simba.eval import ambiguity_fail18, interpretation_runner

PROMPT_VERSION = "answer_unit_witness_v1"
DEFAULT_SOURCE_BASELINE_PATH = pathlib.Path(
    "_gitless/fail18_claude_direct_naive_rag_baseline.json"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_answer_unit_witness_payloads.json"
)
DEFAULT_OUTPUTS_PATH = pathlib.Path("_gitless/fail18_answer_unit_witness_outputs.jsonl")
DEFAULT_REPORT_PATH = pathlib.Path("_gitless/fail18_answer_unit_witness_report.json")
DEFAULT_PROVIDER_COMMAND = (
    "claude -p --no-session-persistence --safe-mode --tools '' "
    "--model 'claude-opus-4-8[1m]' --effort medium --output-format json "
    "--system-prompt 'Return exactly one strict JSON object and no markdown.'"
)
DEFAULT_TIMEOUT_SECONDS = 240
DEFAULT_SESSION_CHAR_LIMIT = 2500

PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_EMPTY = "empty"
PARSE_STATUS_INVALID_JSON = "invalid_json"
PARSE_STATUS_INVALID_SCHEMA = "invalid_schema"

ALLOWED_AGGREGATIONS = {"count_included", "sum_included", "lookup_value"}
ALLOWED_DECISIONS = {"include", "exclude"}
NO_ACQUISITION_REASON_CODES = {
    "no_acquisition_date",
    "no_acquisition_evidence",
    "not_acquired",
}
ACQUISITION_VERBS = (
    "adopted",
    "bought",
    "got",
    "gifted",
    "picked up",
    "received",
)
ACQUISITION_NEGATION_RE = re.compile(
    r"\b(?:didn(?:'|\u2019)?t|did not|never|not|no longer|"
    r"thinking of|considering|planning to|plan to|want to|need to|"
    r"got rid of|get rid of|getting rid of)\b",
    flags=re.I,
)


@dataclasses.dataclass(frozen=True)
class AnswerUnit:
    unit_id: str
    label: str
    decision: str
    borderline: bool
    value: float | None
    unit: str | None
    evidence_session_id: str
    evidence_span: str
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "unit_id": self.unit_id,
            "label": self.label,
            "decision": self.decision,
            "borderline": self.borderline,
            "value": self.value,
            "unit": self.unit,
            "evidence_session_id": self.evidence_session_id,
            "evidence_span": self.evidence_span,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class WitnessParseResult:
    case_id: str
    parse_status: str
    answer_variable: str
    aggregation: str
    units: tuple[AnswerUnit, ...]
    answer_number: float | None
    rationale: str
    parse_errors: tuple[str, ...]
    provider_result_text: str = ""

    def to_output_dict(
        self,
        *,
        provider: str,
        prompt_version: str,
        raw_output: str,
        sample_index: int,
    ) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "sample_index": sample_index,
            "provider": provider,
            "prompt_version": prompt_version,
            "raw_output": raw_output,
            "provider_result_text": self.provider_result_text,
            "parse_status": self.parse_status,
            "answer_variable": self.answer_variable,
            "aggregation": self.aggregation,
            "units": [unit.to_dict() for unit in self.units],
            "answer_number": self.answer_number,
            "rationale": self.rationale,
            "parse_errors": list(self.parse_errors),
        }


def build_fail18_payload_artifact(
    *,
    source_baseline_path: str | pathlib.Path = DEFAULT_SOURCE_BASELINE_PATH,
    corpus_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    session_char_limit: int = DEFAULT_SESSION_CHAR_LIMIT,
) -> dict[str, typing.Any]:
    """Build answer-free witness payloads from the saved direct baseline retrieval."""
    source = _load_json(source_baseline_path)
    corpus_rows = ambiguity_fail18.load_corpus(corpus_path)
    corpus_by_id = {str(row["question_id"]): row for row in corpus_rows}
    payloads: list[dict[str, typing.Any]] = []
    for source_row in source.get("results", []):
        if not isinstance(source_row, dict):
            continue
        case_id = str(source_row.get("case_id", ""))
        corpus_row = corpus_by_id[case_id]
        payloads.append(
            build_payload(
                source_row=source_row,
                corpus_row=corpus_row,
                session_char_limit=session_char_limit,
            )
        )
    return {
        "name": "fail18-answer-unit-witness-payloads",
        "artifact_kind": "provider_payloads",
        "prompt_version": PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_baseline": str(source_baseline_path),
        "source_corpus": str(corpus_path),
        "provider_visibility": {
            "gold_answer_visible": False,
            "answer_session_ids_visible": False,
            "failure_mode_visible": False,
        },
        "retrieval": {
            "method": "reuse saved answer-free lexical top-12 retrieval/session IDs",
            "top_k": int(source.get("retrieval", {}).get("top_k", 12)),
            "chars_per_session": session_char_limit,
        },
        "total": len(payloads),
        "payloads": payloads,
    }


def build_payload(
    *,
    source_row: dict[str, typing.Any],
    corpus_row: dict[str, typing.Any],
    session_char_limit: int = DEFAULT_SESSION_CHAR_LIMIT,
) -> dict[str, typing.Any]:
    id_to_session = dict(
        zip(
            corpus_row.get("haystack_session_ids", []),
            corpus_row.get("haystack_sessions", []),
            strict=True,
        )
    )
    id_to_date = dict(
        zip(
            corpus_row.get("haystack_session_ids", []),
            corpus_row.get("haystack_dates", []),
            strict=True,
        )
    )
    evidence_sessions = []
    for session_id in source_row.get("top_session_ids", []):
        session_id_str = str(session_id)
        messages = id_to_session.get(session_id_str)
        if messages is None:
            continue
        evidence_sessions.append(
            {
                "session_id": session_id_str,
                "date": str(id_to_date.get(session_id_str, "")),
                "text": _session_text(messages, char_limit=session_char_limit),
            }
        )
    return {
        "task": (
            "Answer the question by emitting a checkable answer-unit witness. "
            "Do not emit neutral facts or Datalog. The answer must be the simple "
            "aggregation over the units you list."
        ),
        "prompt_version": PROMPT_VERSION,
        "contract": [
            "Use only the evidence sessions in this payload.",
            "List every plausible answer-bearing unit needed for the numeric answer.",
            "Each unit decision must be include or exclude.",
            "Set borderline=true when the unit is a plausible swing vote.",
            (
                "Every unit must cite exactly one evidence_session_id from the "
                "payload and one short exact evidence_span from that session."
            ),
            (
                "For count questions, aggregation=count_included and "
                "answer_number is the number of include units."
            ),
            (
                "For sum questions, aggregation=sum_included and each included "
                "unit must have a numeric value."
            ),
            (
                "For lookup/scalar questions, aggregation=lookup_value and "
                "exactly one included unit must carry the numeric answer value."
            ),
            (
                "Do not include hidden labels, hidden answer ids, or final "
                "prose outside JSON."
            ),
        ],
        "output_schema": {
            "case_id": str(source_row.get("case_id", "")),
            "answer_variable": "short noun phrase for what is counted/summed/looked up",
            "aggregation": "count_included|sum_included|lookup_value",
            "units": [
                {
                    "unit_id": "stable string unique within this response",
                    "label": "short unit label",
                    "decision": "include|exclude",
                    "borderline": False,
                    "value": "number for sum/lookup units, otherwise null",
                    "unit": "unit label such as days/dollars/hours/points or null",
                    "evidence_session_id": "one provided session_id",
                    "evidence_span": "short exact span copied from that session",
                    "reason_code": "stable snake_case string",
                    "reason": "one short reason for include/exclude",
                }
            ],
            "answer_number": 0,
            "rationale": "one terse sentence explaining the aggregation",
        },
        "case": {
            "id": str(source_row.get("case_id", "")),
            "question": str(source_row.get("question", "")),
            "question_date": str(corpus_row.get("question_date", "")),
            "evidence_sessions": evidence_sessions,
        },
    }


def parse_witness_response(
    raw_output: str,
    *,
    expected_case_id: str | None = None,
) -> WitnessParseResult:
    if not raw_output.strip():
        return _invalid_result(
            case_id=expected_case_id or "",
            status=PARSE_STATUS_EMPTY,
            errors=("empty provider output",),
        )
    result_text = _provider_result_text(raw_output)
    decoded = _decode_first_json_object(result_text)
    if decoded is None:
        return _invalid_result(
            case_id=expected_case_id or "",
            status=PARSE_STATUS_INVALID_JSON,
            errors=("could not parse JSON object from provider output",),
            provider_result_text=result_text,
        )
    if not isinstance(decoded, dict):
        return _invalid_result(
            case_id=expected_case_id or "",
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=("root output must be a JSON object",),
            provider_result_text=result_text,
        )
    return parse_witness_object(
        decoded,
        expected_case_id=expected_case_id,
        provider_result_text=result_text,
    )


def parse_witness_object(
    raw: dict[str, typing.Any],
    *,
    expected_case_id: str | None = None,
    provider_result_text: str = "",
) -> WitnessParseResult:
    fallback_case_id = expected_case_id or ""
    errors: list[str] = []
    case_id = _string_field(raw, "case_id", errors)
    if expected_case_id is not None and case_id and case_id != expected_case_id:
        errors.append(
            f"case_id {case_id!r} does not match expected {expected_case_id!r}"
        )
    answer_variable = _string_field(raw, "answer_variable", errors)
    aggregation = _string_field(raw, "aggregation", errors)
    if aggregation and aggregation not in ALLOWED_AGGREGATIONS:
        errors.append(f"unknown aggregation {aggregation!r}")
    units = _answer_units(raw.get("units"), errors)
    _validate_unit_ids(units, errors)
    answer_number = _number_field(raw, "answer_number", errors, allow_null=True)
    rationale = _string_field(raw, "rationale", errors)
    if errors:
        return _invalid_result(
            case_id=case_id or fallback_case_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=tuple(errors),
            provider_result_text=provider_result_text,
        )
    return WitnessParseResult(
        case_id=case_id,
        parse_status=PARSE_STATUS_PARSED,
        answer_variable=answer_variable,
        aggregation=aggregation,
        units=tuple(units),
        answer_number=answer_number,
        rationale=rationale,
        parse_errors=(),
        provider_result_text=provider_result_text,
    )


def build_provider_prompt(payload: dict[str, typing.Any]) -> str:
    return (
        "Complete this answer-unit witness task.\n"
        "Return exactly one strict JSON object matching output_schema. "
        "No markdown, comments, or trailing text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def run_payloads(
    *,
    payload_artifact: dict[str, typing.Any],
    provider_command: str = DEFAULT_PROVIDER_COMMAND,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    samples: int = 1,
    limit: int = 0,
) -> list[dict[str, typing.Any]]:
    payloads = list(payload_artifact.get("payloads", []))
    if limit > 0:
        payloads = payloads[:limit]
    rows: list[dict[str, typing.Any]] = []
    prompt_version = str(payload_artifact.get("prompt_version", PROMPT_VERSION))
    for sample_index in range(1, samples + 1):
        for payload in payloads:
            case_id = str(payload.get("case", {}).get("id", ""))
            provider_result = interpretation_runner.run_provider(
                command=provider_command,
                prompt=build_provider_prompt(payload),
                timeout_seconds=timeout_seconds,
            )
            parsed = parse_witness_response(
                provider_result.raw_output,
                expected_case_id=case_id,
            )
            row = parsed.to_output_dict(
                provider=provider_command,
                prompt_version=prompt_version,
                raw_output=provider_result.raw_output,
                sample_index=sample_index,
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


def build_report(
    *,
    rows: list[dict[str, typing.Any]],
    payload_artifact: dict[str, typing.Any],
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    pipeline_report_path: str | pathlib.Path | None = None,
) -> dict[str, typing.Any]:
    manifest_by_id = {
        str(row.get("question_id", "")): row
        for row in ambiguity_fail18.load_manifest(manifest_path)
    }
    payload_by_id = {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    }
    expected_case_ids = sorted(payload_by_id)
    parsed_rows = [
        row
        for row in rows
        if row.get("parse_status") == PARSE_STATUS_PARSED
        and not _provider_failed(row)
    ]
    reviews = [
        review_witness_row(
            row,
            manifest_by_id.get(str(row.get("case_id", ""))),
            payload_by_id.get(str(row.get("case_id", ""))),
        )
        for row in rows
    ]
    grouped = _group_reviews(reviews, expected_case_ids)
    pipeline_comparison = _pipeline_comparison(grouped, pipeline_report_path)
    total_latency = sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in rows)
    samples = sorted({int(row.get("sample_index", 0) or 0) for row in rows})
    sample_exact = {
        str(sample): sum(
            1
            for review in reviews
            if review["sample_index"] == sample
            and review["recomputed_answer_matches_gold"]
        )
        for sample in samples
    }
    return {
        "name": "fail18-answer-unit-witness-report",
        "artifact_kind": "answer_unit_witness_report",
        "prompt_version": str(payload_artifact.get("prompt_version", PROMPT_VERSION)),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "provider": str(rows[0].get("provider", "")) if rows else "",
        "source_payload_artifact": str(payloads_path),
        "source_outputs_artifact": str(outputs_path),
        "source_manifest": str(manifest_path),
        "source_baseline": str(payload_artifact.get("source_baseline", "")),
        "rows_total": len(rows),
        "rows_expected": len(expected_case_ids) * max(len(samples), 1),
        "samples": samples,
        "rows_parsed": len(parsed_rows),
        "rows_failed_parse": len(rows) - len(parsed_rows),
        "rows_provider_timed_out": sum(
            1 for row in rows if bool(row.get("provider_timed_out", False))
        ),
        "sample_exact_matches": sample_exact,
        "support_exact_matches": sum(
            1 for case in grouped if case["gold_in_answer_support"]
        ),
        "bucket_counts": _bucket_counts(grouped),
        "verifier_summary": {
            "rows_arithmetic_passed": sum(
                1 for review in reviews if review["answer_matches_recomputed"]
            ),
            "rows_with_exclusion_contradictions": sum(
                1 for review in reviews if review["exclusion_contradictions"]
            ),
            "rows_all_included_spans_resolve": sum(
                1 for review in reviews if review["all_included_spans_resolve"]
            ),
            "rows_all_unit_spans_resolve": sum(
                1 for review in reviews if review["all_unit_spans_resolve"]
            ),
            "issue_counts": _issue_counts(reviews),
        },
        "provider_cost_or_latency_if_available": {
            "total_latency_seconds": round(total_latency, 3),
            "average_latency_seconds": round(total_latency / len(rows), 3)
            if rows
            else 0.0,
        },
        "pipeline_comparison": pipeline_comparison,
        "cases": grouped,
        "sample_reviews": reviews,
        "acceptance": {
            "outputs_cover_expected_case_sample_grid": (
                len(rows) == len(expected_case_ids) * max(len(samples), 1)
            ),
            "raw_provider_output_retained": all("raw_output" in row for row in rows),
            "answer_recomputed_from_units": True,
            "gold_hidden_from_provider_payload": _payloads_hide_gold(payload_artifact),
        },
    }


def review_witness_row(
    row: dict[str, typing.Any],
    manifest_row: dict[str, typing.Any] | None,
    payload: dict[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    issues: list[str] = []
    warnings: list[str] = []
    if _provider_failed(row):
        issues.append("provider_failed")
    if row.get("parse_status") != PARSE_STATUS_PARSED:
        issues.append("parse_failed")
    units = [unit for unit in row.get("units", []) if isinstance(unit, dict)]
    included_units = [unit for unit in units if str(unit.get("decision")) == "include"]
    if not units:
        issues.append("no_units")
    if not included_units:
        warnings.append("no_included_units")
    recomputed_answer, recompute_issues = recompute_answer(row)
    issues.extend(recompute_issues)
    answer_number = _optional_float(row.get("answer_number"))
    answer_matches_recomputed = _numbers_match(answer_number, recomputed_answer)
    if row.get("parse_status") == PARSE_STATUS_PARSED and not answer_matches_recomputed:
        issues.append("answer_number_mismatch")
    span_results = resolve_spans(row, payload)
    unresolved_unit_spans = [
        item["unit_id"] for item in span_results if not item["span_resolves"]
    ]
    unresolved_included_spans = [
        item["unit_id"]
        for item in span_results
        if item["decision"] == "include" and not item["span_resolves"]
    ]
    if unresolved_unit_spans:
        issues.append("unit_span_unresolved")
    exclusion_contradictions = find_exclusion_contradictions(row, payload)
    if exclusion_contradictions:
        issues.append("excluded_unit_contradicted")
    corrected_unit_ids = {
        str(item["unit_id"])
        for item in exclusion_contradictions
        if str(item.get("contradiction_type", "")) == "acquisition_evidence_present"
    }
    effective_answer, effective_issues = recompute_answer(
        row,
        forced_include_unit_ids=corrected_unit_ids,
    )
    issues.extend(effective_issues)
    gold_value = (
        float(ambiguity_fail18.numeric_gold(manifest_row))
        if manifest_row is not None
        and ambiguity_fail18.numeric_gold(manifest_row) is not None
        else None
    )
    if gold_value is None:
        warnings.append("gold_value_unknown")
    recomputed_matches_gold = _numbers_match(effective_answer, gold_value)
    if gold_value is not None and not recomputed_matches_gold:
        issues.append("recomputed_answer_misses_gold")
    return {
        "case_id": str(row.get("case_id", "")),
        "sample_index": int(row.get("sample_index", 0) or 0),
        "question": str(manifest_row.get("question", "")) if manifest_row else "",
        "gold_value": gold_value,
        "answer_variable": str(row.get("answer_variable", "")),
        "aggregation": str(row.get("aggregation", "")),
        "unit_count": len(units),
        "included_unit_count": len(included_units),
        "excluded_unit_count": sum(
            1 for unit in units if str(unit.get("decision")) == "exclude"
        ),
        "borderline_unit_count": sum(
            1 for unit in units if bool(unit.get("borderline"))
        ),
        "provider_answer_number": answer_number,
        "provider_recomputed_answer": recomputed_answer,
        "recomputed_answer": effective_answer,
        "verifier_corrected_answer": effective_answer
        if corrected_unit_ids
        else None,
        "exclusion_contradictions": exclusion_contradictions,
        "corrected_exclusion_count": len(corrected_unit_ids),
        "answer_matches_recomputed": answer_matches_recomputed,
        "recomputed_answer_matches_gold": recomputed_matches_gold,
        "all_unit_spans_resolve": not unresolved_unit_spans,
        "all_included_spans_resolve": not unresolved_included_spans,
        "span_results": span_results,
        "included_units": [
            {
                "unit_id": str(unit.get("unit_id", "")),
                "label": str(unit.get("label", "")),
                "value": unit.get("value"),
                "unit": unit.get("unit"),
                "borderline": bool(unit.get("borderline", False)),
                "evidence_session_id": str(unit.get("evidence_session_id", "")),
                "evidence_span": str(unit.get("evidence_span", "")),
                "reason_code": str(unit.get("reason_code", "")),
            }
            for unit in included_units
        ],
        "quality_issues": sorted(set(issues)),
        "warning_issues": sorted(set(warnings)),
    }


def _recompute_answer_from_units(
    aggregation: str,
    included_units: list[dict[str, typing.Any]],
) -> tuple[float | None, list[str]]:
    if aggregation == "count_included":
        return float(len(included_units)), []
    if aggregation == "sum_included":
        values = [_optional_float(unit.get("value")) for unit in included_units]
        if any(value is None for value in values):
            return None, ["included_sum_unit_missing_numeric_value"]
        return float(sum(typing.cast("list[float]", values))), []
    if aggregation == "lookup_value":
        values = [_optional_float(unit.get("value")) for unit in included_units]
        numeric_values = [value for value in values if value is not None]
        if len(numeric_values) != 1:
            return None, ["lookup_requires_exactly_one_numeric_included_unit"]
        return float(numeric_values[0]), []
    return None, ["unsupported_aggregation"]


def recompute_answer(
    row: dict[str, typing.Any],
    *,
    forced_include_unit_ids: set[str] | None = None,
) -> tuple[float | None, list[str]]:
    aggregation = str(row.get("aggregation", ""))
    units = [unit for unit in row.get("units", []) if isinstance(unit, dict)]
    forced = forced_include_unit_ids or set()
    included_units = [
        unit
        for unit in units
        if str(unit.get("decision")) == "include"
        or str(unit.get("unit_id", "")) in forced
    ]
    return _recompute_answer_from_units(aggregation, included_units)


def resolve_spans(
    row: dict[str, typing.Any],
    payload: dict[str, typing.Any] | None,
) -> list[dict[str, typing.Any]]:
    sessions = {
        str(session.get("session_id", "")): str(session.get("text", ""))
        for session in (payload or {}).get("case", {}).get("evidence_sessions", [])
        if isinstance(session, dict)
    }
    results: list[dict[str, typing.Any]] = []
    for unit in row.get("units", []):
        if not isinstance(unit, dict):
            continue
        session_id = str(unit.get("evidence_session_id", ""))
        span = str(unit.get("evidence_span", ""))
        text = sessions.get(session_id, "")
        results.append(
            {
                "unit_id": str(unit.get("unit_id", "")),
                "decision": str(unit.get("decision", "")),
                "evidence_session_id": session_id,
                "session_exists": session_id in sessions,
                "span_resolves": bool(text) and _normalized_contains(text, span),
            }
        )
    return results


def find_exclusion_contradictions(
    row: dict[str, typing.Any],
    payload: dict[str, typing.Any] | None,
) -> list[dict[str, typing.Any]]:
    if str(row.get("aggregation", "")) != "count_included":
        return []
    sessions = {
        str(session.get("session_id", "")): str(session.get("text", ""))
        for session in (payload or {}).get("case", {}).get("evidence_sessions", [])
        if isinstance(session, dict)
    }
    contradictions: list[dict[str, typing.Any]] = []
    for unit in row.get("units", []):
        if not isinstance(unit, dict):
            continue
        if str(unit.get("decision", "")) != "exclude":
            continue
        reason = _canonical_exclude_reason(str(unit.get("reason_code", "")))
        if reason != "no_acquisition_evidence":
            continue
        session_id = str(unit.get("evidence_session_id", ""))
        session_text = sessions.get(session_id, "")
        span = _find_acquisition_span(str(unit.get("label", "")), session_text)
        if span is None:
            continue
        contradictions.append(
            {
                "unit_id": str(unit.get("unit_id", "")),
                "label": str(unit.get("label", "")),
                "evidence_session_id": session_id,
                "exclude_reason": reason,
                "contradiction_type": "acquisition_evidence_present",
                "contradicting_span": span,
            }
        )
    return contradictions


def _group_reviews(
    reviews: list[dict[str, typing.Any]],
    expected_case_ids: list[str],
) -> list[dict[str, typing.Any]]:
    by_case: dict[str, list[dict[str, typing.Any]]] = {
        case_id: [] for case_id in expected_case_ids
    }
    for review in reviews:
        by_case.setdefault(str(review.get("case_id", "")), []).append(review)
    grouped: list[dict[str, typing.Any]] = []
    for case_id in sorted(by_case):
        case_reviews = sorted(
            by_case[case_id], key=lambda item: int(item.get("sample_index", 0) or 0)
        )
        predictions = [
            review.get("recomputed_answer")
            for review in case_reviews
            if review.get("recomputed_answer") is not None
        ]
        prediction_histogram = dict(
            collections.Counter(_answer_key(p) for p in predictions)
        )
        match_count = sum(
            1 for review in case_reviews if review["recomputed_answer_matches_gold"]
        )
        if case_reviews and match_count == len(case_reviews):
            bucket = "stably_right"
        elif match_count == 0:
            bucket = "stably_wrong"
        else:
            bucket = "flipping"
        grouped.append(
            {
                "case_id": case_id,
                "question": case_reviews[0]["question"] if case_reviews else "",
                "gold_value": case_reviews[0]["gold_value"] if case_reviews else None,
                "sample_count": len(case_reviews),
                "answer_support": sorted(prediction_histogram),
                "prediction_histogram": prediction_histogram,
                "gold_in_answer_support": match_count > 0,
                "support_match_count": match_count,
                "stability_bucket": bucket,
                "unit_decision_histogram": _unit_decision_histogram(case_reviews),
                "unstable_included_labels": _unstable_included_labels(case_reviews),
                "all_samples_arithmetic_passed": all(
                    review["answer_matches_recomputed"] for review in case_reviews
                )
                if case_reviews
                else False,
                "all_samples_included_spans_resolve": all(
                    review["all_included_spans_resolve"] for review in case_reviews
                )
                if case_reviews
                else False,
            }
        )
    return grouped


def _pipeline_comparison(
    grouped: list[dict[str, typing.Any]],
    pipeline_report_path: str | pathlib.Path | None,
) -> dict[str, typing.Any]:
    if not pipeline_report_path:
        return {}
    path = pathlib.Path(pipeline_report_path)
    if not path.exists():
        return {"error": f"pipeline report not found: {path}"}
    report = _load_json(path)
    pipeline_by_id = {
        str(case.get("case_id", "")): bool(case.get("recomputed_answer_matches_gold"))
        for case in report.get("cases", [])
        if isinstance(case, dict)
    }
    baseline_support_only = 0
    pipeline_only = 0
    both = 0
    neither = 0
    for case in grouped:
        baseline = bool(case["gold_in_answer_support"])
        pipeline = pipeline_by_id.get(str(case["case_id"]), False)
        if baseline and pipeline:
            both += 1
        elif baseline:
            baseline_support_only += 1
        elif pipeline:
            pipeline_only += 1
        else:
            neither += 1
    return {
        "baseline_support_only": baseline_support_only,
        "pipeline_only": pipeline_only,
        "both": both,
        "neither": neither,
        "mcnemar_exact_two_sided_p": _exact_mcnemar(
            baseline_support_only, pipeline_only
        ),
    }


def _provider_result_text(raw_output: str) -> str:
    decoded = _decode_json(raw_output)
    if isinstance(decoded, dict) and isinstance(decoded.get("result"), str):
        return str(decoded["result"])
    return raw_output


def _decode_first_json_object(text: str) -> typing.Any:
    stripped = text.strip()
    if not stripped:
        return None
    decoded = _decode_json(stripped)
    if decoded is not None:
        return decoded
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    if fence:
        decoded = _decode_json(fence.group(1))
        if decoded is not None:
            return decoded
    brace = re.search(r"\{.*\}", stripped, flags=re.S)
    if brace:
        return _decode_json(brace.group(0))
    return None


def _decode_json(text: str) -> typing.Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _answer_units(raw_units: typing.Any, errors: list[str]) -> list[AnswerUnit]:
    if not isinstance(raw_units, list) or not raw_units:
        errors.append("units must be a non-empty list")
        return []
    units: list[AnswerUnit] = []
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, dict):
            errors.append(f"units[{index}] must be a JSON object")
            continue
        unit_errors: list[str] = []
        unit_id = _string_field(raw, "unit_id", unit_errors)
        label = _string_field(raw, "label", unit_errors)
        decision = _string_field(raw, "decision", unit_errors)
        if decision and decision not in ALLOWED_DECISIONS:
            unit_errors.append(f"unknown decision {decision!r}")
        borderline = _bool_field(raw, "borderline", unit_errors)
        value = _number_field(raw, "value", unit_errors, allow_null=True)
        unit = _nullable_string_field(raw, "unit", unit_errors)
        evidence_session_id = _string_field(raw, "evidence_session_id", unit_errors)
        evidence_span = _string_field(raw, "evidence_span", unit_errors)
        reason_code = _string_field(raw, "reason_code", unit_errors)
        reason = _string_field(raw, "reason", unit_errors)
        if unit_errors:
            errors.extend(f"units[{index}]: {error}" for error in unit_errors)
            continue
        units.append(
            AnswerUnit(
                unit_id=unit_id,
                label=label,
                decision=decision,
                borderline=borderline,
                value=value,
                unit=unit,
                evidence_session_id=evidence_session_id,
                evidence_span=evidence_span,
                reason_code=reason_code,
                reason=reason,
            )
        )
    return units


def _validate_unit_ids(units: list[AnswerUnit], errors: list[str]) -> None:
    counts = collections.Counter(unit.unit_id for unit in units)
    duplicate_ids = sorted(unit_id for unit_id, count in counts.items() if count > 1)
    if duplicate_ids:
        errors.append("duplicate unit_id values: " + ", ".join(duplicate_ids))


def _invalid_result(
    *,
    case_id: str,
    status: str,
    errors: tuple[str, ...],
    provider_result_text: str = "",
) -> WitnessParseResult:
    return WitnessParseResult(
        case_id=case_id,
        parse_status=status,
        answer_variable="",
        aggregation="",
        units=(),
        answer_number=None,
        rationale="",
        parse_errors=errors,
        provider_result_text=provider_result_text,
    )


def _session_text(
    messages: list[dict[str, typing.Any]],
    *,
    char_limit: int,
) -> str:
    chunks = [
        f"{str(message.get('role', '')).upper()}: {message.get('content', '')!s}"
        for message in messages
        if isinstance(message, dict)
    ]
    text = "\n".join(chunks)
    if len(text) > char_limit:
        return f"{text[:char_limit]}\n...[truncated]"
    return text


def _normalized_contains(text: str, span: str) -> bool:
    if not span.strip():
        return False
    return _normalize_for_span(span) in _normalize_for_span(text)


def _normalize_for_span(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def _canonical_exclude_reason(reason_code: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", reason_code.casefold()).strip("_")
    if normalized in NO_ACQUISITION_REASON_CODES or normalized.startswith(
        "no_acquisition"
    ):
        return "no_acquisition_evidence"
    return normalized


def _find_acquisition_span(label: str, session_text: str) -> str | None:
    label_terms = _label_search_terms(label)
    if not label_terms or not session_text.strip():
        return None
    normalized_text = _normalize_for_span(session_text)
    for term in label_terms:
        escaped_term = re.escape(term)
        verb_pattern = "|".join(re.escape(verb) for verb in ACQUISITION_VERBS)
        patterns = [
            rf"\b{escaped_term}\b\s*,?\s*(?:which|that)\s+i\s+"
            rf"\b(?:{verb_pattern})\b",
            rf"\bi\s+\b(?:{verb_pattern})\b.{{0,80}}\b{escaped_term}\b",
            rf"\b(?:{verb_pattern})\b.{{0,80}}\b{escaped_term}\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized_text, flags=re.I)
            if not match:
                continue
            start = max(0, match.start() - 80)
            end = min(len(normalized_text), match.end() + 120)
            candidate = normalized_text[start:end]
            if _acquisition_span_is_blocked(candidate):
                continue
            return candidate.strip()
    return None


def _label_search_terms(label: str) -> list[str]:
    normalized = _normalize_for_span(re.sub(r"\([^)]*\)", " ", label))
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return []
    terms = [normalized]
    if normalized.endswith(" plant"):
        terms.append(normalized.removesuffix(" plant").strip())
    return [term for term in dict.fromkeys(terms) if len(term) >= 3]


def _acquisition_span_is_blocked(span: str) -> bool:
    return bool(ACQUISITION_NEGATION_RE.search(span))


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


def _bool_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        errors.append(f"{key} must be a boolean")
        return False
    return value


def _number_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
    *,
    allow_null: bool,
) -> float | None:
    value = raw.get(key)
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


def _answer_key(value: typing.Any) -> str:
    if value is None:
        return "null"
    numeric = _optional_float(value)
    if numeric is not None and numeric.is_integer():
        return str(int(numeric))
    return str(value)


def _label_key(value: typing.Any) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _unit_decision_histogram(
    case_reviews: list[dict[str, typing.Any]],
) -> dict[str, dict[str, int]]:
    histogram: dict[str, dict[str, int]] = {}
    for review in case_reviews:
        for unit in review.get("included_units", []):
            label = _label_key(unit.get("label", ""))
            if not label:
                continue
            histogram.setdefault(label, {"include": 0, "exclude": 0})
            histogram[label]["include"] += 1
    return dict(sorted(histogram.items()))


def _unstable_included_labels(case_reviews: list[dict[str, typing.Any]]) -> list[str]:
    sample_count = len(case_reviews)
    if sample_count <= 1:
        return []
    histogram = _unit_decision_histogram(case_reviews)
    return sorted(
        label
        for label, counts in histogram.items()
        if 0 < int(counts.get("include", 0)) < sample_count
    )


def _bucket_counts(grouped: list[dict[str, typing.Any]]) -> dict[str, int]:
    counts = collections.Counter(str(case["stability_bucket"]) for case in grouped)
    return dict(sorted(counts.items()))


def _issue_counts(reviews: list[dict[str, typing.Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for review in reviews:
        for issue in review.get("quality_issues", []):
            counts[str(issue)] = counts.get(str(issue), 0) + 1
    return dict(sorted(counts.items()))


def _payloads_hide_gold(payload_artifact: dict[str, typing.Any]) -> bool:
    banned_keys = {
        "answer_session_ids",
        "failure_mode",
        "gold",
        "gold_answer",
        "gold_count",
        "gold_value",
    }
    return not _contains_banned_key(payload_artifact.get("payloads", []), banned_keys)


def _contains_banned_key(value: typing.Any, banned_keys: set[str]) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in banned_keys:
                return True
            if _contains_banned_key(child, banned_keys):
                return True
    elif isinstance(value, list):
        return any(_contains_banned_key(item, banned_keys) for item in value)
    return False


def _exact_mcnemar(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    low = min(b, c)
    tail = sum(math.comb(total, idx) * (0.5**total) for idx in range(low + 1))
    return round(min(1.0, 2 * tail), 6)


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument(
        "--source-baseline",
        type=pathlib.Path,
        default=DEFAULT_SOURCE_BASELINE_PATH,
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_CORPUS,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument("--payloads", type=pathlib.Path, default=DEFAULT_PAYLOADS_PATH)
    parser.add_argument("--outputs", type=pathlib.Path, default=DEFAULT_OUTPUTS_PATH)
    parser.add_argument("--report", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER_COMMAND)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--session-char-limit",
        type=int,
        default=DEFAULT_SESSION_CHAR_LIMIT,
    )
    parser.add_argument("--pipeline-report", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    if not args.build_payloads and not args.run_provider and not args.build_report:
        parser.print_help()
        return 2

    if args.build_payloads:
        artifact = build_fail18_payload_artifact(
            source_baseline_path=args.source_baseline,
            corpus_path=args.corpus,
            session_char_limit=args.session_char_limit,
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
            samples=args.samples,
            limit=args.limit,
        )
        interpretation_runner.write_jsonl(args.outputs, rows)
    else:
        rows = interpretation_runner.load_jsonl(args.outputs)

    if args.run_provider or args.build_report:
        report = build_report(
            rows=rows,
            payload_artifact=payload_artifact,
            outputs_path=args.outputs,
            payloads_path=args.payloads,
            manifest_path=args.manifest,
            pipeline_report_path=args.pipeline_report,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

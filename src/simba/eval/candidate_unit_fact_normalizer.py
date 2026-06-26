"""Normalize residual ``other`` facts into generic recursive facts."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import typing

from simba.eval import interpretation_runner

PROMPT_VERSION = "candidate_unit_other_fact_normalizer_v1"
DEFAULT_FORMALIZER_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_formalizer_outputs_recursive_v2.jsonl"
)
DEFAULT_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_other_normalizer_payloads_recursive_v2.json"
)
DEFAULT_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_other_normalizer_outputs_recursive_v2.jsonl"
)
DEFAULT_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_other_normalizer_report_recursive_v2.json"
)
DEFAULT_NORMALIZED_FORMALIZER_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_candidate_unit_formalizer_outputs_recursive_v2_normalized.jsonl"
)

PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_INVALID_JSON = "invalid_json"
PARSE_STATUS_INVALID_SCHEMA = "invalid_schema"
PARSE_STATUS_EMPTY = "empty"

DECISION_REPLACE = "replace"
DECISION_KEEP_OTHER = "keep_other"
_ALLOWED_DECISIONS = {DECISION_REPLACE, DECISION_KEEP_OTHER}
_GENERIC_PREDICATES = {
    "action",
    "coreference",
    "distinct",
    "entity",
    "event",
    "object_type",
    "property",
    "quantity",
    "relation",
    "sortal",
    "status",
    "time",
    "value",
}
_FORBIDDEN_TERMS = (
    "supports_answer",
    "contradicts_answer",
    "gold_answer",
    "gold_value",
    "answer_session_ids",
    "raw_session_ids",
    "failure_mode",
)

FACT_SCHEMA = (
    {
        "predicate": "action",
        "arguments": ["subject", "object", "verb", "location", "status"],
        "meaning": "A grounded action, request, plan, task, or obligation.",
    },
    {
        "predicate": "event",
        "arguments": ["event", "type", "participants", "location", "date", "status"],
        "meaning": "A grounded event mention.",
    },
    {
        "predicate": "object_type",
        "arguments": ["entity", "type"],
        "meaning": "A type assertion for an entity or object.",
    },
    {
        "predicate": "sortal",
        "arguments": ["entity", "type", "source", "antecedent", "licensed_by"],
        "meaning": (
            "A sortal/type inherited by bridging or ellipsis without claiming "
            "same-token identity."
        ),
    },
    {
        "predicate": "property",
        "arguments": ["entity", "property", "value"],
        "meaning": "A descriptive property attached to an entity.",
    },
    {
        "predicate": "quantity",
        "arguments": ["entity", "attribute", "value", "unit"],
        "meaning": "A count, amount, or measurement.",
    },
    {
        "predicate": "relation",
        "arguments": ["source", "relation", "target"],
        "meaning": "A grounded relation between two entities.",
    },
    {
        "predicate": "status",
        "arguments": ["entity", "status"],
        "meaning": "A state such as pending, completed, cancelled, owned, or borrowed.",
    },
    {
        "predicate": "time",
        "arguments": ["entity", "date", "time_window"],
        "meaning": "A date, interval, or temporal qualifier.",
    },
    {
        "predicate": "value",
        "arguments": ["entity", "attribute", "value", "unit"],
        "meaning": "A scalar value attached to an entity or event.",
    },
    {
        "predicate": "coreference",
        "arguments": ["entity", "same_as", "reason"],
        "meaning": (
            "A local same-entity identity claim inside one evidence session. "
            "Do not use this for bridging, sortal inheritance, or replacement."
        ),
    },
    {
        "predicate": "distinct",
        "arguments": ["a", "b", "reason"],
        "meaning": (
            "A grounded non-identity claim, especially licensed by contrastive "
            "mentions such as new, another, different, replacement, or old."
        ),
    },
)


@dataclasses.dataclass(frozen=True)
class ReplacementFact:
    fact_id: str
    predicate: str
    arguments: dict[str, typing.Any]
    evidence_span: str
    confidence: float

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "fact_id": self.fact_id,
            "predicate": self.predicate,
            "arguments": dict(self.arguments),
            "evidence_span": self.evidence_span,
            "confidence": self.confidence,
        }


@dataclasses.dataclass(frozen=True)
class NormalizerParseResult:
    normalizer_id: str
    case_id: str
    evidence_session_id: str
    source_fact_id: str
    decision: str
    replacement_fact: ReplacementFact | None
    reason: str
    parse_status: str
    parse_errors: tuple[str, ...]

    def to_output_dict(
        self,
        *,
        provider: str,
        prompt_version: str,
        raw_output: str,
    ) -> dict[str, typing.Any]:
        return {
            "normalizer_id": self.normalizer_id,
            "case_id": self.case_id,
            "evidence_session_id": self.evidence_session_id,
            "source_fact_id": self.source_fact_id,
            "decision": self.decision,
            "replacement_fact": (
                self.replacement_fact.to_dict()
                if self.replacement_fact is not None
                else None
            ),
            "reason": self.reason,
            "parse_status": self.parse_status,
            "parse_errors": list(self.parse_errors),
            "provider": provider,
            "prompt_version": prompt_version,
            "raw_output": raw_output,
        }


def build_payload_artifact(
    *,
    formalizer_outputs_path: str | pathlib.Path = DEFAULT_FORMALIZER_OUTPUTS_PATH,
) -> dict[str, typing.Any]:
    formalizer_rows = interpretation_runner.load_jsonl(formalizer_outputs_path)
    payloads: list[dict[str, typing.Any]] = []
    for row in formalizer_rows:
        if row.get("parse_status") != "parsed" or _provider_failed(row):
            continue
        for fact in row.get("facts", []):
            if not isinstance(fact, dict) or fact.get("predicate") != "other":
                continue
            payloads.append(_build_payload(row=row, fact=fact))
    return {
        "name": "fail18-candidate-unit-other-normalizer-payloads",
        "artifact_kind": "provider_payloads",
        "gate": "candidate_unit_other_fact_normalizer",
        "gate_status": "normalizer_payloads_only_not_run",
        "prompt_version": PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_formalizer_outputs": str(formalizer_outputs_path),
        "provider_visibility": {
            "gold_answer_visible": False,
            "gold_value_visible": False,
            "failure_mode_visible": False,
            "answer_session_ids_visible": False,
            "raw_session_ids_visible": False,
            "candidate_unit_status_visible": False,
            "final_answer_requested": False,
            "answer_support_labels_requested": False,
        },
        "total": len(payloads),
        "case_ids": sorted({str(payload["case_id"]) for payload in payloads}),
        "payloads": payloads,
    }


def parse_normalizer_response(
    raw_output: str,
    *,
    expected_normalizer_id: str | None = None,
) -> NormalizerParseResult:
    fallback_id = expected_normalizer_id or ""
    if not raw_output.strip():
        return _invalid_result(
            normalizer_id=fallback_id,
            status=PARSE_STATUS_EMPTY,
            errors=("empty provider output",),
        )
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return _invalid_result(
            normalizer_id=fallback_id,
            status=PARSE_STATUS_INVALID_JSON,
            errors=(f"invalid JSON: {exc.msg} at char {exc.pos}",),
        )
    if not isinstance(decoded, dict):
        return _invalid_result(
            normalizer_id=fallback_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=("root output must be a JSON object",),
        )
    return parse_normalizer_object(
        decoded,
        expected_normalizer_id=expected_normalizer_id,
    )


def parse_normalizer_object(
    raw: dict[str, typing.Any],
    *,
    expected_normalizer_id: str | None = None,
) -> NormalizerParseResult:
    errors: list[str] = []
    normalizer_id = _string_field(raw, "normalizer_id", errors)
    if (
        expected_normalizer_id is not None
        and normalizer_id
        and normalizer_id != expected_normalizer_id
    ):
        errors.append(
            f"normalizer_id {normalizer_id!r} does not match expected "
            f"{expected_normalizer_id!r}"
        )
    case_id = _string_field(raw, "case_id", errors)
    evidence_session_id = _string_field(raw, "evidence_session_id", errors)
    source_fact_id = _string_field(raw, "source_fact_id", errors)
    decision = _string_field(raw, "decision", errors)
    if decision and decision not in _ALLOWED_DECISIONS:
        errors.append(f"unknown decision {decision!r}")
    reason = _string_field(raw, "reason", errors)
    replacement_fact = _replacement_fact_field(
        raw.get("replacement_fact"),
        decision=decision,
        errors=errors,
    )
    _append_forbidden_term_errors(raw, errors)
    if errors:
        return _invalid_result(
            normalizer_id=normalizer_id or expected_normalizer_id or "",
            case_id=case_id,
            evidence_session_id=evidence_session_id,
            source_fact_id=source_fact_id,
            status=PARSE_STATUS_INVALID_SCHEMA,
            errors=tuple(errors),
        )
    return NormalizerParseResult(
        normalizer_id=normalizer_id,
        case_id=case_id,
        evidence_session_id=evidence_session_id,
        source_fact_id=source_fact_id,
        decision=decision,
        replacement_fact=replacement_fact,
        reason=reason,
        parse_status=PARSE_STATUS_PARSED,
        parse_errors=(),
    )


def build_provider_prompt(payload: dict[str, typing.Any]) -> str:
    return (
        "Complete this residual-fact normalization task.\n"
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
    prompt_version = str(payload_artifact.get("prompt_version", PROMPT_VERSION))
    for payload in payloads:
        normalizer_id = str(payload.get("normalizer_id", ""))
        provider_result = interpretation_runner.run_provider(
            command=provider_command,
            prompt=build_provider_prompt(payload),
            timeout_seconds=timeout_seconds,
        )
        parsed = parse_normalizer_response(
            provider_result.raw_output,
            expected_normalizer_id=normalizer_id,
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


def build_report(
    *,
    payload_artifact: dict[str, typing.Any],
    normalizer_rows: list[dict[str, typing.Any]],
    formalizer_rows: list[dict[str, typing.Any]],
    outputs_path: str | pathlib.Path = DEFAULT_OUTPUTS_PATH,
    payloads_path: str | pathlib.Path = DEFAULT_PAYLOADS_PATH,
    normalized_formalizer_outputs_path: str
    | pathlib.Path = DEFAULT_NORMALIZED_FORMALIZER_OUTPUTS_PATH,
) -> dict[str, typing.Any]:
    expected_ids = [
        str(payload.get("normalizer_id", ""))
        for payload in payload_artifact.get("payloads", [])
        if isinstance(payload, dict)
    ]
    observed_ids = [str(row.get("normalizer_id", "")) for row in normalizer_rows]
    observed_counts = collections.Counter(observed_ids)
    parsed_rows = [
        row
        for row in normalizer_rows
        if row.get("parse_status") == PARSE_STATUS_PARSED
        and not _provider_failed(row)
    ]
    decisions = collections.Counter(str(row.get("decision", "")) for row in parsed_rows)
    predicates = collections.Counter(
        str(row.get("replacement_fact", {}).get("predicate", ""))
        for row in parsed_rows
        if isinstance(row.get("replacement_fact"), dict)
    )
    cases = _case_summary(parsed_rows)
    conversions = [
        _conversion_record(row)
        for row in parsed_rows
        if row.get("decision") == DECISION_REPLACE
    ]
    kept = [
        _conversion_record(row)
        for row in parsed_rows
        if row.get("decision") == DECISION_KEEP_OTHER
    ]
    return {
        "name": "fail18-candidate-unit-other-normalizer-report",
        "artifact_kind": "candidate_unit_other_normalizer_report",
        "gate": "candidate_unit_other_fact_normalizer",
        "gate_status": "normalizer_report_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "prompt_version": str(payload_artifact.get("prompt_version", PROMPT_VERSION)),
        "source_payloads": str(payloads_path),
        "source_outputs": str(outputs_path),
        "source_formalizer_outputs": str(
            payload_artifact.get("source_formalizer_outputs", "")
        ),
        "normalized_formalizer_outputs": str(normalized_formalizer_outputs_path),
        "summary": {
            "source_other_facts": len(expected_ids),
            "provider_rows_total": len(normalizer_rows),
            "provider_rows_parsed": len(parsed_rows),
            "provider_rows_failed": sum(
                1 for row in normalizer_rows if _provider_failed(row)
            ),
            "provider_rows_timed_out": sum(
                1 for row in normalizer_rows if bool(row.get("provider_timed_out"))
            ),
            "missing_normalizer_ids": sorted(set(expected_ids) - set(observed_ids)),
            "extra_normalizer_ids": sorted(set(observed_ids) - set(expected_ids)),
            "duplicate_normalizer_ids": sorted(
                row_id for row_id, count in observed_counts.items() if count > 1
            ),
            "decision_counts": dict(sorted(decisions.items())),
            "replacement_predicate_counts": dict(sorted(predicates.items())),
            "forbidden_label_violations": _forbidden_label_violations(parsed_rows),
            "normalized_other_count": _count_normalized_other_facts(
                formalizer_rows, parsed_rows
            ),
        },
        "cases": cases,
        "conversions": conversions,
        "kept_other": kept,
        "decision": {
            "next_slice": "generic_question_rule_compiler",
            "reason": (
                "This pass only normalizes residual fact shape. It must not be "
                "used as answer relevance, inclusion, exclusion, or aggregation."
            ),
        },
    }


def apply_normalizations(
    *,
    formalizer_rows: list[dict[str, typing.Any]],
    normalizer_rows: list[dict[str, typing.Any]],
) -> list[dict[str, typing.Any]]:
    replacements = {
        (
            str(row.get("case_id", "")),
            str(row.get("evidence_session_id", "")),
            str(row.get("source_fact_id", "")),
        ): row.get("replacement_fact")
        for row in normalizer_rows
        if row.get("parse_status") == PARSE_STATUS_PARSED
        and row.get("decision") == DECISION_REPLACE
        and isinstance(row.get("replacement_fact"), dict)
        and not _provider_failed(row)
    }
    normalized: list[dict[str, typing.Any]] = []
    for row in formalizer_rows:
        new_row = dict(row)
        facts = []
        for fact in row.get("facts", []):
            if not isinstance(fact, dict):
                facts.append(fact)
                continue
            key = (
                str(row.get("case_id", "")),
                str(row.get("evidence_session_id", "")),
                str(fact.get("fact_id", "")),
            )
            replacement = replacements.get(key)
            if replacement is None:
                facts.append(fact)
                continue
            replacement_fact = dict(replacement)
            replacement_fact["normalized_from_fact_id"] = str(fact.get("fact_id", ""))
            facts.append(replacement_fact)
        new_row["facts"] = facts
        normalized.append(new_row)
    return normalized


def _build_payload(
    *,
    row: dict[str, typing.Any],
    fact: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    case_id = str(row.get("case_id", ""))
    evidence_session_id = str(row.get("evidence_session_id", ""))
    fact_id = str(fact.get("fact_id", ""))
    normalizer_id = f"{case_id}::{evidence_session_id}::{fact_id}"
    return {
        "task": (
            "Normalize one residual other fact into a generic recursive fact if "
            "the source span supports it. Otherwise keep it as other."
        ),
        "prompt_version": PROMPT_VERSION,
        "normalizer_id": normalizer_id,
        "case_id": case_id,
        "evidence_session_id": evidence_session_id,
        "source_fact_id": fact_id,
        "fact_schema": list(FACT_SCHEMA),
        "contract": [
            "Use only this source_fact and its evidence_span.",
            (
                "If a generic predicate can faithfully represent the fact, "
                "return decision=replace."
            ),
            (
                "If no generic predicate fits without adding information, "
                "return decision=keep_other."
            ),
            "When decision=keep_other, replacement_fact must be JSON null.",
            (
                "When decision=replace, every replacement argument value must "
                "be a string, number, boolean, or JSON null. Omit unknown "
                "optional fields or use JSON null; never use an empty string "
                "as an unknown placeholder."
            ),
            (
                "Do not decide answer relevance, countability, inclusion, "
                "exclusion, correctness, or final answers."
            ),
            (
                "Do not use answer-path labels, gold labels, raw session labels, "
                "or candidate-unit status labels."
            ),
            (
                "The replacement predicate must be one generic predicate from "
                "fact_schema, never other."
            ),
            (
                "Use coreference only for true same-entity identity. For "
                "bridging or incomplete relational nouns, use sortal; for "
                "contrastive new/another/replacement mentions, use distinct "
                "and relation(..., relation='replaces', ...) when grounded."
            ),
            (
                "Keep entity symbols and type symbols disjoint. Relation endpoints, "
                "distinct arguments, sortal.entity, and sortal.antecedent are "
                "entity-sorted; sortal.type and object_type.type are type-sorted. "
                "Do not use a type word such as 'boots' as the entity endpoint for "
                "the old pair. Use a local handle such as pair_1, then type it with "
                "sortal(entity='pair_1', type='boots')."
            ),
        ],
        "source_fact": {
            "fact_id": fact_id,
            "predicate": "other",
            "arguments": fact.get("arguments", {}),
            "evidence_span": str(fact.get("evidence_span", "")),
            "confidence": fact.get("confidence"),
        },
        "output_schema": {
            "normalizer_id": normalizer_id,
            "case_id": case_id,
            "evidence_session_id": evidence_session_id,
            "source_fact_id": fact_id,
            "decision": "replace|keep_other",
            "replacement_fact": {
                "fact_id": "stable replacement id or null if keep_other",
                "predicate": "|".join(sorted(_GENERIC_PREDICATES)),
                "arguments": {"key": "value or null"},
                "evidence_span": "short exact span",
                "confidence": 0.0,
            },
            "reason": "short reason",
        },
    }


def _replacement_fact_field(
    raw: typing.Any,
    *,
    decision: str,
    errors: list[str],
) -> ReplacementFact | None:
    if decision == DECISION_KEEP_OTHER:
        if raw is not None:
            errors.append("replacement_fact must be null when decision=keep_other")
        return None
    if decision != DECISION_REPLACE:
        return None
    if not isinstance(raw, dict):
        errors.append("replacement_fact must be an object when decision=replace")
        return None
    fact_errors: list[str] = []
    fact_id = _string_field(raw, "fact_id", fact_errors)
    predicate = _string_field(raw, "predicate", fact_errors)
    if predicate and predicate not in _GENERIC_PREDICATES:
        fact_errors.append(f"replacement predicate must be generic, got {predicate!r}")
    arguments = _arguments_field(raw, "arguments", fact_errors)
    evidence_span = _string_field(raw, "evidence_span", fact_errors)
    confidence = _number_field(raw, "confidence", fact_errors)
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        fact_errors.append("confidence must be between 0.0 and 1.0")
    if fact_errors:
        errors.extend(f"replacement_fact: {error}" for error in fact_errors)
        return None
    return ReplacementFact(
        fact_id=fact_id,
        predicate=predicate,
        arguments=arguments,
        evidence_span=evidence_span,
        confidence=typing.cast("float", confidence),
    )


def _case_summary(rows: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
    by_case: dict[str, list[dict[str, typing.Any]]] = collections.defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id", ""))].append(row)
    cases = []
    for case_id, case_rows in sorted(by_case.items()):
        decisions = collections.Counter(
            str(row.get("decision", "")) for row in case_rows
        )
        predicates = collections.Counter(
            str(row.get("replacement_fact", {}).get("predicate", ""))
            for row in case_rows
            if isinstance(row.get("replacement_fact"), dict)
        )
        cases.append(
            {
                "case_id": case_id,
                "source_other_facts": len(case_rows),
                "decision_counts": dict(sorted(decisions.items())),
                "replacement_predicate_counts": dict(sorted(predicates.items())),
            }
        )
    return cases


def _conversion_record(row: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {
        "normalizer_id": str(row.get("normalizer_id", "")),
        "case_id": str(row.get("case_id", "")),
        "evidence_session_id": str(row.get("evidence_session_id", "")),
        "source_fact_id": str(row.get("source_fact_id", "")),
        "decision": str(row.get("decision", "")),
        "replacement_fact": row.get("replacement_fact"),
        "reason": str(row.get("reason", "")),
    }


def _count_normalized_other_facts(
    formalizer_rows: list[dict[str, typing.Any]],
    normalizer_rows: list[dict[str, typing.Any]],
) -> int:
    normalized = apply_normalizations(
        formalizer_rows=formalizer_rows,
        normalizer_rows=normalizer_rows,
    )
    return sum(
        1
        for row in normalized
        for fact in row.get("facts", [])
        if isinstance(fact, dict) and fact.get("predicate") == "other"
    )


def _forbidden_label_violations(rows: list[dict[str, typing.Any]]) -> int:
    count = 0
    for row in rows:
        visible = {
            key: value
            for key, value in row.items()
            if key not in {"raw_output", "provider", "provider_stderr"}
        }
        rendered = json.dumps(visible, sort_keys=True).lower()
        count += sum(1 for term in _FORBIDDEN_TERMS if term in rendered)
    return count


def _append_forbidden_term_errors(
    raw: dict[str, typing.Any],
    errors: list[str],
) -> None:
    visible = {
        key: value
        for key, value in raw.items()
        if key not in {"reason"}
    }
    rendered = json.dumps(visible, sort_keys=True).lower()
    for term in _FORBIDDEN_TERMS:
        if term in rendered:
            errors.append(f"forbidden answer-path term {term!r}")


def _string_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
    *,
    allow_empty: bool = False,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        errors.append(f"{key} must be a string")
        return ""
    if not allow_empty and not value.strip():
        errors.append(f"{key} must be non-empty")
        return ""
    return value


def _arguments_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> dict[str, typing.Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    bad_keys = [item for item in value if not isinstance(item, str)]
    if bad_keys:
        errors.append(f"{key} keys must be strings")
        return {}
    bad_values = [
        item
        for item in value.values()
        if isinstance(item, list | dict)
    ]
    if bad_values:
        errors.append(f"{key} values must be scalar or null")
        return {}
    return {
        str(item_key): _normalize_argument_value(item_value)
        for item_key, item_value in value.items()
    }


def _normalize_argument_value(value: typing.Any) -> typing.Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _number_field(
    raw: dict[str, typing.Any],
    key: str,
    errors: list[str],
) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{key} must be a number")
        return None
    return float(value)


def _invalid_result(
    *,
    normalizer_id: str,
    status: str,
    errors: tuple[str, ...],
    case_id: str = "",
    evidence_session_id: str = "",
    source_fact_id: str = "",
) -> NormalizerParseResult:
    parts = normalizer_id.split("::", 2)
    if len(parts) == 3:
        case_id = case_id or parts[0]
        evidence_session_id = evidence_session_id or parts[1]
        source_fact_id = source_fact_id or parts[2]
    return NormalizerParseResult(
        normalizer_id=normalizer_id,
        case_id=case_id,
        evidence_session_id=evidence_session_id,
        source_fact_id=source_fact_id,
        decision="",
        replacement_fact=None,
        reason="",
        parse_status=status,
        parse_errors=errors,
    )


def _provider_failed(row: dict[str, typing.Any]) -> bool:
    return (
        int(row.get("provider_exit_code", 0) or 0) != 0
        or bool(row.get("provider_timed_out", False))
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--run-provider", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument(
        "--formalizer-outputs",
        type=pathlib.Path,
        default=DEFAULT_FORMALIZER_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--normalizer-payloads",
        type=pathlib.Path,
        default=DEFAULT_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--normalizer-outputs",
        type=pathlib.Path,
        default=DEFAULT_OUTPUTS_PATH,
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--normalized-formalizer-outputs",
        type=pathlib.Path,
        default=DEFAULT_NORMALIZED_FORMALIZER_OUTPUTS_PATH,
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.build_payloads and not args.run_provider and not args.build_report:
        print("Specify --build-payloads, --run-provider, or --build-report")
        return 2

    if args.build_payloads:
        payload_artifact = build_payload_artifact(
            formalizer_outputs_path=args.formalizer_outputs,
        )
        args.normalizer_payloads.parent.mkdir(parents=True, exist_ok=True)
        args.normalizer_payloads.write_text(
            f"{json.dumps(payload_artifact, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        if not args.run_provider and not args.build_report:
            return 0
    else:
        payload_artifact = interpretation_runner.load_payload_artifact(
            args.normalizer_payloads
        )

    if args.run_provider:
        rows = run_payloads(
            payload_artifact=payload_artifact,
            provider_command=args.provider,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
        interpretation_runner.write_jsonl(args.normalizer_outputs, rows)
    else:
        rows = interpretation_runner.load_jsonl(args.normalizer_outputs)

    if args.run_provider or args.build_report:
        formalizer_rows = interpretation_runner.load_jsonl(args.formalizer_outputs)
        normalized_rows = apply_normalizations(
            formalizer_rows=formalizer_rows,
            normalizer_rows=rows,
        )
        interpretation_runner.write_jsonl(
            args.normalized_formalizer_outputs,
            normalized_rows,
        )
        report = build_report(
            payload_artifact=payload_artifact,
            normalizer_rows=rows,
            formalizer_rows=formalizer_rows,
            outputs_path=args.normalizer_outputs,
            payloads_path=args.normalizer_payloads,
            normalized_formalizer_outputs_path=args.normalized_formalizer_outputs,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Parse provider outputs into ambiguity interpretation records."""

from __future__ import annotations

import dataclasses
import json
import typing
from collections import Counter

from simba.eval.interpretation_ir import InterpretationRecord

PARSE_STATUS_PARSED = "parsed"
PARSE_STATUS_INVALID_JSON = "invalid_json"
PARSE_STATUS_INVALID_SCHEMA = "invalid_schema"
PARSE_STATUS_EMPTY = "empty"


@dataclasses.dataclass(frozen=True)
class InterpretationParseResult:
    case_id: str
    parse_status: str
    interpretations: tuple[InterpretationRecord, ...]
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
            "interpretations": [
                item.to_dict() for item in self.interpretations
            ],
            "parse_errors": list(self.parse_errors),
        }


def parse_interpretation_response(
    raw_output: str,
    *,
    expected_case_id: str | None = None,
) -> InterpretationParseResult:
    """Strictly parse one provider response.

    The provider contract requires a single JSON object. Markdown fences,
    prefatory prose, partial interpretation arrays, and schema-invalid
    interpretations are reported as failures instead of being repaired here.
    """
    fallback_case_id = expected_case_id or ""
    if not raw_output.strip():
        return InterpretationParseResult(
            case_id=fallback_case_id,
            parse_status=PARSE_STATUS_EMPTY,
            interpretations=(),
            parse_errors=("empty provider output",),
        )
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return InterpretationParseResult(
            case_id=fallback_case_id,
            parse_status=PARSE_STATUS_INVALID_JSON,
            interpretations=(),
            parse_errors=(f"invalid JSON: {exc.msg} at char {exc.pos}",),
        )
    if not isinstance(decoded, dict):
        return InterpretationParseResult(
            case_id=fallback_case_id,
            parse_status=PARSE_STATUS_INVALID_SCHEMA,
            interpretations=(),
            parse_errors=("root output must be a JSON object",),
        )
    return parse_interpretation_object(
        decoded,
        expected_case_id=expected_case_id,
    )


def parse_interpretation_object(
    raw: dict[str, typing.Any],
    *,
    expected_case_id: str | None = None,
) -> InterpretationParseResult:
    fallback_case_id = expected_case_id or ""
    errors: list[str] = []

    raw_case_id = raw.get("case_id")
    case_id = ""
    if isinstance(raw_case_id, str) and raw_case_id.strip():
        case_id = raw_case_id.strip()
    else:
        errors.append("case_id must be a non-empty string")
    if expected_case_id is not None and case_id and case_id != expected_case_id:
        errors.append(
            f"case_id {case_id!r} does not match expected {expected_case_id!r}"
        )

    raw_interpretations = raw.get("interpretations")
    if not isinstance(raw_interpretations, list):
        errors.append("interpretations must be a list")
        raw_interpretations = []

    interpretations: list[InterpretationRecord] = []
    for index, raw_interpretation in enumerate(raw_interpretations):
        try:
            interpretations.append(
                _parse_single_interpretation(raw_interpretation, index=index)
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"interpretations[{index}]: {exc}")
    duplicate_ids = sorted(
        interpretation_id
        for interpretation_id, count in Counter(
            item.interpretation_id for item in interpretations
        ).items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(
            "duplicate interpretation_id values: " + ", ".join(duplicate_ids)
        )

    if errors:
        return InterpretationParseResult(
            case_id=case_id or fallback_case_id,
            parse_status=PARSE_STATUS_INVALID_SCHEMA,
            interpretations=(),
            parse_errors=tuple(errors),
        )
    return InterpretationParseResult(
        case_id=case_id,
        parse_status=PARSE_STATUS_PARSED,
        interpretations=tuple(interpretations),
        parse_errors=(),
    )


def _parse_single_interpretation(
    raw: typing.Any,
    *,
    index: int,
) -> InterpretationRecord:
    if not isinstance(raw, dict):
        raise TypeError("interpretation must be a JSON object")
    _require_string(raw, "interpretation_id")
    _require_string(raw, "natural_language_interpretation")
    _require_string(raw, "expected_answer_shape")
    ambiguity_types = raw.get("ambiguity_types")
    if not isinstance(ambiguity_types, list) or not ambiguity_types:
        raise ValueError("ambiguity_types must be a non-empty list")
    if not all(isinstance(item, str) for item in ambiguity_types):
        raise TypeError("ambiguity_types must contain only strings")
    assumptions = raw.get("assumptions", [])
    if not isinstance(assumptions, list):
        raise TypeError("assumptions must be a list")
    if not all(isinstance(item, str) for item in assumptions):
        raise TypeError("assumptions must contain only strings")
    try:
        return InterpretationRecord.from_dict(raw)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _require_string(raw: dict[str, typing.Any], key: str) -> None:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")

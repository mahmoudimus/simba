"""Quality review for ambiguous NLIDB interpretation outputs."""

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

from simba.eval import ambiguity_fail18, interpretation_parser, interpretation_runner

DEFAULT_QUALITY_REVIEW_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_quality_review.json"
)

_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
}
_DATE_OR_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?\b")
_EVIDENCE_RE = re.compile(r"\bevidence[_ -]?\d+\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\$?\d[\d,]*(?:\.\d+)?")


@dataclasses.dataclass(frozen=True)
class InterpretationQualityCase:
    case_id: str
    question: str
    failure_mode: str
    gold_value: float | None
    interpretation_count: int
    provider_succeeded: bool
    parse_status: str
    expected_answer_shapes: tuple[str, ...]
    observed_answer_shapes: tuple[str, ...]
    expected_ambiguity_types: tuple[str, ...]
    observed_ambiguity_types: tuple[str, ...]
    missing_ambiguity_types: tuple[str, ...]
    gold_compatible_interpretation_ids: tuple[str, ...]
    duplicate_interpretation_count: int
    quality_issues: tuple[str, ...]
    warning_issues: tuple[str, ...]
    useful_for_compilation: bool

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "failure_mode": self.failure_mode,
            "gold_value": self.gold_value,
            "interpretation_count": self.interpretation_count,
            "provider_succeeded": self.provider_succeeded,
            "parse_status": self.parse_status,
            "expected_answer_shapes": list(self.expected_answer_shapes),
            "observed_answer_shapes": list(self.observed_answer_shapes),
            "expected_ambiguity_types": list(self.expected_ambiguity_types),
            "observed_ambiguity_types": list(self.observed_ambiguity_types),
            "missing_ambiguity_types": list(self.missing_ambiguity_types),
            "gold_compatible_interpretation_ids": list(
                self.gold_compatible_interpretation_ids
            ),
            "duplicate_interpretation_count": self.duplicate_interpretation_count,
            "quality_issues": list(self.quality_issues),
            "warning_issues": list(self.warning_issues),
            "useful_for_compilation": self.useful_for_compilation,
        }


def build_fail18_quality_review(
    *,
    outputs_path: str | pathlib.Path = interpretation_runner.DEFAULT_OUTPUTS_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    report_path: str | pathlib.Path = interpretation_runner.DEFAULT_REPORT_PATH,
) -> dict[str, typing.Any]:
    rows = interpretation_runner.load_jsonl(outputs_path)
    manifest_rows = ambiguity_fail18.load_manifest(manifest_path)
    manifest_by_id = {str(row["question_id"]): row for row in manifest_rows}
    cases = [
        review_interpretation_row(row, manifest_by_id.get(str(row.get("case_id", ""))))
        for row in rows
    ]
    useful_cases = [case for case in cases if case.useful_for_compilation]
    issue_counts: collections.Counter[str] = collections.Counter()
    warning_counts: collections.Counter[str] = collections.Counter()
    for case in cases:
        issue_counts.update(case.quality_issues)
        warning_counts.update(case.warning_issues)

    gate1_report = _load_optional_json(report_path)
    gate1_report_summary = {
        "gate_status": gate1_report.get("gate_status", ""),
        "gate1_passed": gate1_report.get("gate1_passed", False),
        "rows_parsed": gate1_report.get("rows_parsed"),
        "rows_provider_failed": gate1_report.get("rows_provider_failed"),
        "duplicate_interpretation_count": gate1_report.get(
            "duplicate_interpretation_count"
        ),
    }
    return {
        "name": "fail18-ambiguous-nlidb-gate1-quality-review",
        "artifact_kind": "interpretation_quality_review",
        "gate": "gate1",
        "gate_status": "slice1c_quality_review_complete",
        "gate1_quality_passed": len(useful_cases) == len(cases) and bool(cases),
        "gate1_quality_blocker": (
            ""
            if len(useful_cases) == len(cases) and cases
            else (
                "One or more rows lack enough reviewed interpretation quality "
                "for candidate-unit compilation."
            )
        ),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_outputs_artifact": str(outputs_path),
        "source_manifest": str(manifest_path),
        "source_gate1_report": str(report_path),
        "source_gate1_report_summary": gate1_report_summary,
        "review_policy": {
            "provider_success_required": True,
            "parse_status_required": interpretation_parser.PARSE_STATUS_PARSED,
            "minimum_interpretations": 2,
            "gold_compatibility": (
                "Heuristic: extracts explicit numbers and simple ranges from "
                "provider interpretation text. This is a review signal, not an "
                "answer verifier."
            ),
            "missing_expected_ambiguity_is_warning": True,
        },
        "summary": {
            "rows_total": len(cases),
            "rows_useful_for_compilation": len(useful_cases),
            "rows_not_useful_for_compilation": len(cases) - len(useful_cases),
            "rows_with_gold_compatible_interpretation": sum(
                1 for case in cases if case.gold_compatible_interpretation_ids
            ),
            "rows_missing_gold_compatible_interpretation": sum(
                1
                for case in cases
                if case.gold_value is not None
                and not case.gold_compatible_interpretation_ids
            ),
            "rows_with_expected_answer_shape": sum(
                1
                for case in cases
                if set(case.expected_answer_shapes)
                & set(case.observed_answer_shapes)
            ),
            "rows_missing_expected_ambiguity_types": sum(
                1 for case in cases if case.missing_ambiguity_types
            ),
            "issue_counts": dict(sorted(issue_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "cases": [case.to_dict() for case in cases],
    }


def review_interpretation_row(
    row: dict[str, typing.Any],
    manifest_row: dict[str, typing.Any] | None,
) -> InterpretationQualityCase:
    case_id = str(row.get("case_id", ""))
    question = str((manifest_row or {}).get("question", ""))
    failure_mode = str((manifest_row or {}).get("failure_mode", ""))
    interpretations = [
        item for item in row.get("interpretations", []) if isinstance(item, dict)
    ]
    expected_answer_shapes = _expected_answer_shapes(question, failure_mode)
    observed_answer_shapes = tuple(
        sorted(
            {
                str(item.get("expected_answer_shape", ""))
                for item in interpretations
                if item.get("expected_answer_shape")
            }
        )
    )
    expected_ambiguity_types = _expected_ambiguity_types(failure_mode, question)
    observed_ambiguity_types = tuple(
        sorted(
            {
                str(kind)
                for item in interpretations
                for kind in item.get("ambiguity_types", [])
            }
        )
    )
    missing_ambiguity_types = tuple(
        kind
        for kind in expected_ambiguity_types
        if kind not in set(observed_ambiguity_types)
    )
    gold_value = _gold_value(manifest_row or {})
    gold_compatible_ids = _gold_compatible_interpretation_ids(
        interpretations,
        gold_value,
    )
    duplicate_count = _duplicate_interpretation_count(interpretations)
    provider_succeeded = _provider_succeeded(row)
    parse_status = str(row.get("parse_status", ""))
    issues: list[str] = []
    warnings: list[str] = []
    if manifest_row is None:
        issues.append("missing_manifest_row")
    if not provider_succeeded:
        issues.append("provider_failed")
    if parse_status != interpretation_parser.PARSE_STATUS_PARSED:
        issues.append("parse_failed")
    if len(interpretations) < 2:
        issues.append("less_than_two_interpretations")
    if duplicate_count:
        issues.append("duplicate_interpretations")
    if not set(expected_answer_shapes) & set(observed_answer_shapes):
        issues.append("missing_expected_answer_shape")
    if gold_value is not None and not gold_compatible_ids:
        issues.append("no_gold_compatible_interpretation")
    if missing_ambiguity_types:
        warnings.append("missing_expected_ambiguity_types")
    if _mentions_runtime_current_date(interpretations):
        warnings.append("uses_runtime_current_date_anchor")

    blocking_issues = {
        "missing_manifest_row",
        "provider_failed",
        "parse_failed",
        "less_than_two_interpretations",
        "duplicate_interpretations",
        "missing_expected_answer_shape",
        "no_gold_compatible_interpretation",
    }
    return InterpretationQualityCase(
        case_id=case_id,
        question=question,
        failure_mode=failure_mode,
        gold_value=gold_value,
        interpretation_count=len(interpretations),
        provider_succeeded=provider_succeeded,
        parse_status=parse_status,
        expected_answer_shapes=expected_answer_shapes,
        observed_answer_shapes=observed_answer_shapes,
        expected_ambiguity_types=expected_ambiguity_types,
        observed_ambiguity_types=observed_ambiguity_types,
        missing_ambiguity_types=missing_ambiguity_types,
        gold_compatible_interpretation_ids=gold_compatible_ids,
        duplicate_interpretation_count=duplicate_count,
        quality_issues=tuple(issues),
        warning_issues=tuple(warnings),
        useful_for_compilation=not (set(issues) & blocking_issues),
    )


def _provider_succeeded(row: dict[str, typing.Any]) -> bool:
    if "provider_exit_code" not in row:
        return False
    try:
        exit_code = int(row.get("provider_exit_code"))
    except (TypeError, ValueError):
        return False
    return exit_code == 0 and not bool(row.get("provider_timed_out", False))


def _gold_value(row: dict[str, typing.Any]) -> float | None:
    text = str(row.get("gold_answer", ""))
    values = _extract_numeric_values(text)
    if values:
        return values[0]
    raw_count = row.get("gold_count")
    if isinstance(raw_count, int | float):
        return float(raw_count)
    if isinstance(raw_count, str):
        raw_count = raw_count.strip().replace(",", "")
        try:
            return float(raw_count)
        except ValueError:
            return None
    return None


def _gold_compatible_interpretation_ids(
    interpretations: list[dict[str, typing.Any]],
    gold_value: float | None,
) -> tuple[str, ...]:
    if gold_value is None:
        return ()
    compatible: list[str] = []
    for interpretation in interpretations:
        text = _interpretation_text(interpretation)
        values = _extract_numeric_values(text)
        if any(_same_number(value, gold_value) for value in values):
            compatible.append(str(interpretation.get("interpretation_id", "")))
            continue
        if (
            interpretation.get("expected_answer_shape") == "range"
            and _range_contains(values, gold_value)
        ):
            compatible.append(str(interpretation.get("interpretation_id", "")))
    return tuple(item for item in compatible if item)


def _interpretation_text(interpretation: dict[str, typing.Any]) -> str:
    parts = [str(interpretation.get("natural_language_interpretation", ""))]
    parts.extend(str(item) for item in interpretation.get("assumptions", []))
    return "\n".join(parts)


def _extract_numeric_values(text: str) -> tuple[float, ...]:
    scrubbed = _EVIDENCE_RE.sub(" ", text.lower())
    scrubbed = _DATE_OR_YEAR_RE.sub(" ", scrubbed)
    values: list[tuple[int, float]] = []
    for match in _NUMBER_RE.finditer(scrubbed):
        raw = match.group().replace("$", "").replace(",", "")
        try:
            values.append((match.start(), float(raw)))
        except ValueError:
            continue
    for word, value in _NUMBER_WORDS.items():
        for match in re.finditer(rf"\b{word}\b", scrubbed):
            values.append((match.start(), value))
    return tuple(value for _position, value in sorted(values))


def _same_number(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=0.001)


def _range_contains(values: tuple[float, ...], gold_value: float) -> bool:
    if len(values) < 2:
        return False
    low = min(values)
    high = max(values)
    return low <= gold_value <= high


def _duplicate_interpretation_count(
    interpretations: list[dict[str, typing.Any]],
) -> int:
    seen: set[str] = set()
    duplicate_count = 0
    for interpretation in interpretations:
        text = str(interpretation.get("natural_language_interpretation", ""))
        shape = str(interpretation.get("expected_answer_shape", ""))
        ambiguity_types = ",".join(
            str(item) for item in interpretation.get("ambiguity_types", [])
        )
        normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
        key = f"{normalized}|{shape}|{ambiguity_types}"
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)
    return duplicate_count


def _expected_answer_shapes(
    question: str,
    failure_mode: str,
) -> tuple[str, ...]:
    lowered = question.lower()
    mode = failure_mode.lower()
    if "lookup" in mode or ("points" in lowered and "redeem" in lowered):
        return ("lookup", "range")
    if (
        lowered.startswith("how much")
        or " total " in f" {lowered} "
        or "hours" in lowered
        or "days" in lowered
        or "people reached" in lowered
        or mode.startswith("b_sum")
    ):
        return ("sum", "range", "set")
    return ("count", "range", "set")


def _expected_ambiguity_types(
    failure_mode: str,
    question: str,
) -> tuple[str, ...]:
    mode = failure_mode.lower()
    lowered = question.lower()
    expected: list[str] = []
    if "timewindow" in mode or "past" in lowered or "last" in lowered:
        expected.extend(["context_parameter_ambiguous", "scope_ambiguous"])
    if "lookup" in mode:
        expected.extend(["schema_link_ambiguous", "value_mapping_ambiguous"])
    if "sum" in mode:
        expected.extend(["aggregation_view_ambiguous", "source_of_truth_ambiguous"])
    if "overcount" in mode or "overfilter" in mode:
        expected.extend(["scope_ambiguous", "value_mapping_ambiguous"])
    if "underextraction" in mode:
        expected.extend(["source_of_truth_ambiguous", "scope_ambiguous"])
    if not expected:
        expected.append("scope_ambiguous")
    seen: set[str] = set()
    return tuple(item for item in expected if not (item in seen or seen.add(item)))


def _mentions_runtime_current_date(
    interpretations: list[dict[str, typing.Any]],
) -> bool:
    text = "\n".join(_interpretation_text(item) for item in interpretations).lower()
    return "current date" in text or "today's date" in text or "2026-06-19" in text


def _load_optional_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    json_path = pathlib.Path(path)
    if not json_path.exists():
        return {}
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(json_path.read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=pathlib.Path,
        default=interpretation_runner.DEFAULT_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--gate1-report",
        type=pathlib.Path,
        default=interpretation_runner.DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_QUALITY_REVIEW_PATH,
    )
    args = parser.parse_args(argv)

    review = build_fail18_quality_review(
        outputs_path=args.outputs,
        manifest_path=args.manifest,
        report_path=args.gate1_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"{json.dumps(review, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

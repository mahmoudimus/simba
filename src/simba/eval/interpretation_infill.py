"""Missing-interpretation infill for ambiguous NLIDB Gate 1."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

from simba.eval import (
    ambiguity_fail18,
    interpretation_parser,
    interpretation_quality_review,
    interpretation_runner,
)
from simba.eval.interpretation_prompts import PROMPT_VERSION

INFILL_PROMPT_VERSION = f"{PROMPT_VERSION}_infill_v1"
DEFAULT_INFILL_PAYLOADS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_infill_payloads.json"
)
DEFAULT_INFILL_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_infill_outputs.jsonl"
)
DEFAULT_INFILL_REPORT_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_infill_report.json"
)
DEFAULT_INFILLED_OUTPUTS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_infilled_outputs.jsonl"
)
DEFAULT_POST_INFILL_QUALITY_REVIEW_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_post_infill_quality_review.json"
)

INFILL_GENERATION_CONTRACT = (
    "Return exactly one strict JSON object. No markdown.",
    "Generate only missing natural-language interpretations that are not already "
    "covered by existing_interpretations.",
    "Do not compute a final answer and do not choose one interpretation as the "
    "winner.",
    "Use only the question and supplied evidence. Do not use web knowledge.",
    "If the existing interpretations already cover all reasonable readings, "
    "return an empty interpretations list.",
    "Prefer interpretations that clarify answer shape, scope, source of truth, "
    "time anchoring, and aggregation policy.",
)


@dataclasses.dataclass(frozen=True)
class InfillMergeResult:
    case_id: str
    original_count: int
    infill_count: int
    added_count: int
    duplicate_count: int
    added_interpretation_ids: tuple[str, ...]
    duplicate_interpretation_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "original_count": self.original_count,
            "infill_count": self.infill_count,
            "added_count": self.added_count,
            "duplicate_count": self.duplicate_count,
            "added_interpretation_ids": list(self.added_interpretation_ids),
            "duplicate_interpretation_ids": list(
                self.duplicate_interpretation_ids
            ),
        }


def build_fail18_infill_payload_artifact(
    *,
    quality_review_path: str | pathlib.Path = (
        interpretation_quality_review.DEFAULT_QUALITY_REVIEW_PATH
    ),
    gate1_payloads_path: str | pathlib.Path = (
        interpretation_runner.DEFAULT_PAYLOADS_PATH
    ),
    gate1_outputs_path: str | pathlib.Path = (
        interpretation_runner.DEFAULT_OUTPUTS_PATH
    ),
) -> dict[str, typing.Any]:
    """Build provider payloads for rows that failed quality review."""
    quality_review = _load_json(quality_review_path)
    gate1_payloads = _load_json(gate1_payloads_path)
    gate1_rows = interpretation_runner.load_jsonl(gate1_outputs_path)
    payload_by_id = {
        str(payload.get("case", {}).get("id", "")): payload
        for payload in gate1_payloads.get("payloads", [])
    }
    row_by_id = {str(row.get("case_id", "")): row for row in gate1_rows}
    blocked_cases = [
        case
        for case in quality_review.get("cases", [])
        if not bool(case.get("useful_for_compilation", False))
    ]
    payloads = []
    skipped: list[dict[str, str]] = []
    for case in blocked_cases:
        case_id = str(case.get("case_id", ""))
        source_payload = payload_by_id.get(case_id)
        source_row = row_by_id.get(case_id)
        if source_payload is None or source_row is None:
            skipped.append(
                {
                    "case_id": case_id,
                    "reason": "missing source payload or provider output row",
                }
            )
            continue
        payloads.append(
            build_infill_payload(
                quality_case=case,
                source_payload=source_payload,
                source_row=source_row,
            )
        )
    return {
        "name": "fail18-ambiguous-nlidb-gate1-infill-payloads",
        "artifact_kind": "provider_payloads",
        "gate": "gate1",
        "gate_status": "slice2_infill_payloads_only_not_run",
        "prompt_version": INFILL_PROMPT_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_quality_review": str(quality_review_path),
        "source_payload_artifact": str(gate1_payloads_path),
        "source_outputs_artifact": str(gate1_outputs_path),
        "total": len(payloads),
        "blocked_case_ids": [str(case.get("case_id", "")) for case in blocked_cases],
        "skipped_cases": skipped,
        "provider_visibility": {
            "gold_answer_visible": False,
            "gold_value_visible": False,
            "failure_mode_visible": False,
        },
        "commands": [
            (
                "uv run python -m simba.eval.interpretation_infill "
                "--build-payloads "
                "--payload-output "
                "_gitless/fail18_ambiguous_nlidb_gate1_infill_payloads.json"
            ),
            (
                "uv run python -m simba.eval.interpretation_runner "
                "--payloads "
                "_gitless/fail18_ambiguous_nlidb_gate1_infill_payloads.json "
                "--outputs "
                "_gitless/fail18_ambiguous_nlidb_gate1_infill_outputs.jsonl "
                "--report "
                "_gitless/fail18_ambiguous_nlidb_gate1_infill_runner_report.json"
            ),
        ],
        "payloads": payloads,
    }


def build_infill_payload(
    *,
    quality_case: dict[str, typing.Any],
    source_payload: dict[str, typing.Any],
    source_row: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    case = typing.cast("dict[str, typing.Any]", source_payload.get("case", {}))
    return {
        "task": (
            "Generate missing natural-language interpretations for an ambiguous "
            "memory question after reviewing the existing interpretation set."
        ),
        "prompt_version": INFILL_PROMPT_VERSION,
        "generation_contract": list(INFILL_GENERATION_CONTRACT),
        "allowed_ambiguity_types": source_payload.get("allowed_ambiguity_types", []),
        "output_schema": {
            "case_id": str(quality_case.get("case_id", "")),
            "interpretations": [
                {
                    "interpretation_id": "stable string not used before",
                    "natural_language_interpretation": "string",
                    "ambiguity_types": ["one or more allowed_ambiguity_types"],
                    "assumptions": ["string"],
                    "expected_answer_shape": "count|sum|lookup|range|set",
                }
            ],
        },
        "quality_review": {
            "quality_issues": list(quality_case.get("quality_issues", [])),
            "warning_issues": list(quality_case.get("warning_issues", [])),
            "missing_ambiguity_types": list(
                quality_case.get("missing_ambiguity_types", [])
            ),
            "expected_answer_shapes": list(
                quality_case.get("expected_answer_shapes", [])
            ),
            "observed_answer_shapes": list(
                quality_case.get("observed_answer_shapes", [])
            ),
            "instruction": (
                "The prior interpretation set was judged incomplete. Add only "
                "plausible missing readings supported by the evidence. Do not "
                "repeat existing readings."
            ),
        },
        "case": {
            "id": str(case.get("id", "")),
            "question": str(case.get("question", "")),
            "evidence_sessions": case.get("evidence_sessions", []),
            "existing_interpretations": source_row.get("interpretations", []),
        },
    }


def build_fail18_infill_report(
    *,
    original_outputs_path: str | pathlib.Path = (
        interpretation_runner.DEFAULT_OUTPUTS_PATH
    ),
    infill_outputs_path: str | pathlib.Path = DEFAULT_INFILL_OUTPUTS_PATH,
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    quality_review_path: str | pathlib.Path = (
        interpretation_quality_review.DEFAULT_QUALITY_REVIEW_PATH
    ),
    payloads_path: str | pathlib.Path = DEFAULT_INFILL_PAYLOADS_PATH,
    infilled_outputs_path: str | pathlib.Path = DEFAULT_INFILLED_OUTPUTS_PATH,
    post_review_path: str | pathlib.Path = DEFAULT_POST_INFILL_QUALITY_REVIEW_PATH,
) -> dict[str, typing.Any]:
    original_rows = interpretation_runner.load_jsonl(original_outputs_path)
    infill_rows = interpretation_runner.load_jsonl(infill_outputs_path)
    infilled_rows, merge_results = merge_infill_rows(original_rows, infill_rows)
    interpretation_runner.write_jsonl(infilled_outputs_path, infilled_rows)
    post_review = _quality_review_for_rows(
        rows=infilled_rows,
        manifest_path=manifest_path,
        source_outputs_path=infilled_outputs_path,
    )
    pathlib.Path(post_review_path).write_text(
        f"{json.dumps(post_review, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )

    previous_review = _load_json(quality_review_path)
    payload_artifact = _load_json(payloads_path)
    parsed_infill_rows = [
        row
        for row in infill_rows
        if row.get("parse_status") == interpretation_parser.PARSE_STATUS_PARSED
        and _provider_succeeded(row)
    ]
    provider_failed_rows = [row for row in infill_rows if not _provider_succeeded(row)]
    return {
        "name": "fail18-ambiguous-nlidb-gate1-infill-report",
        "artifact_kind": "interpretation_infill_report",
        "gate": "gate1",
        "gate_status": "slice2_infill_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_original_outputs": str(original_outputs_path),
        "source_infill_payloads": str(payloads_path),
        "source_infill_outputs": str(infill_outputs_path),
        "source_previous_quality_review": str(quality_review_path),
        "infilled_outputs_artifact": str(infilled_outputs_path),
        "post_infill_quality_review": str(post_review_path),
        "rows_requested": int(payload_artifact.get("total", len(infill_rows))),
        "rows_infill_total": len(infill_rows),
        "rows_infill_parsed": len(parsed_infill_rows),
        "rows_infill_provider_failed": len(provider_failed_rows),
        "rows_with_added_interpretations": sum(
            1 for item in merge_results if item.added_count > 0
        ),
        "added_interpretation_count": sum(
            item.added_count for item in merge_results
        ),
        "duplicate_interpretation_count": sum(
            item.duplicate_count for item in merge_results
        ),
        "merge_results": [item.to_dict() for item in merge_results],
        "quality_delta": _quality_delta(previous_review, post_review),
        "acceptance": {
            "candidate_unit_compilation_attempted": False,
            "infill_outputs_cover_requested_rows": (
                len(infill_rows) == int(payload_artifact.get("total", len(infill_rows)))
            ),
            "provider_rows_succeeded": not provider_failed_rows,
            "post_infill_quality_review_written": pathlib.Path(
                post_review_path
            ).exists(),
        },
    }


def merge_infill_rows(
    original_rows: list[dict[str, typing.Any]],
    infill_rows: list[dict[str, typing.Any]],
) -> tuple[list[dict[str, typing.Any]], list[InfillMergeResult]]:
    infill_by_id = {str(row.get("case_id", "")): row for row in infill_rows}
    merged_rows: list[dict[str, typing.Any]] = []
    merge_results: list[InfillMergeResult] = []
    for row in original_rows:
        case_id = str(row.get("case_id", ""))
        infill_row = infill_by_id.get(case_id)
        if infill_row is None:
            merged_rows.append(row)
            continue
        merged_row = dict(row)
        original_interpretations = [
            item for item in row.get("interpretations", []) if isinstance(item, dict)
        ]
        original_keys = {_interpretation_key(item) for item in original_interpretations}
        original_ids = {
            str(item.get("interpretation_id", ""))
            for item in original_interpretations
        }
        added: list[dict[str, typing.Any]] = []
        duplicate_ids: list[str] = []
        infill_interpretations = [
            item
            for item in infill_row.get("interpretations", [])
            if isinstance(item, dict)
        ]
        for interpretation in infill_interpretations:
            interpretation_id = str(interpretation.get("interpretation_id", ""))
            key = _interpretation_key(interpretation)
            if key in original_keys or interpretation_id in original_ids:
                duplicate_ids.append(interpretation_id)
                continue
            original_keys.add(key)
            original_ids.add(interpretation_id)
            added.append(interpretation)
        merged_row["interpretations"] = [*original_interpretations, *added]
        merged_row["infill_source_case_id"] = case_id
        merged_row["infill_added_interpretation_ids"] = [
            str(item.get("interpretation_id", "")) for item in added
        ]
        merged_rows.append(merged_row)
        merge_results.append(
            InfillMergeResult(
                case_id=case_id,
                original_count=len(original_interpretations),
                infill_count=len(infill_interpretations),
                added_count=len(added),
                duplicate_count=len(duplicate_ids),
                added_interpretation_ids=tuple(
                    str(item.get("interpretation_id", "")) for item in added
                ),
                duplicate_interpretation_ids=tuple(duplicate_ids),
            )
        )
    return merged_rows, merge_results


def _quality_review_for_rows(
    *,
    rows: list[dict[str, typing.Any]],
    manifest_path: str | pathlib.Path,
    source_outputs_path: str | pathlib.Path,
) -> dict[str, typing.Any]:
    manifest_rows = ambiguity_fail18.load_manifest(manifest_path)
    manifest_by_id = {str(row["question_id"]): row for row in manifest_rows}
    cases = [
        interpretation_quality_review.review_interpretation_row(
            row,
            manifest_by_id.get(str(row.get("case_id", ""))),
        )
        for row in rows
    ]
    issue_counts: collections.Counter[str] = collections.Counter()
    warning_counts: collections.Counter[str] = collections.Counter()
    for case in cases:
        issue_counts.update(case.quality_issues)
        warning_counts.update(case.warning_issues)
    useful_cases = [case for case in cases if case.useful_for_compilation]
    return {
        "name": "fail18-ambiguous-nlidb-gate1-post-infill-quality-review",
        "artifact_kind": "interpretation_quality_review",
        "gate": "gate1",
        "gate_status": "slice2_post_infill_quality_review_complete",
        "gate1_quality_passed": len(useful_cases) == len(cases) and bool(cases),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_outputs_artifact": str(source_outputs_path),
        "source_manifest": str(manifest_path),
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
            "issue_counts": dict(sorted(issue_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "cases": [case.to_dict() for case in cases],
    }


def _quality_delta(
    previous_review: dict[str, typing.Any],
    post_review: dict[str, typing.Any],
) -> dict[str, typing.Any]:
    previous = typing.cast(
        "dict[str, typing.Any]",
        previous_review.get("summary", {}),
    )
    current = typing.cast("dict[str, typing.Any]", post_review.get("summary", {}))
    keys = (
        "rows_useful_for_compilation",
        "rows_not_useful_for_compilation",
        "rows_with_gold_compatible_interpretation",
        "rows_missing_gold_compatible_interpretation",
    )
    return {
        key: {
            "before": previous.get(key),
            "after": current.get(key),
            "delta": _numeric_delta(previous.get(key), current.get(key)),
        }
        for key in keys
    }


def _numeric_delta(first: typing.Any, second: typing.Any) -> int | None:
    if not isinstance(first, int) or not isinstance(second, int):
        return None
    return second - first


def _provider_succeeded(row: dict[str, typing.Any]) -> bool:
    if "provider_exit_code" not in row:
        return False
    try:
        exit_code = int(row.get("provider_exit_code"))
    except (TypeError, ValueError):
        return False
    return exit_code == 0 and not bool(row.get("provider_timed_out", False))


def _interpretation_key(interpretation: dict[str, typing.Any]) -> str:
    text = str(interpretation.get("natural_language_interpretation", ""))
    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    shape = str(interpretation.get("expected_answer_shape", ""))
    ambiguity_types = ",".join(
        str(item) for item in interpretation.get("ambiguity_types", [])
    )
    return f"{normalized_text}|{shape}|{ambiguity_types}"


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-payloads", action="store_true")
    parser.add_argument("--build-report", action="store_true")
    parser.add_argument(
        "--quality-review",
        type=pathlib.Path,
        default=interpretation_quality_review.DEFAULT_QUALITY_REVIEW_PATH,
    )
    parser.add_argument(
        "--gate1-payloads",
        type=pathlib.Path,
        default=interpretation_runner.DEFAULT_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--gate1-outputs",
        type=pathlib.Path,
        default=interpretation_runner.DEFAULT_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--infill-outputs",
        type=pathlib.Path,
        default=DEFAULT_INFILL_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--payload-output",
        type=pathlib.Path,
        default=DEFAULT_INFILL_PAYLOADS_PATH,
    )
    parser.add_argument(
        "--report-output",
        type=pathlib.Path,
        default=DEFAULT_INFILL_REPORT_PATH,
    )
    parser.add_argument(
        "--infilled-outputs",
        type=pathlib.Path,
        default=DEFAULT_INFILLED_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--post-review-output",
        type=pathlib.Path,
        default=DEFAULT_POST_INFILL_QUALITY_REVIEW_PATH,
    )
    args = parser.parse_args(argv)

    if args.build_payloads:
        artifact = build_fail18_infill_payload_artifact(
            quality_review_path=args.quality_review,
            gate1_payloads_path=args.gate1_payloads,
            gate1_outputs_path=args.gate1_outputs,
        )
        args.payload_output.parent.mkdir(parents=True, exist_ok=True)
        args.payload_output.write_text(
            f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    if args.build_report:
        report = build_fail18_infill_report(
            original_outputs_path=args.gate1_outputs,
            infill_outputs_path=args.infill_outputs,
            manifest_path=args.manifest,
            quality_review_path=args.quality_review,
            payloads_path=args.payload_output,
            infilled_outputs_path=args.infilled_outputs,
            post_review_path=args.post_review_output,
        )
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    if not args.build_payloads and not args.build_report:
        raise SystemExit("pass --build-payloads and/or --build-report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

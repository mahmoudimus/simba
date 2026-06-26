"""Diagnose rows still blocked after interpretation infill."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import pathlib
import typing

from simba.eval import (
    ambiguity_fail18,
    interpretation_infill,
    interpretation_quality_review,
    interpretation_runner,
)

DEFAULT_INFILL_DIAGNOSTICS_PATH = pathlib.Path(
    "_gitless/fail18_ambiguous_nlidb_gate1_infill_diagnostics.json"
)


@dataclasses.dataclass(frozen=True)
class BlockedRowDiagnosis:
    case_id: str
    question: str
    failure_mode: str
    gold_value: float | None
    interpretation_count: int
    observed_answer_shapes: tuple[str, ...]
    observed_numeric_values: tuple[float, ...]
    closest_numeric_values: tuple[float, ...]
    issue_tags: tuple[str, ...]
    likely_failure: str
    recommended_next_intervention: str

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "failure_mode": self.failure_mode,
            "gold_value": self.gold_value,
            "interpretation_count": self.interpretation_count,
            "observed_answer_shapes": list(self.observed_answer_shapes),
            "observed_numeric_values": list(self.observed_numeric_values),
            "closest_numeric_values": list(self.closest_numeric_values),
            "issue_tags": list(self.issue_tags),
            "likely_failure": self.likely_failure,
            "recommended_next_intervention": self.recommended_next_intervention,
        }


def build_fail18_infill_diagnostics(
    *,
    post_review_path: str | pathlib.Path = (
        interpretation_infill.DEFAULT_POST_INFILL_QUALITY_REVIEW_PATH
    ),
    infilled_outputs_path: str | pathlib.Path = (
        interpretation_infill.DEFAULT_INFILLED_OUTPUTS_PATH
    ),
    manifest_path: str | pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    infill_report_path: str | pathlib.Path = (
        interpretation_infill.DEFAULT_INFILL_REPORT_PATH
    ),
) -> dict[str, typing.Any]:
    post_review = _load_json(post_review_path)
    infill_report = _load_optional_json(infill_report_path)
    rows = interpretation_runner.load_jsonl(infilled_outputs_path)
    rows_by_id = {str(row.get("case_id", "")): row for row in rows}
    manifest_rows = ambiguity_fail18.load_manifest(manifest_path)
    manifest_by_id = {str(row["question_id"]): row for row in manifest_rows}
    blocked_cases = [
        typing.cast("dict[str, typing.Any]", case)
        for case in post_review.get("cases", [])
        if not bool(case.get("useful_for_compilation", False))
    ]
    diagnoses = [
        diagnose_blocked_row(
            quality_case=case,
            infilled_row=rows_by_id.get(str(case.get("case_id", "")), {}),
            manifest_row=manifest_by_id.get(str(case.get("case_id", "")), {}),
        )
        for case in blocked_cases
    ]
    intervention_counts = collections.Counter(
        item.recommended_next_intervention for item in diagnoses
    )
    issue_tag_counts: collections.Counter[str] = collections.Counter()
    for item in diagnoses:
        issue_tag_counts.update(item.issue_tags)
    return {
        "name": "fail18-ambiguous-nlidb-gate1-infill-diagnostics",
        "artifact_kind": "interpretation_infill_diagnostics",
        "gate": "gate1",
        "gate_status": "slice2b_diagnostics_complete",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_post_infill_quality_review": str(post_review_path),
        "source_infilled_outputs": str(infilled_outputs_path),
        "source_manifest": str(manifest_path),
        "source_infill_report": str(infill_report_path),
        "source_infill_quality_delta": infill_report.get("quality_delta", {}),
        "summary": {
            "blocked_rows": len(diagnoses),
            "issue_tag_counts": dict(sorted(issue_tag_counts.items())),
            "recommended_next_intervention_counts": dict(
                sorted(intervention_counts.items())
            ),
        },
        "decision": {
            "candidate_unit_compilation_should_start": False,
            "another_generic_infill_should_run": False,
            "next_slice": "targeted_diagnostics_driven_infill_or_verifier_probe",
            "reason": (
                "Generic infill improved only a small number of rows. The "
                "remaining misses need targeted evidence enumeration or "
                "answer-variable policy hints before compilation."
            ),
        },
        "blocked_cases": [item.to_dict() for item in diagnoses],
    }


def diagnose_blocked_row(
    *,
    quality_case: dict[str, typing.Any],
    infilled_row: dict[str, typing.Any],
    manifest_row: dict[str, typing.Any],
) -> BlockedRowDiagnosis:
    case_id = str(quality_case.get("case_id", ""))
    question = str(manifest_row.get("question", quality_case.get("question", "")))
    failure_mode = str(
        manifest_row.get("failure_mode", quality_case.get("failure_mode", ""))
    )
    interpretations = [
        item
        for item in infilled_row.get("interpretations", [])
        if isinstance(item, dict)
    ]
    observed_shapes = tuple(
        sorted(
            {
                str(item.get("expected_answer_shape", ""))
                for item in interpretations
                if item.get("expected_answer_shape")
            }
        )
    )
    gold_value = _as_float(quality_case.get("gold_value"))
    values = _observed_numeric_values(interpretations)
    issue_tags = _issue_tags(
        question=question,
        failure_mode=failure_mode,
        quality_case=quality_case,
        interpretations=interpretations,
    )
    closest_values = _closest_values(values, gold_value)
    recommended = _recommended_intervention(
        question=question,
        failure_mode=failure_mode,
        issue_tags=issue_tags,
    )
    return BlockedRowDiagnosis(
        case_id=case_id,
        question=question,
        failure_mode=failure_mode,
        gold_value=gold_value,
        interpretation_count=len(interpretations),
        observed_answer_shapes=observed_shapes,
        observed_numeric_values=values,
        closest_numeric_values=closest_values,
        issue_tags=issue_tags,
        likely_failure=_likely_failure(issue_tags, recommended),
        recommended_next_intervention=recommended,
    )


def _issue_tags(
    *,
    question: str,
    failure_mode: str,
    quality_case: dict[str, typing.Any],
    interpretations: list[dict[str, typing.Any]],
) -> tuple[str, ...]:
    lowered_question = question.lower()
    lowered_mode = failure_mode.lower()
    text = "\n".join(
        interpretation_quality_review._interpretation_text(item)
        for item in interpretations
    ).lower()
    tags: list[str] = []
    if "no_gold_compatible_interpretation" in quality_case.get("quality_issues", []):
        tags.append("missing_gold_numeric_reading")
    if "uses_runtime_current_date_anchor" in quality_case.get("warning_issues", []):
        tags.append("runtime_current_date_anchor")
    if (
        "past " in lowered_question
        or "last " in lowered_question
        or "this year" in lowered_question
        or "timewindow" in lowered_mode
    ):
        tags.append("temporal_anchor_or_window_policy")
    if "sum" in lowered_mode or lowered_question.startswith("how much"):
        tags.append("aggregation_without_enumerated_total")
    if any(term in lowered_question for term in ("different", "currently own")):
        tags.append("canonical_unit_enumeration_needed")
    if any(
        term in lowered_question
        for term in ("art-related", "citrus", "cuisines", "albums", "eps")
    ):
        tags.append("broad_semantic_category_boundary")
    if any(term in lowered_question for term in ("wedding", "events")):
        tags.append("event_instance_coreference_needed")
    if any(term in text for term in ("unknown", "not recorded", "indeterminate")):
        tags.append("provider_deferred_to_unknown_instead_of_enumerating")
    seen: set[str] = set()
    return tuple(item for item in tags if not (item in seen or seen.add(item)))


def _recommended_intervention(
    *,
    question: str,
    failure_mode: str,
    issue_tags: tuple[str, ...],
) -> str:
    lowered_question = question.lower()
    lowered_mode = failure_mode.lower()
    if "aggregation_without_enumerated_total" in issue_tags:
        return "verifier_enumeration_probe"
    if "event_instance_coreference_needed" in issue_tags:
        return "event_instance_coreference_hint"
    if "canonical_unit_enumeration_needed" in issue_tags:
        return "candidate_unit_enumeration_hint"
    if "broad_semantic_category_boundary" in issue_tags:
        return "semantic_boundary_targeted_infill"
    if (
        "temporal_anchor_or_window_policy" in issue_tags
        or "timewindow" in lowered_mode
        or "this year" in lowered_question
    ):
        return "temporal_anchor_policy_hint"
    return "targeted_infill_with_answer_variable_hint"


def _likely_failure(
    issue_tags: tuple[str, ...],
    recommended_intervention: str,
) -> str:
    if recommended_intervention == "verifier_enumeration_probe":
        return (
            "The provider discussed aggregation policies but did not enumerate "
            "the source amounts or durations needed to reach the target reading."
        )
    if recommended_intervention == "candidate_unit_enumeration_hint":
        return (
            "The provider stayed at interpretation level and did not bind the "
            "canonical answer units that the later compiler would need."
        )
    if recommended_intervention == "event_instance_coreference_hint":
        return (
            "The provider did not resolve event instances and cross-session "
            "coreference strongly enough to expose the missing count."
        )
    if recommended_intervention == "semantic_boundary_targeted_infill":
        return (
            "The provider explored category boundaries, but not the boundary "
            "that recovers the missing gold-compatible reading."
        )
    if "runtime_current_date_anchor" in issue_tags:
        return (
            "The provider spent interpretation budget on runtime-date anchoring "
            "instead of the evidence-time reading needed for the eval row."
        )
    return (
        "The provider produced diverse interpretations, but none stated a "
        "gold-compatible numeric answer reading."
    )


def _observed_numeric_values(
    interpretations: list[dict[str, typing.Any]],
) -> tuple[float, ...]:
    values: set[float] = set()
    for interpretation in interpretations:
        values.update(
            interpretation_quality_review._extract_numeric_values(
                interpretation_quality_review._interpretation_text(interpretation)
            )
        )
    return tuple(sorted(values))


def _closest_values(
    values: tuple[float, ...],
    gold_value: float | None,
    *,
    limit: int = 5,
) -> tuple[float, ...]:
    if gold_value is None:
        return values[:limit]
    return tuple(
        sorted(values, key=lambda item: (abs(item - gold_value), item))[:limit]
    )


def _as_float(value: typing.Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    return typing.cast(
        "dict[str, typing.Any]",
        json.loads(pathlib.Path(path).read_text(encoding="utf-8")),
    )


def _load_optional_json(path: str | pathlib.Path) -> dict[str, typing.Any]:
    json_path = pathlib.Path(path)
    if not json_path.exists():
        return {}
    return _load_json(json_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-review",
        type=pathlib.Path,
        default=interpretation_infill.DEFAULT_POST_INFILL_QUALITY_REVIEW_PATH,
    )
    parser.add_argument(
        "--infilled-outputs",
        type=pathlib.Path,
        default=interpretation_infill.DEFAULT_INFILLED_OUTPUTS_PATH,
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=ambiguity_fail18.DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--infill-report",
        type=pathlib.Path,
        default=interpretation_infill.DEFAULT_INFILL_REPORT_PATH,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_INFILL_DIAGNOSTICS_PATH,
    )
    args = parser.parse_args(argv)

    artifact = build_fail18_infill_diagnostics(
        post_review_path=args.post_review,
        infilled_outputs_path=args.infilled_outputs,
        manifest_path=args.manifest,
        infill_report_path=args.infill_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"{json.dumps(artifact, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

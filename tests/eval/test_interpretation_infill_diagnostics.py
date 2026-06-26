from __future__ import annotations

import json
import pathlib

import simba.eval.interpretation_infill_diagnostics as diagnostics


def test_infill_diagnostics_classifies_blocked_rows(
    tmp_path: pathlib.Path,
) -> None:
    post_review = tmp_path / "post_review.json"
    outputs = tmp_path / "outputs.jsonl"
    manifest = tmp_path / "manifest.json"
    infill_report = tmp_path / "infill_report.json"
    post_review.write_text(
        json.dumps(
            {
                "summary": {"rows_not_useful_for_compilation": 2},
                "cases": [
                    {
                        "case_id": "q_sum",
                        "gold_value": 12_000,
                        "quality_issues": ["no_gold_compatible_interpretation"],
                        "warning_issues": [],
                        "useful_for_compilation": False,
                    },
                    {
                        "case_id": "q_entity",
                        "gold_value": 4,
                        "quality_issues": ["no_gold_compatible_interpretation"],
                        "warning_issues": ["uses_runtime_current_date_anchor"],
                        "useful_for_compilation": False,
                    },
                    {
                        "case_id": "q_ok",
                        "gold_value": 2,
                        "quality_issues": [],
                        "warning_issues": [],
                        "useful_for_compilation": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    outputs.write_text(
        "\n".join(
            [
                json.dumps(
                    _row(
                        "q_sum",
                        [
                            _interpretation(
                                "i1",
                                "Add only the Facebook 2,000 reach, excluding "
                                "10,000 followers.",
                                "sum",
                            )
                        ],
                    )
                ),
                json.dumps(
                    _row(
                        "q_entity",
                        [
                            _interpretation(
                                "i1",
                                "Count two instruments under the current date.",
                                "count",
                            )
                        ],
                    )
                ),
                json.dumps(_row("q_ok", [])),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q_sum",
                    "question": "What was the total number of people reached?",
                    "failure_mode": "B_sum_value_readerfixed",
                },
                {
                    "question_id": "q_entity",
                    "question": "How many musical instruments do I currently own?",
                    "failure_mode": "A_overcount_distinct_readerfixed",
                },
                {
                    "question_id": "q_ok",
                    "question": "How many events?",
                    "failure_mode": "test",
                },
            ]
        ),
        encoding="utf-8",
    )
    infill_report.write_text(
        json.dumps({"quality_delta": {"rows_useful_for_compilation": {"delta": 2}}}),
        encoding="utf-8",
    )

    artifact = diagnostics.build_fail18_infill_diagnostics(
        post_review_path=post_review,
        infilled_outputs_path=outputs,
        manifest_path=manifest,
        infill_report_path=infill_report,
    )

    assert artifact["summary"]["blocked_rows"] == 2
    assert artifact["decision"]["candidate_unit_compilation_should_start"] is False
    by_id = {case["case_id"]: case for case in artifact["blocked_cases"]}
    assert by_id["q_sum"]["recommended_next_intervention"] == (
        "verifier_enumeration_probe"
    )
    assert by_id["q_entity"]["recommended_next_intervention"] == (
        "candidate_unit_enumeration_hint"
    )
    assert "runtime_current_date_anchor" in by_id["q_entity"]["issue_tags"]
    assert by_id["q_sum"]["closest_numeric_values"] == [10000.0, 2000.0]


def test_infill_diagnostics_routes_event_coreference() -> None:
    case = diagnostics.diagnose_blocked_row(
        quality_case={
            "case_id": "q_event",
            "gold_value": 3,
            "quality_issues": ["no_gold_compatible_interpretation"],
            "warning_issues": [],
        },
        infilled_row={
            "interpretations": [
                _interpretation("i1", "Count one wedding event.", "count")
            ]
        },
        manifest_row={
            "question_id": "q_event",
            "question": "How many weddings have I attended in this year?",
            "failure_mode": "D_underextraction",
        },
    )

    assert case.recommended_next_intervention == "event_instance_coreference_hint"
    assert "event_instance_coreference_needed" in case.issue_tags


def _row(
    case_id: str,
    interpretations: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "parse_status": "parsed",
        "provider_exit_code": 0,
        "provider_timed_out": False,
        "interpretations": interpretations,
    }


def _interpretation(
    interpretation_id: str,
    text: str,
    expected_answer_shape: str,
) -> dict[str, object]:
    return {
        "interpretation_id": interpretation_id,
        "natural_language_interpretation": text,
        "ambiguity_types": ["scope_ambiguous"],
        "assumptions": [],
        "expected_answer_shape": expected_answer_shape,
    }

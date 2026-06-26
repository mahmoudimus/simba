from __future__ import annotations

import json
import pathlib

import simba.eval.interpretation_quality_review as interpretation_quality_review


def test_quality_review_marks_gold_compatible_row_useful(
    tmp_path: pathlib.Path,
) -> None:
    outputs = tmp_path / "outputs.jsonl"
    manifest = tmp_path / "manifest.json"
    gate1_report = tmp_path / "gate1_report.json"
    outputs.write_text(
        json.dumps(
            {
                "case_id": "q1",
                "parse_status": "parsed",
                "provider_exit_code": 0,
                "provider_timed_out": False,
                "interpretations": [
                    _interpretation(
                        "i1",
                        "Count only completed events, yielding 2.",
                        "count",
                    ),
                    _interpretation(
                        "i2",
                        "Count a broader reading that includes planned events, "
                        "yielding 3.",
                        "range",
                    ),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many events did I attend?",
                    "failure_mode": "A_overcount_distinct",
                    "gold_answer": "3",
                }
            ]
        ),
        encoding="utf-8",
    )
    gate1_report.write_text(
        json.dumps({"gate_status": "slice1b_ready_for_review"}),
        encoding="utf-8",
    )

    review = interpretation_quality_review.build_fail18_quality_review(
        outputs_path=outputs,
        manifest_path=manifest,
        report_path=gate1_report,
    )

    assert review["summary"]["rows_useful_for_compilation"] == 1
    case = review["cases"][0]
    assert case["gold_value"] == 3.0
    assert case["gold_compatible_interpretation_ids"] == ["i2"]
    assert case["quality_issues"] == []
    assert case["useful_for_compilation"] is True


def test_quality_review_blocks_parseable_provider_failures(
    tmp_path: pathlib.Path,
) -> None:
    outputs = tmp_path / "outputs.jsonl"
    manifest = tmp_path / "manifest.json"
    outputs.write_text(
        json.dumps(
            {
                "case_id": "q1",
                "parse_status": "parsed",
                "provider_exit_code": 1,
                "provider_timed_out": False,
                "interpretations": [
                    _interpretation("i1", "Count one event.", "count"),
                    _interpretation("i2", "Count two events.", "count"),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many events did I attend?",
                    "failure_mode": "A_overcount_distinct",
                    "gold_answer": "2",
                }
            ]
        ),
        encoding="utf-8",
    )

    review = interpretation_quality_review.build_fail18_quality_review(
        outputs_path=outputs,
        manifest_path=manifest,
        report_path=tmp_path / "missing.json",
    )

    case = review["cases"][0]
    assert case["quality_issues"] == ["provider_failed"]
    assert case["useful_for_compilation"] is False
    assert review["summary"]["issue_counts"] == {"provider_failed": 1}


def test_quality_review_blocks_rows_without_gold_compatible_interpretation(
    tmp_path: pathlib.Path,
) -> None:
    outputs = tmp_path / "outputs.jsonl"
    manifest = tmp_path / "manifest.json"
    outputs.write_text(
        json.dumps(
            {
                "case_id": "q1",
                "parse_status": "parsed",
                "provider_exit_code": 0,
                "provider_timed_out": False,
                "interpretations": [
                    _interpretation("i1", "Count one event.", "count"),
                    _interpretation("i2", "Count two events.", "range"),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many events did I attend?",
                    "failure_mode": "A_overcount_distinct",
                    "gold_answer": "4",
                }
            ]
        ),
        encoding="utf-8",
    )

    review = interpretation_quality_review.build_fail18_quality_review(
        outputs_path=outputs,
        manifest_path=manifest,
        report_path=tmp_path / "missing.json",
    )

    case = review["cases"][0]
    assert case["gold_compatible_interpretation_ids"] == []
    assert case["quality_issues"] == ["no_gold_compatible_interpretation"]
    assert case["useful_for_compilation"] is False


def test_quality_review_preserves_decimal_gold_values(
    tmp_path: pathlib.Path,
) -> None:
    outputs = tmp_path / "outputs.jsonl"
    manifest = tmp_path / "manifest.json"
    outputs.write_text(
        json.dumps(
            {
                "case_id": "q1",
                "parse_status": "parsed",
                "provider_exit_code": 0,
                "provider_timed_out": False,
                "interpretations": [
                    _interpretation("i1", "No qualifying time, so 0 hours.", "sum"),
                    _interpretation(
                        "i2",
                        "A narrow yoga-only reading gives 0.5 hours.",
                        "sum",
                    ),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many hours of yoga did I do last week?",
                    "failure_mode": "D_underextraction_zero",
                    "gold_answer": "0.5 hours",
                }
            ]
        ),
        encoding="utf-8",
    )

    review = interpretation_quality_review.build_fail18_quality_review(
        outputs_path=outputs,
        manifest_path=manifest,
        report_path=tmp_path / "missing.json",
    )

    case = review["cases"][0]
    assert case["gold_value"] == 0.5
    assert case["gold_compatible_interpretation_ids"] == ["i2"]


def _interpretation(
    interpretation_id: str,
    text: str,
    expected_answer_shape: str,
) -> dict[str, object]:
    return {
        "interpretation_id": interpretation_id,
        "natural_language_interpretation": text,
        "ambiguity_types": ["scope_ambiguous", "value_mapping_ambiguous"],
        "assumptions": [],
        "expected_answer_shape": expected_answer_shape,
    }

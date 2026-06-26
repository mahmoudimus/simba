from __future__ import annotations

import json
import pathlib

import simba.eval.interpretation_infill as interpretation_infill


def test_infill_payloads_target_blocked_rows_without_gold_or_failure_mode(
    tmp_path: pathlib.Path,
) -> None:
    quality_review = tmp_path / "quality.json"
    payloads = tmp_path / "payloads.json"
    outputs = tmp_path / "outputs.jsonl"
    quality_review.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "q1",
                        "useful_for_compilation": False,
                        "quality_issues": ["no_gold_compatible_interpretation"],
                        "warning_issues": [],
                        "missing_ambiguity_types": [],
                        "expected_answer_shapes": ["count", "range"],
                        "observed_answer_shapes": ["count"],
                        "gold_value": 3,
                        "failure_mode": "A_overcount_distinct",
                    },
                    {
                        "case_id": "q2",
                        "useful_for_compilation": True,
                        "quality_issues": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    payloads.write_text(
        json.dumps(
            {
                "payloads": [
                    _source_payload("q1", "How many events count?"),
                    _source_payload("q2", "How many tasks count?"),
                ]
            }
        ),
        encoding="utf-8",
    )
    outputs.write_text(
        json.dumps(_provider_row("q1", [_interpretation("i1", "Count one.", "count")]))
        + "\n",
        encoding="utf-8",
    )

    artifact = interpretation_infill.build_fail18_infill_payload_artifact(
        quality_review_path=quality_review,
        gate1_payloads_path=payloads,
        gate1_outputs_path=outputs,
    )

    assert artifact["total"] == 1
    assert artifact["blocked_case_ids"] == ["q1"]
    assert artifact["provider_visibility"] == {
        "gold_answer_visible": False,
        "gold_value_visible": False,
        "failure_mode_visible": False,
    }
    encoded_payload = json.dumps(artifact["payloads"][0])
    assert "gold_value" not in encoded_payload
    assert "A_overcount_distinct" not in encoded_payload
    assert "no_gold_compatible_interpretation" in encoded_payload


def test_merge_infill_rows_adds_new_and_filters_duplicates() -> None:
    original = [
        _provider_row(
            "q1",
            [
                _interpretation("i1", "Count one event.", "count"),
            ],
        )
    ]
    infill = [
        _provider_row(
            "q1",
            [
                _interpretation("i1", "Count a duplicate id.", "count"),
                _interpretation("i2", "Count two events.", "count"),
            ],
        )
    ]

    merged, results = interpretation_infill.merge_infill_rows(original, infill)

    assert len(merged[0]["interpretations"]) == 2
    assert merged[0]["infill_added_interpretation_ids"] == ["i2"]
    assert results[0].to_dict() == {
        "case_id": "q1",
        "original_count": 1,
        "infill_count": 2,
        "added_count": 1,
        "duplicate_count": 1,
        "added_interpretation_ids": ["i2"],
        "duplicate_interpretation_ids": ["i1"],
    }


def test_infill_report_records_quality_delta(tmp_path: pathlib.Path) -> None:
    original_outputs = tmp_path / "original.jsonl"
    infill_outputs = tmp_path / "infill.jsonl"
    manifest = tmp_path / "manifest.json"
    quality_review = tmp_path / "quality.json"
    payloads = tmp_path / "payloads.json"
    infilled_outputs = tmp_path / "infilled.jsonl"
    post_review = tmp_path / "post_review.json"
    original_outputs.write_text(
        json.dumps(_provider_row("q1", [_interpretation("i1", "Count one.", "count")]))
        + "\n",
        encoding="utf-8",
    )
    infill_outputs.write_text(
        json.dumps(
            _provider_row(
                "q1",
                [_interpretation("i2", "Count the broader reading as 3.", "range")],
            )
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many events count?",
                    "failure_mode": "A_overcount_distinct",
                    "gold_answer": "3",
                }
            ]
        ),
        encoding="utf-8",
    )
    quality_review.write_text(
        json.dumps(
            {
                "summary": {
                    "rows_useful_for_compilation": 0,
                    "rows_not_useful_for_compilation": 1,
                    "rows_with_gold_compatible_interpretation": 0,
                    "rows_missing_gold_compatible_interpretation": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    payloads.write_text(json.dumps({"total": 1}), encoding="utf-8")

    report = interpretation_infill.build_fail18_infill_report(
        original_outputs_path=original_outputs,
        infill_outputs_path=infill_outputs,
        manifest_path=manifest,
        quality_review_path=quality_review,
        payloads_path=payloads,
        infilled_outputs_path=infilled_outputs,
        post_review_path=post_review,
    )

    assert report["added_interpretation_count"] == 1
    assert report["quality_delta"]["rows_useful_for_compilation"] == {
        "before": 0,
        "after": 1,
        "delta": 1,
    }
    assert report["acceptance"]["candidate_unit_compilation_attempted"] is False
    assert infilled_outputs.exists()
    assert post_review.exists()


def _source_payload(case_id: str, question: str) -> dict[str, object]:
    return {
        "allowed_ambiguity_types": ["scope_ambiguous"],
        "case": {
            "id": case_id,
            "question": question,
            "evidence_sessions": [{"session_id": "evidence_001", "text": "user: x"}],
        },
    }


def _provider_row(
    case_id: str,
    interpretations: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "parse_status": "parsed",
        "provider_exit_code": 0,
        "provider_timed_out": False,
        "interpretations": interpretations,
        "parse_errors": [],
    }


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

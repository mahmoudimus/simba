from __future__ import annotations

import json
import pathlib

from simba.eval import candidate_unit_diagnostics


def _write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(f"{json.dumps(value)}\n", encoding="utf-8")


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def test_candidate_unit_diagnostics_flags_retrieval_and_exclusion_gaps(
    tmp_path: pathlib.Path,
) -> None:
    report_path = tmp_path / "report.json"
    outputs_path = tmp_path / "outputs.jsonl"
    provenance_path = tmp_path / "provenance.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        report_path,
        {
            "cases": [
                {
                    "case_id": "q1",
                    "question": "How many items?",
                    "failure_mode": "test",
                    "gold_value": 3,
                    "recomputed_answer": 1,
                    "quality_issues": ["recomputed_answer_misses_gold"],
                    "useful_for_candidate_compilation": False,
                }
            ]
        },
    )
    _write_jsonl(
        outputs_path,
        [
            {
                "case_id": "q1",
                "candidate_units": [
                    {
                        "unit_id": "u1",
                        "label": "included item",
                        "status": "included",
                        "merge_target": None,
                        "reason_code": "included",
                        "reason": "Included.",
                        "evidence_session_ids": ["evidence_001"],
                        "evidence_spans": ["included span"],
                    },
                    {
                        "unit_id": "u2",
                        "label": "wrongly excluded item",
                        "status": "excluded",
                        "merge_target": None,
                        "reason_code": "excluded",
                        "reason": "Excluded.",
                        "evidence_session_ids": ["evidence_002"],
                        "evidence_spans": ["excluded span"],
                    },
                ],
            }
        ],
    )
    _write_json(
        provenance_path,
        {
            "evidence_provenance": {
                "q1": {
                    "evidence_001": {
                        "raw_session_id": "answer_1",
                        "selection_rank": 1,
                        "selection_score": 10,
                    },
                    "evidence_002": {
                        "raw_session_id": "answer_2",
                        "selection_rank": 2,
                        "selection_score": 9,
                    },
                }
            }
        },
    )
    _write_json(
        manifest_path,
        [
            {
                "question_id": "q1",
                "answer_session_ids": ["answer_1", "answer_2", "answer_3"],
            }
        ],
    )

    artifact = candidate_unit_diagnostics.build_candidate_unit_diagnostics(
        report_path=report_path,
        outputs_path=outputs_path,
        payload_provenance_path=provenance_path,
        manifest_path=manifest_path,
    )
    case = artifact["cases"][0]

    assert case["missing_answer_session_ids_from_payload"] == ["answer_3"]
    assert case["present_answer_session_ids_without_included_unit"] == ["answer_2"]
    assert case["excluded_units_from_answer_sessions"][0]["unit_id"] == "u2"
    assert case["recommended_next_intervention"] == "retrieval_or_payload_budget_fix"
    assert artifact["summary"]["issue_counts"] == {
        "excluded_answer_session_unit": 1,
        "included_unit_under_count": 1,
        "missing_answer_session_from_payload": 1,
        "present_answer_session_without_included_unit": 1,
    }


def test_candidate_unit_diagnostics_flags_present_answer_session_without_any_unit(
    tmp_path: pathlib.Path,
) -> None:
    report_path = tmp_path / "report.json"
    outputs_path = tmp_path / "outputs.jsonl"
    provenance_path = tmp_path / "provenance.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        report_path,
        {
            "cases": [
                {
                    "case_id": "q1",
                    "question": "How many items?",
                    "failure_mode": "test",
                    "gold_value": 2,
                    "recomputed_answer": 1,
                    "quality_issues": ["recomputed_answer_misses_gold"],
                    "useful_for_candidate_compilation": False,
                }
            ]
        },
    )
    _write_jsonl(
        outputs_path,
        [
            {
                "case_id": "q1",
                "candidate_units": [
                    {
                        "unit_id": "u1",
                        "label": "included item",
                        "status": "included",
                        "merge_target": None,
                        "reason_code": "included",
                        "reason": "Included.",
                        "evidence_session_ids": ["evidence_001"],
                        "evidence_spans": ["included span"],
                    }
                ],
            }
        ],
    )
    _write_json(
        provenance_path,
        {
            "evidence_provenance": {
                "q1": {
                    "evidence_001": {"raw_session_id": "answer_1"},
                    "evidence_002": {"raw_session_id": "answer_2"},
                }
            }
        },
    )
    _write_json(
        manifest_path,
        [{"question_id": "q1", "answer_session_ids": ["answer_1", "answer_2"]}],
    )

    artifact = candidate_unit_diagnostics.build_candidate_unit_diagnostics(
        report_path=report_path,
        outputs_path=outputs_path,
        payload_provenance_path=provenance_path,
        manifest_path=manifest_path,
    )
    case = artifact["cases"][0]

    assert case["present_answer_session_ids_without_any_unit"] == ["answer_2"]
    assert "present_answer_session_without_candidate_unit" in case[
        "diagnostic_issues"
    ]
    assert (
        case["recommended_next_intervention"]
        == "evidence_to_unit_coverage_check"
    )


def test_candidate_unit_diagnostics_flags_overmerge_risk(
    tmp_path: pathlib.Path,
) -> None:
    report_path = tmp_path / "report.json"
    outputs_path = tmp_path / "outputs.jsonl"
    provenance_path = tmp_path / "provenance.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        report_path,
        {
            "cases": [
                {
                    "case_id": "q1",
                    "question": "How many events?",
                    "failure_mode": "test",
                    "gold_value": 2,
                    "recomputed_answer": 1,
                    "quality_issues": ["recomputed_answer_misses_gold"],
                    "useful_for_candidate_compilation": False,
                }
            ]
        },
    )
    _write_jsonl(
        outputs_path,
        [
            {
                "case_id": "q1",
                "candidate_units": [
                    {
                        "unit_id": "event_1",
                        "label": "first event",
                        "status": "included",
                        "merge_target": None,
                        "reason_code": "included",
                        "reason": "Included.",
                        "evidence_session_ids": ["evidence_001"],
                        "evidence_spans": ["first event span"],
                    },
                    {
                        "unit_id": "event_2",
                        "label": "second event",
                        "status": "merged",
                        "merge_target": "event_1",
                        "reason_code": "same_event",
                        "reason": "Merged.",
                        "evidence_session_ids": ["evidence_002"],
                        "evidence_spans": ["second event span"],
                    },
                ],
            }
        ],
    )
    _write_json(
        provenance_path,
        {
            "evidence_provenance": {
                "q1": {
                    "evidence_001": {"raw_session_id": "answer_1"},
                    "evidence_002": {"raw_session_id": "answer_2"},
                }
            }
        },
    )
    _write_json(
        manifest_path,
        [{"question_id": "q1", "answer_session_ids": ["answer_1", "answer_2"]}],
    )

    artifact = candidate_unit_diagnostics.build_candidate_unit_diagnostics(
        report_path=report_path,
        outputs_path=outputs_path,
        payload_provenance_path=provenance_path,
        manifest_path=manifest_path,
    )
    case = artifact["cases"][0]

    assert case["merged_required_units"][0]["unit_id"] == "event_2"
    assert "overmerged_distinct_unit_possible" in case["diagnostic_issues"]


def test_candidate_unit_diagnostics_flags_included_non_answer_session_units(
    tmp_path: pathlib.Path,
) -> None:
    report_path = tmp_path / "report.json"
    outputs_path = tmp_path / "outputs.jsonl"
    provenance_path = tmp_path / "provenance.json"
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        report_path,
        {
            "cases": [
                {
                    "case_id": "q1",
                    "question": "How much money?",
                    "failure_mode": "test",
                    "gold_value": 1000,
                    "recomputed_answer": 6000,
                    "quality_issues": ["recomputed_answer_misses_gold"],
                    "useful_for_candidate_compilation": False,
                }
            ]
        },
    )
    _write_jsonl(
        outputs_path,
        [
            {
                "case_id": "q1",
                "candidate_units": [
                    {
                        "unit_id": "required_amount",
                        "label": "required fundraiser",
                        "status": "included",
                        "merge_target": None,
                        "reason_code": "included",
                        "reason": "Answer-bearing event.",
                        "evidence_session_ids": ["evidence_001"],
                        "evidence_spans": ["raised $1000"],
                    },
                    {
                        "unit_id": "distractor_amount",
                        "label": "distractor fundraiser",
                        "status": "included",
                        "merge_target": None,
                        "reason_code": "included",
                        "reason": "Wrongly included.",
                        "evidence_session_ids": ["evidence_002"],
                        "evidence_spans": ["raised $5000"],
                    },
                ],
            }
        ],
    )
    _write_json(
        provenance_path,
        {
            "evidence_provenance": {
                "q1": {
                    "evidence_001": {"raw_session_id": "answer_1"},
                    "evidence_002": {"raw_session_id": "distractor_1"},
                }
            }
        },
    )
    _write_json(
        manifest_path,
        [{"question_id": "q1", "answer_session_ids": ["answer_1"]}],
    )

    artifact = candidate_unit_diagnostics.build_candidate_unit_diagnostics(
        report_path=report_path,
        outputs_path=outputs_path,
        payload_provenance_path=provenance_path,
        manifest_path=manifest_path,
    )
    case = artifact["cases"][0]

    assert case["included_units_from_non_answer_sessions"][0]["unit_id"] == (
        "distractor_amount"
    )
    assert "included_unit_over_count" in case["diagnostic_issues"]
    assert "included_non_answer_session_unit" in case["diagnostic_issues"]
    assert case["recommended_next_intervention"] == "inclusion_scope_verifier"

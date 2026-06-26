from __future__ import annotations

import json
import pathlib

from simba.eval import candidate_unit_runner


def _valid_response(**overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "case_id": "q1",
        "answer_variable": "entity",
        "individuation_policy": "canonical_entity",
        "aggregation": "count_distinct",
        "candidate_units": [
            {
                "unit_id": "unit_1",
                "label": "blue blazer",
                "status": "included",
                "merge_target": None,
                "value": None,
                "unit": None,
                "evidence_session_ids": ["evidence_001"],
                "evidence_spans": ["need to pick up my blue blazer"],
                "reason_code": "store_pickup_obligation",
                "reason": "The user reports a store pickup obligation.",
            },
            {
                "unit_id": "unit_2",
                "label": "black jeans",
                "status": "excluded",
                "merge_target": None,
                "value": None,
                "unit": None,
                "evidence_session_ids": ["evidence_002"],
                "evidence_spans": ["wore my new black jeans"],
                "reason_code": "already_owned",
                "reason": "The user already owns the jeans.",
            },
        ],
        "facts": ["action(user, unit_1, pick_up)."],
        "query": "answer(N) :- count_distinct(O, action(user,O,pick_up), N).",
        "computed_answer": 1,
        "rationale": "One clothing item is still a store pickup obligation.",
    }
    response.update(overrides)
    return response


def test_parse_candidate_unit_response_accepts_valid_output() -> None:
    result = candidate_unit_runner.parse_candidate_unit_response(
        json.dumps(_valid_response()),
        expected_case_id="q1",
    )

    assert result.parse_status == candidate_unit_runner.PARSE_STATUS_PARSED
    assert result.case_id == "q1"
    assert result.aggregation == "count_distinct"
    assert [unit.unit_id for unit in result.candidate_units] == ["unit_1", "unit_2"]
    assert result.candidate_units[0].evidence_spans == (
        "need to pick up my blue blazer",
    )


def test_parse_candidate_unit_response_rejects_duplicate_unit_ids() -> None:
    response = _valid_response(
        candidate_units=[
            {
                "unit_id": "dup",
                "label": "first",
                "status": "included",
                "merge_target": None,
                "value": None,
                "unit": None,
                "evidence_session_ids": ["evidence_001"],
                "evidence_spans": ["first span"],
                "reason_code": "included",
                "reason": "Included.",
            },
            {
                "unit_id": "dup",
                "label": "second",
                "status": "included",
                "merge_target": None,
                "value": None,
                "unit": None,
                "evidence_session_ids": ["evidence_002"],
                "evidence_spans": ["second span"],
                "reason_code": "included",
                "reason": "Included.",
            },
        ]
    )

    result = candidate_unit_runner.parse_candidate_unit_response(
        json.dumps(response),
        expected_case_id="q1",
    )

    assert result.parse_status == candidate_unit_runner.PARSE_STATUS_INVALID_SCHEMA
    assert result.candidate_units == ()
    assert "duplicate unit_id values: dup" in result.parse_errors


def test_parse_candidate_unit_response_rejects_bad_merge_target() -> None:
    response = _valid_response(
        candidate_units=[
            {
                "unit_id": "unit_1",
                "label": "first",
                "status": "merged",
                "merge_target": "missing",
                "value": None,
                "unit": None,
                "evidence_session_ids": ["evidence_001"],
                "evidence_spans": ["first span"],
                "reason_code": "same_entity",
                "reason": "Same object.",
            }
        ]
    )

    result = candidate_unit_runner.parse_candidate_unit_response(
        json.dumps(response),
        expected_case_id="q1",
    )

    assert result.parse_status == candidate_unit_runner.PARSE_STATUS_INVALID_SCHEMA
    assert "merge_target 'missing' does not exist" in "\n".join(result.parse_errors)


def test_review_candidate_unit_row_recomputes_count_against_gold() -> None:
    parsed = candidate_unit_runner.parse_candidate_unit_response(
        json.dumps(_valid_response(computed_answer=1)),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="candidate_unit_compiler_v1",
        raw_output="{}",
    )
    row.update({"provider_exit_code": 0, "provider_timed_out": False})

    review = candidate_unit_runner.review_candidate_unit_row(
        row,
        {
            "question_id": "q1",
            "question": "How many items?",
            "failure_mode": "test",
            "gold_answer": 1,
        },
    )

    assert review["provider_answer_matches_recomputed"] is True
    assert review["recomputed_answer_matches_gold"] is True
    assert review["useful_for_candidate_compilation"] is True
    assert review["quality_issues"] == []


def test_review_candidate_unit_row_flags_provider_answer_mismatch() -> None:
    parsed = candidate_unit_runner.parse_candidate_unit_response(
        json.dumps(_valid_response(computed_answer=2)),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="candidate_unit_compiler_v1",
        raw_output="{}",
    )
    row.update({"provider_exit_code": 0, "provider_timed_out": False})

    review = candidate_unit_runner.review_candidate_unit_row(
        row,
        {
            "question_id": "q1",
            "question": "How many items?",
            "failure_mode": "test",
            "gold_answer": 1,
        },
    )

    assert "provider_computed_answer_mismatch" in review["quality_issues"]
    assert review["useful_for_candidate_compilation"] is False


def test_recompute_answer_sums_included_numeric_values() -> None:
    response = _valid_response(
        answer_variable="scalar_value",
        individuation_policy="scalar_value",
        aggregation="sum",
        computed_answer=30,
        candidate_units=[
            {
                "unit_id": "u1",
                "label": "first donation",
                "status": "included",
                "merge_target": None,
                "value": 10,
                "unit": "dollars",
                "evidence_session_ids": ["evidence_001"],
                "evidence_spans": ["raised $10"],
                "reason_code": "reported_amount",
                "reason": "Reported as raised.",
            },
            {
                "unit_id": "u2",
                "label": "second donation",
                "status": "included",
                "merge_target": None,
                "value": 20,
                "unit": "dollars",
                "evidence_session_ids": ["evidence_002"],
                "evidence_spans": ["raised $20"],
                "reason_code": "reported_amount",
                "reason": "Reported as raised.",
            },
        ],
    )
    parsed = candidate_unit_runner.parse_candidate_unit_response(
        json.dumps(response),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="candidate_unit_compiler_v1",
        raw_output="{}",
    )

    assert candidate_unit_runner.recompute_answer(row) == (30.0, [])


def test_build_fail18_candidate_unit_payloads_hide_gold_and_target_blocked_rows(
    tmp_path: pathlib.Path,
) -> None:
    quality_review_path = tmp_path / "quality.json"
    payloads_path = tmp_path / "payloads.json"
    quality_review_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "blocked",
                        "useful_for_compilation": False,
                        "quality_issues": ["no_gold_compatible_interpretation"],
                        "warning_issues": [],
                        "observed_answer_shapes": ["count"],
                    },
                    {
                        "case_id": "ok",
                        "useful_for_compilation": True,
                        "quality_issues": [],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    payloads_path.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "blocked",
                            "question": "How many items?",
                            "evidence_sessions": [
                                {
                                    "session_id": "evidence_001",
                                    "text": "user: I bought one item.",
                                }
                            ],
                        }
                    },
                    {
                        "case": {
                            "id": "ok",
                            "question": "How many ok items?",
                            "evidence_sessions": [],
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = candidate_unit_runner.build_fail18_candidate_unit_payload_artifact(
        quality_review_path=quality_review_path,
        gate1_payloads_path=payloads_path,
    )
    provider_payloads = json.dumps(artifact["payloads"]).lower()

    assert artifact["total"] == 1
    assert artifact["blocked_case_ids"] == ["blocked"]
    assert artifact["provider_visibility"]["gold_answer_visible"] is False
    assert artifact["payloads"][0]["output_schema"]["computed_answer"] == 0
    assert "not a quoted string" in "\n".join(
        artifact["payloads"][0]["compiler_contract"]
    )
    assert "candidate_units" in provider_payloads
    assert "gold_answer" not in provider_payloads

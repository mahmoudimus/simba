from __future__ import annotations

import json
import pathlib

from simba.eval import answer_unit_witness


def _valid_response(**overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "case_id": "q1",
        "answer_variable": "clothing items to pick up or return",
        "aggregation": "count_included",
        "units": [
            {
                "unit_id": "u1",
                "label": "blue blazer",
                "decision": "include",
                "borderline": False,
                "value": None,
                "unit": None,
                "evidence_session_id": "s1",
                "evidence_span": "pick up the blue blazer",
                "reason_code": "pickup_obligation",
                "reason": "The user needs to pick it up.",
            },
            {
                "unit_id": "u2",
                "label": "black jeans",
                "decision": "exclude",
                "borderline": True,
                "value": None,
                "unit": None,
                "evidence_session_id": "s2",
                "evidence_span": "already wearing the black jeans",
                "reason_code": "already_owned",
                "reason": "The jeans are already owned.",
            },
        ],
        "answer_number": 1,
        "rationale": "One included unit.",
    }
    response.update(overrides)
    return response


def test_parse_witness_response_accepts_json_envelope() -> None:
    envelope = {"result": json.dumps(_valid_response())}

    parsed = answer_unit_witness.parse_witness_response(
        json.dumps(envelope),
        expected_case_id="q1",
    )

    assert parsed.parse_status == answer_unit_witness.PARSE_STATUS_PARSED
    assert parsed.case_id == "q1"
    assert parsed.aggregation == "count_included"
    assert parsed.units[0].label == "blue blazer"
    assert parsed.answer_number == 1.0


def test_parse_witness_response_rejects_duplicate_unit_ids() -> None:
    response = _valid_response(
        units=[
            {
                "unit_id": "dup",
                "label": "first",
                "decision": "include",
                "borderline": False,
                "value": None,
                "unit": None,
                "evidence_session_id": "s1",
                "evidence_span": "first span",
                "reason_code": "first",
                "reason": "First.",
            },
            {
                "unit_id": "dup",
                "label": "second",
                "decision": "include",
                "borderline": False,
                "value": None,
                "unit": None,
                "evidence_session_id": "s2",
                "evidence_span": "second span",
                "reason_code": "second",
                "reason": "Second.",
            },
        ]
    )

    parsed = answer_unit_witness.parse_witness_response(
        json.dumps(response),
        expected_case_id="q1",
    )

    assert parsed.parse_status == answer_unit_witness.PARSE_STATUS_INVALID_SCHEMA
    assert "duplicate unit_id values: dup" in parsed.parse_errors


def test_recompute_answer_sums_included_values() -> None:
    parsed = answer_unit_witness.parse_witness_response(
        json.dumps(
            _valid_response(
                aggregation="sum_included",
                answer_number=30,
                units=[
                    {
                        "unit_id": "u1",
                        "label": "first donation",
                        "decision": "include",
                        "borderline": False,
                        "value": 10,
                        "unit": "dollars",
                        "evidence_session_id": "s1",
                        "evidence_span": "raised $10",
                        "reason_code": "reported_amount",
                        "reason": "Reported.",
                    },
                    {
                        "unit_id": "u2",
                        "label": "second donation",
                        "decision": "include",
                        "borderline": False,
                        "value": 20,
                        "unit": "dollars",
                        "evidence_session_id": "s2",
                        "evidence_span": "raised $20",
                        "reason_code": "reported_amount",
                        "reason": "Reported.",
                    },
                ],
            )
        ),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="answer_unit_witness_v1",
        raw_output="{}",
        sample_index=1,
    )

    assert answer_unit_witness.recompute_answer(row) == (30.0, [])


def test_review_witness_row_checks_arithmetic_and_span_grounding() -> None:
    parsed = answer_unit_witness.parse_witness_response(
        json.dumps(_valid_response()),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="answer_unit_witness_v1",
        raw_output="{}",
        sample_index=1,
    )
    row.update({"provider_exit_code": 0, "provider_timed_out": False})
    payload = {
        "case": {
            "id": "q1",
            "evidence_sessions": [
                {
                    "session_id": "s1",
                    "text": "USER: I need to pick up the blue blazer tomorrow.",
                },
                {
                    "session_id": "s2",
                    "text": "USER: I am already wearing the black jeans.",
                },
            ],
        }
    }

    review = answer_unit_witness.review_witness_row(
        row,
        {"question_id": "q1", "question": "How many?", "gold_answer": 1},
        payload,
    )

    assert review["answer_matches_recomputed"] is True
    assert review["recomputed_answer_matches_gold"] is True
    assert review["all_unit_spans_resolve"] is True
    assert review["quality_issues"] == []


def test_review_witness_row_flags_unresolved_span() -> None:
    parsed = answer_unit_witness.parse_witness_response(
        json.dumps(_valid_response()),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="answer_unit_witness_v1",
        raw_output="{}",
        sample_index=1,
    )
    row.update({"provider_exit_code": 0, "provider_timed_out": False})
    payload = {
        "case": {
            "id": "q1",
            "evidence_sessions": [{"session_id": "s1", "text": "different text"}],
        }
    }

    review = answer_unit_witness.review_witness_row(
        row,
        {"question_id": "q1", "question": "How many?", "gold_answer": 1},
        payload,
    )

    assert "unit_span_unresolved" in review["quality_issues"]
    assert review["all_unit_spans_resolve"] is False


def test_review_witness_row_corrects_contradicted_no_acquisition_exclusion() -> None:
    parsed = answer_unit_witness.parse_witness_response(
        json.dumps(
            _valid_response(
                answer_variable="plants acquired in the last month",
                units=[
                    {
                        "unit_id": "u1",
                        "label": "peace lily",
                        "decision": "include",
                        "borderline": False,
                        "value": None,
                        "unit": "plants",
                        "evidence_session_id": "s1",
                        "evidence_span": "got from the nursery two weeks ago",
                        "reason_code": "acquired_in_window",
                        "reason": "Recently acquired.",
                    },
                    {
                        "unit_id": "u2",
                        "label": "succulent",
                        "decision": "include",
                        "borderline": False,
                        "value": None,
                        "unit": "plants",
                        "evidence_session_id": "s1",
                        "evidence_span": "along with a succulent",
                        "reason_code": "acquired_in_window",
                        "reason": "Recently acquired.",
                    },
                    {
                        "unit_id": "u3",
                        "label": "snake plant",
                        "decision": "exclude",
                        "borderline": False,
                        "value": None,
                        "unit": "plants",
                        "evidence_session_id": "s2",
                        "evidence_span": "my snake plant has been doing great",
                        "reason_code": "no_acquisition_evidence",
                        "reason": "No acquisition evidence.",
                    },
                ],
                answer_number=2,
            )
        ),
        expected_case_id="q1",
    )
    row = parsed.to_output_dict(
        provider="test",
        prompt_version="answer_unit_witness_v1",
        raw_output="{}",
        sample_index=1,
    )
    row.update({"provider_exit_code": 0, "provider_timed_out": False})
    payload = {
        "case": {
            "id": "q1",
            "evidence_sessions": [
                {
                    "session_id": "s1",
                    "text": (
                        "USER: I got from the nursery two weeks ago along with "
                        "a succulent."
                    ),
                },
                {
                    "session_id": "s2",
                    "text": (
                        "USER: My snake plant has been doing great. "
                        "I should repot my snake plant, which I got from my "
                        "sister last month."
                    ),
                },
            ],
        }
    }

    review = answer_unit_witness.review_witness_row(
        row,
        {"question_id": "q1", "question": "How many plants?", "gold_answer": 3},
        payload,
    )

    assert review["answer_matches_recomputed"] is True
    assert review["provider_recomputed_answer"] == 2.0
    assert review["recomputed_answer"] == 3.0
    assert review["verifier_corrected_answer"] == 3.0
    assert review["recomputed_answer_matches_gold"] is True
    assert review["corrected_exclusion_count"] == 1
    assert "excluded_unit_contradicted" in review["quality_issues"]
    assert review["exclusion_contradictions"][0]["unit_id"] == "u3"
    assert (
        "got from my sister last month"
        in review["exclusion_contradictions"][0]["contradicting_span"]
    )


def test_exclusion_contradiction_filters_non_acquisition_senses() -> None:
    row = {
        "aggregation": "count_included",
        "units": [
            {
                "unit_id": "u1",
                "label": "snake plant",
                "decision": "exclude",
                "reason_code": "no_acquisition_evidence",
                "evidence_session_id": "s1",
                "evidence_span": "snake plant",
            }
        ],
    }
    blocked_texts = [
        "USER: I got rid of my snake plant last month.",
        "USER: I am thinking of getting a snake plant next month.",
        "USER: I didn't get a snake plant after all.",
    ]

    for text in blocked_texts:
        payload = {
            "case": {
                "id": "q1",
                "evidence_sessions": [{"session_id": "s1", "text": text}],
            }
        }
        assert answer_unit_witness.find_exclusion_contradictions(row, payload) == []


def test_exclusion_contradiction_does_not_cross_bind_other_entities() -> None:
    row = {
        "aggregation": "count_included",
        "units": [
            {
                "unit_id": "u1",
                "label": "basil plant",
                "decision": "exclude",
                "reason_code": "no_acquisition_evidence",
                "evidence_session_id": "s1",
                "evidence_span": "my basil plant on the balcony",
            }
        ],
    }
    payload = {
        "case": {
            "id": "q1",
            "evidence_sessions": [
                {
                    "session_id": "s1",
                    "text": (
                        "USER: My basil plant is on the balcony. "
                        "Should I repot my snake plant, which I got from my "
                        "sister last month?"
                    ),
                }
            ],
        }
    }

    assert answer_unit_witness.find_exclusion_contradictions(row, payload) == []


def test_build_fail18_payload_hides_gold_and_answer_ids(tmp_path: pathlib.Path) -> None:
    source_path = tmp_path / "source.json"
    corpus_path = tmp_path / "corpus.json"
    source_path.write_text(
        json.dumps(
            {
                "retrieval": {"top_k": 1},
                "results": [
                    {
                        "case_id": "q1",
                        "question": "How many items?",
                        "top_session_ids": ["s1"],
                        "answer_session_ids": ["s1"],
                        "gold": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_date": "2026/01/01",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I bought one item."}]
                    ],
                    "answer_session_ids": ["s1"],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = answer_unit_witness.build_fail18_payload_artifact(
        source_baseline_path=source_path,
        corpus_path=corpus_path,
    )
    payload_text = json.dumps(artifact["payloads"]).lower()

    assert artifact["total"] == 1
    assert artifact["provider_visibility"]["gold_answer_visible"] is False
    assert "gold" not in payload_text
    assert "answer_session_ids" not in payload_text
    assert artifact["payloads"][0]["case"]["evidence_sessions"][0]["session_id"] == "s1"


def test_build_report_buckets_support_and_flips(tmp_path: pathlib.Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many items?",
                    "gold_answer": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    payload_artifact = {
        "prompt_version": "answer_unit_witness_v1",
        "payloads": [
            {
                "case": {
                    "id": "q1",
                    "evidence_sessions": [
                        {"session_id": "s1", "text": "USER: included item"}
                    ],
                }
            }
        ],
    }
    rows = []
    for sample, answer in [(1, 1), (2, 2)]:
        units = []
        for unit_index in range(answer):
            units.append(
                {
                    "unit_id": f"u{sample}_{unit_index}",
                    "label": f"item {unit_index}",
                    "decision": "include",
                    "borderline": sample == 2,
                    "value": None,
                    "unit": None,
                    "evidence_session_id": "s1",
                    "evidence_span": "included item",
                    "reason_code": "included",
                    "reason": "Included.",
                }
            )
        parsed = answer_unit_witness.parse_witness_response(
            json.dumps(
                _valid_response(
                    answer_number=answer,
                    units=units,
                )
            ),
            expected_case_id="q1",
        )
        row = parsed.to_output_dict(
            provider="test",
            prompt_version="answer_unit_witness_v1",
            raw_output="{}",
            sample_index=sample,
        )
        row.update({"provider_exit_code": 0, "provider_timed_out": False})
        rows.append(row)

    report = answer_unit_witness.build_report(
        rows=rows,
        payload_artifact=payload_artifact,
        manifest_path=manifest_path,
    )

    assert report["support_exact_matches"] == 1
    assert report["cases"][0]["stability_bucket"] == "flipping"

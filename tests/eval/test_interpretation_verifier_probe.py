from __future__ import annotations

import json
import pathlib

from simba.eval import interpretation_verifier_probe as probe


def test_verifier_probe_separates_payload_retrieval_gap(
    tmp_path: pathlib.Path,
) -> None:
    diagnostics = tmp_path / "diagnostics.json"
    payloads = tmp_path / "payloads.json"
    provenance = tmp_path / "provenance.json"
    corpus = tmp_path / "corpus.json"
    diagnostics.write_text(
        json.dumps(
            {
                "blocked_cases": [
                    {
                        "case_id": "camping_case",
                        "question": (
                            "How many days did I spend on camping trips in the "
                            "United States this year?"
                        ),
                        "failure_mode": "B_sum_policy",
                        "gold_value": 8,
                        "recommended_next_intervention": "verifier_enumeration_probe",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payloads.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "camping_case",
                            "question": "question",
                            "evidence_sessions": [
                                _payload_evidence(
                                    "evidence_001",
                                    (
                                        "user: I just got back from a 3-day solo "
                                        "camping trip to Big Sur in early April."
                                    ),
                                ),
                                _payload_evidence(
                                    "evidence_002",
                                    (
                                        "user: We had a 7-day family road trip in "
                                        "Utah, but not camping for this time."
                                    ),
                                ),
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps(
            {
                "evidence_provenance": {
                    "camping_case": {
                        "evidence_001": {"raw_session_id": "big_sur"},
                        "evidence_002": {"raw_session_id": "utah"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "camping_case",
                    "question": "question",
                    "question_date": "2023/04/29 (Sat) 23:45",
                    "answer": "8 days.",
                    "answer_session_ids": ["yellowstone", "big_sur"],
                    "haystack_session_ids": ["yellowstone", "big_sur", "utah"],
                    "haystack_dates": ["d1", "d2", "d3"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": (
                                    "I just got back from an amazing 5-day "
                                    "camping trip to Yellowstone National Park."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "I just got back from a 3-day solo camping "
                                    "trip to Big Sur in early April."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "We had a 7-day family road trip in Utah, "
                                    "but not camping for this time."
                                ),
                            }
                        ],
                    ],
                    "question_type": "test",
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = probe.build_fail18_verifier_enumeration_probe(
        diagnostics_path=diagnostics,
        payloads_path=payloads,
        payload_provenance_path=provenance,
        corpus_path=corpus,
    )

    case = artifact["cases"][0]
    assert case["payload_result"]["computed_total"] == 3
    assert case["full_corpus_result"]["computed_total"] == 8
    assert case["verdict"] == "payload_missing_required_units"
    assert case["payload_result"]["matches_gold"] is False
    assert case["full_corpus_result"]["matches_gold"] is True
    assert case["missing_payload_units_from_full_corpus"][0]["value"] == 5


def test_verifier_probe_deduplicates_reported_charity_amounts(
    tmp_path: pathlib.Path,
) -> None:
    diagnostics = tmp_path / "diagnostics.json"
    payloads = tmp_path / "payloads.json"
    corpus = tmp_path / "corpus.json"
    diagnostics.write_text(
        json.dumps(
            {
                "blocked_cases": [
                    {
                        "case_id": "charity_case",
                        "question": "How much money did I raise for charity in total?",
                        "failure_mode": "B_sum_partial",
                        "gold_value": 1250,
                        "recommended_next_intervention": "verifier_enumeration_probe",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payloads.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "charity_case",
                            "question": "question",
                            "evidence_sessions": [
                                _payload_evidence(
                                    "evidence_001",
                                    (
                                        "user: I raised $1,000 for the local "
                                        "children's hospital at a charity bake sale. "
                                        "user: Like I said, I helped raise over "
                                        "$1,000 for the local children's hospital."
                                    ),
                                ),
                                _payload_evidence(
                                    "evidence_002",
                                    (
                                        "user: I raised $250 for a local food bank "
                                        "during a charity run."
                                    ),
                                ),
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "charity_case",
                    "question": "question",
                    "question_date": "2023/03/20 (Mon) 23:59",
                    "answer": "$1,250",
                    "answer_session_ids": ["s1", "s2"],
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_dates": ["d1", "d2"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": (
                                    "I raised $1,000 for the local children's "
                                    "hospital at a charity bake sale."
                                ),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "I raised $250 for a local food bank during "
                                    "a charity run."
                                ),
                            }
                        ],
                    ],
                    "question_type": "test",
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = probe.build_fail18_verifier_enumeration_probe(
        diagnostics_path=diagnostics,
        payloads_path=payloads,
        payload_provenance_path=tmp_path / "missing-provenance.json",
        corpus_path=corpus,
    )

    case = artifact["cases"][0]
    assert case["payload_result"]["computed_total"] == 1250
    assert case["payload_result"]["matches_gold"] is True
    assert any(
        unit["reason_code"] == "duplicate_same_charity_event"
        for unit in case["payload_candidate_units"]
    )


def test_verifier_probe_carries_target_destination_context() -> None:
    segments = [
        probe.EvidenceSegment(
            case_id="travel_case",
            source_scope="full_corpus",
            session_id="hawaii_session",
            raw_session_id="hawaii_session",
            evidence_date="2023/05/01",
            segment_index=1,
            text=(
                "I just got back from an amazing island-hopping trip to Hawaii "
                "with my family."
            ),
            is_answer_session=True,
        ),
        probe.EvidenceSegment(
            case_id="travel_case",
            source_scope="full_corpus",
            session_id="hawaii_session",
            raw_session_id="hawaii_session",
            evidence_date="2023/05/01",
            segment_index=2,
            text=(
                "With my family, we had to plan everything out for the 10-day "
                "so far in advance."
            ),
            is_answer_session=True,
        ),
        probe.EvidenceSegment(
            case_id="travel_case",
            source_scope="full_corpus",
            session_id="nyc_session",
            raw_session_id="nyc_session",
            evidence_date="2023/05/02",
            segment_index=1,
            text=(
                "I recently got back from a solo trip to New York City for five days."
            ),
            is_answer_session=True,
        ),
    ]

    units = probe._enumerate_units(
        case_id="travel_case",
        probe_kind="destination_travel_days_sum",
        segments=segments,
    )
    result = probe._summarize_units(units, 15)

    assert result["computed_total"] == 15
    assert result["matches_gold"] is True


def _payload_evidence(session_id: str, text: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "date": "2023/01/01",
        "text": text,
        "truncated": False,
    }

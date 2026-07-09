from __future__ import annotations

import json
import pathlib

from simba.eval import interpretation_retrieval_recall as recall


def test_retrieval_recall_reports_outside_topk_answer_session(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    provenance = tmp_path / "provenance.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 1,
                    "max_evidence_chars": 1000,
                    "max_session_chars": 1000,
                },
                "payloads": [
                    {
                        "case": {
                            "id": "q1",
                            "question": "How many camping days?",
                            "evidence_sessions": [],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps(
            {
                "evidence_provenance": {
                    "q1": {
                        "evidence_001": {
                            "raw_session_id": "distractor",
                            "selection_rank": 1,
                            "selection_score": 4,
                        }
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
                    "question_id": "q1",
                    "question": "How many camping days?",
                    "answer_session_ids": ["answer"],
                    "haystack_session_ids": ["distractor", "answer"],
                    "haystack_dates": ["2023/01/01", "2023/01/02"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": ("Camping days camping days camping days."),
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": "One camping day at the lake.",
                            }
                        ],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = recall.build_fail18_retrieval_recall_probe(
        payloads_path=payloads,
        payload_provenance_path=provenance,
        corpus_path=corpus,
    )

    assert artifact["summary"]["answer_sessions_total"] == 1
    assert artifact["summary"]["answer_sessions_in_payload"] == 0
    case = artifact["cases"][0]
    assert case["missing_answer_sessions"] == ["answer"]
    assert case["answer_sessions"][0]["retrieval_status"] == (
        "outside_max_evidence_sessions"
    )
    assert case["recommendation"]["action"] == (
        "improve_query_or_rerank_before_expanding_budget"
    )


def test_retrieval_recall_reports_char_budget_miss(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    provenance = tmp_path / "provenance.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 2,
                    "max_evidence_chars": 40,
                    "max_session_chars": 40,
                },
                "payloads": [{"case": {"id": "q1", "question": "How many days?"}}],
            }
        ),
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps(
            {
                "evidence_provenance": {
                    "q1": {
                        "evidence_001": {
                            "raw_session_id": "first",
                            "selection_rank": 1,
                            "selection_score": 4,
                        }
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
                    "question_id": "q1",
                    "question": "How many days?",
                    "answer_session_ids": ["answer"],
                    "haystack_session_ids": ["first", "answer"],
                    "haystack_dates": ["2023/01/01", "2023/01/02"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "days " * 30}],
                        [{"role": "user", "content": "days answer"}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = recall.build_fail18_retrieval_recall_probe(
        payloads_path=payloads,
        payload_provenance_path=provenance,
        corpus_path=corpus,
    )

    answer = artifact["cases"][0]["answer_sessions"][0]
    assert answer["retrieval_status"] == "omitted_by_total_char_budget"
    assert artifact["cases"][0]["recommendation"]["action"] == (
        "compress_or_chunk_sessions_before_increasing_budget"
    )


def test_retrieval_recall_reports_included_answer_session(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    provenance = tmp_path / "provenance.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 2,
                    "max_evidence_chars": 1000,
                    "max_session_chars": 1000,
                },
                "payloads": [{"case": {"id": "q1", "question": "How many days?"}}],
            }
        ),
        encoding="utf-8",
    )
    provenance.write_text(
        json.dumps(
            {
                "evidence_provenance": {
                    "q1": {
                        "evidence_001": {
                            "raw_session_id": "answer",
                            "selection_rank": 1,
                            "selection_score": 2,
                        }
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
                    "question_id": "q1",
                    "question": "How many days?",
                    "answer_session_ids": ["answer"],
                    "haystack_session_ids": ["answer"],
                    "haystack_dates": ["2023/01/01"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "Two days camping."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = recall.build_fail18_retrieval_recall_probe(
        payloads_path=payloads,
        payload_provenance_path=provenance,
        corpus_path=corpus,
    )

    assert artifact["summary"]["answer_session_recall"] == 1.0
    assert artifact["cases"][0]["all_answer_sessions_in_payload"] is True

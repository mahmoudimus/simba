from __future__ import annotations

import json
import pathlib

from simba.eval import interpretation_retrieval_tuning as tuning


def test_expanded_query_strategy_can_fix_morphology_and_salience_miss(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 1,
                    "max_evidence_chars": 1000,
                    "max_session_chars": 1000,
                },
                "payloads": [{"case": {"id": "q1"}}],
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
                    "answer_session_ids": ["secret_session_1"],
                    "haystack_session_ids": ["distractor", "secret_session_1"],
                    "haystack_dates": ["2023/01/01", "2023/01/02"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "Camping days gear advice.",
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "By the way, I just got back from a "
                                    "5-day camp trip."
                                ),
                            }
                        ],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = tuning.build_fail18_retrieval_tuning_experiment(
        payloads_path=payloads,
        corpus_path=corpus,
    )

    by_strategy = {item["strategy_name"]: item for item in artifact["strategies"]}
    assert (
        by_strategy["baseline_current_lexical"]["summary"][
            "answer_sessions_in_payload"
        ]
        == 0
    )
    assert (
        by_strategy["expanded_query_v1"]["summary"][
            "answer_sessions_in_payload"
        ]
        == 1
    )
    answer = by_strategy["expanded_query_v1"]["cases"][0]["answer_sessions"][0]
    assert "duration_mention" in answer["boost_reasons"]
    assert "completed_user_fact_marker" in answer["boost_reasons"]


def test_compact_session_strategy_can_fix_char_budget_miss(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 2,
                    "max_evidence_chars": 40,
                    "max_session_chars": 40,
                },
                "payloads": [{"case": {"id": "q1"}}],
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

    artifact = tuning.build_fail18_retrieval_tuning_experiment(
        payloads_path=payloads,
        corpus_path=corpus,
    )

    by_strategy = {item["strategy_name"]: item for item in artifact["strategies"]}
    assert (
        by_strategy["baseline_current_lexical"]["summary"][
            "answer_sessions_in_payload"
        ]
        == 0
    )
    assert (
        by_strategy["baseline_compact_sessions"]["summary"][
            "answer_sessions_in_payload"
        ]
        == 1
    )


def test_tuning_artifact_reports_deltas_against_baseline(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 1,
                    "max_evidence_chars": 1000,
                    "max_session_chars": 1000,
                },
                "payloads": [{"case": {"id": "q1"}}],
            }
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How much money did I raise?",
                    "answer_session_ids": ["answer"],
                    "haystack_session_ids": ["answer"],
                    "haystack_dates": ["2023/01/01"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I raised $10."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = tuning.build_fail18_retrieval_tuning_experiment(
        payloads_path=payloads,
        corpus_path=corpus,
    )

    assert artifact["summary"]["best_strategy"]
    for strategy in artifact["strategies"]:
        assert "delta_vs_baseline" in strategy


def test_tuned_payload_builder_keeps_raw_ids_private(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 1,
                    "max_evidence_chars": 1000,
                    "max_session_chars": 1000,
                },
                "payloads": [{"case": {"id": "q1"}}],
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many camping days?",
                    "gold_answer": "1",
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many camping days?",
                    "answer_session_ids": ["secret_session_1"],
                    "haystack_session_ids": ["distractor", "secret_session_1"],
                    "haystack_dates": ["2023/01/01", "2023/01/02"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "Camping days gear advice."}],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "By the way, I just got back from a "
                                    "5-day camp trip."
                                ),
                            }
                        ],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    provider = tuning.build_fail18_tuned_generation_artifact(
        payloads_path=payloads,
        manifest_path=manifest,
        corpus_path=corpus,
    )
    provenance = tuning.build_fail18_tuned_private_provenance_artifact(
        payloads_path=payloads,
        manifest_path=manifest,
        corpus_path=corpus,
    )

    encoded_provider = json.dumps(provider)
    assert "secret_session_1" not in encoded_provider
    assert "gold_answer" not in encoded_provider
    evidence = provider["payloads"][0]["case"]["evidence_sessions"]
    assert evidence[0]["session_id"] == "evidence_001"
    assert evidence[0]["selection_score"] > 0
    assert (
        provenance["evidence_provenance"]["q1"]["evidence_001"]["raw_session_id"]
        == "secret_session_1"
    )
    assert (
        provenance["evidence_provenance"]["q1"]["evidence_001"]["boost_score"]
        > 0
    )


def test_candidate_unit_coverage_strategy_boosts_event_units(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 1,
                    "max_evidence_chars": 1000,
                    "max_session_chars": 1000,
                },
                "payloads": [{"case": {"id": "q1"}}],
            }
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many times did I bake something?",
                    "answer_session_ids": ["answer"],
                    "haystack_session_ids": ["distractor", "answer"],
                    "haystack_dates": ["2023/01/01", "2023/01/02"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "How many times did this happen?",
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "I tried a new sourdough bread recipe "
                                    "and later baked a cake."
                                ),
                            }
                        ],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    artifact = tuning.build_fail18_retrieval_tuning_experiment(
        payloads_path=payloads,
        corpus_path=corpus,
    )

    by_strategy = {item["strategy_name"]: item for item in artifact["strategies"]}
    assert (
        by_strategy["baseline_current_lexical"]["summary"][
            "answer_sessions_in_payload"
        ]
        == 0
    )
    coverage = by_strategy["candidate_unit_coverage_v1"]
    assert coverage["summary"]["answer_sessions_in_payload"] == 1
    answer = coverage["cases"][0]["answer_sessions"][0]
    assert "baking_event_mention" in answer["boost_reasons"]


def test_candidate_unit_coverage_payload_trim_keeps_late_salient_user_turn(
    tmp_path: pathlib.Path,
) -> None:
    payloads = tmp_path / "payloads.json"
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    payloads.write_text(
        json.dumps(
            {
                "evidence_selection": {
                    "max_evidence_sessions": 1,
                    "max_evidence_chars": 500,
                    "max_session_chars": 500,
                },
                "payloads": [{"case": {"id": "q1"}}],
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": (
                        "How many items of clothing do I need to pick up "
                        "or return from a store?"
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": (
                        "How many items of clothing do I need to pick up "
                        "or return from a store?"
                    ),
                    "answer_session_ids": ["answer"],
                    "haystack_session_ids": ["answer"],
                    "haystack_dates": ["2023/01/01"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "Please give me closet tips.",
                            },
                            {
                                "role": "assistant",
                                "content": "Long advice. " * 200,
                            },
                            {
                                "role": "user",
                                "content": (
                                    "I need to return boots to Zara and pick "
                                    "up the larger exchanged pair."
                                ),
                            },
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    provider = tuning.build_fail18_tuned_generation_artifact(
        strategy_name="candidate_unit_coverage_v1",
        payloads_path=payloads,
        manifest_path=manifest,
        corpus_path=corpus,
    )

    text = provider["payloads"][0]["case"]["evidence_sessions"][0]["text"]
    assert "Zara" in text
    assert "exchanged pair" in text

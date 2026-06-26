from __future__ import annotations

import json
import pathlib

import pytest

import simba.eval.ambiguity_fail18 as ambiguity_fail18
import simba.eval.interpretation_prompts as interpretation_prompts


def test_interpretation_generation_payload_contract_is_domain_general() -> None:
    payload = interpretation_prompts.build_interpretation_generation_payload(
        case_id="q",
        question="How many events count?",
        evidence_sessions=[
            {
                "session_id": "s1",
                "date": "2023/01/01",
                "text": "user: I attended one event.",
            }
        ],
    )
    encoded = json.dumps(payload)
    contract = "\n".join(payload["generation_contract"]).lower()
    forbidden_terms = {
        "zara",
        "boots",
        "blazer",
        "tame impala",
        "wedding",
        "hawaii",
        "new york",
    }

    assert json.loads(encoded)["case"]["id"] == "q"
    assert not any(term in contract for term in forbidden_terms)
    assert "candidate units" in contract
    assert "do not compute the final answer" in contract
    assert "natural_language_interpretation" in encoded
    assert "expected_answer_shape" in encoded


def test_fail18_payload_builder_uses_corpus_evidence_without_gold(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many things happened?",
                    "gold_answer": 2,
                    "clingo_certain": 1,
                    "clingo_possible": 3,
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
                    "haystack_dates": ["2023/01/01"],
                    "haystack_session_ids": ["s1"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I completed a task.",
                            },
                            {
                                "role": "assistant",
                                "content": "That sounds complete.",
                            },
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    payloads = interpretation_prompts.fail18_interpretation_generation_payloads(
        manifest_path=manifest,
        corpus_path=corpus,
    )

    assert len(payloads) == 1
    assert payloads[0]["case"]["id"] == "q1"
    assert "gold" not in payloads[0]["case"]
    assert payloads[0]["case"]["evidence_sessions"] == [
        {
            "session_id": "evidence_001",
            "date": "2023/01/01",
            "selection_rank": 1,
            "selection_score": 0,
            "user_selection_score": 0,
            "assistant_selection_score": 0,
            "truncated": False,
            "text": "user: I completed a task.\nassistant: That sounds complete.",
        }
    ]


def test_fail18_payload_builder_applies_evidence_budget(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many pottery workshops did I attend?",
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
                    "answer_session_ids": ["gold_should_not_drive_selection"],
                    "haystack_dates": [
                        "2023/01/01",
                        "2023/01/02",
                        "2023/01/03",
                    ],
                    "haystack_session_ids": [
                        "irrelevant",
                        "gold_should_not_drive_selection",
                        "lexical_match",
                    ],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I bought groceries."}],
                        [{"role": "user", "content": "A hidden gold session."}],
                        [
                            {
                                "role": "user",
                                "content": (
                                    "I attended a pottery workshop with a very "
                                    "long description."
                                ),
                            }
                        ],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    payloads = interpretation_prompts.fail18_interpretation_generation_payloads(
        manifest_path=manifest,
        corpus_path=corpus,
        max_evidence_sessions=1,
        max_evidence_chars=45,
        max_session_chars=45,
    )
    sessions = payloads[0]["case"]["evidence_sessions"]

    assert [session["session_id"] for session in sessions] == ["evidence_001"]
    assert len(sessions) == 1
    assert len(sessions[0]["text"]) <= 45
    assert sessions[0]["selection_score"] == 2
    assert sessions[0]["user_selection_score"] == 1
    assert sessions[0]["assistant_selection_score"] == 0
    assert sessions[0]["truncated"] is True


def test_fail18_generation_artifact_is_payload_only() -> None:
    if not ambiguity_fail18.DEFAULT_MANIFEST.exists():
        pytest.skip("local clingo_fail18 fixture not present")
    artifact = interpretation_prompts.build_fail18_generation_artifact(limit=0)

    assert artifact["name"] == "fail18-ambiguous-nlidb-gate1-payloads"
    assert artifact["artifact_kind"] == "provider_payloads"
    assert artifact["gate_status"] == "payloads_only_not_run"
    assert "model outputs" in artifact["known_gap"]
    assert artifact["evidence_selection"]["uses_answer_session_ids"] is False
    assert artifact["evidence_selection"]["provider_session_ids"] == (
        "opaque evidence_NNN ids"
    )
    assert "user_term_overlap" in artifact["evidence_selection"]["score_formula"]
    assert (
        "_gitless/fail18_ambiguous_nlidb_gate1_payloads.json"
        in artifact["commands"][0]
    )


def test_fail18_provider_artifact_uses_opaque_ids_and_private_provenance(
    tmp_path: pathlib.Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many pottery workshops did I attend?",
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
                    "answer_session_ids": ["answer_secret"],
                    "haystack_dates": ["2023/01/01"],
                    "haystack_session_ids": ["answer_secret"],
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I attended a pottery workshop.",
                            }
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    provider_artifact = interpretation_prompts.build_fail18_generation_artifact(
        manifest_path=manifest,
        corpus_path=corpus,
    )
    provenance_artifact = (
        interpretation_prompts.build_fail18_private_provenance_artifact(
            manifest_path=manifest,
            corpus_path=corpus,
        )
    )

    assert "answer_secret" not in json.dumps(provider_artifact)
    evidence = provider_artifact["payloads"][0]["case"]["evidence_sessions"]
    assert evidence[0]["session_id"] == "evidence_001"
    assert (
        provenance_artifact["evidence_provenance"]["q1"]["evidence_001"][
            "raw_session_id"
        ]
        == "answer_secret"
    )

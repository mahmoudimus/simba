from __future__ import annotations

import json
import pathlib

from simba.eval import candidate_unit_formalizer


def _write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(f"{json.dumps(value)}\n", encoding="utf-8")


def _valid_response(**overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "formalizer_id": "q1::evidence_001",
        "case_id": "q1",
        "evidence_session_id": "evidence_001",
        "facts": [
            {
                "fact_id": "fact_1",
                "predicate": "action",
                "arguments": {
                    "subject": "user",
                    "object": "blue blazer",
                    "verb": "pick_up",
                    "location": "dry cleaner",
                    "status": "pending",
                },
                "evidence_span": "pick up my blue blazer",
                "confidence": 0.91,
            }
        ],
        "notes": "",
    }
    response.update(overrides)
    return response


def test_parse_formalizer_response_accepts_valid_output() -> None:
    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(_valid_response()),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_PARSED
    assert result.formalizer_id == "q1::evidence_001"
    assert result.facts[0].predicate == "action"
    assert result.facts[0].polarity == ""
    assert "polarity" not in result.facts[0].to_dict()


def test_parse_formalizer_response_accepts_empty_fact_list() -> None:
    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(_valid_response(facts=[])),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_PARSED
    assert result.facts == ()


def test_parse_formalizer_response_normalizes_open_arguments() -> None:
    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(
            _valid_response(
                facts=[
                    {
                        "fact_id": "fact_1",
                        "predicate": "action",
                        "arguments": {
                            "subject": "user",
                            "object": "blue blazer",
                            "verb": "pick_up",
                            "location": None,
                            "status": "",
                        },
                        "evidence_span": "pick up my blue blazer",
                        "confidence": 0.91,
                    }
                ]
            )
        ),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_PARSED
    assert result.facts[0].arguments["location"] is None
    assert result.facts[0].arguments["status"] is None


def test_parse_formalizer_response_accepts_sortal_and_distinct() -> None:
    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(
            _valid_response(
                facts=[
                    {
                        "fact_id": "sortal_1",
                        "predicate": "sortal",
                        "arguments": {
                            "entity": "new pair",
                            "type": "boots",
                            "source": "bridging",
                            "antecedent": "exchanged pair",
                            "licensed_by": "contrastive new",
                        },
                        "evidence_span": "pick up the new pair",
                        "confidence": 0.84,
                    },
                    {
                        "fact_id": "distinct_1",
                        "predicate": "distinct",
                        "arguments": {
                            "a": "new pair",
                            "b": "exchanged pair",
                            "reason": "contrastive new",
                        },
                        "evidence_span": "exchanged a pair ... new pair",
                        "confidence": 0.82,
                    },
                ]
            )
        ),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_PARSED
    assert [fact.predicate for fact in result.facts] == ["sortal", "distinct"]


def test_parse_formalizer_response_rejects_duplicate_fact_ids() -> None:
    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(
            _valid_response(
                facts=[
                    {
                        "fact_id": "dup",
                        "predicate": "other",
                        "arguments": {"description": "one"},
                        "evidence_span": "one",
                        "confidence": 0.5,
                    },
                    {
                        "fact_id": "dup",
                        "predicate": "other",
                        "arguments": {"description": "two"},
                        "evidence_span": "two",
                        "confidence": 0.5,
                    },
                ]
            )
        ),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_INVALID_SCHEMA
    assert "duplicate fact_id 'dup'" in "\n".join(result.parse_errors)


def test_build_formalizer_payload_artifact_hides_private_eval_labels(
    tmp_path: pathlib.Path,
) -> None:
    candidate_payloads_path = tmp_path / "candidate_payloads.json"
    _write_json(
        candidate_payloads_path,
        {
            "prompt_version": "candidate_unit_compiler_v1",
            "payloads": [
                {
                    "case": {
                        "id": "q1",
                        "question": "How many items?",
                        "gold_answer": 3,
                        "evidence_sessions": [
                            {
                                "session_id": "evidence_001",
                                "date": "2026-06-20",
                                "text": "user: pick up my blazer",
                            },
                            {
                                "session_id": "evidence_002",
                                "date": "2026-06-20",
                                "text": "user: unrelated",
                            },
                        ],
                    }
                }
            ],
        },
    )

    artifact = candidate_unit_formalizer.build_formalizer_payload_artifact(
        candidate_payloads_path=candidate_payloads_path,
    )
    rendered = json.dumps(artifact).lower()
    rendered_payloads = json.dumps(artifact["payloads"]).lower()

    assert artifact["total"] == 2
    assert artifact["payloads"][0]["formalizer_id"] == "q1::evidence_001"
    assert artifact["provider_visibility"]["final_answer_requested"] is False
    assert artifact["provider_visibility"]["answer_support_labels_requested"] is False
    assert artifact["prompt_version"] == "candidate_unit_formalizer_recursive_v2"
    assert '"gold_answer": 3' not in rendered_payloads
    assert "raw_session_id" not in rendered_payloads
    assert "supports_answer" not in rendered_payloads
    assert "contradicts_answer" not in rendered_payloads
    assert "clothing_obligation" not in rendered_payloads
    assert "fact_schema" in rendered_payloads
    assert "sortal" in rendered_payloads
    assert "distinct" in rendered_payloads
    assert "use coreference only for true same-entity identity" in rendered_payloads
    assert "keep entity symbols and type symbols disjoint" in rendered_payloads
    assert "omit the argument or use json null" in rendered_payloads
    assert "offline_lexicon_context" in rendered_payloads
    assert "gold_answer_visible" in rendered


def test_build_formalizer_report_flags_verifier_conflicts(
    tmp_path: pathlib.Path,
) -> None:
    provenance_path = tmp_path / "provenance.json"
    _write_json(
        provenance_path,
        {
            "evidence_provenance": {
                "q1": {
                    "evidence_001": {"raw_session_id": "raw_1"},
                    "evidence_002": {"raw_session_id": "raw_2"},
                    "evidence_003": {"raw_session_id": "raw_3"},
                    "evidence_004": {"raw_session_id": "raw_4"},
                }
            }
        },
    )
    formalizer_payload_artifact = {
        "prompt_version": "candidate_unit_formalizer_v1",
        "payloads": [
            {"formalizer_id": f"q1::evidence_00{idx}", "case_id": "q1"}
            for idx in range(1, 5)
        ],
    }
    formalizer_rows = [
        {
            "formalizer_id": "q1::evidence_001",
            "case_id": "q1",
            "evidence_session_id": "evidence_001",
            "parse_status": "parsed",
            "facts": [
                {
                    "fact_id": "support_1",
                    "predicate": "action",
                    "arguments": {
                        "subject": "user",
                        "object": "blazer",
                        "verb": "pick_up",
                    },
                    "evidence_span": "pick up blazer",
                    "confidence": 0.9,
                }
            ],
            "provider_exit_code": 0,
            "provider_timed_out": False,
        },
        {
            "formalizer_id": "q1::evidence_002",
            "case_id": "q1",
            "evidence_session_id": "evidence_002",
            "parse_status": "parsed",
            "facts": [
                {
                    "fact_id": "support_2",
                    "predicate": "action",
                    "arguments": {
                        "subject": "user",
                        "object": "boots",
                        "verb": "return",
                    },
                    "evidence_span": "return boots",
                    "confidence": 0.9,
                }
            ],
            "provider_exit_code": 0,
            "provider_timed_out": False,
        },
        {
            "formalizer_id": "q1::evidence_003",
            "case_id": "q1",
            "evidence_session_id": "evidence_003",
            "parse_status": "parsed",
            "facts": [
                {
                    "fact_id": "contra_1",
                    "predicate": "status",
                    "arguments": {"entity": "already owned item", "status": "borrowed"},
                    "evidence_span": "already own it",
                    "confidence": 0.8,
                }
            ],
            "provider_exit_code": 0,
            "provider_timed_out": False,
        },
    ]
    candidate_rows = [
        {
            "case_id": "q1",
            "candidate_units": [
                {
                    "unit_id": "included_supported",
                    "label": "blazer",
                    "status": "included",
                    "merge_target": None,
                    "reason_code": "included",
                    "evidence_session_ids": ["evidence_001"],
                },
                {
                    "unit_id": "included_unsupported",
                    "label": "unsupported",
                    "status": "included",
                    "merge_target": None,
                    "reason_code": "included",
                    "evidence_session_ids": ["evidence_004"],
                },
                {
                    "unit_id": "included_contradicted",
                    "label": "already owned item",
                    "status": "included",
                    "merge_target": None,
                    "reason_code": "included",
                    "evidence_session_ids": ["evidence_003"],
                },
                {
                    "unit_id": "excluded_supported",
                    "label": "boots",
                    "status": "excluded",
                    "merge_target": None,
                    "reason_code": "excluded",
                    "evidence_session_ids": ["evidence_002"],
                },
                {
                    "unit_id": "included_non_answer_supported",
                    "label": "boots",
                    "status": "included",
                    "merge_target": None,
                    "reason_code": "included",
                    "evidence_session_ids": ["evidence_002"],
                },
                {
                    "unit_id": "merged_supported",
                    "label": "blazer duplicate",
                    "status": "merged",
                    "merge_target": "included_supported",
                    "reason_code": "same_unit",
                    "evidence_session_ids": ["evidence_001"],
                },
            ],
        }
    ]

    report = candidate_unit_formalizer.build_formalizer_report(
        formalizer_payload_artifact=formalizer_payload_artifact,
        formalizer_rows=formalizer_rows,
        candidate_rows=candidate_rows,
        payload_provenance_path=provenance_path,
    )
    case = report["cases"][0]

    assert report["summary"]["rows_total"] == 1
    assert report["summary"]["candidate_units_without_supporting_facts"] == 2
    assert report["summary"]["candidate_units_with_contradicting_facts"] == 1
    assert report["summary"]["excluded_units_with_supporting_facts"] == 1
    assert report["summary"]["merged_units_with_supporting_facts"] == 1
    assert report["summary"]["fact_role_counts"] == {
        "negative_fact": 1,
        "neutral_fact": 2,
    }
    assert "polarity_counts" not in report["summary"]
    assert "included_units_with_non_answer_supporting_facts" not in report["summary"]
    assert "rows_verified_answer_matches_gold" not in report["summary"]
    assert "rejected_included_units" not in report["summary"]
    assert "source_manifest" not in report
    assert "included_units_with_non_answer_supporting_facts" not in case
    assert case["candidate_units"][0]["linked_fact_ids"] == ["support_1"]
    assert case["missing_provider_fact_sessions"] == ["evidence_004"]


def test_parse_formalizer_response_accepts_legacy_v1_polarity() -> None:
    response = _valid_response(
        facts=[
            {
                "fact_id": "legacy_1",
                "predicate": "clothing_obligation",
                "arguments": {"item": "blue blazer"},
                "polarity": "supports_answer",
                "evidence_span": "pick up my blue blazer",
                "confidence": 0.91,
            }
        ]
    )

    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(response),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_PARSED
    assert result.facts[0].polarity == "supports_answer"


def test_parse_formalizer_response_rejects_answer_decision_predicates() -> None:
    response = _valid_response(
        facts=[
            {
                "fact_id": "bad_1",
                "predicate": "included",
                "arguments": {"entity": "blue blazer"},
                "evidence_span": "pick up my blue blazer",
                "confidence": 0.91,
            }
        ]
    )

    result = candidate_unit_formalizer.parse_formalizer_response(
        json.dumps(response),
        expected_formalizer_id="q1::evidence_001",
    )

    assert result.parse_status == candidate_unit_formalizer.PARSE_STATUS_INVALID_SCHEMA
    assert "unknown predicate 'included'" in "\n".join(result.parse_errors)

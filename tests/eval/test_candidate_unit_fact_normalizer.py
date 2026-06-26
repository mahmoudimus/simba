from __future__ import annotations

import json
import pathlib

from simba.eval import candidate_unit_fact_normalizer


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _valid_replace(**overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "normalizer_id": "q1::evidence_001::fact_1",
        "case_id": "q1",
        "evidence_session_id": "evidence_001",
        "source_fact_id": "fact_1",
        "decision": "replace",
        "replacement_fact": {
            "fact_id": "fact_1_normalized",
            "predicate": "action",
            "arguments": {
                "subject": "user",
                "object": "recipe backstory",
                "verb": "add",
            },
            "evidence_span": "back story about how this recipe came about",
            "confidence": 0.87,
        },
        "reason": "The source describes a user request/action.",
    }
    response.update(overrides)
    return response


def test_build_payload_artifact_selects_only_other_facts(
    tmp_path: pathlib.Path,
) -> None:
    formalizer_outputs = tmp_path / "formalizer.jsonl"
    _write_jsonl(
        formalizer_outputs,
        [
            {
                "case_id": "q1",
                "evidence_session_id": "evidence_001",
                "parse_status": "parsed",
                "provider_exit_code": 0,
                "provider_timed_out": False,
                "facts": [
                    {
                        "fact_id": "action_1",
                        "predicate": "action",
                        "arguments": {"subject": "user"},
                        "evidence_span": "did thing",
                        "confidence": 0.8,
                    },
                    {
                        "fact_id": "other_1",
                        "predicate": "other",
                        "arguments": {"description": "recipe backstory request"},
                        "evidence_span": "back story about how this recipe came about",
                        "confidence": 0.8,
                    },
                ],
            }
        ],
    )

    artifact = candidate_unit_fact_normalizer.build_payload_artifact(
        formalizer_outputs_path=formalizer_outputs,
    )
    rendered_payloads = json.dumps(artifact["payloads"]).lower()

    assert artifact["total"] == 1
    assert artifact["provider_visibility"]["candidate_unit_status_visible"] is False
    assert artifact["provider_visibility"]["final_answer_requested"] is False
    assert artifact["payloads"][0]["normalizer_id"] == "q1::evidence_001::other_1"
    assert artifact["payloads"][0]["source_fact"]["predicate"] == "other"
    assert "sortal" in rendered_payloads
    assert "distinct" in rendered_payloads
    assert "coreference only for true same-entity identity" in rendered_payloads
    assert "keep entity symbols and type symbols disjoint" in rendered_payloads
    assert "omit unknown optional fields or use json null" in rendered_payloads
    assert "question" not in artifact["payloads"][0]
    assert "supports_answer" not in rendered_payloads
    assert "contradicts_answer" not in rendered_payloads
    assert "gold_answer" not in rendered_payloads
    assert "candidate_units" not in rendered_payloads


def test_parse_normalizer_response_accepts_replacement() -> None:
    result = candidate_unit_fact_normalizer.parse_normalizer_response(
        json.dumps(_valid_replace()),
        expected_normalizer_id="q1::evidence_001::fact_1",
    )

    assert result.parse_status == candidate_unit_fact_normalizer.PARSE_STATUS_PARSED
    assert result.decision == "replace"
    assert result.replacement_fact is not None
    assert result.replacement_fact.predicate == "action"


def test_parse_normalizer_response_normalizes_open_arguments() -> None:
    result = candidate_unit_fact_normalizer.parse_normalizer_response(
        json.dumps(
            _valid_replace(
                replacement_fact={
                    "fact_id": "fact_1_normalized",
                    "predicate": "action",
                    "arguments": {
                        "subject": "user",
                        "object": "recipe backstory",
                        "verb": "add",
                        "location": None,
                        "status": "",
                    },
                    "evidence_span": "back story about how this recipe came about",
                    "confidence": 0.87,
                }
            )
        ),
        expected_normalizer_id="q1::evidence_001::fact_1",
    )

    assert result.parse_status == candidate_unit_fact_normalizer.PARSE_STATUS_PARSED
    assert result.replacement_fact is not None
    assert result.replacement_fact.arguments["location"] is None
    assert result.replacement_fact.arguments["status"] is None


def test_parse_normalizer_response_accepts_sortal_replacement() -> None:
    result = candidate_unit_fact_normalizer.parse_normalizer_response(
        json.dumps(
            _valid_replace(
                replacement_fact={
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
                    "confidence": 0.87,
                }
            )
        ),
        expected_normalizer_id="q1::evidence_001::fact_1",
    )

    assert result.parse_status == candidate_unit_fact_normalizer.PARSE_STATUS_PARSED
    assert result.replacement_fact is not None
    assert result.replacement_fact.predicate == "sortal"


def test_parse_normalizer_response_accepts_keep_other() -> None:
    result = candidate_unit_fact_normalizer.parse_normalizer_response(
        json.dumps(
            _valid_replace(
                decision="keep_other",
                replacement_fact=None,
                reason="No generic predicate fits without adding information.",
            )
        ),
        expected_normalizer_id="q1::evidence_001::fact_1",
    )

    assert result.parse_status == candidate_unit_fact_normalizer.PARSE_STATUS_PARSED
    assert result.decision == "keep_other"
    assert result.replacement_fact is None


def test_parse_normalizer_response_rejects_non_generic_replacement() -> None:
    result = candidate_unit_fact_normalizer.parse_normalizer_response(
        json.dumps(
            _valid_replace(
                replacement_fact={
                    "fact_id": "legacy_1",
                    "predicate": "fundraiser",
                    "arguments": {"event": "concert"},
                    "evidence_span": "benefit concert",
                    "confidence": 0.9,
                }
            )
        ),
        expected_normalizer_id="q1::evidence_001::fact_1",
    )

    assert (
        result.parse_status
        == candidate_unit_fact_normalizer.PARSE_STATUS_INVALID_SCHEMA
    )
    assert "replacement predicate must be generic" in "\n".join(result.parse_errors)


def test_parse_normalizer_response_rejects_answer_path_labels() -> None:
    result = candidate_unit_fact_normalizer.parse_normalizer_response(
        json.dumps(
            _valid_replace(
                replacement_fact={
                    "fact_id": "bad_1",
                    "predicate": "action",
                    "arguments": {"label": "supports_answer"},
                    "evidence_span": "back story",
                    "confidence": 0.9,
                }
            )
        ),
        expected_normalizer_id="q1::evidence_001::fact_1",
    )

    assert (
        result.parse_status
        == candidate_unit_fact_normalizer.PARSE_STATUS_INVALID_SCHEMA
    )
    assert "forbidden answer-path term" in "\n".join(result.parse_errors)


def test_apply_normalizations_replaces_only_replace_decisions() -> None:
    formalizer_rows = [
        {
            "case_id": "q1",
            "evidence_session_id": "evidence_001",
            "facts": [
                {
                    "fact_id": "other_1",
                    "predicate": "other",
                    "arguments": {"description": "recipe backstory request"},
                    "evidence_span": "back story",
                    "confidence": 0.8,
                },
                {
                    "fact_id": "other_2",
                    "predicate": "other",
                    "arguments": {"description": "residual context"},
                    "evidence_span": "context",
                    "confidence": 0.7,
                },
            ],
        }
    ]
    normalizer_rows = [
        {
            "case_id": "q1",
            "evidence_session_id": "evidence_001",
            "source_fact_id": "other_1",
            "decision": "replace",
            "replacement_fact": {
                "fact_id": "other_1_normalized",
                "predicate": "action",
                "arguments": {"subject": "user", "verb": "add"},
                "evidence_span": "back story",
                "confidence": 0.9,
            },
            "parse_status": "parsed",
            "provider_exit_code": 0,
            "provider_timed_out": False,
        },
        {
            "case_id": "q1",
            "evidence_session_id": "evidence_001",
            "source_fact_id": "other_2",
            "decision": "keep_other",
            "replacement_fact": None,
            "parse_status": "parsed",
            "provider_exit_code": 0,
            "provider_timed_out": False,
        },
    ]

    normalized = candidate_unit_fact_normalizer.apply_normalizations(
        formalizer_rows=formalizer_rows,
        normalizer_rows=normalizer_rows,
    )

    facts = normalized[0]["facts"]
    assert facts[0]["predicate"] == "action"
    assert facts[0]["normalized_from_fact_id"] == "other_1"
    assert facts[1]["predicate"] == "other"

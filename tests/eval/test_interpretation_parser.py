from __future__ import annotations

import json

import simba.eval.interpretation_parser as interpretation_parser


def test_parse_interpretation_response_accepts_valid_record() -> None:
    result = interpretation_parser.parse_interpretation_response(
        json.dumps(
            {
                "case_id": "q1",
                "interpretations": [
                    {
                        "interpretation_id": "i1",
                        "natural_language_interpretation": (
                            "Count only explicitly completed pickup tasks."
                        ),
                        "ambiguity_types": ["scope_ambiguous"],
                        "assumptions": ["recommendations are not tasks"],
                        "expected_answer_shape": "count",
                    }
                ],
            }
        ),
        expected_case_id="q1",
    )

    assert result.parse_status == "parsed"
    assert result.case_id == "q1"
    assert result.parse_errors == ()
    assert result.interpretations[0].to_dict()["ambiguity_types"] == [
        "scope_ambiguous"
    ]


def test_parse_interpretation_response_rejects_markdown_wrapped_json() -> None:
    result = interpretation_parser.parse_interpretation_response(
        '```json\n{"case_id": "q1", "interpretations": []}\n```',
        expected_case_id="q1",
    )

    assert result.parse_status == "invalid_json"
    assert result.case_id == "q1"
    assert result.interpretations == ()


def test_parse_interpretation_response_rejects_partial_schema() -> None:
    result = interpretation_parser.parse_interpretation_response(
        json.dumps(
            {
                "case_id": "q1",
                "interpretations": [
                    {
                        "interpretation_id": "i1",
                        "natural_language_interpretation": "Count tasks.",
                        "ambiguity_types": "scope_ambiguous",
                        "expected_answer_shape": "count",
                    }
                ],
            }
        ),
        expected_case_id="q1",
    )

    assert result.parse_status == "invalid_schema"
    assert result.interpretations == ()
    assert "ambiguity_types must be a non-empty list" in result.parse_errors[0]


def test_parse_interpretation_response_allows_zero_interpretations() -> None:
    result = interpretation_parser.parse_interpretation_response(
        json.dumps({"case_id": "q1", "interpretations": []}),
        expected_case_id="q1",
    )

    assert result.parse_status == "parsed"
    assert result.interpretations == ()


def test_parse_interpretation_response_rejects_duplicate_interpretation_ids() -> None:
    result = interpretation_parser.parse_interpretation_response(
        json.dumps(
            {
                "case_id": "q1",
                "interpretations": [
                    {
                        "interpretation_id": "i1",
                        "natural_language_interpretation": "Count tasks.",
                        "ambiguity_types": ["scope_ambiguous"],
                        "assumptions": [],
                        "expected_answer_shape": "count",
                    },
                    {
                        "interpretation_id": "i1",
                        "natural_language_interpretation": "Count all tasks.",
                        "ambiguity_types": ["scope_ambiguous"],
                        "assumptions": [],
                        "expected_answer_shape": "count",
                    },
                ],
            }
        ),
        expected_case_id="q1",
    )

    assert result.parse_status == "invalid_schema"
    assert result.interpretations == ()
    assert "duplicate interpretation_id values: i1" in result.parse_errors

from __future__ import annotations

import json

import simba.eval.interpretation_runner as interpretation_runner


def test_build_gate1_report_counts_parse_and_duplicate_metrics() -> None:
    payload_artifact = {
        "prompt_version": "interpretation_generator_v1",
        "payloads": [
            _payload("q1"),
            _payload("q2"),
        ],
    }
    rows = [
        {
            "case_id": "q1",
            "provider": "fake-provider",
            "prompt_version": "interpretation_generator_v1",
            "raw_output": "{}",
            "parse_status": "parsed",
            "parse_errors": [],
            "latency_seconds": 1.25,
            "interpretations": [
                _interpretation("i1", "Count completed tasks."),
                _interpretation("i2", "count completed tasks"),
                _interpretation(
                    "i3",
                    "Count recommended tasks too.",
                    ambiguity_types=["scope_ambiguous", "false_assumption"],
                    expected_answer_shape="range",
                ),
            ],
        },
        {
            "case_id": "q2",
            "provider": "fake-provider",
            "prompt_version": "interpretation_generator_v1",
            "raw_output": "not json",
            "parse_status": "invalid_json",
            "parse_errors": ["invalid JSON"],
            "latency_seconds": 2.0,
            "interpretations": [],
        },
    ]

    report = interpretation_runner.build_gate1_report(
        rows=rows,
        payload_artifact=payload_artifact,
        outputs_path="_gitless/out.jsonl",
        payloads_path="_gitless/payloads.json",
        provenance_path="_gitless/provenance.json",
    )

    assert report["gate1_passed"] is False
    assert report["rows_expected"] == 2
    assert report["rows_total"] == 2
    assert report["rows_parse_status_parsed"] == 1
    assert report["rows_parsed"] == 1
    assert report["rows_failed_parse"] == 1
    assert report["rows_provider_failed"] == 0
    assert report["rows_with_multiple_interpretations"] == 1
    assert report["average_interpretations_per_row"] == 1.5
    assert report["duplicate_interpretation_count"] == 1
    assert report["duplicate_interpretation_rows"] == {"q1": ["i2"]}
    assert report["ambiguity_type_distribution"] == {
        "false_assumption": 1,
        "scope_ambiguous": 3,
    }
    assert report["expected_answer_shape_distribution"] == {
        "count": 2,
        "range": 1,
    }
    assert report["case_coverage"]["covers_exactly_expected_cases"] is True
    assert report["acceptance"]["outputs_cover_exactly_expected_cases"] is True
    assert (
        report["acceptance"]["accepted_provider_outputs_cover_exactly_expected_cases"]
        is True
    )
    assert report["acceptance"]["outputs_cover_exactly_fail18_rows"] is False
    assert (
        report["acceptance"]["accepted_provider_outputs_cover_exactly_fail18_rows"]
        is False
    )
    assert report["acceptance"]["candidate_unit_compilation_attempted"] is False


def test_build_gate1_report_rejects_parseable_provider_failures() -> None:
    report = interpretation_runner.build_gate1_report(
        rows=[
            {
                "case_id": "q1",
                "provider": "fake-provider",
                "raw_output": json.dumps({"case_id": "q1", "interpretations": []}),
                "parse_status": "parsed",
                "parse_errors": ["provider exited with code 1"],
                "provider_exit_code": 1,
                "provider_timed_out": False,
                "interpretations": [],
            }
        ],
        payload_artifact={"payloads": [_payload("q1")]},
    )

    assert report["gate_status"] == "slice1b_incomplete"
    assert report["rows_parse_status_parsed"] == 1
    assert report["rows_parsed"] == 0
    assert report["rows_failed_parse"] == 1
    assert report["rows_provider_failed"] == 1
    assert report["provider_failure_case_ids"] == ["q1"]
    assert report["case_coverage"]["covers_exactly_expected_cases"] is True
    assert (
        report["acceptance"]["accepted_provider_outputs_cover_exactly_expected_cases"]
        is False
    )
    assert (
        report["acceptance"]["accepted_provider_outputs_cover_exactly_fail18_rows"]
        is False
    )
    assert report["acceptance"]["provider_rows_succeeded"] is False


def test_build_gate1_report_detects_case_coverage_gaps() -> None:
    report = interpretation_runner.build_gate1_report(
        rows=[
            {
                "case_id": "q1",
                "provider": "fake-provider",
                "raw_output": json.dumps({"case_id": "q1"}),
                "parse_status": "parsed",
                "parse_errors": [],
                "interpretations": [],
            },
            {
                "case_id": "extra",
                "provider": "fake-provider",
                "raw_output": json.dumps({"case_id": "extra"}),
                "parse_status": "parsed",
                "parse_errors": [],
                "interpretations": [],
            },
        ],
        payload_artifact={"payloads": [_payload("q1"), _payload("q2")]},
    )

    coverage = report["case_coverage"]
    assert coverage["covers_exactly_expected_cases"] is False
    assert coverage["missing_case_ids"] == ["q2"]
    assert coverage["extra_case_ids"] == ["extra"]


def test_build_provider_prompt_embeds_payload_contract() -> None:
    prompt = interpretation_runner.build_provider_prompt(_payload("q1"))

    assert "Return exactly one strict JSON object" in prompt
    assert '"case_id": "q1"' in prompt


def test_fail18_noun_leakage_check_derives_question_terms() -> None:
    report = interpretation_runner.build_gate1_report(
        rows=[],
        payload_artifact={
            "payloads": [
                {
                    "task": "Generate pottery interpretations.",
                    "generation_contract": ["Return strict JSON."],
                    "output_schema": {"case_id": "q1"},
                    "case": {
                        "id": "q1",
                        "question": "How many pottery workshops did I attend?",
                    },
                }
            ]
        },
    )

    leakage = report["fail18_noun_leakage_check"]
    assert leakage["check_kind"] == ("derived_question_terms_against_prompt_contract")
    assert "pottery" in leakage["forbidden_terms"]
    assert leakage["found_terms"] == ["pottery"]
    assert leakage["passed"] is False


def _payload(case_id: str) -> dict[str, object]:
    return {
        "task": "Generate candidate natural-language interpretations.",
        "generation_contract": ["Do not compute the final answer."],
        "output_schema": {"case_id": case_id, "interpretations": []},
        "case": {"id": case_id, "question": "How many tasks count?"},
    }


def _interpretation(
    interpretation_id: str,
    text: str,
    *,
    ambiguity_types: list[str] | None = None,
    expected_answer_shape: str = "count",
) -> dict[str, object]:
    return {
        "interpretation_id": interpretation_id,
        "natural_language_interpretation": text,
        "ambiguity_types": ambiguity_types or ["scope_ambiguous"],
        "assumptions": [],
        "expected_answer_shape": expected_answer_shape,
    }

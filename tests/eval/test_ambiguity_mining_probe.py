from __future__ import annotations

import json
import pathlib

from simba.eval import ambiguity_mining_probe as probe
from simba.eval import interpretation_runner


def test_build_payload_artifact_uses_heldout_and_wider_answer_sessions(
    tmp_path: pathlib.Path,
) -> None:
    dataset = tmp_path / "longmemeval_s.json"
    heldout = tmp_path / "heldout.json"
    fail18 = tmp_path / "fail18.json"
    dataset.write_text(
        json.dumps(
            [
                _dataset_row("held", answer="2"),
                _dataset_row("fail18_row", answer="4"),
                _dataset_row("wide", answer="3"),
            ]
        ),
        encoding="utf-8",
    )
    heldout.write_text(json.dumps([{"question_id": "held"}]), encoding="utf-8")
    fail18.write_text(json.dumps([{"question_id": "fail18_row"}]), encoding="utf-8")

    artifact = probe.build_payload_artifact(
        dataset_path=dataset,
        heldout_manifest_path=heldout,
        fail18_manifest_path=fail18,
        wider_limit=1,
        session_char_limit=500,
    )

    assert artifact["total"] == 2
    assert artifact["selection"]["heldout_count"] == 1
    assert artifact["selection"]["wider_count"] == 1
    assert artifact["provider_visibility"]["official_answer_visible"] is True
    assert artifact["provider_visibility"]["system_answer_visible"] is False
    case_ids = [payload["case"]["id"] for payload in artifact["payloads"]]
    assert case_ids == ["held", "wide"]
    rendered = json.dumps(artifact["payloads"])
    assert "official_answer" in rendered
    assert "provider_answer_number" not in rendered
    assert "witness" not in rendered.casefold()


def test_parse_adjudication_response_rejects_duplicate_reading_ids() -> None:
    parsed = probe.parse_adjudication_response(
        json.dumps(
            {
                "case_id": "q1",
                "verdict": "contestable",
                "bucket": "semantic_collapsed_gold_contestable",
                "axis_type": "set_scope",
                "readings": [
                    _reading("r1", "2", True),
                    _reading("r1", "3", False),
                ],
                "contestability_reason": "two readings",
            }
        ),
        expected_case_id="q1",
    )

    assert parsed.parse_status == probe.PARSE_STATUS_INVALID_SCHEMA
    assert "duplicate reading_id values" in " ".join(parsed.parse_errors)


def test_parse_adjudication_response_requires_bucket_and_axis_type() -> None:
    parsed = probe.parse_adjudication_response(
        json.dumps(
            {
                "case_id": "q1",
                "verdict": "contestable",
                "readings": [
                    _reading("r1", "2", True),
                    _reading("r2", "3", False),
                ],
                "contestability_reason": "two readings",
            }
        ),
        expected_case_id="q1",
    )

    assert parsed.parse_status == probe.PARSE_STATUS_INVALID_SCHEMA
    assert "bucket must be a non-empty string" in " ".join(parsed.parse_errors)
    assert "axis_type must be a non-empty string" in " ".join(parsed.parse_errors)


def test_report_confirms_contestable_gold_only_when_gate_passes() -> None:
    payload_artifact = _payload_artifact()
    rows = [
        {
            "case_id": "q1",
            "provider": "test",
            "raw_output": "{}",
            "parse_status": probe.PARSE_STATUS_PARSED,
            "provider_exit_code": 0,
            "provider_timed_out": False,
            "verdict": "contestable",
            "readings": [
                _reading("official", "2", True, span="two weddings"),
                _reading("alternative", "3", False, span="three wedding events"),
            ],
            "contestability_reason": "event scope can include reception separately",
        }
    ]

    report = probe.build_report(
        rows=rows,
        payload_artifact=payload_artifact,
        go_min_cases=1,
        go_min_rate=0.0,
    )

    assert report["confirmed_contestable_gold_count"] == 1
    assert report["go_no_go"]["decision"] == "go"
    assert report["cases"][0]["all_readings_have_resolved_pivot"] is True


def test_report_does_not_confirm_contestable_when_pivot_span_is_unresolved() -> None:
    payload_artifact = _payload_artifact()
    rows = [
        {
            "case_id": "q1",
            "provider": "test",
            "raw_output": "{}",
            "parse_status": probe.PARSE_STATUS_PARSED,
            "provider_exit_code": 0,
            "provider_timed_out": False,
            "verdict": "contestable",
            "readings": [
                _reading("official", "2", True, span="two weddings"),
                _reading("alternative", "3", False, span="not in payload"),
            ],
            "contestability_reason": "event scope can include reception separately",
        }
    ]

    report = probe.build_report(
        rows=rows,
        payload_artifact=payload_artifact,
        go_min_cases=1,
        go_min_rate=0.0,
    )

    assert report["confirmed_contestable_gold_count"] == 0
    assert report["go_no_go"]["decision"] == "no_go"
    assert "reading_pivot_span_unresolved" in report["cases"][0]["quality_issues"]


def test_report_buckets_official_answer_accepts_both_values() -> None:
    payload_artifact = _payload_artifact()
    payload_artifact["payloads"][0]["case"]["official_answer"] = (
        "2 days. 3 days is also acceptable."
    )
    rows = [
        {
            "case_id": "q1",
            "provider": "test",
            "raw_output": "{}",
            "parse_status": probe.PARSE_STATUS_PARSED,
            "provider_exit_code": 0,
            "provider_timed_out": False,
            "verdict": "contestable",
            "readings": [
                _reading("exclusive", "2", True, span="two weddings"),
                _reading("inclusive", "3", True, span="three wedding events"),
            ],
            "contestability_reason": "official answer accepts both values",
        }
    ]

    report = probe.build_report(
        rows=rows,
        payload_artifact=payload_artifact,
        go_min_cases=1,
        go_min_rate=0.0,
    )

    assert report["confirmed_contestable_gold_count"] == 0
    assert report["gold_already_articulated_count"] == 1
    assert report["cases"][0]["bucket"] == "gold_already_articulated"
    assert report["go_no_go"]["decision"] == "no_go"


def test_run_payloads_streams_and_filters_case_ids(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    payload_artifact = _payload_artifact()
    payload_artifact["payloads"].append(
        {
            "case": {
                "id": "q2",
                "source_split": "heldout",
                "question": "How many events?",
                "official_answer": "3",
                "evidence_sessions": [
                    {"session_id": "s1", "text": "USER: three events"}
                ],
            }
        }
    )
    output_path = tmp_path / "outputs.jsonl"

    def fake_run_provider(
        *,
        command: str,
        prompt: str,
        timeout_seconds: int,
    ) -> interpretation_runner.ProviderResult:
        assert "q2" in prompt
        return interpretation_runner.ProviderResult(
            raw_output=json.dumps(
                {
                    "case_id": "q2",
                    "verdict": "not_contestable",
                    "bucket": "not_contestable",
                    "axis_type": "not_ambiguous",
                    "readings": [_reading("r1", "3", True, span="three events")],
                    "contestability_reason": "single reading",
                }
            ),
            stderr="",
            exit_code=0,
            latency_seconds=0.1,
        )

    monkeypatch.setattr(
        probe.interpretation_runner,
        "run_provider",
        fake_run_provider,
    )

    rows = probe.run_payloads(
        payload_artifact=payload_artifact,
        provider_command="provider",
        case_ids={"q2"},
        stream_outputs_path=output_path,
    )

    streamed = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["case_id"] for row in rows] == ["q2"]
    assert [row["case_id"] for row in streamed] == ["q2"]
    assert streamed[0]["parse_status"] == probe.PARSE_STATUS_PARSED


def _dataset_row(question_id: str, *, answer: str) -> dict:
    return {
        "question_id": question_id,
        "question_type": "count",
        "question": "How many weddings did I attend?",
        "answer": answer,
        "question_date": "2023/05/01",
        "answer_session_ids": [f"{question_id}_answer"],
        "haystack_session_ids": [f"{question_id}_answer", f"{question_id}_distractor"],
        "haystack_dates": ["2023/04/20", "2023/04/21"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I attended two weddings last month.",
                }
            ],
            [{"role": "assistant", "content": "irrelevant"}],
        ],
    }


def _payload_artifact() -> dict:
    return {
        "provider_visibility": {
            "official_answer_visible": True,
            "system_answer_visible": False,
        },
        "payloads": [
            {
                "case": {
                    "id": "q1",
                    "source_split": "heldout",
                    "question": "How many weddings?",
                    "official_answer": "2",
                    "evidence_sessions": [
                        {
                            "session_id": "s1",
                            "text": (
                                "USER: I attended two weddings. "
                                "There were three wedding events if the "
                                "reception is counted separately."
                            ),
                        }
                    ],
                }
            }
        ],
    }


def _reading(
    reading_id: str,
    answer_value: str,
    official: bool,
    *,
    span: str = "two weddings",
) -> dict:
    return {
        "reading_id": reading_id,
        "interpretation": f"answer is {answer_value}",
        "answer_value": answer_value,
        "compatible_with_official_answer": official,
        "pivot_spans": [
            {
                "evidence_session_id": "s1",
                "span": span,
                "why_pivot": "supports the reading",
            }
        ],
        "assumptions": ["scope choice"],
        "defensibility": "moderate",
    }

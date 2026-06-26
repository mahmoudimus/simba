from __future__ import annotations

import json
import pathlib

from simba.eval import answer_unit_witness_diagnostics as diagnostics


def test_longmemeval_heldout_excludes_fail18_and_hides_gold(
    tmp_path: pathlib.Path,
) -> None:
    dataset_path = tmp_path / "longmemeval_s.json"
    manifest_path = tmp_path / "fail18_manifest.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "fail18_row",
                    "question_type": "multi-session",
                    "question": "How many plants did I buy?",
                    "answer": "3",
                    "question_date": "2026/01/03",
                    "answer_session_ids": ["answer_secret"],
                    "haystack_session_ids": ["answer_secret", "distractor"],
                    "haystack_dates": ["2026/01/01", "2026/01/02"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I bought three plants."}],
                        [{"role": "user", "content": "Unrelated."}],
                    ],
                },
                {
                    "question_id": "heldout_row",
                    "question_type": "multi-session",
                    "question": "How many mugs did I buy?",
                    "answer": "2",
                    "question_date": "2026/01/03",
                    "answer_session_ids": ["raw_answer_session"],
                    "haystack_session_ids": ["raw_answer_session", "distractor"],
                    "haystack_dates": ["2026/01/01", "2026/01/02"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I bought two mugs."}],
                        [{"role": "assistant", "content": "General mug advice."}],
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps([{"question_id": "fail18_row"}]),
        encoding="utf-8",
    )

    artifacts = diagnostics.build_longmemeval_s_heldout_artifacts(
        dataset_path=dataset_path,
        fail18_manifest_path=manifest_path,
        seed="test",
        limit=10,
        top_k=2,
    )

    payloads = artifacts["payloads"]["payloads"]
    assert [payload["case"]["id"] for payload in payloads] == ["heldout_row"]
    assert artifacts["manifest"][0]["question_id"] == "heldout_row"
    payload_text = json.dumps(artifacts["payloads"]["payloads"]).lower()
    assert "raw_answer_session" not in payload_text
    assert "answer_session_ids" not in payload_text
    assert '"answer": "2"' not in payload_text
    assert payloads[0]["case"]["evidence_sessions"][0]["session_id"] == "evidence_001"
    provenance = artifacts["provenance"]["cases"]["heldout_row"]
    assert provenance["answer_session_ids"] == ["raw_answer_session"]


def test_reasoning_mechanism_diagnostic_splits_missing_and_wrong_choice(
    tmp_path: pathlib.Path,
) -> None:
    stable_path = tmp_path / "stable.json"
    outputs_path = tmp_path / "outputs.jsonl"
    stable_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "missing_case",
                        "question": "How many items?",
                        "gold_value": 2.0,
                        "answer_support": ["1"],
                        "classification": "reasoning_or_enumeration_capped",
                        "subtype": "wrong_unit_boundary_or_value",
                        "answer_session_ids": ["s1", "s2"],
                        "included_label_histogram": {"first item": 1},
                        "excluded_label_histogram": {},
                        "sample_summaries": [],
                    },
                    {
                        "case_id": "lookup_case",
                        "question": "How many points?",
                        "gold_value": 100.0,
                        "answer_support": ["50"],
                        "classification": "reasoning_or_enumeration_capped",
                        "subtype": "wrong_unit_boundary_or_value",
                        "answer_session_ids": ["s3"],
                        "included_label_histogram": {"50 point reward": 1},
                        "excluded_label_histogram": {"100 point reward": 1},
                        "sample_summaries": [],
                    },
                    {
                        "case_id": "retrieval_case",
                        "classification": "retrieval_capped",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "case_id": "missing_case",
            "sample_index": 1,
            "units": [
                {
                    "unit_id": "u1",
                    "label": "first item",
                    "decision": "include",
                    "value": None,
                    "evidence_session_id": "s1",
                    "evidence_span": "first item",
                }
            ],
        },
        {
            "case_id": "lookup_case",
            "sample_index": 1,
            "aggregation": "lookup_value",
            "units": [
                {
                    "unit_id": "u1",
                    "label": "50 point reward",
                    "decision": "include",
                    "value": 50,
                    "evidence_session_id": "s3",
                    "evidence_span": "50 points",
                },
                {
                    "unit_id": "u2",
                    "label": "100 point reward",
                    "decision": "exclude",
                    "value": 100,
                    "evidence_session_id": "s3",
                    "evidence_span": "100 points",
                },
            ],
        },
        {
            "case_id": "sum_case",
            "sample_index": 1,
            "aggregation": "sum_included",
            "units": [
                {
                    "unit_id": "u1",
                    "label": "30 minute jog",
                    "decision": "exclude",
                    "value": 0.5,
                    "evidence_session_id": "s4",
                    "evidence_span": "30-minute jog",
                }
            ],
        },
    ]
    stable = json.loads(stable_path.read_text(encoding="utf-8"))
    stable["cases"].append(
        {
            "case_id": "sum_case",
            "question": "How many hours?",
            "gold_value": 0.5,
            "answer_support": ["0"],
            "classification": "reasoning_or_enumeration_capped",
            "subtype": "missing_matching_unit",
            "answer_session_ids": ["s4"],
            "included_label_histogram": {},
            "excluded_label_histogram": {"30 minute jog": 1},
            "sample_summaries": [],
        }
    )
    stable_path.write_text(json.dumps(stable), encoding="utf-8")
    outputs_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    artifact = diagnostics.build_fail18_reasoning_mechanism_diagnostic(
        stable_wrong_diagnostic_path=stable_path,
        outputs_path=outputs_path,
    )

    assert artifact["summary"]["mechanism_counts"] == {
        "all_candidate_units_excluded": 1,
        "missing_answer_session_unit": 1,
        "wrong_lookup_value_choice": 1,
    }
    by_id = {case["case_id"]: case for case in artifact["cases"]}
    assert by_id["missing_case"]["never_covered_answer_sessions"] == ["s2"]
    assert by_id["lookup_case"]["excluded_gold_value_units"][0]["value"] == 100


def test_inclusion_policy_payload_filters_cases_and_preserves_secrecy(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "payloads.json"
    source_path.write_text(
        json.dumps(
            {
                "name": "source",
                "prompt_version": "answer_unit_witness_v1",
                "payloads": [
                    {
                        "task": "Answer with units.",
                        "prompt_version": "answer_unit_witness_v1",
                        "contract": ["Use only evidence."],
                        "case": {
                            "id": "keep",
                            "question": "How many plants?",
                            "evidence_sessions": [
                                {
                                    "session_id": "evidence_001",
                                    "text": "USER: I bought basil.",
                                }
                            ],
                        },
                    },
                    {
                        "task": "Answer with units.",
                        "prompt_version": "answer_unit_witness_v1",
                        "contract": ["Use only evidence."],
                        "case": {
                            "id": "drop",
                            "question": "How many mugs?",
                            "evidence_sessions": [
                                {"session_id": "evidence_002", "text": "USER: two"}
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = diagnostics.build_inclusion_policy_payload_artifact(
        source_payloads_path=source_path,
        case_ids=("keep",),
    )

    assert artifact["prompt_version"] == diagnostics.INCLUSION_POLICY_PROMPT_VERSION
    assert artifact["total"] == 1
    assert artifact["payloads"][0]["case"]["id"] == "keep"
    assert artifact["payloads"][0]["prompt_version"] == (
        diagnostics.INCLUSION_POLICY_PROMPT_VERSION
    )
    assert any(
        "scan every evidence session" in line
        for line in artifact["payloads"][0]["contract"]
    )
    rendered = json.dumps(artifact["payloads"]).lower()
    assert "gold_answer" not in rendered
    assert "answer_session_ids" not in rendered


def test_inclusion_policy_ab_report_marks_deltas(tmp_path: pathlib.Path) -> None:
    fail18_base = tmp_path / "fail18_base.json"
    fail18_candidate = tmp_path / "fail18_candidate.json"
    heldout_base = tmp_path / "heldout_base.json"
    heldout_candidate = tmp_path / "heldout_candidate.json"
    fail18_base.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "q1",
                        "question": "How many?",
                        "gold_value": 2,
                        "answer_support": ["1"],
                        "gold_in_answer_support": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fail18_candidate.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "q1",
                        "question": "How many?",
                        "gold_value": 2,
                        "answer_support": ["2"],
                        "gold_in_answer_support": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    heldout_base.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "h1",
                        "question": "How many?",
                        "gold_value": 1,
                        "answer_support": ["1"],
                        "gold_in_answer_support": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    heldout_candidate.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "h1",
                        "question": "How many?",
                        "gold_value": 1,
                        "answer_support": ["1"],
                        "gold_in_answer_support": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = diagnostics.build_inclusion_policy_ab_report(
        fail18_baseline_report_path=fail18_base,
        fail18_candidate_report_path=fail18_candidate,
        heldout_baseline_report_path=heldout_base,
        heldout_candidate_report_path=heldout_candidate,
        target_case_ids=("q1",),
    )

    assert artifact["summary"]["fail18"]["support_delta"] == 1
    assert artifact["summary"]["heldout"]["support_delta"] == 0
    assert artifact["summary"]["decision"] == "candidate_supported_for_next_k3_probe"


def test_span_survival_report_distinguishes_truncation_from_retrieval(
    tmp_path: pathlib.Path,
) -> None:
    payloads_path = tmp_path / "payloads.json"
    needles_path = tmp_path / "needles.json"
    corpus_path = tmp_path / "corpus.json"
    stable_path = tmp_path / "stable.json"
    payloads_path.write_text(
        json.dumps(
            {
                "prompt_version": "answer_unit_witness_v1",
                "retrieval": {"chars_per_session": 20},
                "payloads": [
                    {
                        "case": {
                            "id": "q1",
                            "evidence_sessions": [
                                {
                                    "session_id": "s1",
                                    "text": "USER: early clue\n...[truncated]",
                                }
                            ],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    needles_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "q1",
                        "question": "How many plants?",
                        "needles": [
                            {
                                "label": "late acquisition",
                                "session_id": "s1",
                                "needle": "got from my sister last month",
                            },
                            {
                                "label": "missing session",
                                "session_id": "s2",
                                "needle": "bought two mugs",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "early clue"},
                            {
                                "role": "user",
                                "content": "got from my sister last month",
                            },
                        ],
                        [{"role": "user", "content": "bought two mugs"}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    stable_path.write_text(
        json.dumps({"cases": [{"case_id": "q1"}]}),
        encoding="utf-8",
    )

    artifact = diagnostics.build_fail18_span_survival_report(
        payloads_path=payloads_path,
        gold_span_needles_path=needles_path,
        corpus_path=corpus_path,
        stable_wrong_diagnostic_path=stable_path,
    )

    assert artifact["summary"]["status_counts"] == {
        "absent_payload_truncated": 1,
        "session_not_retrieved": 1,
    }
    case = artifact["cases"][0]
    assert case["has_truncation_loss"] is True
    assert case["has_retrieval_loss"] is True
    assert case["needles"][0]["full_session_contains_exact"] is True


def test_payload_budget_ab_report_tracks_fail18_and_heldout_deltas(
    tmp_path: pathlib.Path,
) -> None:
    fail18_base = tmp_path / "fail18_base.json"
    fail18_candidate = tmp_path / "fail18_candidate.json"
    heldout_base = tmp_path / "heldout_base.json"
    heldout_candidate = tmp_path / "heldout_candidate.json"
    fail18_base.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "q1", "gold_in_answer_support": False},
                    {"case_id": "q2", "gold_in_answer_support": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    fail18_candidate.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "q1", "gold_in_answer_support": True},
                    {"case_id": "q2", "gold_in_answer_support": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    heldout_base.write_text(
        json.dumps({"cases": [{"case_id": "h1", "gold_in_answer_support": True}]}),
        encoding="utf-8",
    )
    heldout_candidate.write_text(
        json.dumps({"cases": [{"case_id": "h1", "gold_in_answer_support": False}]}),
        encoding="utf-8",
    )

    artifact = diagnostics.build_payload_budget_ab_report(
        fail18_baseline_report_path=fail18_base,
        fail18_candidate_report_path=fail18_candidate,
        heldout_baseline_report_path=heldout_base,
        heldout_candidate_report_path=heldout_candidate,
    )

    assert artifact["summary"]["fail18"]["support_delta"] == 1
    assert artifact["summary"]["fail18"]["improved_case_ids"] == ["q1"]
    assert artifact["summary"]["heldout"]["support_delta"] == -1
    assert artifact["summary"]["heldout"]["regressed_case_ids"] == ["h1"]
    assert artifact["summary"]["decision"] == "reject_or_rework_possible_overfit"


def test_payload_budget_regression_diagnostic_splits_flip_and_displacement(
    tmp_path: pathlib.Path,
) -> None:
    budget_report = tmp_path / "budget.json"
    baseline_payloads = tmp_path / "baseline_payloads.json"
    candidate_payloads = tmp_path / "candidate_payloads.json"
    baseline_outputs = tmp_path / "baseline_outputs.jsonl"
    candidate_outputs = tmp_path / "candidate_outputs.jsonl"
    budget_report.write_text(
        json.dumps(
            {
                "summary": {
                    "fail18": {
                        "regressed_case_ids": ["flip_case", "displace_case"]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    baseline_payloads.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "flip_case",
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {"session_id": "s1", "text": "gold art tour"}
                            ],
                        }
                    },
                    {
                        "case": {
                            "id": "displace_case",
                            "question": "How many projects?",
                            "evidence_sessions": [
                                {"session_id": "s2", "text": "gold project"}
                            ],
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_payloads.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "flip_case",
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "s1",
                                    "text": "gold art tour plus distractors",
                                }
                            ],
                        }
                    },
                    {
                        "case": {
                            "id": "displace_case",
                            "question": "How many projects?",
                            "evidence_sessions": [
                                {
                                    "session_id": "s2",
                                    "text": "gold project and late wrong project",
                                }
                            ],
                        }
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_rows = [
        {
            "case_id": "flip_case",
            "answer_number": 1,
            "units": [
                {
                    "label": "gold art tour",
                    "decision": "include",
                    "evidence_session_id": "s1",
                    "evidence_span": "gold art tour",
                }
            ],
        },
        {
            "case_id": "displace_case",
            "answer_number": 1,
            "units": [
                {
                    "label": "gold project",
                    "decision": "include",
                    "evidence_session_id": "s2",
                    "evidence_span": "gold project",
                }
            ],
        },
    ]
    candidate_rows = [
        {
            "case_id": "flip_case",
            "answer_number": 0,
            "units": [
                {
                    "label": "gold art tour",
                    "decision": "exclude",
                    "evidence_session_id": "s1",
                    "evidence_span": "gold art tour",
                    "reason_code": "ambiguous",
                }
            ],
        },
        {
            "case_id": "displace_case",
            "answer_number": 2,
            "units": [
                {
                    "label": "gold project",
                    "decision": "include",
                    "evidence_session_id": "s2",
                    "evidence_span": "gold project",
                },
                {
                    "label": "late wrong project",
                    "decision": "include",
                    "evidence_session_id": "s2",
                    "evidence_span": "late wrong project",
                },
            ],
        },
    ]
    baseline_outputs.write_text(
        "\n".join(json.dumps(row) for row in baseline_rows) + "\n",
        encoding="utf-8",
    )
    candidate_outputs.write_text(
        "\n".join(json.dumps(row) for row in candidate_rows) + "\n",
        encoding="utf-8",
    )

    artifact = diagnostics.build_payload_budget_regression_diagnostic(
        budget_ab_report_path=budget_report,
        baseline_payloads_path=baseline_payloads,
        candidate_payloads_path=candidate_payloads,
        baseline_outputs_path=baseline_outputs,
        candidate_outputs_path=candidate_outputs,
    )

    assert artifact["summary"]["mechanism_counts"] == {
        "displacement_new_window_span": 1,
        "same_visible_span_decision_flip": 1,
    }
    by_id = {case["case_id"]: case for case in artifact["cases"]}
    assert by_id["flip_case"]["candidate_excluded_baseline_span_count"] == 1
    assert by_id["displace_case"]["candidate_added_included_new_window_count"] == 1

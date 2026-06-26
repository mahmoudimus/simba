from __future__ import annotations

import json
import pathlib

from simba.eval import answer_window_selector as selector


def test_default_config_is_promoted_a5_augment() -> None:
    assert selector.SELECTOR_VERSION == "answer_window_selector_a5"
    assert selector.DEFAULT_SELECTED_PAYLOADS_PATH.name.endswith(
        "selector_a5_augment.json"
    )
    assert selector.DEFAULT_PRE_METRICS_PATH.name.endswith(
        "selector_a5_augment_pre_metrics.json"
    )
    assert selector.DEFAULT_CONFIG.name == "selector_a5_augment"
    assert selector.DEFAULT_CONFIG.prefix_floor_chars == 2500
    assert selector.DEFAULT_CONFIG.type_cue_weight == 0.25
    assert selector.DEFAULT_CONFIG.operation_cue_weight == 0.5
    assert selector.DEFAULT_CONFIG.operation_cue_radius_chars == 120


def test_cli_defaults_use_promoted_a5_augment(monkeypatch) -> None:
    captured: dict[str, selector.WindowSelectorConfig] = {}

    def fake_build_selected_payload_artifact(
        *,
        source_payloads_path: pathlib.Path,
        config: selector.WindowSelectorConfig,
    ) -> dict:
        captured["config"] = config
        return {"payloads": [], "source": str(source_payloads_path)}

    monkeypatch.setattr(
        selector,
        "build_selected_payload_artifact",
        fake_build_selected_payload_artifact,
    )
    monkeypatch.setattr(selector, "_write_json", lambda path, artifact: None)

    assert selector.main(["--build-payloads"]) == 0

    config = captured["config"]
    assert config.name == "selector_a5_augment"
    assert config.prefix_floor_chars == 2500
    assert config.type_cue_weight == 0.25
    assert config.operation_cue_weight == 0.5
    assert config.operation_cue_radius_chars == 120


def test_user_window_beats_assistant_window_with_same_question_terms() -> None:
    text = (
        "ASSISTANT: The museum has art events and gallery tours every week.\n"
        "USER: I attended an art event at the small gallery yesterday."
    )
    config = selector.WindowSelectorConfig(
        window_radius_chars=120,
        max_windows_per_session=1,
        max_chars_per_session=400,
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many art events did I attend?"
        ),
        term_weights={
            "art": 1.0,
            "events": 1.0,
            "attend": 1.0,
        },
        config=config,
    )

    assert len(windows) == 1
    assert windows[0].role == "USER"
    assert "attended an art event" in text[windows[0].start : windows[0].end]


def test_build_selected_payload_artifact_keeps_payload_gold_free(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "payloads.json"
    source_path.write_text(
        json.dumps(
            {
                "name": "source",
                "prompt_version": "answer_unit_witness_v1",
                "provider_visibility": {"gold_answer_visible": False},
                "retrieval": {"chars_per_session": 8000},
                "payloads": [
                    {
                        "task": "Answer with units.",
                        "prompt_version": "answer_unit_witness_v1",
                        "contract": ["Use only evidence."],
                        "case": {
                            "id": "q1",
                            "question": "How many art events did I attend?",
                            "evidence_sessions": [
                                {
                                    "session_id": "s1",
                                    "date": "2026/01/01",
                                    "text": (
                                        "ASSISTANT: I can recommend many art "
                                        "events and museum tours.\n"
                                        "USER: I attended an art event at the "
                                        "local gallery last night."
                                    ),
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = selector.build_selected_payload_artifact(
        source_payloads_path=source_path,
        config=selector.WindowSelectorConfig(
            window_radius_chars=80,
            max_windows_per_session=1,
            max_chars_per_session=400,
        ),
    )

    payload_text = json.dumps(artifact["payloads"])
    assert "gold" not in payload_text.lower()
    assert "attended an art event" in payload_text
    assert "selection_metadata" in artifact
    assert artifact["selector"]["config"]["name"] == "selector_a2"


def test_pre_metrics_reports_selector_drop_and_salience_margin(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.json"
    selected_path = tmp_path / "selected.json"
    needles_path = tmp_path / "needles.json"
    source_path.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "q1",
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "gold_session",
                                    "text": "USER: I attended the gold art event.",
                                },
                                {
                                    "session_id": "dropped_session",
                                    "text": "USER: I attended the dropped art event.",
                                },
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selected_path.write_text(
        json.dumps(
            {
                "selector": {"version": "test"},
                "payloads": [
                    {
                        "case": {
                            "id": "q1",
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "gold_session",
                                    "text": "USER: I attended the gold art event.",
                                },
                                {
                                    "session_id": "dropped_session",
                                    "text": "USER: unrelated text",
                                },
                            ],
                        }
                    }
                ],
                "selection_metadata": {
                    "cases": {
                        "q1": {
                            "sessions": {
                                "gold_session": {
                                    "windows": [
                                        {
                                            "start": 0,
                                            "end": 36,
                                            "score": 4.0,
                                        }
                                    ]
                                },
                                "dropped_session": {
                                    "windows": [
                                        {
                                            "start": 0,
                                            "end": 20,
                                            "score": 8.0,
                                        }
                                    ]
                                },
                            }
                        }
                    }
                },
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
                        "question": "How many art events?",
                        "needles": [
                            {
                                "label": "gold",
                                "session_id": "gold_session",
                                "needle": "gold art event",
                            },
                            {
                                "label": "dropped",
                                "session_id": "dropped_session",
                                "needle": "dropped art event",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = selector.build_fail18_selector_pre_metrics(
        selected_payloads_path=selected_path,
        source_payloads_path=source_path,
        gold_span_needles_path=needles_path,
    )

    case = artifact["cases"][0]
    assert artifact["summary"]["selector_dropped_count"] == 1
    assert artifact["summary"]["kill_gate_passed"] is False
    assert case["selector_dropped_count"] == 1
    gold = case["needles"][0]
    assert gold["selected_contains_exact"] is True
    assert gold["salience_margin"] == -4.0


def test_type_cue_creates_window_for_typed_instance_without_literal_question_term(
    tmp_path: pathlib.Path,
) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_lexicon(
        lexicon_path,
        [
            _record(
                "wordnet:spathiphyllum.n.01",
                "spathiphyllum.n.01",
                "peace lily",
                aliases=("peace lily", "spathiphyllum"),
                parents=("flower.n.01",),
                definition="any of various plants of the genus Spathiphyllum",
            ),
            _record(
                "wordnet:flower.n.01",
                "flower.n.01",
                "flower",
                aliases=("flower",),
                parents=("plant.n.02",),
                definition="a plant cultivated for its blooms or blossoms",
            ),
            _record(
                "wordnet:plant.n.02",
                "plant.n.02",
                "flora",
                aliases=("plant", "flora"),
                parents=(),
                definition="a living organism lacking the power of locomotion",
            ),
        ],
    )
    text = (
        "ASSISTANT: Garden care can be tricky in dry rooms.\n"
        "USER: I got a peace lily from the nursery yesterday."
    )
    config = selector.WindowSelectorConfig(
        window_radius_chars=80,
        max_windows_per_session=1,
        max_chars_per_session=400,
        type_cue_weight=4.0,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many plants did I acquire?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many plants did I acquire?"
        ),
        term_weights={"plant": 1.0, "plants": 1.0, "acquire": 1.0},
        config=config,
    )

    assert len(windows) == 1
    assert windows[0].role == "USER"
    assert "peace lily" in text[windows[0].start : windows[0].end]
    assert windows[0].score_components["type_cue"] > 0
    assert windows[0].type_cue_matches[0]["source_type"] == "peace lily"
    assert windows[0].type_cue_matches[0]["target_type"] == "plant"
    assert windows[0].score_components["role_weight"] == 0
    assert windows[0].score_components["compactness"] == 0


def test_type_only_window_does_not_unlock_role_or_compactness(
    tmp_path: pathlib.Path,
) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_lexicon(
        lexicon_path,
        [
            _record(
                "wordnet:spathiphyllum.n.01",
                "spathiphyllum.n.01",
                "peace lily",
                aliases=("peace lily",),
                parents=("plant.n.02",),
                definition="a plant with white flowers",
            ),
            _record(
                "wordnet:plant.n.02",
                "plant.n.02",
                "plant",
                aliases=("plant",),
                parents=(),
                definition="a living organism lacking locomotion",
            ),
        ],
    )
    config = selector.WindowSelectorConfig(
        window_radius_chars=80,
        max_windows_per_session=1,
        max_chars_per_session=400,
        type_cue_weight=4.0,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        "USER: I got a peace lily from the nursery.",
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many plants did I acquire?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many plants did I acquire?"
        ),
        term_weights={"plant": 1.0, "plants": 1.0, "acquire": 1.0},
        config=config,
    )

    assert len(windows) == 1
    assert windows[0].score_components["base_question_overlap"] == 0
    assert windows[0].score_components["type_cue"] > 0
    assert windows[0].score_components["role_weight"] == 0
    assert windows[0].score_components["compactness"] == 0


def test_operation_type_conjunction_unlocks_tight_window(
    tmp_path: pathlib.Path,
) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_plant_lexicon(lexicon_path)
    text = (
        "USER: I got a peace lily from the nursery yesterday. "
        + "Care notes can be verbose. " * 20
    )
    config = selector.WindowSelectorConfig(
        window_radius_chars=400,
        max_windows_per_session=1,
        max_chars_per_session=1000,
        type_cue_weight=0.5,
        operation_cue_weight=0.5,
        operation_cue_radius_chars=40,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many plants did I acquire?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many plants did I acquire?"
        ),
        term_weights={"plant": 1.0, "plants": 1.0, "acquire": 1.0},
        config=config,
    )

    assert len(windows) == 1
    window_text = text[windows[0].start : windows[0].end]
    assert "got a peace lily" in window_text
    assert len(window_text) < 160
    assert windows[0].score_components["base_question_overlap"] == 0
    assert windows[0].score_components["type_cue"] > 0
    assert windows[0].score_components["operation_cue"] > 0
    assert windows[0].score_components["role_weight"] > 0
    assert windows[0].score_components["compactness"] > 0


def test_blocked_acquisition_sense_does_not_unlock_operation(
    tmp_path: pathlib.Path,
) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_plant_lexicon(lexicon_path)
    config = selector.WindowSelectorConfig(
        window_radius_chars=80,
        max_windows_per_session=1,
        max_chars_per_session=400,
        type_cue_weight=0.5,
        operation_cue_weight=0.5,
        operation_cue_radius_chars=40,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        "USER: I got rid of a peace lily last month.",
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many plants did I acquire?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many plants did I acquire?"
        ),
        term_weights={"plant": 1.0, "plants": 1.0, "acquire": 1.0},
        config=config,
    )

    assert len(windows) == 1
    assert windows[0].score_components["base_question_overlap"] == 0
    assert windows[0].score_components["type_cue"] > 0
    assert windows[0].score_components["operation_cue"] == 0
    assert windows[0].score_components["role_weight"] == 0
    assert windows[0].score_components["compactness"] == 0


def test_prefix_floor_preserves_early_answer_and_adds_late_window(
    tmp_path: pathlib.Path,
) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_plant_lexicon(lexicon_path)
    text = (
        "USER: I attended the gold art event yesterday.\n"
        + ("ASSISTANT: filler about art.\n" * 20)
        + "USER: I got a peace lily from the nursery last week."
    )
    config = selector.WindowSelectorConfig(
        prefix_floor_chars=80,
        window_radius_chars=120,
        max_windows_per_session=1,
        max_chars_per_session=500,
        type_cue_weight=0.5,
        operation_cue_weight=0.5,
        operation_cue_radius_chars=40,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many plants did I acquire?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many plants did I acquire?"
        ),
        term_weights={"plant": 1.0, "plants": 1.0, "acquire": 1.0},
        config=config,
    )

    selected_text = "\n".join(text[window.start : window.end] for window in windows)
    assert windows[0].role == "PREFIX"
    assert windows[0].start == 0
    assert windows[0].end == 80
    assert "gold art event" in selected_text
    assert "got a peace lily" in selected_text
    assert all(window.start >= 80 for window in windows[1:])


def test_prefix_floor_does_not_consume_extra_window_budget(
    tmp_path: pathlib.Path,
) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_plant_lexicon(lexicon_path)
    text = (
        "USER: early answer in the guaranteed prefix.\n"
        + ("ASSISTANT: filler.\n" * 20)
        + "USER: I got a peace lily from the nursery last week.\n"
        + ("ASSISTANT: more filler.\n" * 20)
        + "USER: I received a peace lily from a friend yesterday."
    )
    config = selector.WindowSelectorConfig(
        prefix_floor_chars=80,
        window_radius_chars=120,
        max_windows_per_session=2,
        max_chars_per_session=900,
        type_cue_weight=0.5,
        operation_cue_weight=0.5,
        operation_cue_radius_chars=40,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many plants did I acquire?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many plants did I acquire?"
        ),
        term_weights={"plant": 1.0, "plants": 1.0, "acquire": 1.0},
        config=config,
    )

    assert len(windows) == 3
    assert windows[0].role == "PREFIX"
    assert [window.start >= 80 for window in windows[1:]] == [True, True]


def test_adjacent_windows_do_not_merge_away_answer_span() -> None:
    left = selector.CandidateWindow(
        session_id="s1",
        role="USER",
        start=10,
        end=20,
        score=1.0,
        score_components={},
    )
    right = selector.CandidateWindow(
        session_id="s1",
        role="USER",
        start=20,
        end=30,
        score=2.0,
        score_components={},
    )

    assert selector._overlaps_mergeable(right, [left]) is False


def test_compactness_is_capped() -> None:
    config = selector.WindowSelectorConfig(
        window_radius_chars=80,
        max_windows_per_session=1,
        max_chars_per_session=400,
        compactness_cap=2.0,
    )

    windows = selector.select_session_windows(
        "USER: art event.",
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many art events did I attend?"
        ),
        term_weights={"art": 10.0, "event": 10.0, "events": 10.0, "attend": 10.0},
        config=config,
    )

    assert len(windows) == 1
    assert windows[0].score_components["compactness"] == 2.0


def test_type_cue_prefers_maximal_question_type_targets(tmp_path: pathlib.Path) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_lexicon(
        lexicon_path,
        [
            _record(
                "wordnet:drum.n.01",
                "drum.n.01",
                "drum",
                aliases=("drum",),
                parents=("percussion_instrument.n.01",),
                definition="a musical percussion instrument",
            ),
            _record(
                "wordnet:percussion_instrument.n.01",
                "percussion_instrument.n.01",
                "percussion instrument",
                aliases=("percussion instrument",),
                parents=("musical_instrument.n.01",),
                definition="a musical instrument that is struck",
            ),
            _record(
                "wordnet:musical_instrument.n.01",
                "musical_instrument.n.01",
                "musical instrument",
                aliases=("musical instrument",),
                parents=(),
                definition="an instrument used to make music",
            ),
            _record(
                "wordnet:legal_document.n.01",
                "legal_document.n.01",
                "legal document",
                aliases=("instrument",),
                parents=(),
                definition="a legal instrument or document",
            ),
        ],
    )
    text = (
        "USER: I signed the instrument yesterday.\n"
        "USER: I still own an old drum set."
    )
    config = selector.WindowSelectorConfig(
        window_radius_chars=80,
        max_windows_per_session=1,
        max_chars_per_session=400,
        type_cue_weight=4.0,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many musical instruments do I own?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many musical instruments do I own?"
        ),
        term_weights={"musical": 1.0, "instrument": 1.0, "own": 1.0},
        config=config,
    )

    assert len(windows) == 1
    assert "old drum set" in text[windows[0].start : windows[0].end]
    assert all(
        match["target_type"] == "musical instrument"
        for match in windows[0].type_cue_matches
    )


def test_type_cue_is_role_gated_to_user_evidence(tmp_path: pathlib.Path) -> None:
    lexicon_path = tmp_path / "nltk_lexicon.jsonl"
    _write_lexicon(
        lexicon_path,
        [
            _record(
                "wordnet:piano.n.01",
                "piano.n.01",
                "piano",
                aliases=("piano",),
                parents=("musical_instrument.n.01",),
                definition="a keyboard musical instrument",
            ),
            _record(
                "wordnet:drum.n.01",
                "drum.n.01",
                "drum",
                aliases=("drum",),
                parents=("musical_instrument.n.01",),
                definition="a percussion musical instrument",
            ),
            _record(
                "wordnet:musical_instrument.n.01",
                "musical_instrument.n.01",
                "musical instrument",
                aliases=("musical instrument",),
                parents=(),
                definition="an instrument used to make music",
            ),
        ],
    )
    text = (
        "ASSISTANT: A piano needs regular tuning.\n"
        "USER: I own an old drum set."
    )
    config = selector.WindowSelectorConfig(
        window_radius_chars=80,
        max_windows_per_session=2,
        max_chars_per_session=400,
        type_cue_weight=4.0,
        lexicon_path=str(lexicon_path),
    )

    windows = selector.select_session_windows(
        text,
        session_id="s1",
        question_terms=selector.question_terms_from_text(
            "How many musical instruments do I own?"
        ),
        question_type_targets=selector.question_type_targets_from_text(
            "How many musical instruments do I own?"
        ),
        term_weights={"own": 1.0},
        config=config,
    )

    assert len(windows) == 1
    assert windows[0].role == "USER"
    assert "old drum set" in text[windows[0].start : windows[0].end]
    assert windows[0].score_components["type_cue"] > 0


def test_margin_regression_blocks_provider_even_without_span_drops(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.json"
    selected_path = tmp_path / "selected.json"
    needles_path = tmp_path / "needles.json"
    baseline_path = tmp_path / "baseline.json"
    source_path.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": "q1",
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "gold_session",
                                    "text": "USER: I attended the gold art event.",
                                },
                                {
                                    "session_id": "distractor_session",
                                    "text": "USER: unrelated but high scoring text",
                                },
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selected_path.write_text(
        json.dumps(
            {
                "selector": {"version": "test"},
                "payloads": [
                    {
                        "case": {
                            "id": "q1",
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "gold_session",
                                    "text": "USER: I attended the gold art event.",
                                },
                                {
                                    "session_id": "distractor_session",
                                    "text": "USER: unrelated but high scoring text",
                                },
                            ],
                        }
                    }
                ],
                "selection_metadata": {
                    "cases": {
                        "q1": {
                            "sessions": {
                                "gold_session": {
                                    "windows": [
                                        {
                                            "start": 0,
                                            "end": 36,
                                            "score": 2.0,
                                        }
                                    ]
                                },
                                "distractor_session": {
                                    "windows": [
                                        {
                                            "start": 0,
                                            "end": 39,
                                            "score": 5.0,
                                        }
                                    ]
                                },
                            }
                        }
                    }
                },
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
                        "question": "How many art events?",
                        "needles": [
                            {
                                "label": "gold",
                                "session_id": "gold_session",
                                "needle": "gold art event",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "q1",
                        "salience_margin_min": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = selector.build_fail18_selector_pre_metrics(
        selected_payloads_path=selected_path,
        source_payloads_path=source_path,
        gold_span_needles_path=needles_path,
        baseline_pre_metrics_path=baseline_path,
    )

    assert artifact["summary"]["selector_dropped_count"] == 0
    assert artifact["summary"]["span_survival_gate_passed"] is True
    assert artifact["summary"]["margin_regression_count"] == 1
    assert artifact["summary"]["margin_gate_passed"] is False
    assert artifact["summary"]["provider_run_allowed"] is False


def test_positive_margin_decrease_does_not_block_provider(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.json"
    selected_path = tmp_path / "selected.json"
    needles_path = tmp_path / "needles.json"
    baseline_path = tmp_path / "baseline.json"
    _write_margin_fixture(
        source_path=source_path,
        selected_path=selected_path,
        needles_path=needles_path,
        case_id="q1",
        gold_score=3.0,
        distractor_score=2.0,
    )
    baseline_path.write_text(
        json.dumps({"cases": [{"case_id": "q1", "salience_margin_min": 2.0}]}),
        encoding="utf-8",
    )

    artifact = selector.build_fail18_selector_pre_metrics(
        selected_payloads_path=selected_path,
        source_payloads_path=source_path,
        gold_span_needles_path=needles_path,
        baseline_pre_metrics_path=baseline_path,
    )

    case = artifact["cases"][0]
    assert case["salience_margin_min"] == 1.0
    assert case["salience_margin_delta_from_baseline"] == -1.0
    assert case["salience_margin_crossed_nonpositive"] is False
    assert artifact["summary"]["margin_regression_count"] == 0
    assert artifact["summary"]["margin_gate_passed"] is True
    assert artifact["summary"]["provider_run_allowed"] is True


def test_margin_gate_excludes_retrieval_capped_cases(
    tmp_path: pathlib.Path,
) -> None:
    source_path = tmp_path / "source.json"
    selected_path = tmp_path / "selected.json"
    needles_path = tmp_path / "needles.json"
    baseline_path = tmp_path / "baseline.json"
    _write_margin_fixture(
        source_path=source_path,
        selected_path=selected_path,
        needles_path=needles_path,
        case_id="gpt4_194be4b3",
        gold_score=2.0,
        distractor_score=5.0,
    )
    baseline_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "gpt4_194be4b3",
                        "salience_margin_min": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = selector.build_fail18_selector_pre_metrics(
        selected_payloads_path=selected_path,
        source_payloads_path=source_path,
        gold_span_needles_path=needles_path,
        baseline_pre_metrics_path=baseline_path,
    )

    case = artifact["cases"][0]
    assert case["salience_margin_min"] == -3.0
    assert case["salience_margin_crossed_nonpositive"] is True
    assert case["margin_gate_excluded"] is True
    assert artifact["summary"]["margin_regression_count"] == 0
    assert artifact["summary"]["margin_gate_exclusion_count"] == 1
    assert artifact["summary"]["margin_gate_passed"] is True
    assert artifact["summary"]["provider_run_allowed"] is True


def _write_margin_fixture(
    *,
    source_path: pathlib.Path,
    selected_path: pathlib.Path,
    needles_path: pathlib.Path,
    case_id: str,
    gold_score: float,
    distractor_score: float,
) -> None:
    source_path.write_text(
        json.dumps(
            {
                "payloads": [
                    {
                        "case": {
                            "id": case_id,
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "gold_session",
                                    "text": "USER: I attended the gold art event.",
                                },
                                {
                                    "session_id": "distractor_session",
                                    "text": "USER: unrelated but high scoring text",
                                },
                            ],
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selected_path.write_text(
        json.dumps(
            {
                "selector": {"version": "test"},
                "payloads": [
                    {
                        "case": {
                            "id": case_id,
                            "question": "How many art events?",
                            "evidence_sessions": [
                                {
                                    "session_id": "gold_session",
                                    "text": "USER: I attended the gold art event.",
                                },
                                {
                                    "session_id": "distractor_session",
                                    "text": "USER: unrelated but high scoring text",
                                },
                            ],
                        }
                    }
                ],
                "selection_metadata": {
                    "cases": {
                        case_id: {
                            "sessions": {
                                "gold_session": {
                                    "windows": [
                                        {
                                            "start": 0,
                                            "end": 36,
                                            "score": gold_score,
                                        }
                                    ]
                                },
                                "distractor_session": {
                                    "windows": [
                                        {
                                            "start": 0,
                                            "end": 39,
                                            "score": distractor_score,
                                        }
                                    ]
                                },
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    needles_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": case_id,
                        "question": "How many art events?",
                        "needles": [
                            {
                                "label": "gold",
                                "session_id": "gold_session",
                                "needle": "gold art event",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_plant_lexicon(path: pathlib.Path) -> None:
    _write_lexicon(
        path,
        [
            _record(
                "wordnet:spathiphyllum.n.01",
                "spathiphyllum.n.01",
                "peace lily",
                aliases=("peace lily", "spathiphyllum"),
                parents=("plant.n.02",),
                definition="a plant with white flowers",
            ),
            _record(
                "wordnet:plant.n.02",
                "plant.n.02",
                "plant",
                aliases=("plant",),
                parents=(),
                definition="a living organism lacking locomotion",
            ),
        ],
    )


def _write_lexicon(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(
    concept_id: str,
    provider_ref: str,
    label: str,
    *,
    aliases: tuple[str, ...],
    parents: tuple[str, ...],
    definition: str,
) -> dict[str, object]:
    return {
        "id": concept_id,
        "provider": "wordnet",
        "kind": "concept",
        "provider_ref": provider_ref,
        "label": label,
        "aliases_json": json.dumps(list(aliases)),
        "parents_json": json.dumps(list(parents)),
        "definition": definition,
    }

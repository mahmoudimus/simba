"""Tests for the judge config section (grader separate from answerer)."""

from __future__ import annotations

import pathlib
import tempfile


def test_judge_section_registered_in_config_registry() -> None:
    import simba.config
    import simba.llm.judge_config

    assert "judge" in simba.config.list_sections()


def test_judge_config_defaults_differ_from_llm_defaults() -> None:
    import simba.config
    import simba.llm.config
    import simba.llm.judge_config

    with tempfile.TemporaryDirectory() as td:
        cfg_j = simba.config.load("judge", root=pathlib.Path(td))
        cfg_l = simba.config.load("llm", root=pathlib.Path(td))
    assert cfg_j.provider != cfg_l.provider or cfg_j.model != cfg_l.model
    assert cfg_j.timeout_seconds == 90.0
    assert cfg_j.max_tokens == 512


def test_get_judge_client_returns_llm_client_instance() -> None:
    import simba.llm.client as lc
    import simba.llm.judge_config as jc

    client = jc.get_judge_client()
    assert isinstance(client, lc.LlmClient)


def test_load_judge_config_override_takes_precedence() -> None:
    import simba.llm.judge_config as jc

    cfg = jc.load_judge_config(model="my-judge-model")
    assert cfg.model == "my-judge-model"


def test_judge_config_toml_round_trip(tmp_path) -> None:
    import simba.config
    import simba.llm.judge_config

    simba.config.set_value("judge", "model", "phi-3-mini", scope="local", root=tmp_path)
    loaded = simba.config.load("judge", root=tmp_path)
    assert loaded.model == "phi-3-mini"

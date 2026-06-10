"""Tests for the rlm config section."""

from __future__ import annotations

import simba.config
import simba.rlm.config


def test_rlm_section_registered():
    assert "rlm" in simba.config.list_sections()


def test_defaults():
    cfg = simba.config.load("rlm")
    assert cfg.max_search_matches == 20
    assert cfg.search_context_chars == 200
    assert cfg.regex_timeout_seconds == 2.0
    assert cfg.max_pattern_length == 500
    assert cfg.lru_documents == 4
    assert cfg.transcript_source == "md"
    assert cfg.default_max_pointers == 5
    assert cfg.engine == "claude-cli"


def test_engine_defaults():
    cfg = simba.config.load("rlm")
    assert cfg.engine == "claude-cli"
    assert cfg.engine_model == "haiku"
    assert cfg.engine_base_url == ""
    assert cfg.engine_api_key_env == "ANTHROPIC_API_KEY"
    assert "mcp__neuron__rlm_grep" in cfg.engine_allowed_tools
    assert cfg.engine_max_turns == 12
    assert cfg.engine_max_pointers == 5
    assert cfg.engine_min_new_exchanges == 20


def test_digest_stale_after_seconds_default():
    # 0 => None => no stale-reclaim => preserves current rlm dedup behavior.
    cfg = simba.config.load("rlm")
    assert cfg.digest_stale_after_seconds == 0

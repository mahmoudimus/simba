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
    assert cfg.engine == "claude"

"""Tests for the sync config section defaults."""

from __future__ import annotations

import simba.config
import simba.sync.config


def test_sync_section_registered() -> None:
    assert "sync" in simba.config.list_sections()


def test_extraction_defaults() -> None:
    cfg = simba.sync.config.SyncConfig()
    # LLM extraction is the primary KG feed by default (degrades to regex with
    # no provider); bounded per cycle to keep the first backlog sweep cheap.
    assert cfg.extract_strategy == "llm+regex"
    assert cfg.llm_extract_enabled is True
    assert cfg.llm_extract_max_per_cycle > 0

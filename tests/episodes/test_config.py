"""Tests for the episodes config section."""

from __future__ import annotations

import simba.config
import simba.episodes.config


def test_episodes_section_registered() -> None:
    assert "episodes" in simba.config.list_sections()


def test_defaults() -> None:
    cfg = simba.episodes.config.EpisodesConfig()
    assert cfg.enabled is True
    assert cfg.min_memories == 5
    assert cfg.max_members == 50
    assert cfg.auto_on_precompact is True
    assert cfg.scheduler_enabled is True
    assert cfg.job_timeout_hours == 4

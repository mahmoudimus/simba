"""Tests for episode-consolidation wiring (PreCompact + scheduler gating)."""

from __future__ import annotations

import simba.config
import simba.episodes.config as ec_cfg
import simba.episodes.consolidate as ec
import simba.hooks.pre_compact as pc
from simba.sync.scheduler import SyncScheduler


def _only_episodes(cfg_obj):
    """Replacement for simba.config.load that returns cfg_obj for 'episodes'."""
    real = simba.config.load

    def fake(section):
        return cfg_obj if section == "episodes" else real(section)

    return fake


class TestPreCompactWiring:
    def test_disabled_is_noop(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            simba.config,
            "load",
            _only_episodes(ec_cfg.EpisodesConfig(auto_on_precompact=False)),
        )
        monkeypatch.setattr(ec, "consolidate_eligible", lambda *a, **k: calls.append(1))
        pc._maybe_consolidate_episodes("/proj")
        assert calls == []

    def test_enabled_calls_consolidate(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            simba.config,
            "load",
            _only_episodes(ec_cfg.EpisodesConfig(auto_on_precompact=True)),
        )
        monkeypatch.setattr(
            ec,
            "consolidate_eligible",
            lambda cwd, **k: calls.append(cwd) or {"dispatched": [], "skipped": 0},
        )
        pc._maybe_consolidate_episodes("/proj")
        assert calls == ["/proj"]


class TestSchedulerWiring:
    def test_disabled_is_noop(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            simba.config,
            "load",
            _only_episodes(ec_cfg.EpisodesConfig(scheduler_enabled=False)),
        )
        monkeypatch.setattr(ec, "consolidate_eligible", lambda *a, **k: calls.append(1))
        result = SyncScheduler(cwd="/proj")._maybe_consolidate()
        assert result == {"dispatched": [], "skipped": 0}
        assert calls == []

    def test_enabled_calls_consolidate(self, monkeypatch) -> None:
        monkeypatch.setattr(
            simba.config,
            "load",
            _only_episodes(ec_cfg.EpisodesConfig(scheduler_enabled=True)),
        )
        monkeypatch.setattr(
            ec,
            "consolidate_eligible",
            lambda cwd, **k: {"dispatched": ["s1"], "skipped": 0},
        )
        result = SyncScheduler(cwd="/proj")._maybe_consolidate()
        assert result["dispatched"] == ["s1"]

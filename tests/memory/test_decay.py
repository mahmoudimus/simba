"""Tests for the scheduler decay pass (src/simba/memory/decay.py)."""

from __future__ import annotations

import pathlib
import types

import simba.db
import simba.memory.usage as usage
from simba.memory.decay import run_decay_pass

_DAY = 86400.0
_NOW = 1_000_000_000.0


def _cfg(**kw):
    base = dict(
        decay_enabled=True,
        decay_half_life_days=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
        outcome_quality_weight=0.0,
        strength_dormancy_threshold=0.1,
        decay_capacity_per_type=0,
        arousal_decay_multiplier=1.0,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_decay_pass_reduces_strength_of_old_memory(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_old", now=_NOW - 60 * _DAY)
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg())
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_old"])["mem_old"]
    assert row.strength < 0.5
    assert result.processed == 1
    assert result.updated == 1


def test_decay_pass_sets_dormant_when_below_threshold(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_ancient", now=_NOW - 200 * _DAY)
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg())
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_ancient"])["mem_ancient"]
    assert result.newly_dormant == 1
    assert row.dormant is True


def test_decay_pass_revives_when_strength_recovers(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_revive", now=_NOW - 60 * _DAY)
        usage.set_strength("mem_revive", 0.05)
        usage.set_dormant("mem_revive", dormant=True)
        # Simulate many past accesses → reinforcement pulls strength up.
        usage.MemoryUsage.update(access_count=20).where(
            usage.MemoryUsage.memory_id == "mem_revive"
        ).execute()
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg())
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_revive"])["mem_revive"]
    assert result.revived == 1
    assert row.dormant is False


def test_capacity_cap_requests_minimal_field_projection(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Root cause of the 2026-07-10 CPU/RSS incident: the capacity-cap join
    never reads anything but id/type/projectPath, so the request must carry
    that projection (and never ask for vectors) --- an unprojected /list over
    the whole corpus was pulling every 1024-dim embedding into Python for a
    join that only needed three scalar columns."""
    import httpx

    from simba.memory.decay import _apply_capacity_cap

    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"memories": []}

    def _fake_get(url, *, params=None, **kw):
        captured["params"] = params
        return _FakeResponse()

    monkeypatch.setattr(httpx, "get", _fake_get)

    cfg = _cfg(decay_capacity_per_type=5, port=8741)
    _apply_capacity_cap(cfg, tmp_path, dry_run=True)

    assert captured["params"] == {
        "limit": 100000,
        "fields": "id,type,projectPath",
    }


def test_decay_pass_skips_when_disabled(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_x", now=_NOW - 60 * _DAY)
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg(decay_enabled=False))
    assert result is None


def test_decay_pass_is_deterministic(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_d", now=_NOW - 45 * _DAY)
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg())
    with simba.db.connect(tmp_path):
        first = usage.get_many(["mem_d"])["mem_d"].strength
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg())
    with simba.db.connect(tmp_path):
        second = usage.get_many(["mem_d"])["mem_d"].strength
    assert first == second


def test_decay_pass_no_rows_is_noop(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        pass
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg())
    assert result.processed == 0
    assert result.errors == 0


def test_decay_pass_default_multiplier_matches_baseline(
    tmp_path: pathlib.Path,
) -> None:
    """A cfg without arousal_decay_multiplier behaves identically to mult=1.0."""
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_a", now=_NOW - 45 * _DAY)
        usage.get_or_create("mem_b", now=_NOW - 45 * _DAY)
    # cfg WITHOUT the field at all (getattr default path).
    cfg_missing = types.SimpleNamespace(
        decay_enabled=True,
        decay_half_life_days=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
        strength_dormancy_threshold=0.1,
        decay_capacity_per_type=0,
    )
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=cfg_missing)
    with simba.db.connect(tmp_path):
        missing_strength = usage.get_many(["mem_a"])["mem_a"].strength
    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg(arousal_decay_multiplier=1.0))
    with simba.db.connect(tmp_path):
        explicit_strength = usage.get_many(["mem_b"])["mem_b"].strength
    assert missing_strength == explicit_strength


def test_decay_pass_high_arousal_retains_longer(tmp_path: pathlib.Path) -> None:
    """mult < 1.0 keeps an old memory's strength higher than the baseline pass."""
    base_dir = tmp_path / "base"
    arousal_dir = tmp_path / "arousal"
    base_dir.mkdir()
    arousal_dir.mkdir()
    with simba.db.connect(base_dir):
        usage.get_or_create("mem_x", now=_NOW - 60 * _DAY)
    with simba.db.connect(arousal_dir):
        usage.get_or_create("mem_x", now=_NOW - 60 * _DAY)
    run_decay_pass(now=_NOW, cwd=base_dir, cfg=_cfg(arousal_decay_multiplier=1.0))
    run_decay_pass(now=_NOW, cwd=arousal_dir, cfg=_cfg(arousal_decay_multiplier=0.5))
    with simba.db.connect(base_dir):
        base_strength = usage.get_many(["mem_x"])["mem_x"].strength
    with simba.db.connect(arousal_dir):
        arousal_strength = usage.get_many(["mem_x"])["mem_x"].strength
    assert arousal_strength > base_strength


def test_decay_pass_outcome_quality_weight_lifts_used_memory(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_used", now=_NOW - 30 * _DAY)
        usage.get_or_create("mem_neutral", now=_NOW - 30 * _DAY)
        usage.bump_quality("mem_used", _NOW, use=3)

    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg(outcome_quality_weight=0.5))

    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_used", "mem_neutral"])
    assert rows["mem_used"].strength > rows["mem_neutral"].strength


def test_decay_pass_outcome_quality_weight_demotes_noisy_memory(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_noise", now=_NOW - 30 * _DAY)
        usage.get_or_create("mem_neutral", now=_NOW - 30 * _DAY)
        usage.bump_quality("mem_noise", _NOW, noise=3)

    run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg(outcome_quality_weight=0.5))

    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_noise", "mem_neutral"])
    assert rows["mem_noise"].strength < rows["mem_neutral"].strength


def test_decay_pass_dry_run_reports_without_persisting(
    tmp_path: pathlib.Path,
) -> None:
    """Shadow mode (spec 33): count would-be changes, write nothing."""
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_dry", now=_NOW - 200 * _DAY)
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg(), dry_run=True)
    assert result.dry_run is True
    assert result.processed == 1
    assert result.updated == 1
    assert result.newly_dormant == 1
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_dry"])["mem_dry"]
    assert row.strength == 1.0
    assert row.dormant is False


def test_parse_type_multipliers() -> None:
    from simba.memory.decay import parse_type_multipliers

    assert parse_type_multipliers("PATTERN:4,GOTCHA:1.5") == {
        "PATTERN": 4.0,
        "GOTCHA": 1.5,
    }
    assert parse_type_multipliers("") == {}
    assert parse_type_multipliers("garbage") == {}
    assert parse_type_multipliers("BAD:x,OK:2") == {"OK": 2.0}
    assert parse_type_multipliers(" pattern : 4 ") == {"PATTERN": 4.0}


def test_decay_type_multiplier_slows_configured_type(
    tmp_path: pathlib.Path,
) -> None:
    """Type-aware half-lives (spec 33 Phase 2): PATTERN at 4x outlives GOTCHA."""
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_pat", now=_NOW - 45 * _DAY)
        usage.get_or_create("mem_got", now=_NOW - 45 * _DAY)
    cfg = _cfg(decay_type_multipliers="PATTERN:4")
    run_decay_pass(
        now=_NOW,
        cwd=tmp_path,
        cfg=cfg,
        type_map={"mem_pat": "PATTERN", "mem_got": "GOTCHA"},
    )
    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_pat", "mem_got"])
    assert rows["mem_pat"].strength > rows["mem_got"].strength


def test_decay_type_multiplier_noop_without_type_map(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_a", now=_NOW - 45 * _DAY)
        usage.get_or_create("mem_b", now=_NOW - 45 * _DAY)
    run_decay_pass(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(decay_type_multipliers="PATTERN:4"),
        type_map=None,
    )
    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_a", "mem_b"])
    assert rows["mem_a"].strength == rows["mem_b"].strength


def test_decay_pass_dry_run_counts_would_be_revivals(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_rev", now=_NOW - 60 * _DAY)
        usage.set_strength("mem_rev", 0.05)
        usage.set_dormant("mem_rev", dormant=True)
        usage.MemoryUsage.update(access_count=20).where(
            usage.MemoryUsage.memory_id == "mem_rev"
        ).execute()
    result = run_decay_pass(now=_NOW, cwd=tmp_path, cfg=_cfg(), dry_run=True)
    assert result.revived == 1
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_rev"])["mem_rev"]
    assert row.dormant is True  # nothing persisted
    assert row.strength == 0.05

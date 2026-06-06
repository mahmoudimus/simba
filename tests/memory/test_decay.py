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
        strength_dormancy_threshold=0.1,
        decay_capacity_per_type=0,
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

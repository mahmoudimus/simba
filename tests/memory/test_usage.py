"""Tests for the memory_usage mutable sidecar store (src/simba/memory/usage.py)."""

from __future__ import annotations

import pathlib

import simba.db
import simba.memory.usage as usage


def test_get_or_create_creates_row(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        row = usage.get_or_create("mem_abc", now=1000.0)
        assert row.memory_id == "mem_abc"
        assert row.access_count == 0
        assert row.strength == 1.0
        assert row.dormant is False
        assert row.feedback_score == 0.0
        assert row.match_count == 0
        assert row.inject_count == 0
        assert row.use_count == 0
        assert row.noise_count == 0
        assert row.save_count == 0


def test_get_or_create_is_idempotent(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_abc", now=1000.0)
        usage.get_or_create("mem_abc", now=2000.0)
        count = usage.MemoryUsage.select().count()
        assert count == 1
        row = usage.get_many(["mem_abc"])["mem_abc"]
        assert row.created_at == 1000.0


def test_bump_access_increments(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.bump_access("mem_abc", now=100.0)
        usage.bump_access("mem_abc", now=200.0)
        row = usage.get_many(["mem_abc"])["mem_abc"]
        assert row.access_count == 2
        assert row.last_accessed == 200.0


def test_bump_access_upserts_missing_row(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.bump_access("mem_new", now=50.0)
        row = usage.get_many(["mem_new"])["mem_new"]
        assert row.access_count == 1


def test_bump_quality_counters(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_q", now=100.0, match=1, inject=1, save=1)
        usage.bump_quality("mem_q", now=200.0, use=1, noise=1)
        row = usage.get_many(["mem_q"])["mem_q"]
        assert row.match_count == 1
        assert row.inject_count == 1
        assert row.save_count == 1
        assert row.use_count == 1
        assert row.noise_count == 1


def test_set_dormant_true_false(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_x", now=0.0)
        usage.set_dormant("mem_x", dormant=True)
        assert usage.get_many(["mem_x"])["mem_x"].dormant is True
        usage.set_dormant("mem_x", dormant=False)
        assert usage.get_many(["mem_x"])["mem_x"].dormant is False


def test_apply_feedback_clamps(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_y", now=0.0)
        usage.apply_feedback("mem_y", 0.9, now=1.0)
        usage.apply_feedback("mem_y", 0.9, now=2.0)  # would exceed +1.0
        assert usage.get_many(["mem_y"])["mem_y"].feedback_score == 1.0

        usage.apply_feedback("mem_y", -3.0, now=3.0)
        assert usage.get_many(["mem_y"])["mem_y"].feedback_score == -1.0


def test_set_strength_clamps(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_z", now=0.0)
        usage.set_strength("mem_z", 1.5)  # above 1.0
        assert usage.get_many(["mem_z"])["mem_z"].strength == 1.0
        usage.set_strength("mem_z", -0.1)
        assert usage.get_many(["mem_z"])["mem_z"].strength == 0.0


def test_get_all_for_decay_excludes_dormant_by_default(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_live", now=0.0)
        usage.get_or_create("mem_sleep", now=0.0)
        usage.set_dormant("mem_sleep", dormant=True)
        assert len(usage.get_all_for_decay()) == 1
        assert len(usage.get_all_for_decay(include_dormant=True)) == 2

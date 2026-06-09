"""Tests for the append-only memory_conflicts store (memory/conflict_store.py).

Mirrors the simba.db.connect() fixture style of test_usage.py — every helper runs
inside a ``simba.db.connect(tmp_path)`` context against a throwaway repo DB.
"""

from __future__ import annotations

import pathlib

import simba.db
import simba.memory.conflict_store as cstore

PROJ = "proj_a"


def test_record_and_read_round_trip(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        cstore.record_conflict(
            "mem_a", "mem_b", "two cities", project_path=PROJ, now=1000.0
        )
        rows = cstore.conflicts_among(["mem_a", "mem_b"], project_path=PROJ)
        assert len(rows) == 1
        row = rows[0]
        assert {row.memory_a, row.memory_b} == {"mem_a", "mem_b"}
        assert row.description == "two cities"
        assert row.detected_at == 1000.0
        assert row.project_path == PROJ


def test_pair_order_is_normalized(tmp_path: pathlib.Path) -> None:
    # (a, b) and (b, a) are the same conflict; stored as min/max so both query.
    with simba.db.connect(tmp_path):
        cstore.record_conflict("mem_z", "mem_a", "desc", project_path=PROJ, now=1.0)
        row = cstore.conflicts_among(["mem_a", "mem_z"], project_path=PROJ)[0]
        assert row.memory_a == "mem_a"  # min
        assert row.memory_b == "mem_z"  # max
        # Querying with the ids in either order still finds it.
        assert len(cstore.conflicts_among(["mem_z", "mem_a"], project_path=PROJ)) == 1


def test_idempotent_duplicate_pair(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        cstore.record_conflict("a", "b", "first", project_path=PROJ, now=1.0)
        # Same normalized pair (reversed order) → skipped, no second row.
        cstore.record_conflict("b", "a", "second", project_path=PROJ, now=2.0)
        rows = cstore.conflicts_among(["a", "b"], project_path=PROJ)
        assert len(rows) == 1
        assert rows[0].description == "first"  # first write wins (append-only)
        assert rows[0].detected_at == 1.0


def test_idempotent_is_scoped_per_project(tmp_path: pathlib.Path) -> None:
    # The same pair in a DIFFERENT project is a distinct conflict.
    with simba.db.connect(tmp_path):
        cstore.record_conflict("a", "b", "p1", project_path="p1", now=1.0)
        cstore.record_conflict("a", "b", "p2", project_path="p2", now=2.0)
        assert len(cstore.conflicts_among(["a", "b"], project_path="p1")) == 1
        assert len(cstore.conflicts_among(["a", "b"], project_path="p2")) == 1


def test_conflicts_among_requires_both_ids_in_set(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        cstore.record_conflict("a", "b", "ab", project_path=PROJ, now=1.0)
        cstore.record_conflict("c", "d", "cd", project_path=PROJ, now=2.0)
        # Only "a" present (not "b") → the a/b conflict is NOT returned.
        assert cstore.conflicts_among(["a", "c"], project_path=PROJ) == []
        # Both of a pair present → returned.
        rows = cstore.conflicts_among(["a", "b", "c"], project_path=PROJ)
        assert len(rows) == 1
        assert {rows[0].memory_a, rows[0].memory_b} == {"a", "b"}


def test_conflicts_among_scoped_by_project(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        cstore.record_conflict("a", "b", "ab", project_path=PROJ, now=1.0)
        # Querying a different project returns nothing.
        assert cstore.conflicts_among(["a", "b"], project_path="other") == []


def test_conflicts_among_fail_open_on_empty(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        assert cstore.conflicts_among([], project_path=PROJ) == []


def test_append_only_distinct_pairs_accumulate(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        cstore.record_conflict("a", "b", "ab", project_path=PROJ, now=1.0)
        cstore.record_conflict("a", "c", "ac", project_path=PROJ, now=2.0)
        cstore.record_conflict("b", "c", "bc", project_path=PROJ, now=3.0)
        rows = cstore.conflicts_among(["a", "b", "c"], project_path=PROJ)
        assert len(rows) == 3

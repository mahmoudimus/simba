"""Tests for the resumable exactly-once projection (projection.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.workflow.projection as projection
import simba.workflow.store as store


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_run_processes_all_and_advances_checkpoint():
    seen: list = []
    proj = projection.Projection("p", seen.append)
    n = proj.run([(1, "a"), (2, "b"), (3, "c")], now="2026-01-01T00:00:00Z")
    assert n == 3
    assert seen == ["a", "b", "c"]
    with simba.db.connect():
        assert store.WfCheckpoint.get(store.WfCheckpoint.name == "p").position == 3


def test_rerun_resumes_past_checkpoint():
    seen: list = []
    proj = projection.Projection("p", seen.append)
    proj.run([(1, "a"), (2, "b")], now="2026-01-01T00:00:00Z")
    seen.clear()
    n = proj.run([(2, "b"), (3, "c"), (4, "d")], now="2026-01-01T00:00:01Z")
    assert n == 2  # only positions 3 and 4 are new
    assert seen == ["c", "d"]
    with simba.db.connect():
        assert store.WfCheckpoint.get(store.WfCheckpoint.name == "p").position == 4


def test_event_at_or_below_checkpoint_skipped():
    seen: list = []
    proj = projection.Projection("p", seen.append)
    proj.run([(5, "a")], now="2026-01-01T00:00:00Z")
    seen.clear()
    n = proj.run([(3, "old"), (5, "same")], now="2026-01-01T00:00:01Z")
    assert n == 0
    assert seen == []


def test_distinct_projections_have_distinct_cursors():
    a_seen: list = []
    b_seen: list = []
    projection.Projection("a", a_seen.append).run(
        [(1, "x")], now="2026-01-01T00:00:00Z"
    )
    projection.Projection("b", b_seen.append).run(
        [(1, "y"), (2, "z")], now="2026-01-01T00:00:00Z"
    )
    assert a_seen == ["x"]
    assert b_seen == ["y", "z"]


def test_rebuild_resets_and_replays():
    derived: list = []
    reset_calls: list = []

    proj = projection.Projection("p", derived.append)
    proj.run([(1, "a"), (2, "b")], now="2026-01-01T00:00:00Z")
    assert derived == ["a", "b"]

    def reset_fn():
        reset_calls.append(True)
        derived.clear()

    n = proj.rebuild(
        [(1, "a"), (2, "b"), (3, "c")],
        reset_fn=reset_fn,
        now="2026-01-01T00:00:02Z",
    )
    assert reset_calls == [True]
    assert n == 3
    assert derived == ["a", "b", "c"]  # replayed from zero
    with simba.db.connect():
        assert store.WfCheckpoint.get(store.WfCheckpoint.name == "p").position == 3


def test_rebuild_without_reset_fn():
    derived: list = []
    proj = projection.Projection("p", derived.append)
    proj.run([(1, "a")], now="2026-01-01T00:00:00Z")
    n = proj.rebuild([(1, "a"), (2, "b")], now="2026-01-01T00:00:01Z")
    assert n == 2

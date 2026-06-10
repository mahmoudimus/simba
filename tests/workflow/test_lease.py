"""Tests for the lease primitive (lease.py).

A lease is acquire-key-at-most-once + release-by-key over the shared
``WfTask`` table, with optional expiry-based stale-reclaim. Time-dependent ops
always receive an injected ``now`` (ISO string); no test calls the wall clock.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.workflow.lease as lease
import simba.workflow.store as store

T0 = "2026-01-01T00:00:00Z"
T_LATER = "2026-01-01T02:00:00Z"  # 2h after T0


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _row(queue: str, key: str) -> store.WfTask | None:
    with simba.db.connect():
        return store.WfTask.get_or_none(
            (store.WfTask.queue == queue) & (store.WfTask.dedup_key == key)
        )


# ── acquire ──────────────────────────────────────────────────────────────


def test_acquire_absent_creates_running_and_returns_true():
    assert lease.acquire("q", "k", now=T0) is True
    row = _row("q", "k")
    assert row is not None
    assert row.status == "running"
    assert row.started_at == T0


def test_acquire_twice_second_is_false():
    assert lease.acquire("q", "k", now=T0) is True
    assert lease.acquire("q", "k", now=T0) is False  # held by a live worker


def test_acquire_after_release_is_false_durable_dedup():
    lease.acquire("q", "k", now=T0)
    lease.release("q", "k", now=T0)
    # done is never reclaimed, even with a stale window set.
    assert lease.acquire("q", "k", stale_after_seconds=0, now=T_LATER) is False


def test_acquire_none_stale_never_reclaims():
    lease.acquire("q", "k", now=T0)
    # No stale window => a running lock blocks forever.
    assert lease.acquire("q", "k", now=T_LATER) is False


def test_acquire_stale_running_is_reclaimed():
    lease.acquire("q", "k", now=T0)
    # started_at (T0) older than now (T_LATER) - 3600s => reclaim.
    assert lease.acquire("q", "k", stale_after_seconds=3600, now=T_LATER) is True
    row = _row("q", "k")
    assert row.status == "running"
    assert row.started_at == T_LATER  # re-stamped on reclaim


def test_acquire_fresh_running_not_reclaimed():
    lease.acquire("q", "k", now=T_LATER)
    # A running lock younger than the window is not reclaimed.
    assert lease.acquire("q", "k", stale_after_seconds=3600, now=T_LATER) is False


def test_release_stores_result():
    lease.acquire("q", "k", now=T0)
    lease.release("q", "k", result={"n": 7}, now=T0)
    row = _row("q", "k")
    assert row.status == "done"
    assert row.finished_at == T0
    import json

    assert json.loads(row.result) == {"n": 7}


def test_release_absent_key_is_noop():
    # No row for the key => no error, nothing created.
    lease.release("q", "absent", now=T0)
    assert _row("q", "absent") is None


def test_two_queues_same_key_are_independent():
    assert lease.acquire("q1", "k", now=T0) is True
    assert lease.acquire("q2", "k", now=T0) is True  # different queue
    assert lease.acquire("q1", "k", now=T0) is False


def test_acquire_stores_payload():
    import json

    lease.acquire("q", "k", payload={"engine": "claude-cli"}, now=T0)
    row = _row("q", "k")
    assert json.loads(row.payload) == {"engine": "claude-cli"}

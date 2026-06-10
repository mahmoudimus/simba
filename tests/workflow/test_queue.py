"""Tests for the durable task queue (queue.py).

Time-dependent ops always receive an injected ``now`` (ISO string); no test
calls the wall clock.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.workflow.queue as q
import simba.workflow.store as store

T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-01T00:00:01Z"
T_FUTURE = "2026-01-01T01:00:00Z"


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _count(queue: str) -> int:
    with simba.db.connect():
        return store.WfTask.select().where(store.WfTask.queue == queue).count()


# ── enqueue ────────────────────────────────────────────────────────────────


def test_enqueue_returns_id_and_inserts():
    tid = q.enqueue("q", {"a": 1}, now=T0)
    assert isinstance(tid, int)
    assert _count("q") == 1


def test_enqueue_dedup_idempotent():
    first = q.enqueue("q", {"a": 1}, dedup_key="k", now=T0)
    second = q.enqueue("q", {"a": 2}, dedup_key="k", now=T0)
    assert first == second  # same row id returned, no second insert
    assert _count("q") == 1


def test_enqueue_without_key_always_inserts():
    q.enqueue("q", {"a": 1}, now=T0)
    q.enqueue("q", {"a": 1}, now=T0)
    assert _count("q") == 2


def test_enqueue_delay_sets_future_available_at():
    tid = q.enqueue("q", {}, delay_seconds=3600, now=T0)
    with simba.db.connect():
        row = store.WfTask.get_by_id(tid)
    assert row.available_at == T_FUTURE


def test_enqueue_default_max_attempts_from_config():
    tid = q.enqueue("q", {}, now=T0)
    with simba.db.connect():
        row = store.WfTask.get_by_id(tid)
    assert row.max_attempts == 3  # workflow.default_max_attempts


# ── claim ────────────────────────────────────────────────────────────────


def test_claim_returns_task_and_flips_running():
    tid = q.enqueue("q", {"a": 1}, now=T0)
    claimed = q.claim("q", now=T1)
    assert claimed is not None
    assert claimed["id"] == tid
    assert claimed["status"] == "running"
    assert claimed["started_at"] == T1
    assert claimed["payload"] == {"a": 1}


def test_second_claim_is_none_exactly_once():
    q.enqueue("q", {}, now=T0)
    assert q.claim("q", now=T1) is not None
    assert q.claim("q", now=T1) is None  # nothing pending left


def test_claim_skips_future_available_at():
    q.enqueue("q", {}, delay_seconds=3600, now=T0)
    assert q.claim("q", now=T1) is None  # available_at is in the future
    assert q.claim("q", now=T_FUTURE) is not None  # now reached


def test_claim_oldest_first():
    first = q.enqueue("q", {"n": 1}, now=T0)
    q.enqueue("q", {"n": 2}, now=T1)
    claimed = q.claim("q", now=T_FUTURE)
    assert claimed["id"] == first


def test_claim_scopes_to_queue():
    q.enqueue("other", {}, now=T0)
    assert q.claim("q", now=T1) is None


# ── complete ────────────────────────────────────────────────────────────────


def test_complete_marks_done():
    tid = q.enqueue("q", {}, now=T0)
    q.claim("q", now=T1)
    q.complete(tid, result={"ok": True}, now=T1)
    with simba.db.connect():
        row = store.WfTask.get_by_id(tid)
    assert row.status == "done"
    assert row.finished_at == T1
    assert row.result == '{"ok": true}'


# ── fail / retry / dead ──────────────────────────────────────────────────────


def test_fail_retries_with_backoff_then_dead():
    tid = q.enqueue("q", {}, max_attempts=2, now=T0)
    q.claim("q", now=T0)
    # attempt 1: attempts 0 -> 1 < 2 -> pending, future available_at
    status = q.fail(tid, error="boom", now=T0)
    assert status == "pending"
    with simba.db.connect():
        row = store.WfTask.get_by_id(tid)
    assert row.attempts == 1
    assert row.available_at > T0  # backoff pushed it into the future
    assert row.error == "boom"
    # attempt 2: attempts 1 -> 2 == 2 -> dead
    status = q.fail(tid, error="boom2", now=T0)
    assert status == "dead"
    with simba.db.connect():
        row = store.WfTask.get_by_id(tid)
    assert row.attempts == 2
    assert row.finished_at == T0


def test_fail_backoff_is_exponential_and_capped():
    # base=2.0, cap=300.0: attempt1 -> +2s, attempt2 -> +4s
    tid = q.enqueue("q", {}, max_attempts=5, now=T0)
    q.fail(tid, error="e", now=T0)
    with simba.db.connect():
        a1 = store.WfTask.get_by_id(tid).available_at
    q.fail(tid, error="e", now=T0)
    with simba.db.connect():
        a2 = store.WfTask.get_by_id(tid).available_at
    assert a1 == "2026-01-01T00:00:02Z"  # 2^1 = 2s
    assert a2 == "2026-01-01T00:00:04Z"  # 2^2 = 4s


# ── reclaim_stale ────────────────────────────────────────────────────────────


def test_reclaim_stale_reclaims_old_running_leaves_fresh():
    old = q.enqueue("q", {"n": "old"}, now=T0)
    fresh = q.enqueue("q", {"n": "fresh"}, now=T0)
    # both running, started long ago vs just now
    q.claim("q", now="2026-01-01T00:00:00Z")  # claims old (oldest first)
    q.claim("q", now="2026-01-01T02:00:00Z")  # claims fresh
    n = q.reclaim_stale("q", stale_after_seconds=3600, now="2026-01-01T02:00:30Z")
    assert n == 1
    with simba.db.connect():
        assert store.WfTask.get_by_id(old).status == "pending"
        assert store.WfTask.get_by_id(fresh).status == "running"


def test_reclaim_stale_uses_config_default(monkeypatch):
    tid = q.enqueue("q", {}, now=T0)
    q.claim("q", now=T0)
    # default stale_after_seconds = 3600; 2h later -> reclaimed
    n = q.reclaim_stale("q", now="2026-01-01T02:00:00Z")
    assert n == 1
    with simba.db.connect():
        assert store.WfTask.get_by_id(tid).status == "pending"

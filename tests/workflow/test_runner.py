"""Tests for the execution helpers (runner.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.workflow.queue as q
import simba.workflow.runner as runner
import simba.workflow.store as store

T0 = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


# ── run_sync ────────────────────────────────────────────────────────────────


def test_run_sync_invokes_handler_with_task():
    captured = {}

    def handler(task):
        captured["task"] = task
        return task["payload"]["x"] * 2

    result = runner.run_sync(handler, {"id": 1, "payload": {"x": 21}})
    assert result == 42
    assert captured["task"]["id"] == 1


# ── dispatch_detached ────────────────────────────────────────────────────────


def test_dispatch_detached_popen(monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    runner.dispatch_detached(["uv", "run", "x"], cwd="/proj")

    assert captured["argv"] == ["uv", "run", "x"]
    assert captured["kw"]["cwd"] == "/proj"
    assert captured["kw"]["start_new_session"] is True
    import subprocess

    assert captured["kw"]["stdin"] == subprocess.DEVNULL
    assert captured["kw"]["stdout"] == subprocess.DEVNULL
    assert captured["kw"]["stderr"] == subprocess.DEVNULL


def test_dispatch_detached_passes_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(env=kw.get("env")) or object(),
    )
    runner.dispatch_detached(["x"], cwd="/p", env={"K": "V"})
    assert captured["env"] == {"K": "V"}


# ── fan_out ────────────────────────────────────────────────────────────────


def test_fan_out_returns_all_results():
    results = runner.fan_out([1, 2, 3, 4], lambda n: n * n, max_workers=2)
    assert results == [1, 4, 9, 16]  # order preserved


def test_fan_out_isolates_failing_item():
    def fn(n):
        if n == 2:
            raise ValueError("boom")
        return n * 10

    results = runner.fan_out([1, 2, 3], fn, max_workers=2)
    assert results == [10, None, 30]  # failure -> None, others survive


# ── worker_loop ──────────────────────────────────────────────────────────────


def test_worker_loop_drains_queue_and_completes():
    q.enqueue("q", {"x": 1}, now=T0)
    q.enqueue("q", {"x": 2}, now=T0)
    handled = []

    def handler(task):
        handled.append(task["payload"]["x"])
        return "ok"

    processed = runner.worker_loop("q", handler, now=T0)
    assert processed == 2
    assert sorted(handled) == [1, 2]
    with simba.db.connect():
        done = store.WfTask.select().where(store.WfTask.status == "done").count()
    assert done == 2


def test_worker_loop_routes_raising_handler_to_fail():
    tid = q.enqueue("q", {}, max_attempts=1, now=T0)

    def handler(task):
        raise RuntimeError("kaboom")

    processed = runner.worker_loop("q", handler, now=T0)
    assert processed == 1
    with simba.db.connect():
        row = store.WfTask.get_by_id(tid)
    assert row.status == "dead"  # attempts (1) == max_attempts (1)
    assert "kaboom" in row.error


def test_worker_loop_respects_max_tasks():
    q.enqueue("q", {}, now=T0)
    q.enqueue("q", {}, now=T0)
    q.enqueue("q", {}, now=T0)
    processed = runner.worker_loop("q", lambda t: None, max_tasks=2, now=T0)
    assert processed == 2
    with simba.db.connect():
        pending = store.WfTask.select().where(
            store.WfTask.status == "pending"
        ).count()
    assert pending == 1


def test_worker_loop_empty_queue_returns_zero():
    assert runner.worker_loop("q", lambda t: None, now=T0) == 0

"""Tests for shutdown-aware daemon lifecycle (handoff item 10).

Live 2026-07-08: SIGTERM breached uvicorn's graceful window ("Cancel 54
running task(s), timeout graceful shutdown exceeded" + CancelledError
tracebacks from in-flight ASGI stacks) because (a) fire-and-forget
``asyncio.create_task`` call sites in routes.py were never tracked or
awaited at shutdown, and (b) background passes self-HTTP the daemon's own
endpoints, which can never complete once uvicorn stops serving.

These tests simulate shutdown by invoking the extracted ``_shutdown_daemon``
helper (and, for the end-to-end case, the real ``lifespan`` context manager)
directly --- no real uvicorn server is spawned, per the acceptance criteria.
"""

from __future__ import annotations

import asyncio
import sys
import time

import fastapi
import pytest

import simba.memory.background as background
import simba.memory.config
import simba.memory.server as server


def _make_app(**config_kwargs) -> fastapi.FastAPI:
    app = fastapi.FastAPI()
    app.state.config = simba.memory.config.MemoryConfig(**config_kwargs)
    app.state.sync_scheduler = None
    app.state.maintenance_scheduler = None
    return app


class _FakeScheduler:
    """Minimal stand-in for SyncScheduler/MaintenanceScheduler.

    Real scheduler semantics (startup delay, interval waits, stop-event
    races) are already covered by test_maintenance.py; this file only needs
    something whose ``stop()`` can unblock (or, for ``obeys_stop=False``,
    deliberately NOT unblock) a long-running task, to exercise
    ``_shutdown_daemon``'s own stop/await/force-cancel sequencing.
    """

    def __init__(self, *, obeys_stop: bool = True) -> None:
        self._event = asyncio.Event()
        self.obeys_stop = obeys_stop
        self.stop_called = False

    def stop(self) -> None:
        self.stop_called = True
        if self.obeys_stop:
            self._event.set()

    async def run_forever(self) -> None:
        if self.obeys_stop:
            await self._event.wait()
        else:
            await asyncio.sleep(3600)  # ignores stop() --- forces the cancel path


class TestShutdownDaemon:
    @pytest.mark.asyncio
    async def test_marks_shutdown_flag(self) -> None:
        app = _make_app(shutdown_timeout=1.0)
        assert background.is_shutting_down() is False
        await server._shutdown_daemon(app, sync_task=None, maintenance_task=None)
        assert background.is_shutting_down() is True

    @pytest.mark.asyncio
    async def test_drains_tracked_background_tasks(self) -> None:
        app = _make_app(shutdown_timeout=1.0)
        completed = []

        async def _quick() -> None:
            await asyncio.sleep(0.01)
            completed.append(True)

        background.spawn(_quick())
        await server._shutdown_daemon(app, sync_task=None, maintenance_task=None)
        assert completed == [True]
        assert len(background.TASKS) == 0

    @pytest.mark.asyncio
    async def test_cancels_hung_background_task_without_hanging_shutdown(
        self,
    ) -> None:
        app = _make_app(shutdown_timeout=0.1)
        task = background.spawn(asyncio.sleep(3600))
        t0 = time.monotonic()
        await server._shutdown_daemon(app, sync_task=None, maintenance_task=None)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0
        assert task.done()

    @pytest.mark.asyncio
    async def test_stops_sync_scheduler_cleanly(self) -> None:
        app = _make_app(shutdown_timeout=1.0)
        fake = _FakeScheduler(obeys_stop=True)
        app.state.sync_scheduler = fake
        task = asyncio.create_task(fake.run_forever())
        await asyncio.sleep(0.01)

        await server._shutdown_daemon(app, sync_task=task, maintenance_task=None)
        assert fake.stop_called is True
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_stops_maintenance_scheduler_cleanly(self) -> None:
        app = _make_app(shutdown_timeout=1.0)
        fake = _FakeScheduler(obeys_stop=True)
        app.state.maintenance_scheduler = fake
        task = asyncio.create_task(fake.run_forever())
        await asyncio.sleep(0.01)

        await server._shutdown_daemon(app, sync_task=None, maintenance_task=task)
        assert fake.stop_called is True
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_stops_rss_watchdog_cleanly(self) -> None:
        """The RSS watchdog task (handoff: memory watchdog + transient-alloc
        bounds) must be stopped the same way sync/maintenance are ---
        ``rss_watchdog_task`` is an optional trailing kwarg so every
        pre-existing 2-kwarg call site above keeps working unchanged."""
        app = _make_app(shutdown_timeout=1.0)
        fake = _FakeScheduler(obeys_stop=True)
        app.state.rss_watchdog = fake
        task = asyncio.create_task(fake.run_forever())
        await asyncio.sleep(0.01)

        await server._shutdown_daemon(
            app, sync_task=None, maintenance_task=None, rss_watchdog_task=task
        )
        assert fake.stop_called is True
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_force_cancels_rss_watchdog_task_when_it_ignores_stop(self) -> None:
        app = _make_app(shutdown_timeout=0.2)  # sync_timeout floors at 1s
        fake = _FakeScheduler(obeys_stop=False)
        app.state.rss_watchdog = fake
        task = asyncio.create_task(fake.run_forever())
        await asyncio.sleep(0.01)

        t0 = time.monotonic()
        await server._shutdown_daemon(
            app, sync_task=None, maintenance_task=None, rss_watchdog_task=task
        )
        elapsed = time.monotonic() - t0

        assert fake.stop_called is True
        assert task.done()
        assert task.cancelled()
        assert elapsed < 5.0  # bounded, not hung

    @pytest.mark.asyncio
    async def test_force_cancels_maintenance_task_when_scheduler_ignores_stop(
        self,
    ) -> None:
        """Pre-existing fallback (unchanged by this refactor): a scheduler
        task that doesn't honor stop() within the budget gets force-cancelled
        rather than hanging shutdown forever."""
        app = _make_app(shutdown_timeout=0.2)  # sync_timeout floors at 1s (max(1, ...))
        fake = _FakeScheduler(obeys_stop=False)
        app.state.maintenance_scheduler = fake
        task = asyncio.create_task(fake.run_forever())
        await asyncio.sleep(0.01)

        t0 = time.monotonic()
        await server._shutdown_daemon(app, sync_task=None, maintenance_task=task)
        elapsed = time.monotonic() - t0

        assert fake.stop_called is True
        assert task.done()
        assert task.cancelled()
        assert elapsed < 5.0  # bounded, not hung

    @pytest.mark.asyncio
    async def test_marks_flag_before_stopping_schedulers(self) -> None:
        """The shutdown flag must be set at the START of shutdown --- before
        scheduler.stop() --- so a pass already mid-flight sees it on its very
        next self-HTTP attempt (root cause (b))."""
        app = _make_app(shutdown_timeout=1.0)
        observed_flag_in_stop: list[bool] = []

        class _ObservingScheduler(_FakeScheduler):
            def stop(self) -> None:
                observed_flag_in_stop.append(background.is_shutting_down())
                super().stop()

        fake = _ObservingScheduler(obeys_stop=True)
        app.state.sync_scheduler = fake
        task = asyncio.create_task(fake.run_forever())
        await asyncio.sleep(0.01)

        await server._shutdown_daemon(app, sync_task=task, maintenance_task=None)
        assert observed_flag_in_stop == [True]


@pytest.mark.asyncio
async def test_lifespan_shutdown_marks_flag_and_drains(monkeypatch) -> None:
    """End-to-end: enter/exit the real ``lifespan`` context manager (the
    actual FastAPI shutdown hook), with startup internals stubbed so no real
    DB/embedding model is touched --- only shutdown sequencing is under test,
    per the acceptance criterion to simulate shutdown via the lifespan
    handler directly (no real uvicorn server)."""

    async def _fake_init_database(app) -> None:
        return None

    async def _fake_init_embeddings(app):
        return None

    async def _fake_shutdown_embeddings(app) -> None:
        return None

    monkeypatch.setattr(server, "init_database", _fake_init_database)
    monkeypatch.setattr(server, "init_embeddings", _fake_init_embeddings)
    monkeypatch.setattr(server, "shutdown_embeddings", _fake_shutdown_embeddings)

    cfg = simba.memory.config.MemoryConfig(
        shutdown_timeout=0.2,
        sync_interval=0,  # keep the real schedulers gated off
        maintenance_interval_hours=0,
    )
    app = server.create_app(cfg, use_lifespan=True)

    completed = []

    async def _quick() -> None:
        await asyncio.sleep(0.01)
        completed.append(True)

    assert background.is_shutting_down() is False
    async with server.lifespan(app):
        background.spawn(_quick())
        assert background.is_shutting_down() is False

    assert background.is_shutting_down() is True
    assert completed == [True]
    assert len(background.TASKS) == 0


def test_main_passes_shutdown_timeout_to_uvicorn(monkeypatch) -> None:
    """One knob (memory.shutdown_timeout) governs both uvicorn's own
    graceful window and drain()'s budget --- verify main() wires it through
    to uvicorn.run's timeout_graceful_shutdown."""
    import uvicorn

    captured: dict = {}

    def _fake_run(app, **kwargs) -> None:
        captured.update(kwargs)

    fixed_config = simba.memory.config.MemoryConfig(port=18741, shutdown_timeout=7.5)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(
        simba.memory.config, "load_config", lambda **overrides: fixed_config
    )
    monkeypatch.setattr(sys, "argv", ["simba-memory-daemon"])
    # main() now funnels a clean uvicorn.run() return through the hard-exit
    # guarantee (server._run_server); stub the os._exit seam so this test
    # can make its assertions instead of ending the test process.
    monkeypatch.setattr(server, "_os_exit", lambda code: None)

    server.main()

    assert captured["timeout_graceful_shutdown"] == 7.5
    assert captured["port"] == 18741

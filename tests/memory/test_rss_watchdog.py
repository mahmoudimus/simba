"""Tests for the RSS watchdog (src/simba/memory/rss_watchdog.py).

The daemon floor is ~1.5-3.6GB (bge-large GGUF embedder + cross-encoder
reranker in-process, LanceDB, Python); a 45GB leak incident (fixed in PRs
#90-92) motivated a hard cap. macOS has no enforceable RSS rlimit
(RLIMIT_RSS is a no-op; RLIMIT_AS would abort LanceDB's mmap-based reads),
so the cap is this self-watchdog: poll RSS, relieve allocator pressure past
a soft limit, self-restart (reusing the existing os.execv exec seam) past a
hard limit once the process is old enough to rule out a startup transient.

``RssWatchdog`` takes its RSS reader and restart trigger as injectable
constructor seams, so these tests drive ``check_once()`` directly with
fakes --- no real process RSS, no real ``os.execv``, no real event-loop
timing beyond ``run_forever``'s own stop-cleanly test.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import simba.memory.rss_watchdog as rss_watchdog

_LOGGER_NAME = "simba.memory"


def _make_watchdog(
    *,
    soft_limit_mb: float = 0,
    hard_limit_mb: float = 0,
    min_uptime_seconds: float = 300.0,
    start_time: float | None = None,
    boot_argv: list[str] | None = None,
    rss_values: list[float | None],
    restart_calls: list[list[str]] | None = None,
    restart_error: Exception | None = None,
) -> tuple[rss_watchdog.RssWatchdog, list[list[str]]]:
    """Build a watchdog whose reader replays ``rss_values`` and whose
    restart trigger records its argv into a list the test can assert on."""
    calls: list[list[str]] = restart_calls if restart_calls is not None else []
    reader_values = list(rss_values)

    def _reader() -> float | None:
        return reader_values.pop(0) if reader_values else None

    async def _restart(argv: list[str]) -> None:
        if restart_error is not None:
            raise restart_error
        calls.append(argv)

    watchdog = rss_watchdog.RssWatchdog(
        soft_limit_mb=soft_limit_mb,
        hard_limit_mb=hard_limit_mb,
        interval_seconds=0.01,
        min_uptime_seconds=min_uptime_seconds,
        start_time=start_time if start_time is not None else time.time(),
        boot_argv=boot_argv,
        restart=_restart,
        rss_reader=_reader,
    )
    return watchdog, calls


class TestCurrentRssMb:
    def test_returns_positive_value_for_self(self) -> None:
        # No mocking: exercises the real strategy chain end-to-end on
        # whatever platform CI runs on. The `ps` fallback alone guarantees
        # this succeeds on any POSIX box even if the native reader fails.
        value = rss_watchdog.current_rss_mb()
        assert value is not None
        assert value > 0

    def test_accepts_explicit_pid(self) -> None:
        import os

        value = rss_watchdog.current_rss_mb(os.getpid())
        assert value is not None
        assert value > 0

    def test_all_strategies_failing_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(
            rss_watchdog,
            "_STRATEGIES",
            (lambda pid: None, lambda pid: None),
        )
        assert rss_watchdog.current_rss_mb() is None

    def test_strategy_exception_falls_through_to_next(self, monkeypatch) -> None:
        def _boom(pid: int) -> float | None:
            raise RuntimeError("nope")

        monkeypatch.setattr(rss_watchdog, "_STRATEGIES", (_boom, lambda pid: 42.0))
        assert rss_watchdog.current_rss_mb() == 42.0


class TestPeakRssMb:
    def test_returns_positive_value(self) -> None:
        value = rss_watchdog.peak_rss_mb()
        assert value is not None
        assert value > 0


class TestBelowSoftDoesNothing:
    @pytest.mark.asyncio
    async def test_no_relief_no_restart_no_log(self, monkeypatch, caplog) -> None:
        relief_calls = []
        monkeypatch.setattr(
            rss_watchdog, "_release_free_memory_to_os", lambda: relief_calls.append(1)
        )
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=100.0,
            hard_limit_mb=200.0,
            boot_argv=["python", "-m", "simba.memory.server"],
            rss_values=[50.0],
        )
        with caplog.at_level("WARNING", logger=_LOGGER_NAME):
            await watchdog.check_once()
        assert relief_calls == []
        assert restart_calls == []
        assert caplog.records == []
        assert watchdog.last_rss_mb == 50.0

    @pytest.mark.asyncio
    async def test_disabled_limits_are_always_a_noop_even_at_huge_rss(
        self, monkeypatch
    ) -> None:
        relief_calls = []
        monkeypatch.setattr(
            rss_watchdog, "_release_free_memory_to_os", lambda: relief_calls.append(1)
        )
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=0,
            hard_limit_mb=0,
            boot_argv=["python", "-m", "simba.memory.server"],
            rss_values=[999_999.0],
        )
        await watchdog.check_once()
        assert relief_calls == []
        assert restart_calls == []

    @pytest.mark.asyncio
    async def test_reader_returning_none_is_a_noop(self, monkeypatch) -> None:
        relief_calls = []
        monkeypatch.setattr(
            rss_watchdog, "_release_free_memory_to_os", lambda: relief_calls.append(1)
        )
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=1.0,
            hard_limit_mb=2.0,
            boot_argv=["python"],
            rss_values=[None],
        )
        await watchdog.check_once()
        assert relief_calls == []
        assert restart_calls == []
        assert watchdog.last_rss_mb is None


class TestOverSoft:
    @pytest.mark.asyncio
    async def test_relief_called_once_per_crossing_and_logged(
        self, monkeypatch, caplog
    ) -> None:
        relief_calls = []
        gc_calls = []
        monkeypatch.setattr(
            rss_watchdog, "_release_free_memory_to_os", lambda: relief_calls.append(1)
        )
        monkeypatch.setattr(rss_watchdog.gc, "collect", lambda: gc_calls.append(1))

        # Stays above soft for two consecutive checks, drops below, then
        # crosses again -- relief must fire on crossings 1 and 3 only.
        watchdog, _ = _make_watchdog(
            soft_limit_mb=100.0,
            hard_limit_mb=0,  # hard disabled: isolate the soft path
            rss_values=[150.0, 150.0, 50.0, 150.0],
        )
        with caplog.at_level("WARNING", logger=_LOGGER_NAME):
            await watchdog.check_once()  # crossing 1: fires
            await watchdog.check_once()  # still above: silent
            await watchdog.check_once()  # back below: resets
            await watchdog.check_once()  # crossing 2: fires

        assert len(relief_calls) == 2
        assert len(gc_calls) == 2
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 2
        assert "soft limit crossed" in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_soft_alone_never_calls_restart(self, monkeypatch) -> None:
        monkeypatch.setattr(rss_watchdog, "_release_free_memory_to_os", lambda: None)
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=100.0,
            hard_limit_mb=0,
            boot_argv=["python"],
            rss_values=[150.0],
        )
        await watchdog.check_once()
        assert restart_calls == []


class TestOverHard:
    @pytest.mark.asyncio
    async def test_young_process_does_not_restart(self, monkeypatch, caplog) -> None:
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=0,
            hard_limit_mb=100.0,
            min_uptime_seconds=300.0,
            start_time=time.time(),  # uptime ~0s < 300s
            boot_argv=["python", "-m", "simba.memory.server"],
            rss_values=[150.0],
        )
        with caplog.at_level("CRITICAL", logger=_LOGGER_NAME):
            await watchdog.check_once()
        assert restart_calls == []
        criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert len(criticals) == 1
        assert "younger than min_uptime" in criticals[0].getMessage()

    @pytest.mark.asyncio
    async def test_mature_process_fires_restart_seam(self, monkeypatch, caplog) -> None:
        argv = ["python", "-m", "simba.memory.server", "--port", "18741"]
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=0,
            hard_limit_mb=100.0,
            min_uptime_seconds=300.0,
            start_time=time.time() - 3600.0,  # uptime 1h >> min_uptime
            boot_argv=argv,
            rss_values=[150.0],
        )
        with caplog.at_level("CRITICAL", logger=_LOGGER_NAME):
            await watchdog.check_once()
        assert restart_calls == [argv]
        criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert len(criticals) == 1
        assert "triggering self-restart" in criticals[0].getMessage()

    @pytest.mark.asyncio
    async def test_no_boot_argv_is_critical_only_no_restart_no_crash(
        self, caplog
    ) -> None:
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=0,
            hard_limit_mb=100.0,
            min_uptime_seconds=0.0,  # would otherwise be mature enough to restart
            start_time=time.time() - 3600.0,
            boot_argv=None,
            rss_values=[150.0],
        )
        with caplog.at_level("CRITICAL", logger=_LOGGER_NAME):
            await watchdog.check_once()  # must not raise
        assert restart_calls == []
        criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert len(criticals) == 1
        assert "no boot argv on record" in criticals[0].getMessage()

    @pytest.mark.asyncio
    async def test_empty_boot_argv_list_is_treated_as_unavailable(self) -> None:
        watchdog, restart_calls = _make_watchdog(
            hard_limit_mb=100.0,
            min_uptime_seconds=0.0,
            start_time=time.time() - 3600.0,
            boot_argv=[],
            rss_values=[150.0],
        )
        await watchdog.check_once()
        assert restart_calls == []

    @pytest.mark.asyncio
    async def test_restart_exception_is_swallowed_and_logged(self, caplog) -> None:
        watchdog, restart_calls = _make_watchdog(
            hard_limit_mb=100.0,
            min_uptime_seconds=0.0,
            start_time=time.time() - 3600.0,
            boot_argv=["python"],
            rss_values=[150.0],
            restart_error=RuntimeError("exec seam exploded"),
        )
        with caplog.at_level("CRITICAL", logger=_LOGGER_NAME):
            await watchdog.check_once()  # must not raise
        assert restart_calls == []
        assert any("restart trigger failed" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_soft_and_hard_both_fire_when_rss_exceeds_both(
        self, monkeypatch
    ) -> None:
        relief_calls = []
        monkeypatch.setattr(
            rss_watchdog, "_release_free_memory_to_os", lambda: relief_calls.append(1)
        )
        watchdog, restart_calls = _make_watchdog(
            soft_limit_mb=10.0,
            hard_limit_mb=100.0,
            min_uptime_seconds=0.0,
            start_time=time.time() - 3600.0,
            boot_argv=["python"],
            rss_values=[150.0],
        )
        await watchdog.check_once()
        assert relief_calls == [1]
        assert restart_calls == [["python"]]


class TestRunForever:
    @pytest.mark.asyncio
    async def test_stops_cleanly(self) -> None:
        watchdog, _ = _make_watchdog(rss_values=[])
        watchdog.interval_seconds = 0.01
        task = asyncio.create_task(watchdog.run_forever())
        await asyncio.sleep(0.03)
        watchdog.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()
        assert not task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_before_first_wait_is_not_lost(self) -> None:
        """A stop() issued immediately after creation must still end the loop
        promptly, mirroring MaintenanceScheduler's same guarantee."""
        watchdog, _ = _make_watchdog(rss_values=[])
        watchdog.interval_seconds = 3600.0
        task = asyncio.create_task(watchdog.run_forever())
        await asyncio.sleep(0)  # let the task start and reach the first _wait
        watchdog.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert task.done()

    @pytest.mark.asyncio
    async def test_check_failure_does_not_kill_the_loop(self, monkeypatch) -> None:
        calls = []

        def _boom() -> float | None:
            calls.append(1)
            raise RuntimeError("reader exploded")

        watchdog = rss_watchdog.RssWatchdog(
            soft_limit_mb=1.0,
            hard_limit_mb=0,
            interval_seconds=0.01,
            min_uptime_seconds=300.0,
            start_time=time.time(),
            boot_argv=None,
            restart=lambda argv: asyncio.sleep(0),
            rss_reader=_boom,
        )
        task = asyncio.create_task(watchdog.run_forever())
        await asyncio.sleep(0.05)
        watchdog.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert len(calls) >= 1  # the loop kept calling check_once despite the raise


class TestServerWiring:
    """``_start_rss_watchdog`` — the app-lifespan glue (config-gated start,
    boot_argv/start_time wiring, and NOT registering the task in
    ``background.TASKS``). Mirrors test_maintenance.py's
    ``test_server_gates_scheduler_on_interval`` /
    ``test_server_starts_scheduler_when_enabled`` for the maintenance
    heartbeat starter."""

    @pytest.mark.asyncio
    async def test_gated_off_when_both_limits_zero(self, tmp_path) -> None:
        import simba.memory.config
        import simba.memory.server

        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(rss_soft_limit_mb=0, rss_hard_limit_mb=0)
        )
        app.state.cwd = tmp_path
        task = await simba.memory.server._start_rss_watchdog(app)
        assert task is None
        assert getattr(app.state, "rss_watchdog", None) is None

    @pytest.mark.asyncio
    async def test_starts_when_soft_limit_set(self, tmp_path) -> None:
        import simba.memory.config
        import simba.memory.server

        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(
                rss_soft_limit_mb=1.0, rss_check_interval_seconds=3600.0
            )
        )
        app.state.cwd = tmp_path
        task = await simba.memory.server._start_rss_watchdog(app)
        try:
            assert task is not None
            assert app.state.rss_watchdog is not None
        finally:
            app.state.rss_watchdog.stop()
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_starts_when_hard_limit_set(self, tmp_path) -> None:
        import simba.memory.config
        import simba.memory.server

        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(
                rss_hard_limit_mb=1.0, rss_check_interval_seconds=3600.0
            )
        )
        app.state.cwd = tmp_path
        task = await simba.memory.server._start_rss_watchdog(app)
        try:
            assert task is not None
        finally:
            app.state.rss_watchdog.stop()
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_task_is_not_registered_in_background_tasks(self, tmp_path) -> None:
        """Per the ground rules: the watchdog loop must survive
        ``background.drain()`` (it lives outside the request/response
        lifecycle), so it must never be added to ``background.TASKS`` ---
        mirrors how sync_task/maintenance_task are already excluded."""
        import simba.memory.background as background
        import simba.memory.config
        import simba.memory.server

        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(
                rss_soft_limit_mb=1.0, rss_check_interval_seconds=3600.0
            )
        )
        app.state.cwd = tmp_path
        task = await simba.memory.server._start_rss_watchdog(app)
        try:
            assert task not in background.TASKS
        finally:
            app.state.rss_watchdog.stop()
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_wires_boot_argv_and_start_time_from_app_state(
        self, tmp_path
    ) -> None:
        import simba.memory.config
        import simba.memory.server

        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(
                rss_soft_limit_mb=1.0, rss_check_interval_seconds=3600.0
            )
        )
        app.state.cwd = tmp_path
        app.state.boot_argv = ["python", "-m", "simba.memory.server"]
        app.state.start_time = 12345.0
        task = await simba.memory.server._start_rss_watchdog(app)
        try:
            watchdog = app.state.rss_watchdog
            assert watchdog.boot_argv == ["python", "-m", "simba.memory.server"]
            assert watchdog.start_time == 12345.0
        finally:
            watchdog.stop()
            await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_hard_breach_restart_reuses_run_restart_sequence(
        self, tmp_path, monkeypatch
    ) -> None:
        """The watchdog's restart seam must call the EXISTING
        ``routes._run_restart_sequence`` (reused, not reimplemented) ---
        verified end to end through the real glue by monkeypatching the
        exec seam it eventually reaches (never the real os.execv)."""
        import simba.memory.background as background
        import simba.memory.config
        import simba.memory.server

        calls: list[list[str]] = []
        monkeypatch.setattr(
            background, "_execv", lambda path, argv: calls.append(list(argv))
        )

        app = simba.memory.server.create_app(
            simba.memory.config.MemoryConfig(
                rss_hard_limit_mb=1.0,
                rss_check_interval_seconds=3600.0,
                rss_restart_min_uptime_seconds=0.0,
                shutdown_timeout=1.0,
            )
        )
        app.state.cwd = tmp_path
        app.state.boot_argv = ["python", "-m", "simba.memory.server"]
        app.state.start_time = time.time() - 3600.0

        task = await simba.memory.server._start_rss_watchdog(app)
        watchdog = app.state.rss_watchdog
        monkeypatch.setattr(watchdog, "_rss_reader", lambda: 999.0)
        try:
            await watchdog.check_once()
        finally:
            watchdog.stop()
            await asyncio.wait_for(task, timeout=2.0)

        assert calls == [app.state.boot_argv]

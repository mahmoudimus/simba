"""Tests for the shutdown-aware background-task registry (handoff item 10).

Live 2026-07-08 SIGTERM breach: uvicorn's graceful window was exceeded
("Cancel 54 running task(s)") because fire-and-forget ``asyncio.create_task``
call sites in routes.py were never tracked or awaited at shutdown. This
module (``simba.memory.background``) gives every fire-and-forget task a
registry entry (``spawn``) and a bounded drain at shutdown (``drain``), plus
a process-level shutdown flag so self-HTTP helpers can bail out immediately
once the daemon stops serving requests.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import simba.memory.background as background


class TestSpawn:
    @pytest.mark.asyncio
    async def test_spawn_returns_a_running_task(self) -> None:
        release = asyncio.Event()

        async def _work() -> None:
            await release.wait()

        task = background.spawn(_work())
        try:
            assert isinstance(task, asyncio.Task)
            assert not task.done()
        finally:
            release.set()
            await task

    @pytest.mark.asyncio
    async def test_spawn_tracks_task_in_registry_while_pending(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def _work() -> None:
            started.set()
            await release.wait()

        task = background.spawn(_work())
        await started.wait()
        assert task in background.TASKS
        release.set()
        await task

    @pytest.mark.asyncio
    async def test_completion_auto_discards_from_registry(self) -> None:
        task = background.spawn(asyncio.sleep(0))
        await task
        # The done-callback runs via call_soon; yield once so it lands.
        await asyncio.sleep(0)
        assert task not in background.TASKS

    @pytest.mark.asyncio
    async def test_exception_still_auto_discards_from_registry(self) -> None:
        async def _boom() -> None:
            raise RuntimeError("boom")

        task = background.spawn(_boom())
        with pytest.raises(RuntimeError):
            await task
        await asyncio.sleep(0)
        assert task not in background.TASKS

    @pytest.mark.asyncio
    async def test_registry_empties_after_several_tasks_complete(self) -> None:
        for _ in range(5):
            background.spawn(asyncio.sleep(0))
        await asyncio.sleep(0.05)
        assert len(background.TASKS) == 0


class TestDrain:
    @pytest.mark.asyncio
    async def test_drain_with_empty_registry_returns_immediately(self) -> None:
        t0 = time.monotonic()
        await background.drain(5.0)
        assert time.monotonic() - t0 < 0.5

    @pytest.mark.asyncio
    async def test_drain_lets_fast_tasks_finish_naturally(self) -> None:
        completed = []

        async def _fast() -> None:
            await asyncio.sleep(0.01)
            completed.append(True)

        background.spawn(_fast())
        await background.drain(2.0)
        assert completed == [True]
        assert len(background.TASKS) == 0

    @pytest.mark.asyncio
    async def test_drain_cancels_a_hung_task_after_grace(self) -> None:
        task = background.spawn(asyncio.sleep(3600))
        t0 = time.monotonic()
        await background.drain(0.05, tail_seconds=0.3)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0
        assert task.done()
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_drain_abandons_cancellation_swallowing_task_within_tail(
        self,
    ) -> None:
        """A task that swallows CancelledError must not hang drain forever.

        Concern (c) from the observed failure: asyncio.to_thread workers
        blocked on the process-global LLAMA lock are uncancellable.
        drain() must give up after the bounded tail rather than await such a
        task forever --- the process is exiting either way.
        """
        entered = asyncio.Event()

        async def _stubborn() -> None:
            entered.set()
            swallows = 0
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    swallows += 1
                    if swallows >= 10:
                        raise
                    continue

        task = background.spawn(_stubborn())
        await entered.wait()
        t0 = time.monotonic()
        await background.drain(0.05, tail_seconds=0.1)
        elapsed = time.monotonic() - t0

        # Bounded: drain gave up well short of the swallow loop's own patience.
        assert elapsed < 1.0
        assert not task.done()  # still running --- abandoned, not awaited forever

        # Cleanup: finish the job of killing it off so no task leaks past the test.
        for _ in range(10):
            task.cancel()
            await asyncio.sleep(0)
            if task.done():
                break
        if not task.done():
            task.cancel()

    @pytest.mark.asyncio
    async def test_drain_returns_within_grace_plus_tail_bound(self) -> None:
        """drain() must not exceed ~grace+tail even with multiple stragglers."""
        for _ in range(3):
            background.spawn(asyncio.sleep(3600))
        grace, tail = 0.05, 0.2
        t0 = time.monotonic()
        await background.drain(grace, tail_seconds=tail)
        elapsed = time.monotonic() - t0
        assert elapsed < grace + tail + 1.0  # generous slack for scheduling jitter

    @pytest.mark.asyncio
    async def test_drain_default_tail_is_bounded_for_small_grace(self) -> None:
        """Without an explicit tail, drain must still return quickly for tests
        that pass a tiny grace --- the auto-derived tail should scale down."""
        background.spawn(asyncio.sleep(3600))
        t0 = time.monotonic()
        await background.drain(0.05)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0


class TestShutdownFlag:
    def test_flag_starts_clear(self) -> None:
        assert background.is_shutting_down() is False

    def test_mark_shutting_down_sets_flag(self) -> None:
        background.mark_shutting_down()
        assert background.is_shutting_down() is True

    def test_flag_is_readable_from_a_worker_thread(self) -> None:
        """The self-HTTP helper guards run inside asyncio.to_thread workers,
        so the flag must be a plain thread-safe read, not event-loop-bound."""
        import concurrent.futures

        background.mark_shutting_down()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(background.is_shutting_down).result(timeout=2.0)
        assert result is True

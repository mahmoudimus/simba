"""Tests for the SyncScheduler (simba.sync.scheduler)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from simba.sync.extractor import ExtractResult
from simba.sync.indexer import IndexResult
from simba.sync.scheduler import SyncScheduler

_IDX = "simba.sync.scheduler.run_index"
_EXT = "simba.sync.scheduler.run_extract"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_index(**kwargs) -> IndexResult:
    defaults = dict(
        tables_polled=1, rows_indexed=2, rows_exported=1,
        duplicates=0, errors=0,
    )
    defaults.update(kwargs)
    return IndexResult(**defaults)


def _ok_extract(**kwargs) -> ExtractResult:
    defaults = dict(
        memories_processed=3, facts_extracted=1,
        facts_duplicate=0, agent_dispatched=False, errors=0,
    )
    defaults.update(kwargs)
    return ExtractResult(**defaults)


def _idx_side_effect(*a, **kw):
    return _ok_index()


def _ext_side_effect(*a, **kw):
    return _ok_extract()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunOnce:
    @pytest.mark.asyncio
    @patch(_EXT, side_effect=_ext_side_effect)
    @patch(_IDX, side_effect=_idx_side_effect)
    async def test_returns_summary(
        self, mock_index, mock_extract,
    ) -> None:
        scheduler = SyncScheduler(interval_seconds=1)
        summary = await scheduler.run_once()

        assert summary["cycle"] == 1
        assert summary["index"]["rows_indexed"] == 2
        assert summary["extract"]["facts_extracted"] == 1
        assert summary["total_errors"] == 0

    @pytest.mark.asyncio
    @patch(_EXT, side_effect=_ext_side_effect)
    @patch(_IDX, side_effect=_idx_side_effect)
    async def test_cycle_count_increments(
        self, mock_index, mock_extract,
    ) -> None:
        scheduler = SyncScheduler(interval_seconds=1)
        await scheduler.run_once()
        await scheduler.run_once()
        assert scheduler.cycle_count == 2

    @pytest.mark.asyncio
    @patch(_EXT, side_effect=lambda *a, **kw: _ok_extract(errors=1))
    @patch(_IDX, side_effect=lambda *a, **kw: _ok_index(errors=2))
    async def test_total_errors_aggregated(
        self, mock_index, mock_extract,
    ) -> None:
        scheduler = SyncScheduler(interval_seconds=1)
        summary = await scheduler.run_once()
        assert summary["total_errors"] == 3


class TestRunForever:
    @pytest.mark.asyncio
    @patch(_EXT, side_effect=_ext_side_effect)
    @patch(_IDX, side_effect=_idx_side_effect)
    async def test_stop_ends_run_forever(
        self, mock_index, mock_extract,
    ) -> None:
        scheduler = SyncScheduler(interval_seconds=600)

        async def _stop_soon():
            # Wait until at least one cycle completes, then stop
            while scheduler.cycle_count < 1:
                await asyncio.sleep(0.01)
            scheduler.stop()

        await asyncio.gather(
            scheduler.run_forever(),
            _stop_soon(),
        )
        assert scheduler.cycle_count >= 1
        assert scheduler.running is False

    @pytest.mark.asyncio
    @patch(_EXT, side_effect=RuntimeError("boom"))
    @patch(_IDX, side_effect=RuntimeError("boom"))
    async def test_run_forever_handles_errors(
        self, mock_index, mock_extract,
    ) -> None:
        """run_forever should not crash when run_once raises."""
        scheduler = SyncScheduler(interval_seconds=600)

        async def _stop_after_delay():
            # Give it time to attempt at least one cycle
            await asyncio.sleep(0.1)
            scheduler.stop()

        # Should not raise
        await asyncio.gather(
            scheduler.run_forever(),
            _stop_after_delay(),
        )
        assert scheduler.running is False

    @pytest.mark.asyncio
    @patch(_EXT, side_effect=_ext_side_effect)
    @patch(_IDX, side_effect=_idx_side_effect)
    async def test_running_property(
        self, mock_index, mock_extract,
    ) -> None:
        scheduler = SyncScheduler(interval_seconds=600)
        assert scheduler.running is False

        running_during: list[bool] = []

        async def _observe_and_stop():
            while scheduler.cycle_count < 1:
                await asyncio.sleep(0.01)
            running_during.append(scheduler.running)
            scheduler.stop()

        await asyncio.gather(
            scheduler.run_forever(),
            _observe_and_stop(),
        )
        assert running_during == [True]
        assert scheduler.running is False

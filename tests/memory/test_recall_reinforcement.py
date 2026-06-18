"""Tests for recall-time usage reinforcement (src/simba/memory/routes.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.memory.usage as usage
from simba.memory.routes import _bump_usage


@pytest.mark.asyncio
async def test_bump_usage_writes_to_sqlite(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        pass  # pre-init the DB / schema

    await _bump_usage(["mem_a", "mem_b"], now=1000.0, cwd=tmp_path)

    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_a", "mem_b"])
        assert "mem_a" in rows
        assert "mem_b" in rows
        assert rows["mem_a"].access_count == 1
        assert rows["mem_a"].last_accessed == 1000.0
        assert rows["mem_a"].match_count == 1
        assert rows["mem_a"].inject_count == 1


@pytest.mark.asyncio
async def test_bump_usage_increments_on_repeated_recall(
    tmp_path: pathlib.Path,
) -> None:
    await _bump_usage(["mem_x"], now=100.0, cwd=tmp_path)
    await _bump_usage(["mem_x"], now=200.0, cwd=tmp_path)

    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_x"])
        assert rows["mem_x"].access_count == 2
        assert rows["mem_x"].last_accessed == 200.0
        assert rows["mem_x"].match_count == 2
        assert rows["mem_x"].inject_count == 2


@pytest.mark.asyncio
async def test_bump_usage_handles_empty_list(tmp_path: pathlib.Path) -> None:
    await _bump_usage([], now=0.0, cwd=tmp_path)
    with simba.db.connect(tmp_path):
        assert usage.get_all_for_decay(include_dormant=True) == []

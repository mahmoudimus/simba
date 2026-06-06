"""Tests for the outcome-feedback route (src/simba/memory/routes.py)."""

from __future__ import annotations

import pathlib

import httpx
import pytest
import pytest_asyncio

import simba.db
import simba.memory.config
import simba.memory.server
import simba.memory.usage as usage


@pytest_asyncio.fixture
async def feedback_client(tmp_path: pathlib.Path):
    """Async client backed by a minimal app whose usage store is in tmp_path."""
    cfg = simba.memory.config.MemoryConfig(feedback_default_weight=0.3)
    app = simba.memory.server.create_app(cfg)
    app.state.cwd = tmp_path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, tmp_path


@pytest.mark.asyncio
async def test_feedback_good_increments_score(feedback_client) -> None:
    ac, _ = feedback_client
    resp = await ac.post("/memory/mem_abc/feedback", json={"signal": "good"})
    assert resp.status_code == 200
    assert resp.json()["feedback_score"] == 0.3


@pytest.mark.asyncio
async def test_feedback_bad_decrements_score(feedback_client) -> None:
    ac, _ = feedback_client
    resp = await ac.post("/memory/mem_abc/feedback", json={"signal": "bad"})
    assert resp.json()["feedback_score"] == -0.3


@pytest.mark.asyncio
async def test_feedback_repeated_good_clamps_at_one(feedback_client) -> None:
    ac, _ = feedback_client
    await ac.post("/memory/mem_x/feedback", json={"signal": "good", "weight": 0.8})
    resp = await ac.post(
        "/memory/mem_x/feedback", json={"signal": "good", "weight": 0.8}
    )
    assert resp.json()["feedback_score"] == 1.0


@pytest.mark.asyncio
async def test_feedback_invalid_signal_returns_400(feedback_client) -> None:
    ac, _ = feedback_client
    resp = await ac.post("/memory/mem_abc/feedback", json={"signal": "neutral"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_feedback_custom_weight(feedback_client) -> None:
    ac, _ = feedback_client
    resp = await ac.post(
        "/memory/mem_abc/feedback", json={"signal": "good", "weight": 0.1}
    )
    assert resp.json()["feedback_score"] == 0.1


@pytest.mark.asyncio
async def test_feedback_missing_usage_row_creates_it(feedback_client) -> None:
    ac, tmp_path = feedback_client
    resp = await ac.post("/memory/mem_new/feedback", json={"signal": "good"})
    assert resp.json()["feedback_score"] == 0.3
    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_new"])
    assert "mem_new" in rows
    assert rows["mem_new"].memory_id == "mem_new"

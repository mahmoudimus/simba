"""Daemon-side usage-signal tests (spec 33 Phase 0): match/inject split + ack.

``match`` = returned by search (daemon bumps at recall). ``inject`` = actually
placed into a hook's context — only the hook knows, so it acks via
``POST /recall/ack``. Before the split both were bumped on the same line,
making the ledger degenerate (match == inject always).
"""

from __future__ import annotations

import asyncio

import pytest

import simba.db
import simba.memory.usage as usage


def _make_memory(memory_id: str) -> dict:
    return {
        "id": memory_id,
        "type": "GOTCHA",
        "content": f"memory {memory_id}",
        "context": "",
        "tags": "[]",
        "confidence": 0.85,
        "sessionSource": "",
        "projectPath": "",
        "createdAt": "2025-01-01T00:00:00Z",
        "lastAccessedAt": "2025-01-01T00:00:00Z",
        "accessCount": 0,
        "vector": [0.1] * 768,
    }


async def _drain_background_tasks(timeout: float = 2.0) -> None:
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


@pytest.mark.asyncio
async def test_recall_bumps_match_and_access_but_not_inject(
    async_client, lance_table, tmp_path
) -> None:
    await lance_table.add([_make_memory("mem_m")])
    resp = await async_client.post("/recall", json={"query": "memory"})
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 1
    await _drain_background_tasks()

    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_m"])["mem_m"]
    assert row.access_count == 1
    assert row.match_count == 1
    assert row.inject_count == 0


@pytest.mark.asyncio
async def test_cached_recall_still_bumps_usage(
    tmp_path, lance_table, mock_embed
) -> None:
    """A cache-served recall is still a LOGICAL recall (measured live:
    inject=2 while match=1 — acks counted, the ledger didn't). The sidecar
    bump must fire on the cached path too; only the search is skipped."""
    import httpx

    import simba.memory.config
    import simba.memory.fts
    import simba.memory.recall_cache
    import simba.memory.server

    cfg = simba.memory.config.MemoryConfig(recall_cache_ttl_seconds=60.0)
    app = simba.memory.server.create_app(cfg)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.cwd = tmp_path
    fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
    simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
    app.state.fts_path = str(fts_path)
    app.state.recall_cache = simba.memory.recall_cache.RecallCache(
        max_entries=16, ttl_seconds=60.0
    )
    await lance_table.add([_make_memory("mem_cached")])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        first = await ac.post("/recall", json={"query": "memory"})
        assert first.status_code == 200
        second = await ac.post("/recall", json={"query": "memory"})
        assert second.json().get("cached") is True
    await _drain_background_tasks()

    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_cached"])["mem_cached"]
    assert row.access_count == 2
    assert row.match_count == 2


@pytest.mark.asyncio
async def test_recall_ack_bumps_inject(async_client, tmp_path) -> None:
    resp = await async_client.post("/recall/ack", json={"ids": ["mem_i", "mem_j"]})
    assert resp.status_code == 200
    assert resp.json()["acked"] == 2

    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_i", "mem_j"])
    assert rows["mem_i"].inject_count == 1
    assert rows["mem_j"].inject_count == 1
    assert rows["mem_i"].match_count == 0


@pytest.mark.asyncio
async def test_recall_ack_empty_is_noop(async_client) -> None:
    resp = await async_client.post("/recall/ack", json={"ids": []})
    assert resp.status_code == 200
    assert resp.json()["acked"] == 0


@pytest.mark.asyncio
async def test_recall_includes_last_used_at(
    async_client, lance_table, tmp_path
) -> None:
    """Recall surfaces consumption freshness (spec 33 Phase 2) so the rule-TTL
    refresh can key off max(createdAt, lastUsedAt)."""
    await lance_table.add([_make_memory("mem_lu")])
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_lu", 1_700_000_000.0, use=1)
    resp = await async_client.post("/recall", json={"query": "memory"})
    mem = resp.json()["memories"][0]
    assert mem["lastUsedAt"].startswith("2023-11-14")


@pytest.mark.asyncio
async def test_recall_omits_last_used_when_never_used(
    async_client, lance_table
) -> None:
    await lance_table.add([_make_memory("mem_nu")])
    resp = await async_client.post("/recall", json={"query": "memory"})
    assert "lastUsedAt" not in resp.json()["memories"][0]


def test_bump_quality_use_sets_last_used(tmp_path) -> None:
    """A ``use`` signal stamps ``last_used`` — the freshness rules (rule TTL
    refresh, spec 33 Phase 2) key off consumption, not retrieval."""
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_u", 123.0, use=1)
        row = usage.get_many(["mem_u"])["mem_u"]
    assert row.last_used == 123.0


def test_bump_quality_noise_leaves_last_used(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_n", 123.0, noise=1)
        row = usage.get_many(["mem_n"])["mem_n"]
    assert row.last_used == 0.0

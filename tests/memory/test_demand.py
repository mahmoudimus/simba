"""Recall demand log + knowledge gaps (spec 33 v2, yantrikdb borrow).

The usage ledger instruments the MEMORY side of the loop; this instruments
the QUERY side — an O(1) aggregate per normalized query so "asked often,
answered poorly" is queryable. Feeds health trends and promotion drafting
(what memory SHOULD exist). Default-off; internal daemon self-calls and
TOOL_RULE gate probes never count.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

import simba.db
import simba.memory.config
import simba.memory.demand as demand
import simba.memory.fts
import simba.memory.server


def test_record_upserts_normalized_aggregate(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        demand.record(
            "How do I run   system tests?", result_count=0, best_score=0.0, now=100.0
        )
        demand.record(
            "how do i run system tests?", result_count=2, best_score=0.42, now=200.0
        )
        rows = demand.gaps(min_asks=1, max_best=1.1)
    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "how do i run system tests?"
    assert row["askCount"] == 2
    assert row["zeroCount"] == 1
    assert row["bestScoreMax"] == 0.42
    assert row["lastAsked"] == 200.0


def test_gaps_excludes_well_answered_queries(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        for _ in range(3):
            demand.record("well answered query", 3, 0.9, now=1.0)
            demand.record("poorly answered query", 1, 0.2, now=1.0)
        rows = demand.gaps(min_asks=2, max_best=0.5)
    assert [r["query"] for r in rows] == ["poorly answered query"]


def test_gaps_min_asks_floor(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        demand.record("asked once only", 0, 0.0, now=1.0)
        assert demand.gaps(min_asks=2, max_best=1.1) == []


# ---------------------------------------------------------------------------
# Daemon wiring
# ---------------------------------------------------------------------------


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


def _client(tmp_path, lance_table, mock_embed, cfg) -> httpx.AsyncClient:
    app = simba.memory.server.create_app(cfg)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.cwd = tmp_path
    fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
    simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
    app.state.fts_path = str(fts_path)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def _drain(timeout: float = 2.0) -> None:
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


@pytest.mark.asyncio
async def test_recall_records_demand_when_enabled(
    tmp_path, lance_table, mock_embed
) -> None:
    cfg = simba.memory.config.MemoryConfig(
        demand_log_enabled=True, recall_cache_ttl_seconds=0.0
    )
    await lance_table.add([_make_memory("mem_d")])
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        resp = await ac.post(
            "/recall",
            json={"query": "where is the docker runner script"},
            headers={"X-Simba-Client": "claude-code"},
        )
        assert resp.status_code == 200
    await _drain()
    with simba.db.connect(tmp_path):
        rows = demand.gaps(min_asks=1, max_best=1.1)
    assert len(rows) == 1
    assert rows[0]["askCount"] == 1


@pytest.mark.asyncio
async def test_recall_demand_off_by_default(tmp_path, lance_table, mock_embed) -> None:
    cfg = simba.memory.config.MemoryConfig(recall_cache_ttl_seconds=0.0)
    await lance_table.add([_make_memory("mem_d2")])
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        await ac.post(
            "/recall",
            json={"query": "where is the docker runner script"},
            headers={"X-Simba-Client": "claude-code"},
        )
    await _drain()
    with simba.db.connect(tmp_path):
        assert demand.gaps(min_asks=1, max_best=1.1) == []


@pytest.mark.asyncio
async def test_recall_demand_skips_tool_rule_probes(
    tmp_path, lance_table, mock_embed
) -> None:
    cfg = simba.memory.config.MemoryConfig(
        demand_log_enabled=True, recall_cache_ttl_seconds=0.0
    )
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        await ac.post(
            "/recall",
            json={
                "query": "pytest tests/system full command line",
                "filters": {"types": ["TOOL_RULE"]},
            },
            headers={"X-Simba-Client": "claude-code"},
        )
    await _drain()
    with simba.db.connect(tmp_path):
        assert demand.gaps(min_asks=1, max_best=1.1) == []


@pytest.mark.asyncio
async def test_recall_demand_skips_internal_daemon_client(
    tmp_path, lance_table, mock_embed
) -> None:
    cfg = simba.memory.config.MemoryConfig(
        demand_log_enabled=True, recall_cache_ttl_seconds=0.0
    )
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        await ac.post(
            "/recall",
            json={"query": "internal conflict recheck lookup"},
            headers={"X-Simba-Client": "daemon"},
        )
    await _drain()
    with simba.db.connect(tmp_path):
        assert demand.gaps(min_asks=1, max_best=1.1) == []


@pytest.mark.asyncio
async def test_gaps_endpoint(tmp_path, lance_table, mock_embed) -> None:
    cfg = simba.memory.config.MemoryConfig()
    with simba.db.connect(tmp_path):
        for _ in range(3):
            demand.record("unanswered deploy question", 0, 0.0, now=5.0)
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        resp = await ac.get("/demand/gaps", params={"minAsks": 2, "maxBest": 0.5})
        assert resp.status_code == 200
        body = resp.json()
    assert body["gaps"][0]["query"] == "unanswered deploy question"
    assert body["gaps"][0]["askCount"] == 3

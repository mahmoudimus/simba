"""Per-session use attribution (spec 33 v2 rule R2).

Counters roll up totals; events carry WHICH session used a memory — the
spec's real promotion trigger is "used in ≥2 DISTINCT sessions", and every
week without events is promotion evidence lost forever.
"""

from __future__ import annotations

import pathlib

import httpx
import pytest
import pytest_asyncio

import simba.db
import simba.memory.config
import simba.memory.server
import simba.memory.usage as usage
import simba.memory.usage_events as usage_events


def test_record_and_distinct_sessions(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage_events.record("mem_e", "s1", "use", now=1.0)
        usage_events.record("mem_e", "s1", "use", now=2.0)
        usage_events.record("mem_e", "s2", "use", now=3.0)
        usage_events.record("mem_e", "s3", "noise", now=4.0)
        assert usage_events.distinct_use_sessions("mem_e") == 2
        assert usage_events.use_sessions_for(["mem_e", "mem_x"]) == {"mem_e": 2}


@pytest_asyncio.fixture
async def feedback_client(tmp_path: pathlib.Path):
    cfg = simba.memory.config.MemoryConfig(feedback_default_weight=0.3)
    app = simba.memory.server.create_app(cfg)
    app.state.cwd = tmp_path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, tmp_path


@pytest.mark.asyncio
async def test_feedback_with_session_records_event(feedback_client) -> None:
    ac, tmp_path = feedback_client
    await ac.post(
        "/memory/mem_ev/feedback", json={"signal": "good", "sessionSource": "s1"}
    )
    await ac.post(
        "/memory/mem_ev/feedback", json={"signal": "good", "sessionSource": "s2"}
    )
    with simba.db.connect(tmp_path):
        assert usage_events.distinct_use_sessions("mem_ev") == 2


@pytest.mark.asyncio
async def test_feedback_without_session_records_no_event(feedback_client) -> None:
    ac, tmp_path = feedback_client
    await ac.post("/memory/mem_nv/feedback", json={"signal": "good"})
    with simba.db.connect(tmp_path):
        assert usage_events.distinct_use_sessions("mem_nv") == 0


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


@pytest.mark.asyncio
async def test_promotions_report_and_gate_on_distinct_sessions(
    tmp_path, lance_table, mock_embed
) -> None:
    import simba.memory.fts

    cfg = simba.memory.config.MemoryConfig(promotion_min_sessions=2)
    app = simba.memory.server.create_app(cfg)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.cwd = tmp_path
    fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
    simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
    app.state.fts_path = str(fts_path)
    await lance_table.add([_make_memory("mem_multi"), _make_memory("mem_single")])
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_multi", 1.0, use=4)
        usage.bump_quality("mem_single", 1.0, use=4)
        usage_events.record("mem_multi", "s1", "use", now=1.0)
        usage_events.record("mem_multi", "s2", "use", now=2.0)
        usage_events.record("mem_single", "s1", "use", now=3.0)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        resp = await ac.get("/promotions/candidates")
    body = resp.json()
    assert [c["id"] for c in body["candidates"]] == ["mem_multi"]
    assert body["candidates"][0]["sessions"] == 2

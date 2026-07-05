"""Boot digest (spec 33 v2, yantrikdb-server borrow): the consumption surface.

One call returns everything a fresh session should see acted-on-or-visible —
heartbeat, promotion inbox, supersession pendings, knowledge gaps — so the
new surfaces can't become the next inbox nobody reads (the animaworks
lesson: surfaced candidates re-appear until consumed).
"""

from __future__ import annotations

import httpx
import pytest

import simba.db
import simba.memory.config
import simba.memory.demand as demand
import simba.memory.fts
import simba.memory.server
import simba.memory.supersession as supersession
import simba.memory.usage as usage


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
    app.state.last_maintenance = {
        "at": "2026-07-04T00:00:00Z",
        "apply": False,
        "decay": {"updated": 4245, "newly_dormant": 0},
    }
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


@pytest.mark.asyncio
async def test_digest_aggregates_lifecycle_state(
    tmp_path, lance_table, mock_embed
) -> None:
    cfg = simba.memory.config.MemoryConfig()
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_hot", 1.0, use=4, noise=0)
        supersession.append_event(
            old_id="mem_a",
            new_id="mem_b",
            project_path="",
            memory_type="GOTCHA",
            similarity=0.9,
            reason="near_duplicate_same_type",
            provenance="{}",
            status=supersession.STATUS_PENDING,
            now=1.0,
        )
        for _ in range(3):
            demand.record("how do we rotate staging certs", 0, 0.1, now=5.0)

    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        resp = await ac.get("/digest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["heartbeat"]["at"] == "2026-07-04T00:00:00Z"
    assert body["promotions"]["total"] == 1
    assert body["supersessions"]["pending"] == 1
    assert body["gaps"]["total"] == 1
    assert "rotate staging certs" in body["gaps"]["top"][0]


@pytest.mark.asyncio
async def test_digest_empty_state(tmp_path, lance_table, mock_embed) -> None:
    cfg = simba.memory.config.MemoryConfig()
    async with _client(tmp_path, lance_table, mock_embed, cfg) as ac:
        # No last_maintenance either.
        del ac  # silence linters; re-open a bare app without heartbeat
    app = simba.memory.server.create_app(cfg)
    app.state.cwd = tmp_path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        resp = await ac.get("/digest")
    body = resp.json()
    assert resp.status_code == 200
    assert body["heartbeat"] is None
    assert body["promotions"]["total"] == 0
    assert body["supersessions"]["pending"] == 0
    assert body["gaps"]["total"] == 0

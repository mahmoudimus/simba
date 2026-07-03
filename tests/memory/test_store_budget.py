"""Per-session store-budget gate (spec 33 Phase 2): the inflow throttle.

The audit measured ~400 stores/day with no cap — over-capture (raw error
output stored as rules) is real. When ``memory.store_budget_per_session`` > 0,
non-EPISODE stores beyond the budget for one ``sessionSource`` get 429.
Default 0 = off (byte-identical).
"""

from __future__ import annotations

import httpx
import pytest

import simba.memory.config
import simba.memory.fts
import simba.memory.server


def _cfg(budget: int) -> simba.memory.config.MemoryConfig:
    # duplicate_threshold > 1 disables dup detection for the constant-vector
    # mock embedder (every text embeds identically at similarity 1.0).
    return simba.memory.config.MemoryConfig(
        store_budget_per_session=budget, duplicate_threshold=1.01
    )


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


async def _store(ac, *, content: str, session: str = "", mtype: str = "GOTCHA"):
    payload: dict = {"type": mtype, "content": content}
    if session:
        payload["sessionSource"] = session
    return await ac.post("/store", json=payload)


@pytest.mark.asyncio
async def test_budget_rejects_after_limit(tmp_path, lance_table, mock_embed) -> None:
    async with _client(tmp_path, lance_table, mock_embed, _cfg(2)) as ac:
        for i in range(2):
            resp = await _store(ac, content=f"unique fact {i}", session="s1")
            assert resp.status_code == 200
        over = await _store(ac, content="unique fact 2", session="s1")
        assert over.status_code == 429
        # A different session is unaffected.
        other = await _store(ac, content="unique fact 3", session="s2")
        assert other.status_code == 200


@pytest.mark.asyncio
async def test_budget_exempts_episodes(tmp_path, lance_table, mock_embed) -> None:
    async with _client(tmp_path, lance_table, mock_embed, _cfg(1)) as ac:
        assert (await _store(ac, content="fact one", session="s1")).status_code == 200
        episode = await _store(
            ac, content="session digest", session="s1", mtype="EPISODE"
        )
        assert episode.status_code == 200


@pytest.mark.asyncio
async def test_budget_zero_is_unlimited(tmp_path, lance_table, mock_embed) -> None:
    async with _client(tmp_path, lance_table, mock_embed, _cfg(0)) as ac:
        for i in range(4):
            resp = await _store(ac, content=f"fact number {i}", session="s1")
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_budget_ignores_sessionless_stores(
    tmp_path, lance_table, mock_embed
) -> None:
    async with _client(tmp_path, lance_table, mock_embed, _cfg(1)) as ac:
        for i in range(3):
            resp = await _store(ac, content=f"anonymous fact {i}")
            assert resp.status_code == 200

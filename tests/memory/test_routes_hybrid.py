"""Route-level tests for hybrid recall wiring (/store, /recall, /delete, /patch)."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

import simba.memory.config
import simba.memory.fts as fts
import simba.memory.server


@pytest_asyncio.fixture
async def hybrid_client(memory_config, lance_table, mock_embed, tmp_path):
    """Async client backed by a real LanceDB table + a real FTS mirror."""
    fts_path = tmp_path / fts.FTS_FILENAME
    fts.init(fts_path)
    app = simba.memory.server.create_app(memory_config)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    app.state.fts_path = str(fts_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, str(fts_path), app


def _mirror_count(fts_path: str) -> int:
    conn = fts.connect(fts_path)
    try:
        return fts.count(conn)
    finally:
        conn.close()


async def _store(ac, content, *, project="proj-1", mtype="GOTCHA"):
    resp = await ac.post(
        "/store",
        json={"type": mtype, "content": content, "projectPath": project},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestStoreSync:
    @pytest.mark.asyncio
    async def test_store_dual_writes_to_mirror(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        await _store(ac, "ruff lints the python source tree")
        assert _mirror_count(fts_path) == 1
        conn = fts.connect(fts_path)
        try:
            hits = fts.search(conn, "lints", project_path="proj-1")
            assert len(hits) == 1
        finally:
            conn.close()


class TestRecallHybrid:
    @pytest.mark.asyncio
    async def test_keyword_arm_supplies_hit_below_vector_cutoff(
        self, hybrid_client
    ) -> None:
        ac, _, _ = hybrid_client
        await _store(ac, "the unique_zeta_marker keyword lives here")
        # minSimilarity 1.1 makes the vector arm return nothing (cosine maxes at
        # 1.0); only the keyword arm can surface the memory.
        resp = await ac.post(
            "/recall",
            json={
                "query": "unique_zeta_marker",
                "projectPath": "proj-1",
                "minSimilarity": 1.1,
            },
        )
        assert resp.status_code == 200
        memories = resp.json()["memories"]
        contents = [m["content"] for m in memories]
        assert any("unique_zeta_marker" in c for c in contents)


class TestDeleteSync:
    @pytest.mark.asyncio
    async def test_delete_removes_from_mirror(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        stored = await _store(ac, "deletable kappa memory content")
        assert _mirror_count(fts_path) == 1
        resp = await ac.delete(f"/memory/{stored['id']}")
        assert resp.status_code == 200
        assert _mirror_count(fts_path) == 0


class TestPatchSync:
    @pytest.mark.asyncio
    async def test_patch_moves_project_in_mirror(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        stored = await _store(ac, "movable lambda memory content", project="proj-1")
        resp = await ac.patch(
            f"/memory/{stored['id']}", json={"projectPath": "proj-2"}
        )
        assert resp.status_code == 200
        conn = fts.connect(fts_path)
        try:
            assert fts.search(conn, "lambda", project_path="proj-1") == []
            moved = fts.search(conn, "lambda", project_path="proj-2")
            assert [m["memory_id"] for m in moved] == [stored["id"]]
        finally:
            conn.close()


class TestHybridDisabled:
    @pytest.mark.asyncio
    async def test_disabled_uses_vector_path(self, hybrid_client) -> None:
        ac, _, app = hybrid_client
        app.state.config.hybrid_enabled = False
        await _store(ac, "vector only mu memory content")
        # Constant mock embeddings -> cosine 1.0 -> vector arm returns the row.
        resp = await ac.post(
            "/recall", json={"query": "anything", "projectPath": "proj-1"}
        )
        assert resp.status_code == 200
        assert len(resp.json()["memories"]) >= 1


class TestReindex:
    @pytest.mark.asyncio
    async def test_reindex_rebuilds_from_lancedb(self, hybrid_client) -> None:
        ac, fts_path, _ = hybrid_client
        await _store(ac, "reindexable nu memory content")
        # Simulate drift by wiping the mirror out of band.
        conn = fts.connect(fts_path)
        try:
            conn.execute("DELETE FROM memory_fts")
            conn.commit()
        finally:
            conn.close()
        assert _mirror_count(fts_path) == 0

        resp = await ac.post("/reindex")
        assert resp.status_code == 200
        assert resp.json()["indexed"] == 1
        assert _mirror_count(fts_path) == 1

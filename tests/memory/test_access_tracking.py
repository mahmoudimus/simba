"""Tests for access tracking — lastAccessedAt and accessCount updates on recall.

Uses real LanceDB via the lance_table and async_client fixtures from conftest.py.
"""

from __future__ import annotations

import asyncio
import unittest.mock

import pytest

import simba.memory.vector_db


def _make_memory(
    memory_id: str,
    *,
    created_at: str = "2025-01-01T00:00:00Z",
    access_count: int = 0,
) -> dict:
    """Create a memory row with tracking fields set."""
    return {
        "id": memory_id,
        "type": "GOTCHA",
        "content": f"memory {memory_id}",
        "context": "",
        "tags": "[]",
        "confidence": 0.85,
        "sessionSource": "",
        "projectPath": "",
        "createdAt": created_at,
        "lastAccessedAt": created_at,
        "accessCount": access_count,
        "vector": [0.1] * 768,
    }


async def _drain_background_tasks(timeout: float = 2.0) -> None:
    """Allow pending asyncio tasks to complete.

    The recall endpoint fires off update_access_tracking via
    asyncio.create_task.  We give those tasks time to finish.
    """
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


@pytest.fixture
def memory_config():
    """Override conftest's config to the LEGACY mode so the write happens.

    ``lancedb_version_retention_seconds=0`` is legacy/unbounded: the per-recall
    LanceDB access-tracking write is performed (and versions never pruned). The
    default (>0) suppresses the write — that fix is verified in the gate tests.
    """
    import simba.memory.config

    return simba.memory.config.MemoryConfig(
        max_content_length=200,
        duplicate_threshold=0.92,
        lancedb_version_retention_seconds=0,
    )


class TestAccessTrackingGate:
    """The LanceDB access write is folded into the retention knob.

    retention>0 (bounded) suppresses the write; retention=0 (legacy) performs it.
    """

    async def _build_app(self, tmp_path, lance_table, mock_embed, *, retention: int):
        import httpx

        import simba.memory.config
        import simba.memory.fts
        import simba.memory.server

        cfg = simba.memory.config.MemoryConfig(
            max_content_length=200,
            duplicate_threshold=0.92,
            lancedb_version_retention_seconds=retention,
        )
        app = simba.memory.server.create_app(cfg)
        app.state.table = lance_table
        app.state.embed = mock_embed
        app.state.embed_query = mock_embed
        app.state.db_path = None
        app.state.cwd = tmp_path
        fts_path = tmp_path / simba.memory.fts.FTS_FILENAME
        simba.memory.fts.init(fts_path, tokenize=cfg.fts_tokenize)
        app.state.fts_path = str(fts_path)
        return app, httpx.ASGITransport(app=app)

    @pytest.mark.asyncio
    async def test_bounded_retention_skips_lancedb_write(
        self, tmp_path, lance_table, mock_embed, monkeypatch
    ) -> None:
        import httpx

        await lance_table.add([_make_memory("mem_off")])
        called = unittest.mock.AsyncMock()
        monkeypatch.setattr(simba.memory.vector_db, "update_access_tracking", called)
        _app, transport = await self._build_app(
            tmp_path, lance_table, mock_embed, retention=86_400
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/recall", json={"query": "x"})
        assert resp.status_code == 200
        await _drain_background_tasks()
        called.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_retention_does_lancedb_write(
        self, tmp_path, lance_table, mock_embed, monkeypatch
    ) -> None:
        import httpx

        await lance_table.add([_make_memory("mem_on")])
        called = unittest.mock.AsyncMock()
        monkeypatch.setattr(simba.memory.vector_db, "update_access_tracking", called)
        _app, transport = await self._build_app(
            tmp_path, lance_table, mock_embed, retention=0
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/recall", json={"query": "x"})
        assert resp.status_code == 200
        await _drain_background_tasks()
        called.assert_called_once()


class TestAccessTracking:
    """Verify that recalling memories updates lastAccessedAt and accessCount."""

    @pytest.mark.asyncio
    async def test_recall_updates_last_accessed_at(
        self, async_client, lance_table
    ) -> None:
        """After recall, lastAccessedAt should be updated to current time."""
        original_time = "2025-01-01T00:00:00Z"
        await lance_table.add([_make_memory("mem_aaa", created_at=original_time)])

        resp = await async_client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["memories"]) == 1

        # Give the fire-and-forget task a chance to complete.
        await _drain_background_tasks()

        # The lastAccessedAt field should now be different from original.
        rows = await lance_table.query().where("id = 'mem_aaa'").limit(1).to_list()
        assert len(rows) == 1
        assert rows[0]["lastAccessedAt"] != original_time, (
            f"lastAccessedAt was not updated: still {rows[0]['lastAccessedAt']}"
        )

    @pytest.mark.asyncio
    async def test_recall_increments_access_count(
        self, async_client, lance_table
    ) -> None:
        """After recall, accessCount should be incremented by 1."""
        await lance_table.add([_make_memory("mem_bbb", access_count=0)])

        resp = await async_client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        assert len(resp.json()["memories"]) == 1

        await _drain_background_tasks()

        rows = await lance_table.query().where("id = 'mem_bbb'").limit(1).to_list()
        assert len(rows) == 1
        assert rows[0]["accessCount"] == 1, (
            f"accessCount was not incremented: got {rows[0]['accessCount']}"
        )

    @pytest.mark.asyncio
    async def test_recall_updates_only_returned_memories(
        self, async_client, lance_table
    ) -> None:
        """Memories NOT in the recall results should keep their original values."""
        original_time = "2025-01-01T00:00:00Z"
        await lance_table.add([_make_memory("mem_returned", created_at=original_time)])

        # SYSTEM memories are filtered out by search_memories.
        system_mem = _make_memory("mem_untouched", created_at=original_time)
        system_mem["type"] = "SYSTEM"
        await lance_table.add([system_mem])

        resp = await async_client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        returned_ids = [m["id"] for m in resp.json()["memories"]]
        assert "mem_returned" in returned_ids
        assert "mem_untouched" not in returned_ids

        await _drain_background_tasks()

        # The untouched SYSTEM memory should still have original values.
        rows = (
            await lance_table.query().where("id = 'mem_untouched'").limit(1).to_list()
        )
        assert len(rows) == 1
        assert rows[0]["lastAccessedAt"] == original_time
        assert rows[0]["accessCount"] == 0

    @pytest.mark.asyncio
    async def test_multiple_recalls_increment_count(
        self, async_client, lance_table
    ) -> None:
        """Calling recall twice should set accessCount to 2."""
        await lance_table.add([_make_memory("mem_ccc", access_count=0)])

        # First recall
        resp1 = await async_client.post("/recall", json={"query": "test"})
        assert resp1.status_code == 200
        assert len(resp1.json()["memories"]) == 1
        await _drain_background_tasks()

        # Second recall
        resp2 = await async_client.post("/recall", json={"query": "test"})
        assert resp2.status_code == 200
        assert len(resp2.json()["memories"]) >= 1
        await _drain_background_tasks()

        rows = await lance_table.query().where("id = 'mem_ccc'").limit(1).to_list()
        assert len(rows) == 1
        assert rows[0]["accessCount"] == 2, (
            f"Expected accessCount=2 after two recalls, got {rows[0]['accessCount']}"
        )

    @pytest.mark.asyncio
    async def test_recall_update_is_fire_and_forget(
        self, async_client, lance_table
    ) -> None:
        """The recall response should not be delayed by a slow/failing update.

        We patch update_access_tracking to raise, and verify the recall
        endpoint still returns successfully without propagating the error.
        """
        await lance_table.add([_make_memory("mem_ddd")])

        with unittest.mock.patch.object(
            simba.memory.vector_db,
            "update_access_tracking",
            side_effect=RuntimeError("simulated tracking failure"),
        ):
            resp = await async_client.post("/recall", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["memories"]) == 1
            # The response should include queryTimeMs — no error leaked.
            assert "error" not in data
            assert "queryTimeMs" in data

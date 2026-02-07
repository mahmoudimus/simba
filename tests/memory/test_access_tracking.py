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

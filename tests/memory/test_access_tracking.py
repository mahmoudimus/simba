"""Tests for access tracking — lastAccessedAt and accessCount updates on recall."""

from __future__ import annotations

import asyncio
import typing
import unittest.mock

import fastapi.testclient
import pytest

import simba.memory.config
import simba.memory.routes
import simba.memory.server
import simba.memory.vector_db


class TrackingMockSearch:
    """Chainable mock for table.search().where(...).limit(...).to_list()."""

    def __init__(self, rows: list[dict[str, typing.Any]]) -> None:
        self._rows = rows
        self._where_expr: str | None = None
        self._limit_val: int = 10

    def where(self, expr: str) -> TrackingMockSearch:
        self._where_expr = expr
        return self

    def limit(self, n: int) -> TrackingMockSearch:
        self._limit_val = n
        return self

    async def to_list(self) -> list[dict[str, typing.Any]]:
        results = list(self._rows)
        if self._where_expr and self._where_expr.startswith("id = '"):
            target_id = self._where_expr[6:-1]
            results = [r for r in results if r.get("id") == target_id]
        return results[: self._limit_val]


class TrackingMockVectorSearch:
    """Mock for vector_search() that returns rows with fake distances."""

    def __init__(self, rows: list[dict[str, typing.Any]]) -> None:
        self._rows = rows
        self._limit_val = 10

    def column(self, name: str) -> TrackingMockVectorSearch:
        return self

    def distance_type(self, dtype: str) -> TrackingMockVectorSearch:
        return self

    def limit(self, n: int) -> TrackingMockVectorSearch:
        self._limit_val = n
        return self

    async def to_list(self) -> list[dict[str, typing.Any]]:
        results = []
        for row in self._rows[: self._limit_val]:
            results.append({**row, "_distance": 0.1})
        return results


class TrackingMockTable:
    """In-memory mock of a LanceDB table that supports update and search.

    Extends the base MockTable pattern with update() and search() methods
    required by update_access_tracking.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, typing.Any]] = []

    async def add(self, rows: list[dict[str, typing.Any]]) -> None:
        self._rows.extend(rows)

    async def count_rows(self) -> int:
        return len(self._rows)

    async def to_list(self) -> list[dict[str, typing.Any]]:
        return list(self._rows)

    def vector_search(self, embedding: list[float]) -> TrackingMockVectorSearch:
        return TrackingMockVectorSearch(self._rows)

    def search(self) -> TrackingMockSearch:
        return TrackingMockSearch(self._rows)

    async def update(
        self,
        where: str | None = None,
        values: dict[str, typing.Any] | None = None,
    ) -> None:
        if where is None or values is None:
            return
        for row in self._rows:
            # Parse simple where: id = 'xxx'
            if where.startswith("id = '"):
                target_id = where[6:-1]
                if row.get("id") != target_id:
                    continue
            for key, val in values.items():
                row[key] = val

    async def delete(self, filter_expr: str) -> None:
        if filter_expr.startswith("id = '"):
            target_id = filter_expr[6:-1]
            self._rows = [r for r in self._rows if r.get("id") != target_id]


def _make_memory(
    memory_id: str,
    *,
    created_at: str = "2025-01-01T00:00:00Z",
    access_count: int = 0,
) -> dict[str, typing.Any]:
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


@pytest.fixture
def config() -> simba.memory.config.MemoryConfig:
    return simba.memory.config.MemoryConfig(
        max_content_length=200,
        duplicate_threshold=0.92,
    )


@pytest.fixture
def tracking_table() -> TrackingMockTable:
    return TrackingMockTable()


@pytest.fixture
def mock_embed() -> typing.Callable:
    async def _embed(text: str) -> list[float]:
        return [0.1] * 768

    return _embed


@pytest.fixture
def client(
    config: simba.memory.config.MemoryConfig,
    tracking_table: TrackingMockTable,
    mock_embed: typing.Callable,
) -> fastapi.testclient.TestClient:
    app = simba.memory.server.create_app(config)
    app.state.table = tracking_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    return fastapi.testclient.TestClient(app)


def _drain_background_tasks(timeout: float = 1.0) -> None:
    """Allow pending asyncio tasks to complete.

    The recall endpoint fires off update_access_tracking via
    asyncio.create_task.  In the test-client environment the event
    loop may not run those tasks automatically, so we give them a
    moment to finish.
    """
    loop = asyncio.get_event_loop()
    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
    if pending:
        loop.run_until_complete(asyncio.wait(pending, timeout=timeout))


class TestAccessTracking:
    """Verify that recalling memories updates lastAccessedAt and accessCount."""

    def test_recall_updates_last_accessed_at(
        self, client: fastapi.testclient.TestClient, tracking_table: TrackingMockTable
    ) -> None:
        """After recall, lastAccessedAt should be updated to current time."""
        original_time = "2025-01-01T00:00:00Z"
        tracking_table._rows.append(_make_memory("mem_aaa", created_at=original_time))

        resp = client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["memories"]) == 1

        # Give the fire-and-forget task a chance to complete.
        _drain_background_tasks()

        # The lastAccessedAt field in the underlying table should now
        # be different from the original createdAt value.
        row = tracking_table._rows[0]
        assert row["lastAccessedAt"] != original_time, (
            f"lastAccessedAt was not updated: still {row['lastAccessedAt']}"
        )

    def test_recall_increments_access_count(
        self, client: fastapi.testclient.TestClient, tracking_table: TrackingMockTable
    ) -> None:
        """After recall, accessCount should be incremented by 1."""
        tracking_table._rows.append(_make_memory("mem_bbb", access_count=0))

        resp = client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        assert len(resp.json()["memories"]) == 1

        _drain_background_tasks()

        row = tracking_table._rows[0]
        assert row["accessCount"] == 1, (
            f"accessCount was not incremented: got {row['accessCount']}"
        )

    def test_recall_updates_only_returned_memories(
        self, client: fastapi.testclient.TestClient, tracking_table: TrackingMockTable
    ) -> None:
        """Memories NOT in the recall results should keep their original values."""
        original_time = "2025-01-01T00:00:00Z"
        tracking_table._rows.append(
            _make_memory("mem_returned", created_at=original_time)
        )
        # This memory will NOT be returned because it is type SYSTEM
        # (search_memories filters out SYSTEM memories).
        system_mem = _make_memory("mem_untouched", created_at=original_time)
        system_mem["type"] = "SYSTEM"
        tracking_table._rows.append(system_mem)

        resp = client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        # Only the non-SYSTEM memory should appear in results.
        returned_ids = [m["id"] for m in resp.json()["memories"]]
        assert "mem_returned" in returned_ids
        assert "mem_untouched" not in returned_ids

        _drain_background_tasks()

        # The untouched memory should still have original values.
        untouched = next(r for r in tracking_table._rows if r["id"] == "mem_untouched")
        assert untouched["lastAccessedAt"] == original_time
        assert untouched["accessCount"] == 0

    def test_multiple_recalls_increment_count(
        self, client: fastapi.testclient.TestClient, tracking_table: TrackingMockTable
    ) -> None:
        """Calling recall twice should set accessCount to 2."""
        tracking_table._rows.append(_make_memory("mem_ccc", access_count=0))

        # First recall
        resp1 = client.post("/recall", json={"query": "test"})
        assert resp1.status_code == 200
        assert len(resp1.json()["memories"]) == 1
        _drain_background_tasks()

        # Second recall
        resp2 = client.post("/recall", json={"query": "test"})
        assert resp2.status_code == 200
        assert len(resp2.json()["memories"]) == 1
        _drain_background_tasks()

        row = tracking_table._rows[0]
        assert row["accessCount"] == 2, (
            f"Expected accessCount=2 after two recalls, got {row['accessCount']}"
        )

    def test_recall_update_is_fire_and_forget(
        self, client: fastapi.testclient.TestClient, tracking_table: TrackingMockTable
    ) -> None:
        """The recall response should not be delayed by a slow/failing update.

        We patch update_access_tracking to raise, and verify the recall
        endpoint still returns successfully without propagating the error.
        """
        tracking_table._rows.append(_make_memory("mem_ddd"))

        with unittest.mock.patch.object(
            simba.memory.vector_db,
            "update_access_tracking",
            side_effect=RuntimeError("simulated tracking failure"),
        ):
            resp = client.post("/recall", json={"query": "test"})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["memories"]) == 1
            # The response should include queryTimeMs — no error leaked.
            assert "error" not in data
            assert "queryTimeMs" in data

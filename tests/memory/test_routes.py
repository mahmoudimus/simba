"""Tests for memory routes -- all 6 endpoints via real LanceDB + httpx AsyncClient."""

from __future__ import annotations

import json
import time

import pytest


class TestStoreEndpoint:
    @pytest.mark.asyncio
    async def test_stores_memory(self, async_client):
        resp = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "test memory"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["id"].startswith("mem_")
        assert data["embedding_dims"] == 768

    @pytest.mark.asyncio
    async def test_rejects_missing_type(self, async_client):
        resp = await async_client.post("/store", json={"content": "test"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_missing_content(self, async_client):
        resp = await async_client.post("/store", json={"type": "GOTCHA"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rejects_invalid_type(self, async_client):
        resp = await async_client.post(
            "/store",
            json={"type": "INVALID", "content": "test"},
        )
        assert resp.status_code == 400
        assert "invalid type" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_rejects_long_content(self, async_client):
        resp = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "x" * 300},
        )
        assert resp.status_code == 400
        assert "too long" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_stores_with_context(self, async_client, lance_table):
        resp = await async_client.post(
            "/store",
            json={
                "type": "PATTERN",
                "content": "test pattern",
                "context": "in module X",
            },
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        # Verify in the real table
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        assert rows[0]["context"] == "in module X"

    @pytest.mark.asyncio
    async def test_stores_with_tags(self, async_client, lance_table):
        resp = await async_client.post(
            "/store",
            json={
                "type": "DECISION",
                "content": "chose FastAPI",
                "tags": ["python", "api"],
            },
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        assert json.loads(rows[0]["tags"]) == ["python", "api"]

    @pytest.mark.asyncio
    async def test_default_confidence(self, async_client, lance_table):
        resp = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "test confidence"},
        )
        assert resp.status_code == 200
        stored_id = resp.json()["id"]
        rows = await lance_table.query().where(f"id = '{stored_id}'").to_list()
        assert len(rows) == 1
        assert rows[0]["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, async_client):
        """Storing the same content twice flags the second as duplicate."""
        resp1 = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "duplicate content"},
        )
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "stored"

        # Second store with same embedding vector -> flagged as duplicate
        resp2 = await async_client.post(
            "/store",
            json={"type": "GOTCHA", "content": "duplicate content again"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["status"] == "duplicate"
        assert "existing_id" in data2
        assert data2["similarity"] >= 0.92


class TestRecallEndpoint:
    @pytest.mark.asyncio
    async def test_recall_empty(self, async_client):
        """Recall with no user-added memories returns empty list."""
        resp = await async_client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "queryTimeMs" in data
        # Only the SYSTEM init row exists, which is filtered out
        assert len(data["memories"]) == 0

    @pytest.mark.asyncio
    async def test_recall_returns_results(self, async_client, lance_table):
        """Add a memory directly to the table, then recall it."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_test1",
                    "type": "GOTCHA",
                    "content": "test memory for recall",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.85,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                }
            ]
        )
        resp = await async_client.post("/recall", json={"query": "test"})
        data = resp.json()
        assert len(data["memories"]) >= 1
        ids = [m["id"] for m in data["memories"]]
        assert "mem_test1" in ids

    @pytest.mark.asyncio
    async def test_recall_with_project_path_filters(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_a",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-a",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_b",
                    "type": "GOTCHA",
                    "content": "project B memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-b",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.post(
            "/recall",
            json={"query": "test", "projectPath": "/path/to/project-a"},
        )
        data = resp.json()
        # Only project A memories should pass the filter
        for mem in data["memories"]:
            assert mem["id"] != "mem_b"

    @pytest.mark.asyncio
    async def test_recall_without_project_path_returns_all(
        self, async_client, lance_table
    ):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_a",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-a",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_b",
                    "type": "GOTCHA",
                    "content": "project B memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "/path/to/project-b",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = await async_client.post("/recall", json={"query": "test"})
        data = resp.json()
        # Without projectPath filter, both memories are returned
        ids = [m["id"] for m in data["memories"]]
        assert "mem_a" in ids
        assert "mem_b" in ids

    @pytest.mark.asyncio
    async def test_recall_requires_query(self, async_client):
        resp = await async_client.post("/recall", json={})
        assert resp.status_code == 422


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_ok(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime" in data
        assert "memoryCount" in data
        assert data["embeddingModel"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_health_memory_count(self, async_client, lance_table):
        """Health endpoint reports the real row count from LanceDB."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_health",
                    "type": "GOTCHA",
                    "content": "health check memory",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                }
            ]
        )
        resp = await async_client.get("/health")
        # 1 SYSTEM init row + 1 added row = 2
        assert resp.json()["memoryCount"] == 2


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_empty_stats(self, async_client):
        """Stats endpoint with only the SYSTEM init row."""
        resp = await async_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        # The SYSTEM init row is included in stats
        assert data["total"] == 1
        assert data["byType"]["SYSTEM"] == 1
        assert data["avgConfidence"] == 1.0

    @pytest.mark.asyncio
    async def test_stats_with_data(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "g1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "2",
                    "type": "GOTCHA",
                    "content": "g2",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.8,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "3",
                    "type": "PATTERN",
                    "content": "p1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.7,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/stats")
        data = resp.json()
        # 3 added + 1 SYSTEM init row = 4 total
        assert data["total"] == 4
        assert data["byType"]["GOTCHA"] == 2
        assert data["byType"]["PATTERN"] == 1
        assert data["byType"]["SYSTEM"] == 1
        # avg confidence: (1.0 + 0.9 + 0.8 + 0.7) / 4 = 0.85
        assert data["avgConfidence"] == pytest.approx(0.85, rel=0.01)


class TestListEndpoint:
    @pytest.mark.asyncio
    async def test_empty_list(self, async_client):
        """List with only the SYSTEM init row still returns it."""
        resp = await async_client.get("/list")
        assert resp.status_code == 200
        data = resp.json()
        # The SYSTEM init row is listed
        assert data["total"] == 1
        assert data["memories"][0]["type"] == "SYSTEM"

    @pytest.mark.asyncio
    async def test_list_with_data(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "memory one",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "2",
                    "type": "PATTERN",
                    "content": "memory two",
                    "context": "ctx",
                    "tags": "[]",
                    "confidence": 0.8,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 1,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list")
        data = resp.json()
        # 2 added + 1 SYSTEM init = 3
        assert data["total"] == 3
        assert len(data["memories"]) == 3

    @pytest.mark.asyncio
    async def test_list_type_filter(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "g1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
                {
                    "id": "2",
                    "type": "PATTERN",
                    "content": "p1",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.8,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                },
            ]
        )
        resp = await async_client.get("/list?type=GOTCHA")
        data = resp.json()
        assert data["total"] == 1
        assert data["memories"][0]["type"] == "GOTCHA"

    @pytest.mark.asyncio
    async def test_list_pagination(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for i in range(5):
            await lance_table.add(
                [
                    {
                        "id": str(i),
                        "type": "GOTCHA",
                        "content": f"mem {i}",
                        "context": "",
                        "tags": "[]",
                        "confidence": 0.9,
                        "sessionSource": "",
                        "projectPath": "",
                        "createdAt": now,
                        "lastAccessedAt": now,
                        "accessCount": 0,
                        "vector": [0.0] * 768,
                    }
                ]
            )
        resp = await async_client.get("/list?limit=2&offset=1")
        data = resp.json()
        # 5 added + 1 SYSTEM init = 6 total
        assert data["total"] == 6
        assert len(data["memories"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 1


class TestDeleteEndpoint:
    @pytest.mark.asyncio
    async def test_deletes_memory(self, async_client, lance_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await lance_table.add(
            [
                {
                    "id": "mem_abc",
                    "type": "GOTCHA",
                    "content": "to be deleted",
                    "context": "",
                    "tags": "[]",
                    "confidence": 0.9,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": now,
                    "lastAccessedAt": now,
                    "accessCount": 0,
                    "vector": [0.0] * 768,
                }
            ]
        )
        count_before = await lance_table.count_rows()
        assert count_before == 2  # 1 SYSTEM init + 1 added

        resp = await async_client.delete("/memory/mem_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["id"] == "mem_abc"

        count_after = await lance_table.count_rows()
        assert count_after == 1  # Only SYSTEM init remains

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, async_client):
        resp = await async_client.delete("/memory/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

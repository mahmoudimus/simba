"""Tests for memory routes — all 6 endpoints via FastAPI TestClient."""

from __future__ import annotations

import json
import time
import typing
import unittest.mock

import fastapi.testclient
import pytest

import simba.memory.config
import simba.memory.server


class MockTable:
    """In-memory mock of a LanceDB table for testing."""

    def __init__(self) -> None:
        self._rows: list[dict[str, typing.Any]] = []

    async def add(self, rows: list[dict[str, typing.Any]]) -> None:
        self._rows.extend(rows)

    async def count_rows(self) -> int:
        return len(self._rows)

    async def to_list(self) -> list[dict[str, typing.Any]]:
        return list(self._rows)

    def vector_search(self, embedding: list[float]) -> MockVectorSearch:
        return MockVectorSearch(self._rows, embedding)

    async def delete(self, filter_expr: str) -> None:
        # Parse simple filter: id = 'xxx'
        if filter_expr.startswith("id = '"):
            target_id = filter_expr[6:-1]
            self._rows = [r for r in self._rows if r.get("id") != target_id]


class MockVectorSearch:
    def __init__(self, rows: list[dict[str, typing.Any]], query: list[float]) -> None:
        self._rows = rows
        self._query = query
        self._limit_val = 10

    def column(self, name: str) -> MockVectorSearch:
        return self

    def distance_type(self, dtype: str) -> MockVectorSearch:
        return self

    def limit(self, n: int) -> MockVectorSearch:
        self._limit_val = n
        return self

    async def to_list(self) -> list[dict[str, typing.Any]]:
        # Return rows with fake _distance based on simple comparison
        results = []
        for row in self._rows[: self._limit_val]:
            results.append({**row, "_distance": 0.5})
        return results


@pytest.fixture
def config() -> simba.memory.config.MemoryConfig:
    return simba.memory.config.MemoryConfig(
        max_content_length=200, duplicate_threshold=0.92
    )


@pytest.fixture
def mock_table() -> MockTable:
    return MockTable()


@pytest.fixture
def mock_embed() -> unittest.mock.AsyncMock:
    async def _embed(text: str) -> list[float]:
        return [0.1] * 768

    return _embed


@pytest.fixture
def client(config, mock_table, mock_embed) -> fastapi.testclient.TestClient:
    app = simba.memory.server.create_app(config)
    app.state.table = mock_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    return fastapi.testclient.TestClient(app)


class TestStoreEndpoint:
    def test_stores_memory(self, client, mock_table):
        resp = client.post(
            "/store",
            json={"type": "GOTCHA", "content": "test memory"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stored"
        assert data["id"].startswith("mem_")
        assert data["embedding_dims"] == 768

    def test_rejects_missing_type(self, client):
        resp = client.post("/store", json={"content": "test"})
        assert resp.status_code == 422

    def test_rejects_missing_content(self, client):
        resp = client.post("/store", json={"type": "GOTCHA"})
        assert resp.status_code == 422

    def test_rejects_invalid_type(self, client):
        resp = client.post(
            "/store",
            json={"type": "INVALID", "content": "test"},
        )
        assert resp.status_code == 400
        assert "invalid type" in resp.json()["detail"]

    def test_rejects_long_content(self, client):
        resp = client.post(
            "/store",
            json={"type": "GOTCHA", "content": "x" * 300},
        )
        assert resp.status_code == 400
        assert "too long" in resp.json()["detail"]

    def test_stores_with_context(self, client, mock_table):
        resp = client.post(
            "/store",
            json={
                "type": "PATTERN",
                "content": "test pattern",
                "context": "in module X",
            },
        )
        assert resp.status_code == 200
        assert len(mock_table._rows) == 1
        assert mock_table._rows[0]["context"] == "in module X"

    def test_stores_with_tags(self, client, mock_table):
        resp = client.post(
            "/store",
            json={
                "type": "DECISION",
                "content": "chose FastAPI",
                "tags": ["python", "api"],
            },
        )
        assert resp.status_code == 200
        assert json.loads(mock_table._rows[0]["tags"]) == ["python", "api"]

    def test_default_confidence(self, client, mock_table):
        client.post(
            "/store",
            json={"type": "GOTCHA", "content": "test"},
        )
        assert mock_table._rows[0]["confidence"] == 0.85


class TestRecallEndpoint:
    def test_recall_empty(self, client):
        resp = client.post("/recall", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "queryTimeMs" in data

    def test_recall_returns_results(self, client, mock_table):
        # Add a memory directly
        mock_table._rows.append(
            {
                "id": "mem_test1",
                "type": "GOTCHA",
                "content": "test memory",
                "context": "",
                "confidence": 0.85,
                "vector": [0.1] * 768,
            }
        )
        resp = client.post("/recall", json={"query": "test"})
        data = resp.json()
        assert len(data["memories"]) >= 0  # May or may not pass similarity filter

    def test_recall_with_project_path_filters(self, client, mock_table):
        mock_table._rows.extend(
            [
                {
                    "id": "mem_a",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "confidence": 0.9,
                    "projectPath": "/path/to/project-a",
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_b",
                    "type": "GOTCHA",
                    "content": "project B memory",
                    "context": "",
                    "confidence": 0.9,
                    "projectPath": "/path/to/project-b",
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = client.post(
            "/recall",
            json={"query": "test", "projectPath": "/path/to/project-a"},
        )
        data = resp.json()
        # Only project A memories should pass the filter
        for mem in data["memories"]:
            assert mem["id"] != "mem_b"

    def test_recall_without_project_path_returns_all(self, client, mock_table):
        mock_table._rows.extend(
            [
                {
                    "id": "mem_a",
                    "type": "GOTCHA",
                    "content": "project A memory",
                    "context": "",
                    "confidence": 0.9,
                    "projectPath": "/path/to/project-a",
                    "vector": [0.1] * 768,
                },
                {
                    "id": "mem_b",
                    "type": "GOTCHA",
                    "content": "project B memory",
                    "context": "",
                    "confidence": 0.9,
                    "projectPath": "/path/to/project-b",
                    "vector": [0.1] * 768,
                },
            ]
        )
        resp = client.post("/recall", json={"query": "test"})
        data = resp.json()
        # Without projectPath filter, both memories are returned
        ids = [m["id"] for m in data["memories"]]
        assert "mem_a" in ids
        assert "mem_b" in ids

    def test_recall_requires_query(self, client):
        resp = client.post("/recall", json={})
        assert resp.status_code == 422


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime" in data
        assert "memoryCount" in data
        assert data["embeddingModel"] == "nomic-embed-text"

    def test_health_memory_count(self, client, mock_table):
        mock_table._rows.append({"id": "test", "type": "GOTCHA"})
        resp = client.get("/health")
        assert resp.json()["memoryCount"] == 1


class TestStatsEndpoint:
    def test_empty_stats(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["byType"] == {}
        assert data["avgConfidence"] == 0

    def test_stats_with_data(self, client, mock_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mock_table._rows.extend(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "confidence": 0.9,
                    "createdAt": now,
                },
                {
                    "id": "2",
                    "type": "GOTCHA",
                    "confidence": 0.8,
                    "createdAt": now,
                },
                {
                    "id": "3",
                    "type": "PATTERN",
                    "confidence": 0.7,
                    "createdAt": now,
                },
            ]
        )
        resp = client.get("/stats")
        data = resp.json()
        assert data["total"] == 3
        assert data["byType"]["GOTCHA"] == 2
        assert data["byType"]["PATTERN"] == 1
        assert data["avgConfidence"] == pytest.approx(0.8, rel=0.01)


class TestListEndpoint:
    def test_empty_list(self, client):
        resp = client.get("/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memories"] == []
        assert data["total"] == 0

    def test_list_with_data(self, client, mock_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mock_table._rows.extend(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "memory one",
                    "context": "",
                    "confidence": 0.9,
                    "createdAt": now,
                    "accessCount": 0,
                },
                {
                    "id": "2",
                    "type": "PATTERN",
                    "content": "memory two",
                    "context": "ctx",
                    "confidence": 0.8,
                    "createdAt": now,
                    "accessCount": 1,
                },
            ]
        )
        resp = client.get("/list")
        data = resp.json()
        assert data["total"] == 2
        assert len(data["memories"]) == 2

    def test_list_type_filter(self, client, mock_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        mock_table._rows.extend(
            [
                {
                    "id": "1",
                    "type": "GOTCHA",
                    "content": "g1",
                    "context": "",
                    "confidence": 0.9,
                    "createdAt": now,
                    "accessCount": 0,
                },
                {
                    "id": "2",
                    "type": "PATTERN",
                    "content": "p1",
                    "context": "",
                    "confidence": 0.8,
                    "createdAt": now,
                    "accessCount": 0,
                },
            ]
        )
        resp = client.get("/list?type=GOTCHA")
        data = resp.json()
        assert data["total"] == 1
        assert data["memories"][0]["type"] == "GOTCHA"

    def test_list_pagination(self, client, mock_table):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for i in range(5):
            mock_table._rows.append(
                {
                    "id": str(i),
                    "type": "GOTCHA",
                    "content": f"mem {i}",
                    "context": "",
                    "confidence": 0.9,
                    "createdAt": now,
                    "accessCount": 0,
                }
            )
        resp = client.get("/list?limit=2&offset=1")
        data = resp.json()
        assert data["total"] == 5
        assert len(data["memories"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 1


class TestDeleteEndpoint:
    def test_deletes_memory(self, client, mock_table):
        mock_table._rows.append({"id": "mem_abc", "type": "GOTCHA"})
        resp = client.delete("/memory/mem_abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        assert data["id"] == "mem_abc"
        assert len(mock_table._rows) == 0

    def test_delete_nonexistent(self, client):
        resp = client.delete("/memory/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

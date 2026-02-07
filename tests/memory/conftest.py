"""Shared fixtures for memory daemon tests."""

from __future__ import annotations

import time

import httpx
import pytest
import pytest_asyncio

import simba.memory.config
import simba.memory.server


@pytest_asyncio.fixture
async def lance_table(tmp_path):
    """Create a real LanceDB AsyncTable for memory tests."""
    import lancedb

    db = await lancedb.connect_async(str(tmp_path / "test.lance"))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    table = await db.create_table(
        "memories",
        data=[
            {
                "id": "init_0",
                "type": "SYSTEM",
                "content": "Memory system initialized",
                "context": "",
                "tags": "[]",
                "confidence": 1.0,
                "sessionSource": "",
                "projectPath": "",
                "createdAt": now,
                "lastAccessedAt": now,
                "accessCount": 0,
                "vector": [0.0] * 768,
            }
        ],
    )
    yield table


@pytest.fixture
def memory_config() -> simba.memory.config.MemoryConfig:
    """Standard test config for memory daemon."""
    return simba.memory.config.MemoryConfig(
        max_content_length=200,
        duplicate_threshold=0.92,
    )


@pytest_asyncio.fixture
async def mock_embed():
    """Mock embedding function returning deterministic 768-dim vectors."""

    async def _embed(text: str) -> list[float]:
        return [0.1] * 768

    return _embed


@pytest_asyncio.fixture
async def async_client(memory_config, lance_table, mock_embed):
    """Async HTTP client backed by a real LanceDB table."""
    app = simba.memory.server.create_app(memory_config)
    app.state.table = lance_table
    app.state.embed = mock_embed
    app.state.embed_query = mock_embed
    app.state.db_path = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

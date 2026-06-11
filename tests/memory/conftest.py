"""Shared fixtures for memory daemon tests."""

from __future__ import annotations

import time

import httpx
import pytest
import pytest_asyncio

import simba.memory.config
import simba.memory.reranker
import simba.memory.server


@pytest.fixture(autouse=True)
def _block_gguf_reranker_loads(monkeypatch):
    """Forbid real GGUF reranker model downloads/loads in unit tests.

    The default ``reranker_mode="cross-encoder"`` would otherwise load a real
    GGUF on the rerank hot path. The GGUF accessors are patched to raise, so the
    reranker fail-opens (candidates unchanged) — restoring the prior "no reorder
    unless a scorer is wired" behavior. Tests that exercise the reorder logic
    inject a fake scorer via their own ``monkeypatch.setattr``, which overrides
    this. The opt-in real-GGUF integration test installs its own (non-raising)
    accessor before any reload. No model is ever fetched in CI.
    """

    def _forbidden(cfg):
        raise RuntimeError("GGUF reranker load blocked in unit tests")

    monkeypatch.setattr(simba.memory.reranker, "_get_cross_encoder", _forbidden)
    monkeypatch.setattr(simba.memory.reranker, "_get_local_llm", _forbidden)


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

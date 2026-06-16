"""Daemon query-embed cache: identical recall queries embed once, not N times.

Under the process-global llama lock, re-embedding a repeated query (e.g. the
conflict detector firing the same pairwise check many times) serializes and
inflates tail latency. The persistent cache collapses identical queries to a
single GGUF embed.
"""

from __future__ import annotations

import asyncio

import simba.config
import simba.memory.config  # registers the "memory" config section
import simba.memory.embedding_cache as ec
import simba.memory.server as server


class _FakeService:
    """Stand-in EmbeddingService whose async embed counts how often it runs."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str, task=None) -> list[float]:
        self.calls += 1
        return [float(len(text)), 0.25]


def test_cached_query_embedder_dedups_identical_queries(tmp_path):
    svc = _FakeService()
    cache = ec.EmbeddingCache(tmp_path / "embed_cache.db")
    embed_query = server._cached_query_embedder(svc, cache, "model-x")

    async def run():
        a = await embed_query("checking two memories")
        b = await embed_query("checking two memories")  # identical -> cache hit
        c = await embed_query("a different query")
        return a, b, c

    a, b, c = asyncio.run(run())
    assert a == b  # identical text -> identical vector
    assert a != c
    assert svc.calls == 2  # "checking..." embedded once, "different" once
    cache.close()


def test_cached_query_embedder_persists_across_restart(tmp_path):
    # A fresh daemon (new EmbeddingCache on the same file) still hits — survives
    # the frequent restarts.
    path = tmp_path / "embed_cache.db"

    svc1 = _FakeService()
    c1 = ec.EmbeddingCache(path)
    asyncio.run(server._cached_query_embedder(svc1, c1, "m")("q"))
    assert svc1.calls == 1
    c1.close()

    svc2 = _FakeService()
    c2 = ec.EmbeddingCache(path)
    asyncio.run(server._cached_query_embedder(svc2, c2, "m")("q"))
    assert svc2.calls == 0  # served from the persisted cache, no re-embed
    c2.close()


def test_cache_key_separates_models(tmp_path):
    # A model change must not serve a stale vector (key includes model_id).
    path = tmp_path / "embed_cache.db"
    cache = ec.EmbeddingCache(path)
    svc = _FakeService()
    asyncio.run(server._cached_query_embedder(svc, cache, "model-A")("q"))
    asyncio.run(server._cached_query_embedder(svc, cache, "model-B")("q"))
    assert svc.calls == 2  # different model_id -> different key -> fresh embed
    cache.close()


def test_embed_cache_config_defaults():
    cfg = simba.config.load("memory")
    assert cfg.embed_cache_enabled is True
    assert cfg.embed_cache_path == ""

"""Tests for the persistent content-hash embedding cache."""

from __future__ import annotations

import pathlib

import simba.memory.embedding_cache as ec


def test_key_is_stable_and_varies(tmp_path: pathlib.Path) -> None:
    k1 = ec.EmbeddingCache.key("m1", "doc: ", "hello")
    assert k1 == ec.EmbeddingCache.key("m1", "doc: ", "hello")  # stable
    assert k1 != ec.EmbeddingCache.key("m2", "doc: ", "hello")  # model matters
    assert k1 != ec.EmbeddingCache.key("m1", "query: ", "hello")  # prefix matters
    assert k1 != ec.EmbeddingCache.key("m1", "doc: ", "world")  # content matters


def test_get_miss_then_hit(tmp_path: pathlib.Path) -> None:
    cache = ec.EmbeddingCache(tmp_path / "emb.db")
    assert cache.get("m1", "doc: ", "hello") is None
    cache.put("m1", "doc: ", "hello", [0.1, 0.2, 0.3])
    assert cache.get("m1", "doc: ", "hello") == [0.1, 0.2, 0.3]


def test_persists_across_instances(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "emb.db"
    c1 = ec.EmbeddingCache(path)
    c1.put("m1", "", "x", [1.0, 2.0])
    c1.close()
    c2 = ec.EmbeddingCache(path)
    assert c2.get("m1", "", "x") == [1.0, 2.0]


def test_cached_embedder_wraps_and_skips_recompute(tmp_path: pathlib.Path) -> None:
    cache = ec.EmbeddingCache(tmp_path / "emb.db")
    calls: list[str] = []

    def raw(text: str) -> list[float]:
        calls.append(text)
        return [float(len(text))]

    embed = ec.cached_embedder(raw, cache, model_id="m1", prefix="doc: ")
    assert embed("hello") == [5.0]
    assert embed("hello") == [5.0]  # second call served from cache
    assert calls == ["hello"]  # raw embedder invoked once

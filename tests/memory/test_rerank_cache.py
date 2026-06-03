"""Tests for the in-process rerank cache (LRU, query+candidate-set keyed)."""

from __future__ import annotations

import simba.memory.rerank_cache as rc


def test_signature_is_order_insensitive_on_ids() -> None:
    c = rc.RerankCache()
    assert c.signature("q", ["a", "b", "c"]) == c.signature("q", ["c", "a", "b"])


def test_signature_depends_on_query() -> None:
    c = rc.RerankCache()
    assert c.signature("q1", ["a", "b"]) != c.signature("q2", ["a", "b"])


def test_signature_normalizes_query_whitespace_case() -> None:
    c = rc.RerankCache()
    assert c.signature("  Fix  GH ", ["a"]) == c.signature("fix gh", ["a"])


def test_get_miss_returns_none() -> None:
    assert rc.RerankCache().get("nope") is None


def test_put_then_get_roundtrip() -> None:
    c = rc.RerankCache()
    key = c.signature("q", ["a", "b"])
    c.put(key, ["b", "a"])
    assert c.get(key) == ["b", "a"]


def test_lru_eviction() -> None:
    c = rc.RerankCache(max_entries=2)
    c.put("k1", ["a"])
    c.put("k2", ["b"])
    c.put("k3", ["c"])  # evicts k1 (oldest)
    assert c.get("k1") is None
    assert c.get("k2") == ["b"]
    assert c.get("k3") == ["c"]


def test_get_refreshes_recency() -> None:
    c = rc.RerankCache(max_entries=2)
    c.put("k1", ["a"])
    c.put("k2", ["b"])
    c.get("k1")  # k1 now most-recent
    c.put("k3", ["c"])  # evicts k2, not k1
    assert c.get("k1") == ["a"]
    assert c.get("k2") is None

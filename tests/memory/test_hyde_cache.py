"""Tests for the in-process HyDE cache (LRU, query-keyed, str values)."""

from __future__ import annotations

import simba.memory.hyde_cache as hc


def test_cache_miss_returns_none() -> None:
    assert hc.HydeCache().get("k") is None


def test_cache_put_then_get_returns_text() -> None:
    c = hc.HydeCache()
    c.put("k", "hello")
    assert c.get("k") == "hello"


def test_cache_evicts_lru_when_full() -> None:
    c = hc.HydeCache(max_entries=2)
    c.put("a", "A")
    c.put("b", "B")
    c.put("c", "C")  # evicts a (oldest)
    assert c.get("a") is None
    assert c.get("b") == "B"
    assert c.get("c") == "C"


def test_signature_is_stable_across_whitespace() -> None:
    c = hc.HydeCache()
    assert c.signature("hello world") == c.signature("hello  world")
    assert c.signature("Hello World") == c.signature("hello world")


def test_get_promotes_to_mru() -> None:
    c = hc.HydeCache(max_entries=2)
    c.put("a", "A")
    c.put("b", "B")
    c.get("a")  # promote a to MRU
    c.put("c", "C")  # evicts b (LRU)
    assert c.get("a") == "A"
    assert c.get("b") is None

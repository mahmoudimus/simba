"""Tests for the persistent content-hash embedding cache."""

from __future__ import annotations

import itertools
import json
import logging
import pathlib
import sqlite3
import struct

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
    # Exactly representable in float32 so equality holds after the binary
    # roundtrip (see test_binary_roundtrip_float32_precision for the general
    # case).
    cache.put("m1", "doc: ", "hello", [0.5, 0.25, 0.125])
    assert cache.get("m1", "doc: ", "hello") == [0.5, 0.25, 0.125]


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


# --------------------------------------------------------------------------
# Binary storage (2026-07-18 bound-cache fix)
# --------------------------------------------------------------------------


def test_binary_roundtrip_float32_precision(tmp_path: pathlib.Path) -> None:
    """Vectors are packed as float32 BLOBs -- roundtrip is exact to float32 ULP,
    not float64 bit-identical."""
    cache = ec.EmbeddingCache(tmp_path / "emb.db")
    vector = [0.1, -0.2, 3.14159265, 1e-3, 123.456]
    cache.put("m1", "doc: ", "hello", vector)
    got = cache.get("m1", "doc: ", "hello")
    assert got is not None
    assert len(got) == len(vector)
    for expected, actual in zip(vector, got, strict=True):
        assert actual == struct.unpack("f", struct.pack("f", expected))[0]

    # Underlying column really is a BLOB, not the legacy TEXT/JSON encoding.
    row = cache._conn.execute(
        "SELECT vector FROM embeddings WHERE key = ?",
        (ec.EmbeddingCache.key("m1", "doc: ", "hello"),),
    ).fetchone()
    assert isinstance(row[0], bytes)
    cache.close()


def test_legacy_schema_reset_on_open(tmp_path: pathlib.Path, caplog) -> None:
    """A pre-existing legacy TEXT-vector cache is dropped and recreated, not
    migrated row-by-row -- it's a rebuildable cache, not memory data."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO embeddings (key, vector) VALUES (?, ?)",
        ("legacykey", json.dumps([0.1, 0.2, 0.3])),
    )
    conn.commit()
    conn.close()

    with caplog.at_level(logging.INFO, logger="simba.memory.embedding_cache"):
        cache = ec.EmbeddingCache(path)

    # Old row is gone -- the legacy table was dropped, not converted.
    row = cache._conn.execute(
        "SELECT * FROM embeddings WHERE key = ?", ("legacykey",)
    ).fetchone()
    assert row is None

    # New schema is usable.
    cache.put("m1", "", "x", [1.0])
    assert cache.get("m1", "", "x") == [1.0]

    messages = [r.message for r in caplog.records]
    assert any("reset" in m.lower() or "schema" in m.lower() for m in messages), (
        f"expected an INFO log about the schema reset, got: {messages}"
    )
    cache.close()


def test_lru_eviction_evicts_oldest_and_keeps_touched(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    clock = itertools.count()
    monkeypatch.setattr(ec, "_now", lambda: next(clock))

    cache = ec.EmbeddingCache(tmp_path / "emb.db", max_entries=10)
    for i in range(10):
        cache.put("m1", "", f"k{i}", [float(i)])

    # Touch k0 so its last_used becomes the most recent -- it must survive
    # eviction even though it was inserted first.
    assert cache.get("m1", "", "k0") == [0.0]

    for i in range(10, 15):
        cache.put("m1", "", f"k{i}", [float(i)])

    assert cache.get("m1", "", "k0") is not None  # touched -> survives
    assert cache.get("m1", "", "k1") is None  # untouched + oldest -> evicted
    cache.close()


def test_lru_eviction_batches_down_to_90pct_of_bound(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    clock = itertools.count()
    monkeypatch.setattr(ec, "_now", lambda: next(clock))

    cache = ec.EmbeddingCache(tmp_path / "emb.db", max_entries=10)
    for i in range(15):
        cache.put("m1", "", f"k{i}", [float(i)])

    count = cache._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count <= 9  # 90% of bound(10), batch eviction (not trickle-to-bound)
    cache.close()


def test_max_entries_zero_never_evicts(tmp_path: pathlib.Path) -> None:
    cache = ec.EmbeddingCache(tmp_path / "emb.db", max_entries=0)
    for i in range(300):
        cache.put("m1", "", f"k{i}", [float(i)])
    count = cache._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert count == 300
    cache.close()

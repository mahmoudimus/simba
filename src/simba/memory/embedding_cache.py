"""Persistent content-hash embedding cache.

Re-running a benchmark re-embeds identical corpora — the single biggest cost of
iterating on the eval. This caches ``sha1(model_id | prefix | content) -> vector``
on disk (sqlite) so reruns with the same model skip the GGUF embed entirely.

Append-only-safe: ``INSERT OR REPLACE`` keyed by content hash; nothing is ever
deleted, and a changed model/prefix simply lands under a new key.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import typing

EmbedFn = typing.Callable[[str], list[float]]

_SEP = "\x00"


class EmbeddingCache:
    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def key(model_id: str, prefix: str, content: str) -> str:
        h = hashlib.sha1()
        h.update(f"{model_id}{_SEP}{prefix}{_SEP}{content}".encode())
        return h.hexdigest()

    def get(self, model_id: str, prefix: str, content: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE key = ?",
            (self.key(model_id, prefix, content),),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self, model_id: str, prefix: str, content: str, vector: list[float]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            (self.key(model_id, prefix, content), json.dumps(vector)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def cached_embedder(
    raw: EmbedFn, cache: EmbeddingCache, *, model_id: str, prefix: str
) -> EmbedFn:
    """Wrap an embed function with read-through caching keyed by content."""

    def embed(text: str) -> list[float]:
        hit = cache.get(model_id, prefix, text)
        if hit is not None:
            return hit
        vector = raw(text)
        cache.put(model_id, prefix, text, vector)
        return vector

    return embed

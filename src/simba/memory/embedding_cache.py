"""Persistent content-hash embedding cache.

Re-running a benchmark re-embeds identical corpora — the single biggest cost of
iterating on the eval. This caches ``sha1(model_id | prefix | content) -> vector``
on disk (sqlite) so reruns with the same model skip the GGUF embed entirely.

Append-only-safe for that benchmark use: ``INSERT OR REPLACE`` keyed by content
hash; with ``max_entries=0`` (the default when constructed directly, e.g. by
the eval harness) nothing is ever deleted, and a changed model/prefix simply
lands under a new key.

The daemon (``simba.memory.server``) reuses this same cache for LIVE query
embeds, which is a different workload: raw prompts/tool-inputs/task
notifications are near-unbounded cardinality, not a fixed benchmark corpus.
Live 2026-07-18: an unbounded daemon cache grew to 1.3GB / 65,866 rows of
JSON-TEXT vectors with no eviction. So the daemon wiring passes an explicit
``max_entries`` bound (``memory.embed_cache_max_entries``, default 50000) and
this module LRU-evicts (by ``last_used``) in batches down to 90% of the bound
whenever a ``put`` pushes the row count past it. This cache is derived,
rebuildable data (like the FTS keyword mirror) -- pruning it is safe and, for
the daemon, the point.

Vectors are packed as float32 BLOBs (``array('f')``), not JSON text: ~4KB for
a 1024-dim vector instead of ~20KB of JSON, with no numpy dependency needed.
"""

from __future__ import annotations

import array
import hashlib
import logging
import pathlib
import sqlite3
import time
import typing

EmbedFn = typing.Callable[[str], list[float]]

_SEP = "\x00"

# Row-count check cadence for eviction. Checking on every ``put`` is wasteful
# once the bound is large (the production default is 50000), so the check
# only runs every Nth put -- except when the bound itself is small (<= this
# interval), in which case checking every put is both cheap (a tiny table)
# and necessary for the bound to actually hold at that scale.
_EVICT_CHECK_INTERVAL = 256

# Batch eviction target: when a ``put`` pushes the row count past the bound,
# evict the oldest-``last_used`` rows in one batch down to this fraction of
# the bound, rather than trickling evictions down to the bound itself.
_EVICT_TARGET_RATIO = 0.9

logger = logging.getLogger("simba.memory.embedding_cache")


def _now() -> float:
    """Wall-clock seconds, as a seam tests can monkeypatch for determinism."""
    return time.time()


def _pack(vector: list[float]) -> bytes:
    return array.array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return arr.tolist()


class EmbeddingCache:
    def __init__(self, path: str | pathlib.Path, *, max_entries: int = 0) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self._evict_every = (
            1 if 0 < max_entries <= _EVICT_CHECK_INTERVAL else _EVICT_CHECK_INTERVAL
        )
        self._put_count = 0

        is_new = not self._path.exists()
        self._conn = sqlite3.connect(str(self._path))
        if is_new:
            # Only takes effect on an empty database -- exactly the case here.
            self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            self._create_schema()
        else:
            self._migrate_if_legacy()
            self._create_schema()  # no-op (IF NOT EXISTS) once migrated/current
        self._conn.commit()

    def _create_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(key TEXT PRIMARY KEY, vector BLOB NOT NULL, last_used REAL NOT NULL)"
        )

    def _migrate_if_legacy(self) -> None:
        """Detect the pre-2026-07-18 TEXT/JSON-vector schema and reset it.

        This cache is rebuildable, derived data -- the simplest correct
        migration is to drop the legacy table and start fresh rather than
        convert up to 65k JSON rows row-by-row at daemon startup.
        """
        exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone()
        if exists is None:
            return

        cols = self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
        col_types = {row[1]: (row[2] or "").upper() for row in cols}
        is_legacy = col_types.get("vector") == "TEXT" or "last_used" not in col_types
        if not is_legacy:
            return

        old_rows = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        old_bytes = self._path.stat().st_size
        logger.info(
            "[embed-cache] schema reset for upgrade: legacy TEXT-vector cache "
            "(%d rows, %d bytes) dropped and recreated -- rebuildable, not "
            "migrated row-by-row",
            old_rows,
            old_bytes,
        )
        self._conn.execute("DROP TABLE embeddings")
        self._conn.commit()
        self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self._conn.execute("VACUUM")

    @staticmethod
    def key(model_id: str, prefix: str, content: str) -> str:
        h = hashlib.sha1()
        h.update(f"{model_id}{_SEP}{prefix}{_SEP}{content}".encode())
        return h.hexdigest()

    def get(self, model_id: str, prefix: str, content: str) -> list[float] | None:
        k = self.key(model_id, prefix, content)
        row = self._conn.execute(
            "SELECT vector FROM embeddings WHERE key = ?", (k,)
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE embeddings SET last_used = ? WHERE key = ?", (_now(), k)
        )
        self._conn.commit()
        return _unpack(row[0])

    def put(
        self, model_id: str, prefix: str, content: str, vector: list[float]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector, last_used) "
            "VALUES (?, ?, ?)",
            (self.key(model_id, prefix, content), _pack(vector), _now()),
        )
        self._conn.commit()
        if self._max_entries > 0:
            self._put_count += 1
            if self._put_count % self._evict_every == 0:
                self._maybe_evict()

    def _maybe_evict(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if count <= self._max_entries:
            return
        target = int(self._max_entries * _EVICT_TARGET_RATIO)
        to_delete = count - target
        self._conn.execute(
            "DELETE FROM embeddings WHERE key IN ("
            "SELECT key FROM embeddings ORDER BY last_used ASC LIMIT ?)",
            (to_delete,),
        )
        self._conn.commit()
        self._conn.execute("PRAGMA incremental_vacuum")
        self._conn.commit()
        logger.info(
            "[embed-cache] evicted %d entries (%d -> %d, bound %d)",
            to_delete,
            count,
            count - to_delete,
            self._max_entries,
        )

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

"""Short-TTL recall result cache (daemon, per-process).

Collapses identical-query storms — multi-runtime hooks (claude-code + its daemon
loopback + codex/pi), or a reasoning-verify / conflict-surfacing loop recalling
the same text repeatedly — that would otherwise each re-run search AND the
cross-encoder rerank (which serializes on the process-global ``LLAMA_LOCK``),
backing the queue up to tens of seconds. The first recall does the real work;
identical recalls within ``ttl_seconds`` return its result instantly.

Keyed on the full recall input. Entries expire after a few seconds so results
never go stale across turns. Fail-open: any miss/expiry just recomputes. LRU
(OrderedDict) bounded, mirroring RerankCache / HydeCache.
"""

from __future__ import annotations

import collections
import hashlib
import json
import typing


class RecallCache:
    def __init__(self, max_entries: int = 256, ttl_seconds: float = 5.0) -> None:
        self._max = max(1, max_entries)
        self._ttl = ttl_seconds
        self._d: collections.OrderedDict[str, tuple[float, list]] = (
            collections.OrderedDict()
        )

    @staticmethod
    def key(
        *,
        query: str,
        project_path: str | None,
        min_similarity: float | None,
        max_results: int | None,
        filters: dict[str, typing.Any],
        project_scopes: list[str] | None,
    ) -> str:
        """Stable hash of every input that determines the recall result."""
        payload = json.dumps(
            [query, project_path, min_similarity, max_results, filters, project_scopes],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(payload.encode()).hexdigest()

    def get(self, key: str, *, now: float) -> list | None:
        """Return cached memories if present and unexpired, else ``None``."""
        if self._ttl <= 0:
            return None
        hit = self._d.get(key)
        if hit is None:
            return None
        ts, value = hit
        if now - ts > self._ttl:
            del self._d[key]
            return None
        self._d.move_to_end(key)
        return value

    def put(self, key: str, value: list, *, now: float) -> None:
        """Store ``value`` for ``key`` stamped at ``now`` (LRU-evicting)."""
        if self._ttl <= 0:
            return
        self._d[key] = (now, value)
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached recalls (call when the corpus changes, e.g. a store)."""
        self._d.clear()

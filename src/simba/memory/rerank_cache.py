"""In-process LRU cache for LLM rerank results (daemon hot-path optimization).

Reranking is read-time and query-specific, so it can't be precomputed in the
background like sync extraction. Instead the daemon serves the fast RRF+composite
order immediately and reranks *off the hot path*, caching the result keyed by
(normalized query, candidate set) so a recurring query/candidate-set is served
the reranked order instantly. Ephemeral by design (per daemon process).
"""

from __future__ import annotations

import collections
import hashlib
import re

_WS = re.compile(r"\s+")


class RerankCache:
    def __init__(self, max_entries: int = 256) -> None:
        self._max = max(1, max_entries)
        self._d: collections.OrderedDict[str, list[str]] = collections.OrderedDict()

    def signature(self, query: str, candidate_ids: list[str]) -> str:
        """Stable key from the normalized query + the (order-insensitive) ids."""
        q = _WS.sub(" ", (query or "").strip().lower())
        ids = ",".join(sorted(str(i) for i in candidate_ids))
        return hashlib.sha1(f"{q}|{ids}".encode()).hexdigest()

    def get(self, key: str) -> list[str] | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, order: list[str]) -> None:
        self._d[key] = [str(i) for i in order]
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

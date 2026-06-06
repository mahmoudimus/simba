"""In-process LRU cache for LLM HyDE answers (daemon hot-path optimization).

HyDE generation is read-time and query-specific, so it can't be precomputed in
the background like sync extraction. Instead the daemon serves the keyword
fallback immediately and generates the hypothetical answer *off the hot path*,
caching it keyed by the normalized query so a recurring query gets the HyDE text
free. Ephemeral by design (per daemon process). Structural twin of
``RerankCache`` with a ``str`` value type and a query-only signature.
"""

from __future__ import annotations

import collections
import hashlib
import re

_WS = re.compile(r"\s+")


class HydeCache:
    def __init__(self, max_entries: int = 256) -> None:
        self._max = max(1, max_entries)
        self._d: collections.OrderedDict[str, str] = collections.OrderedDict()

    def signature(self, query: str) -> str:
        """SHA-1 of the whitespace-normalised lowercase query."""
        q = _WS.sub(" ", (query or "").strip().lower())
        return hashlib.sha1(q.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, text: str) -> None:
        self._d[key] = str(text)
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

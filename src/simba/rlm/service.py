"""Composition layer — load-then-search operations returning serializable dicts.

This is what the MCP tools call.  Each operation that targets a transcript
loads it lazily (LRU) and returns either a result dict or an ``{"error": ...}``
dict — it never raises into the tool layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba.config
import simba.rlm.config  # registers the "rlm" section
import simba.rlm.context
import simba.rlm.recall
import simba.rlm.transcripts

if TYPE_CHECKING:
    import pathlib


class RlmService:
    def __init__(self, transcripts_root: pathlib.Path | None = None) -> None:
        self._cfg = simba.config.load("rlm")
        self._provider = simba.rlm.transcripts.TranscriptProvider(
            self._cfg, root=transcripts_root
        )
        self._searcher = simba.rlm.context.DocumentSearcher(
            self._provider.store, self._cfg
        )

    def _ensure(self, transcript_id: str) -> str | None:
        """Load the transcript; return an error string or None on success."""
        try:
            self._provider.load(transcript_id)
            return None
        except simba.rlm.context.DocumentNotFoundError:
            return f"transcript not available: {transcript_id}"

    def grep(self, transcript_id, pattern, max_matches=None) -> dict:
        err = self._ensure(transcript_id)
        if err:
            return {"error": err}
        try:
            matches = self._searcher.grep(transcript_id, pattern, max_matches)
        except simba.rlm.context.SearchError as exc:
            return {"error": str(exc)}
        return {"matches": [m.to_dict() for m in matches]}

    def peek(self, transcript_id, start_char, end_char) -> dict:
        err = self._ensure(transcript_id)
        if err:
            return {"error": err}
        return {"text": self._searcher.peek(transcript_id, start_char, end_char)}

    def window(self, transcript_id, around_char, radius=None) -> dict:
        err = self._ensure(transcript_id)
        if err:
            return {"error": err}
        if radius is None:
            radius = self._cfg.search_context_chars * 4
        return {"text": self._searcher.window(transcript_id, around_char, radius)}

    def head(self, transcript_id, n_lines=20) -> dict:
        err = self._ensure(transcript_id)
        if err:
            return {"error": err}
        return {"text": self._searcher.head(transcript_id, n_lines)}

    def tail(self, transcript_id, n_lines=20) -> dict:
        err = self._ensure(transcript_id)
        if err:
            return {"error": err}
        return {"text": self._searcher.tail(transcript_id, n_lines)}

    def recall(self, query, cwd, max_pointers=None) -> dict:
        pointers = simba.rlm.recall.route(
            query, cwd, max_pointers=max_pointers, provider=self._provider
        )
        return {"pointers": [p.to_dict() for p in pointers]}


_SERVICE: RlmService | None = None


def get_service() -> RlmService:
    """Process-level cached service (used by the MCP tools)."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RlmService()
    return _SERVICE

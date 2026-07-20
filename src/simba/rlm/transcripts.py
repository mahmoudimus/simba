"""Lazy, LRU-bounded loading of PreCompact-exported session transcripts."""

from __future__ import annotations

import collections
import pathlib

import simba.rlm.context

_TRANSCRIPTS_ROOT = pathlib.Path.home() / ".claude" / "transcripts"


class TranscriptProvider:
    """Resolve a transcript_id (== session id) to its exported text, cached LRU."""

    def __init__(self, cfg, root: pathlib.Path | None = None) -> None:
        self._cfg = cfg
        self._root = root or _TRANSCRIPTS_ROOT
        self._store = simba.rlm.context.DocumentStore(
            max_document_mb=getattr(cfg, "max_document_mb", 64.0),
            store_budget_mb=getattr(cfg, "store_budget_mb", 256.0),
        )
        self._lru: collections.OrderedDict[str, None] = collections.OrderedDict()

    @property
    def store(self) -> simba.rlm.context.DocumentStore:
        return self._store

    def path_for(self, transcript_id: str) -> pathlib.Path | None:
        name = (
            "transcript.md"
            if self._cfg.transcript_source == "md"
            else "transcript.jsonl"
        )
        path = self._root / transcript_id / name
        return path if path.is_file() else None

    def available(self, transcript_id: str) -> bool:
        return self.path_for(transcript_id) is not None

    def load(self, transcript_id: str) -> str:
        """Ensure the transcript is resident; return its id. Raises if missing."""
        if self._store.has(transcript_id):
            self._lru.move_to_end(transcript_id)
            return transcript_id
        path = self.path_for(transcript_id)
        if path is None:
            raise simba.rlm.context.DocumentNotFoundError(transcript_id)
        # add_path() -- not add(path.read_text(...)) -- so a huge transcript is
        # never read into memory whole; see the 2026-07-20 RSS incident note
        # in rlm/config.py (max_document_mb).
        self._store.add_path(transcript_id, path, errors="replace")
        self._lru[transcript_id] = None
        while len(self._lru) > self._cfg.lru_documents:
            evicted, _ = self._lru.popitem(last=False)
            self._store.remove(evicted)
        return transcript_id

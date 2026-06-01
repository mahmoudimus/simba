"""Recall router — map a query to candidate transcripts to navigate.

Reuses the existing project-scoped LanceDB recall (so it inherits the
leak-free, exact-project filtering); turns each hit's sessionSource into a
pointer into the full transcript.
"""

from __future__ import annotations

import contextlib
import dataclasses


@dataclasses.dataclass
class Pointer:
    snippet: str
    transcript_id: str | None
    project_path: str
    similarity: float
    available: bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def pointers_from_memories(memories, cwd, *, provider=None):
    """Turn recalled memory dicts into transcript pointers (no recall I/O).

    Lets a caller that already has recall results (e.g. the UserPromptSubmit
    hook) build pointers without issuing a second recall.
    """
    import simba.config
    import simba.rlm.config  # ensures the "rlm" section is registered
    import simba.rlm.transcripts

    if provider is None:
        provider = simba.rlm.transcripts.TranscriptProvider(simba.config.load("rlm"))

    pointers: list[Pointer] = []
    for m in memories:
        tid = m.get("sessionSource") or None
        pointers.append(
            Pointer(
                snippet=m.get("content", ""),
                transcript_id=tid,
                project_path=m.get("projectPath", "") or (cwd or ""),
                similarity=m.get("similarity", 0.0),
                available=bool(tid) and provider.available(tid),
            )
        )
    return pointers


def route(query, cwd, *, max_pointers=None, provider=None):
    """Recall for *query* (project-scoped to *cwd*) and return transcript pointers."""
    import simba.config
    import simba.hooks._memory_client
    import simba.rlm.config  # ensures the "rlm" section is registered

    cfg = simba.config.load("rlm")
    if max_pointers is None:
        max_pointers = cfg.default_max_pointers

    memories: list[dict] = []
    with contextlib.suppress(Exception):
        memories = simba.hooks._memory_client.recall_memories(
            query, project_path=cwd, max_results=max_pointers
        )
    return pointers_from_memories(memories, cwd, provider=provider)

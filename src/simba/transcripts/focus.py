"""Deterministic keyword-overlap scoring for ``/compact <text>`` focus steering.

``/compact <text>`` -- Claude Code's PreCompact hook receives that text as
``custom_instructions`` on stdin. ``hooks/pre_compact.py`` persists it (as
``compactFocus``, truncated) into the session's exported ``metadata.json`` and
forwards it to the detached transcript distiller as ``--focus``. Two
consumers then rank/reorder failure->fix arcs by relevance to that focus
text:

* ``hooks/session_start.py``'s compact-relay leg (``_compact_relay_arc_block``)
  ranks candidate arcs pulled from the ``failure_arc`` sidecar table.
* ``transcripts/distill.py``'s ``distill_transcript`` reorders the
  ``<failure-arcs>`` section of a freshly distilled transcript.

Both share this module so the matching is identical and trivially auditable:
plain lowercase word-token overlap, no LLM, no embeddings.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_MIN_TOKEN_LEN = 3


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens from *text*, dropping tokens shorter than 3 chars.

    Deterministic, in first-seen order (duplicates kept -- callers that want
    a set for membership tests build one themselves, e.g. ``set(tokenize(...))``).
    """
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LEN]


def score_overlap(focus_tokens: set[str], text: str) -> int:
    """Count of *focus_tokens* present (as a set -- repeats don't inflate the
    score) in *text*'s tokens. 0 when either side is empty."""
    if not focus_tokens or not text:
        return 0
    return len(focus_tokens & set(tokenize(text)))

"""Query-intent classification for recall (read-path, no LLM).

A query is "broad" (aggregation / history / exploration) or "precise" (point
fact).  Broad queries widen the cosine floor so more candidates reach RRF
fusion; precise queries keep the strict floor.  Pure and lexicon-driven, like
the keyword stop-word set — unit-testable, no dependencies.

Classification is marker-driven, *not* length-driven: thinking blocks are long
but usually point-seeking, so length alone must never push them to "broad".
"""

from __future__ import annotations

import re

# Whole-word markers that signal an aggregation / history / exploration query.
_BROAD_MARKERS = frozenset(
    {
        "all",
        "every",
        "everything",
        "list",
        "history",
        "historical",
        "ever",
        "always",
        "usually",
        "often",
        "tend",
        "tends",
        "summary",
        "summarize",
        "overview",
        "across",
        "compare",
        "comparison",
        "pattern",
        "patterns",
        "recurring",
        "general",
        "generally",
        "multiple",
        "various",
        "themes",
        "examples",
        "instances",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")


def classify(query: str) -> str:
    """Return "broad" if the query reads as aggregation/exploration, else "precise"."""
    words = {m.group().lower() for m in _WORD_RE.finditer(query or "")}
    return "broad" if words & _BROAD_MARKERS else "precise"

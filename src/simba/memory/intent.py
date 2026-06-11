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


# Counting an open class ("how many korean restaurants") is recall-BREADTH-bound: it
# needs every member, so it wants a wide candidate pool. Excluded: temporal
# ("how many days between") and frequency ("how many times a week" = latest/state),
# which are NOT instance-counting and don't want breadth.
_COUNT_RE = re.compile(
    r"\bhow many\b|\bnumber of\b|\bhow much\b|\bcount of\b|\btotal (?:number|count)\b",
    re.IGNORECASE,
)
_COUNT_EXCLUDE_RE = re.compile(
    r"\bhow many (?:days?|weeks?|months?|years?|hours?|minutes?|times)\b",
    re.IGNORECASE,
)


def is_count(query: str) -> bool:
    """Whether ``query`` is an instance-counting question (recall-breadth-bound).

    True for "how many X / number of X / total number of X"; False for temporal
    span ("how many days between …") and frequency ("how many times a week …"),
    which are latest/state, not class-counting.
    """
    q = query or ""
    if _COUNT_EXCLUDE_RE.search(q):
        return False
    return _COUNT_RE.search(q) is not None

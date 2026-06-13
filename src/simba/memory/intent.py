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


# A knowledge-update / current-value query asks for the PRESENT value of a fact
# ("what is my X now / latest / most recent?"). Such a query retrieves BOTH the
# old and the new value, which the answer-time conflict detector flags as a
# "conflict" and then (wrongly) tells the answerer not to pick a side — when the
# correct behaviour is most-recent-wins. Recognising these by INTENT lets the
# conflict layer skip its directive (recency resolves them); the strict
# surfacing path stays on for everything else. Marker-driven, like the count and
# broad/precise lexicons — whole-word/phrase, case-insensitive, no LLM.
#
# Why intent and not date-disjointness: the ARM3 date-disjoint carve-out FAILED
# its SubtleMemory gate (0.722 < 0.9) — genuine *preference* conflicts ("which
# do I prefer") are ALSO date-disjoint, so date-disjointness can't discriminate
# update-vs-conflict. The discriminator is the query asking for the *current*
# value, not the dates on the memories.
_CURRENT_VALUE_RE = re.compile(
    r"\b(?:current|currently|latest|most recent|now|nowadays|presently|"
    r"these days|right now|at present|still)\b|\bas of (?:now|today)\b",
    re.IGNORECASE,
)


# Multi-session aggregation that the count predicate does NOT catch: questions that
# sum or span events spread ACROSS sessions ("how many days … in total", "how often",
# "across all the events", "list all", "every time"). Measured on LongMemEval-S:
# these are recall-BREADTH-bound exactly like counting — multi-session evidence sets
# reach complete@80 = 0.90 vs complete@20 = 0.33, and widening the answer context
# k=20 -> k=80 lifted the multi-session category 0.557 -> 0.686 (+0.13). Be
# conservative — firing widens a costlier retrieval — so the markers are precise
# multi-session/span shapes, and the bounded "total/average/difference of X and Y"
# arithmetic over two named items (pointwise, not breadth) is deliberately NOT
# matched (it carries no span/recurrence marker).
_AGGREGATION_RE = re.compile(
    r"\bin total\b"  # summation adverbial over an open set of events
    r"|\bhow often\b"  # frequency over history
    r"|\bacross (?:all|the|my|every|both)\b"  # spanning multiple sessions/events
    r"|\bthroughout\b"
    r"|\bevery time\b"  # per-occurrence recurrence
    r"|\blist all\b"  # explicit enumeration
    r"|\ball (?:the|my) \w+s\b"  # "all the events / all my trips" enumeration
    r"|\b(?:over|in|during) the (?:past|last) (?:year|month|week|day|few|couple)"
    r"|\bwhich \w+s\b(?=.*\bdid I\b)",  # "which <plural> … did I" enumeration shape
    re.IGNORECASE,
)
# Aggregation exclude — narrower than the count exclude on purpose. Only the
# frequency-RATE shape ("how many times a week", "how often per day") and the
# temporal-SPAN shape ("how many days between …") are latest/state, not breadth.
# Note: "how many times … across all the events / in the past two weeks" is NOT
# excluded here (it IS a cross-session aggregation) even though the count predicate
# drops it — the two predicates answer different questions about the same string.
_AGGREGATION_EXCLUDE_RE = re.compile(
    r"\b(?:how many|how often)\b.*\b(?:a|per)\s+(?:week|day|month|year|night|hour)\b"
    r"|\bhow many (?:days?|weeks?|months?|years?|hours?|minutes?)\b.*\bbetween\b",
    re.IGNORECASE,
)


def is_knowledge_update(query: str) -> bool:
    """Whether ``query`` asks for the CURRENT value of a fact (most-recent-wins).

    True for current-value markers — "current / currently / latest / most recent
    / now / nowadays / presently / these days / right now / at present / still /
    as of now". False otherwise, so genuine simultaneous-conflict / preference
    questions ("which do I prefer", "cats or dogs") stay on the strict conflict
    surfacing path. Substrings of a marker do not trigger (whole-word).
    """
    return _CURRENT_VALUE_RE.search(query or "") is not None


def is_aggregation(query: str) -> bool:
    """Whether ``query`` is a multi-session / aggregation question (breadth-bound).

    True for span/total/frequency questions that gather evidence across sessions
    ("… in total", "how often", "across all the events", "list all", "every time",
    "which <plural> … did I"); False for bounded arithmetic over two named items
    ("total cost of X and Y"), for the frequency-rate shape ("how many times a
    week"), and for temporal spans ("how many days between …"). Conservative by
    design — it gates a wider, costlier retrieval, and ``is_count`` takes
    precedence in :func:`plan_recall` when a query matches both.
    """
    q = query or ""
    if _AGGREGATION_EXCLUDE_RE.search(q):
        return False
    return _AGGREGATION_RE.search(q) is not None

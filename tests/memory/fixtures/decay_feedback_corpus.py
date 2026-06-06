"""Deterministic decay/feedback corpus for ranking-semantics eval tests.

An in-module factory (not a JSON file) so the eval tests stay pure: no file I/O,
no LanceDB, no daemon, no embedding. ``make_corpus(now)`` returns five entries
with a known expected strength ordering under the default decay parameters
(half_life=30, reinforcement_scale=0.5, feedback_weight=0.2).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class CorpusEntry:
    memory_id: str
    created_at_epoch: float  # epoch seconds
    access_count: int
    feedback_score: float
    label: str  # human label for assertions ("strong", "weak", ...)


def make_corpus(now: float) -> list[CorpusEntry]:
    """Return a fixed corpus of 5 entries with a known expected ranking order."""
    day = 86400.0
    return [
        CorpusEntry("mem_fresh", now - 1 * day, 0, 0.0, "strong"),
        CorpusEntry("mem_accessed", now - 30 * day, 5, 0.0, "strong"),
        CorpusEntry("mem_loved", now - 60 * day, 1, 1.0, "strong"),
        CorpusEntry("mem_old", now - 90 * day, 0, 0.0, "weak"),
        CorpusEntry("mem_hated", now - 10 * day, 0, -1.0, "weak"),
    ]

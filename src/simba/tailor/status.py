"""Error analysis/statistics functions for reflections data.

Ported from claude-tailor/test/nano-status.test.js logic.
"""

from __future__ import annotations


def count_by_error_type(reflections: list[dict]) -> dict[str, int]:
    """Count reflections grouped by error_type."""
    counts: dict[str, int] = {}
    for r in reflections:
        t = r.get("error_type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def get_sorted_counts(counts: dict[str, int]) -> list[tuple[str, int]]:
    """Sort error type counts by frequency (descending)."""
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def get_top_n(sorted_counts: list[tuple[str, int]], n: int) -> list[tuple[str, int]]:
    """Return top N entries from sorted counts."""
    return sorted_counts[:n]


def analyze_co_occurrence(reflections: list[dict]) -> dict[str, int]:
    """Analyze sequential co-occurrence of error types."""
    co: dict[str, int] = {}
    for i in range(len(reflections) - 1):
        pair = sorted([reflections[i]["error_type"], reflections[i + 1]["error_type"]])
        key = "|".join(pair)
        co[key] = co.get(key, 0) + 1
    return co


def get_top_co_occurrences(
    co_occurrence: dict[str, int], n: int
) -> list[tuple[str, int]]:
    """Return top N co-occurrence pairs with count > 1."""
    filtered = [(k, v) for k, v in co_occurrence.items() if v > 1]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:n]


def get_last_reflection(reflections: list[dict]) -> dict | None:
    """Return the last reflection in the list, or None if empty."""
    return reflections[-1] if reflections else None

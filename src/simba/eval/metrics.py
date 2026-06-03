"""Information-retrieval metrics for the eval harness (pure functions).

Each takes a ranked list of result ids (best first), the set of relevant ids,
and (where applicable) a cutoff ``k``. Binary relevance throughout. All return
0.0 for degenerate inputs (empty relevant set, ``k <= 0``) so they aggregate
cleanly without special-casing at the call site.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _relset(relevant: Iterable[str]) -> set[str]:
    return relevant if isinstance(relevant, set) else set(relevant)


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant items found in the top ``k``."""
    rel = _relset(relevant)
    if not rel or k <= 0:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in rel)
    return hits / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    rel = _relset(relevant)
    if not rel or k <= 0:
        return 0.0
    hits = sum(1 for r in ranked[:k] if r in rel)
    return hits / k


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant item is in the top ``k``, else 0.0."""
    rel = _relset(relevant)
    if not rel or k <= 0:
        return 0.0
    return 1.0 if any(r in rel for r in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / (1-indexed rank of the first relevant item); 0.0 if none found."""
    rel = _relset(relevant)
    if not rel:
        return 0.0
    for i, r in enumerate(ranked, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` (binary gains)."""
    rel = _relset(relevant)
    if not rel or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, r in enumerate(ranked[:k], start=1)
        if r in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0

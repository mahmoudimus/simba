"""Deterministic dev/test splitting of eval cases.

A held-out split keeps tuning (on dev) from silently overfitting the number you
report (on test). The split is a stable hash of the case id — no RNG, no state,
reproducible across runs and machines — unless a case pins its own ``split``.
"""

from __future__ import annotations

import hashlib
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from simba.eval.dataset import EvalCase


def effective_split(case: EvalCase, *, test_ratio: float = 0.5) -> str:
    """Return the case's split: its explicit one, else a stable hash bucket."""
    if case.split in ("dev", "test"):
        return case.split
    digest = hashlib.sha1(case.id.encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if frac < test_ratio else "dev"


def select(
    cases: Iterable[EvalCase],
    split: str | None,
    *,
    test_ratio: float = 0.5,
) -> list[EvalCase]:
    """Filter cases to a split; ``None``/``""`` returns all."""
    cases = list(cases)
    if not split:
        return cases
    return [c for c in cases if effective_split(c, test_ratio=test_ratio) == split]

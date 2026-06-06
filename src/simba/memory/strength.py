"""Pure, deterministic strength model for memory decay / reinforcement.

``compute_strength`` maps a memory's age, access count and feedback score onto a
strength in ``[0.0, 1.0]``.  The model is the spaced-repetition intuition from
Generative Agents: pure time decay provides a floor, each access pulls strength
back toward 1.0, and outcome feedback applies a final multiplicative nudge.

No side effects, no I/O, no clock reads — ``now`` is always a parameter so the
scheduler and the tests stay deterministic.  This module imports only stdlib.
"""

from __future__ import annotations

import math

_SECONDS_PER_DAY = 86400.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def decay_factor(age_days: float, half_life: float) -> float:
    """Exponential time decay: ``0.5 ** (age_days / half_life)``.

    Returns ``1.0`` when ``half_life <= 0`` (decay disabled). ``age_days`` is
    clamped to ``>= 0`` so a future ``created_at`` never inflates strength.
    """
    if half_life <= 0:
        return 1.0
    age_days = max(0.0, age_days)
    return 0.5 ** (age_days / half_life)


def reinforcement_factor(access_count: int, scale: float) -> float:
    """Logistic reinforcement in ``[0, 1)``: ``1 - exp(-access_count / scale)``.

    Returns ``0.0`` when ``access_count == 0`` (never accessed). When
    ``scale <= 0`` and there is at least one access, returns ``1.0`` (treat as
    "always reinforced").
    """
    if access_count == 0:
        return 0.0
    if scale <= 0:
        return 1.0
    return 1.0 - math.exp(-access_count / scale)


def compute_strength(
    *,
    created_at_epoch: float,
    now: float,
    access_count: int,
    feedback_score: float,
    half_life: float,
    reinforcement_scale: float,
    feedback_weight: float,
) -> float:
    """Combine decay + reinforcement + feedback into a strength in ``[0.0, 1.0]``.

    Formula::

        age_days = (now - created_at_epoch) / 86400
        d = decay_factor(age_days, half_life)          # pure time decay
        r = reinforcement_factor(access_count, scale)  # [0, 1), 0 if never accessed
        base = d + (1 - d) * r                          # r pulls base toward 1.0
        feedback_term = 1 + feedback_weight * clamp(feedback_score, -1, 1)
        raw = base * feedback_term
        return clamp(raw, 0.0, 1.0)

    With 0 accesses ``base == d``; with many accesses ``base → 1.0`` regardless
    of age (the spaced-repetition lift). ``feedback_weight=0.2`` and
    ``feedback_score=+1.0`` lifts by 20%; ``-1.0`` cuts by 20%.
    """
    age_days = (now - created_at_epoch) / _SECONDS_PER_DAY
    d = decay_factor(age_days, half_life)
    r = reinforcement_factor(access_count, reinforcement_scale)
    base = d + (1.0 - d) * r
    feedback_term = 1.0 + feedback_weight * _clamp(feedback_score, -1.0, 1.0)
    raw = base * feedback_term
    return _clamp(raw, 0.0, 1.0)

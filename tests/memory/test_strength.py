"""Tests for the pure strength model (src/simba/memory/strength.py)."""

from __future__ import annotations

from simba.memory.strength import (
    compute_strength,
    decay_factor,
    reinforcement_factor,
)

_DAY = 86400.0


def test_decay_factor_at_zero_age() -> None:
    assert decay_factor(0.0, 30.0) == 1.0


def test_decay_factor_at_half_life() -> None:
    assert abs(decay_factor(30.0, 30.0) - 0.5) < 1e-9


def test_decay_factor_disabled_when_halflife_zero() -> None:
    assert decay_factor(100.0, 0.0) == 1.0


def test_reinforcement_zero_when_no_accesses() -> None:
    assert reinforcement_factor(0, 0.5) == 0.0


def test_reinforcement_approaches_one() -> None:
    assert reinforcement_factor(10, 0.5) > 0.999


def test_reinforcement_scale_zero_and_accesses() -> None:
    assert reinforcement_factor(1, 0.0) == 1.0


def test_compute_strength_brand_new() -> None:
    result = compute_strength(
        created_at_epoch=0.0,
        now=0.0,
        access_count=0,
        feedback_score=0.0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )
    assert result == 1.0


def test_compute_strength_after_one_half_life_no_access() -> None:
    result = compute_strength(
        created_at_epoch=0.0,
        now=30 * _DAY,
        access_count=0,
        feedback_score=0.0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )
    assert abs(result - 0.5) < 1e-6


def test_compute_strength_reinforcement_lifts_above_decay() -> None:
    result = compute_strength(
        created_at_epoch=0.0,
        now=30 * _DAY,
        access_count=5,
        feedback_score=0.0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )
    assert result > 0.5


def test_compute_strength_positive_feedback_lifts() -> None:
    kw = dict(
        created_at_epoch=0.0,
        now=30 * _DAY,
        access_count=0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )
    result_with = compute_strength(feedback_score=1.0, **kw)
    result_without = compute_strength(feedback_score=0.0, **kw)
    assert result_with > result_without


def test_compute_strength_negative_feedback_lowers() -> None:
    kw = dict(
        created_at_epoch=0.0,
        now=30 * _DAY,
        access_count=0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )
    result_neg = compute_strength(feedback_score=-1.0, **kw)
    result_neutral = compute_strength(feedback_score=0.0, **kw)
    assert result_neg < result_neutral


def test_compute_strength_clamps_to_zero() -> None:
    result = compute_strength(
        created_at_epoch=0.0,
        now=400 * _DAY,
        access_count=0,
        feedback_score=-1.0,
        half_life=1.0,
        reinforcement_scale=0.5,
        feedback_weight=1.0,
    )
    assert result >= 0.0


def test_compute_strength_clamps_to_one() -> None:
    result = compute_strength(
        created_at_epoch=0.0,
        now=0.0,
        access_count=50,
        feedback_score=1.0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )
    assert result == 1.0


# --- Arousal-modulated decay (default no-op) -------------------------------


def _arousal_kw() -> dict:
    return dict(
        created_at_epoch=0.0,
        now=30 * _DAY,  # one half-life of age, no access/feedback
        access_count=0,
        feedback_score=0.0,
        half_life=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
    )


def test_arousal_multiplier_default_is_noop() -> None:
    """Omitting the multiplier matches passing 1.0 exactly."""
    baseline = compute_strength(**_arousal_kw())
    explicit_one = compute_strength(arousal_decay_multiplier=1.0, **_arousal_kw())
    assert explicit_one == baseline


def test_arousal_multiplier_below_one_retains_longer() -> None:
    """High arousal (mult < 1.0) decays slower → higher strength than baseline."""
    baseline = compute_strength(**_arousal_kw())
    high_arousal = compute_strength(arousal_decay_multiplier=0.5, **_arousal_kw())
    assert high_arousal > baseline


def test_arousal_multiplier_above_one_decays_faster() -> None:
    """Low arousal (mult > 1.0) decays faster → lower strength than baseline."""
    baseline = compute_strength(**_arousal_kw())
    low_arousal = compute_strength(arousal_decay_multiplier=2.0, **_arousal_kw())
    assert low_arousal < baseline


def test_arousal_multiplier_stays_in_unit_range() -> None:
    """Modulated strength is still clamped to [0, 1]."""
    for mult in (0.1, 0.5, 1.0, 2.0, 3.0):
        result = compute_strength(arousal_decay_multiplier=mult, **_arousal_kw())
        assert 0.0 <= result <= 1.0

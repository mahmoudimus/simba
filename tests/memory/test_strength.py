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

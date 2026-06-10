"""Tests for asset + freshness policy (asset.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.workflow.asset as asset

T0 = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def test_stale_when_never_materialized():
    policy = asset.FreshnessPolicy(stale_after_events=5)
    assert asset.is_stale("a", policy, current_source_position=0, now=T0) is True


def test_stale_at_event_count_threshold():
    policy = asset.FreshnessPolicy(stale_after_events=5)
    asset.mark_materialized("a", source_position=10, now=T0)
    # 14 - 10 = 4 < 5 -> fresh
    assert asset.is_stale("a", policy, current_source_position=14, now=T0) is False
    # 15 - 10 = 5 >= 5 -> stale
    assert asset.is_stale("a", policy, current_source_position=15, now=T0) is True


def test_stale_at_time_threshold():
    policy = asset.FreshnessPolicy(stale_after_seconds=60)
    asset.mark_materialized("a", source_position=0, now=T0)
    fresh = "2026-01-01T00:00:30Z"  # 30s < 60s
    stale = "2026-01-01T00:01:00Z"  # 60s >= 60s
    assert asset.is_stale("a", policy, current_source_position=0, now=fresh) is False
    assert asset.is_stale("a", policy, current_source_position=0, now=stale) is True


def test_fresh_when_both_axes_satisfied():
    policy = asset.FreshnessPolicy(stale_after_events=100, stale_after_seconds=3600)
    asset.mark_materialized("a", source_position=5, now=T0)
    assert (
        asset.is_stale(
            "a",
            policy,
            current_source_position=10,
            now="2026-01-01T00:10:00Z",
        )
        is False
    )


def test_either_axis_triggers_staleness():
    policy = asset.FreshnessPolicy(stale_after_events=100, stale_after_seconds=60)
    asset.mark_materialized("a", source_position=5, now=T0)
    # event axis fresh (5 < 100) but time axis stale (120s >= 60s)
    assert (
        asset.is_stale(
            "a", policy, current_source_position=10, now="2026-01-01T00:02:00Z"
        )
        is True
    )


def test_mark_materialized_resets_both_axes():
    policy = asset.FreshnessPolicy(stale_after_events=5, stale_after_seconds=60)
    asset.mark_materialized("a", source_position=0, now=T0)
    assert (
        asset.is_stale(
            "a", policy, current_source_position=10, now="2026-01-01T00:02:00Z"
        )
        is True
    )
    # re-materialize -> both axes reset
    asset.mark_materialized("a", source_position=10, now="2026-01-01T00:02:00Z")
    assert (
        asset.is_stale(
            "a", policy, current_source_position=11, now="2026-01-01T00:02:30Z"
        )
        is False
    )


def test_no_policy_axes_never_stale_once_materialized():
    policy = asset.FreshnessPolicy()
    asset.mark_materialized("a", source_position=0, now=T0)
    assert (
        asset.is_stale(
            "a", policy, current_source_position=999, now="2030-01-01T00:00:00Z"
        )
        is False
    )

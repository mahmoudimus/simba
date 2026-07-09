"""Tests for the usage-influence term in composite_rescore (cognee borrow).

See src/simba/memory/scoring.py::composite_rescore and
memory.usage_influence_weight (src/simba/memory/config.py). Ships DATA-GATED
at 0.0 (default): a memory's use_count/noise_count history has to accumulate
before this lever can be measured, per the SoTA-lever rule.
"""

from __future__ import annotations

import datetime
import types
from unittest import mock

import simba.memory.config as mc
import simba.memory.scoring as scoring

_NOW = datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC).timestamp()


def _cfg(**kw):
    cfg = mc.MemoryConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _rec(rid, rrf, *, created="2026-06-01T00:00:00Z", confidence=0.85):
    return {
        "id": rid,
        "rrf_score": rrf,
        "createdAt": created,
        "confidence": confidence,
    }


def _usage(use_count=0, noise_count=0):
    return types.SimpleNamespace(use_count=use_count, noise_count=noise_count)


class TestZeroWeightShortCircuits:
    """weight <= 0 must be byte-identical to the pre-lever function: no usage
    reads, no score changes."""

    def test_matches_baseline_scores_with_usage_map_present(self) -> None:
        cfg_off = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_recency=0.5,
            score_weight_importance=0.3,
            score_weight_strength=0.0,
            usage_influence_weight=0.0,
        )
        records_with_map = [_rec("hi", 0.05), _rec("mid", 0.03), _rec("lo", 0.01)]
        records_no_map = [_rec("hi", 0.05), _rec("mid", 0.03), _rec("lo", 0.01)]

        # Extreme usage values that WOULD change the outcome if the gate were
        # broken (e.g. "lo" heavily used, "hi" heavily noised).
        usage_map = {
            "hi": _usage(use_count=0, noise_count=999),
            "mid": _usage(use_count=0, noise_count=0),
            "lo": _usage(use_count=999, noise_count=0),
        }

        with_map = scoring.composite_rescore(
            records_with_map, cfg=cfg_off, now=_NOW, usage_map=usage_map
        )
        without_map = scoring.composite_rescore(
            records_no_map, cfg=cfg_off, now=_NOW, usage_map=None
        )

        assert [r["id"] for r in with_map] == [r["id"] for r in without_map]
        assert [r["composite_score"] for r in with_map] == [
            r["composite_score"] for r in without_map
        ]

    def test_usage_map_get_is_never_called(self) -> None:
        cfg_off = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_recency=0.0,
            score_weight_importance=0.0,
            score_weight_strength=0.0,
            usage_influence_weight=0.0,
        )
        records = [_rec("a", 0.05), _rec("b", 0.02)]
        sentinel = mock.MagicMock()

        scoring.composite_rescore(records, cfg=cfg_off, now=_NOW, usage_map=sentinel)

        sentinel.get.assert_not_called()

    def test_negative_weight_also_short_circuits(self) -> None:
        cfg = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_strength=0.0,
            usage_influence_weight=-0.5,
        )
        sentinel = mock.MagicMock()
        records = [_rec("a", 0.05), _rec("b", 0.02)]
        scoring.composite_rescore(records, cfg=cfg, now=_NOW, usage_map=sentinel)
        sentinel.get.assert_not_called()


class TestPositiveWeightBlendsUsageSignal:
    def test_use_heavy_outranks_noise_heavy_at_equal_base_score(self) -> None:
        cfg = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_recency=0.0,
            score_weight_importance=0.0,
            score_weight_strength=0.0,
            usage_influence_weight=0.5,
        )
        # Equal rrf -> _normalize gives equal relevance -> equal base composite.
        records = [_rec("use_heavy", 0.5), _rec("noise_heavy", 0.5)]
        usage_map = {
            "use_heavy": _usage(use_count=10, noise_count=0),
            "noise_heavy": _usage(use_count=0, noise_count=10),
        }
        result = scoring.composite_rescore(
            records, cfg=cfg, now=_NOW, usage_map=usage_map
        )
        assert [r["id"] for r in result] == ["use_heavy", "noise_heavy"]
        assert result[0]["composite_score"] > result[1]["composite_score"]

    def test_ordering_monotonic_in_usage_signal(self) -> None:
        cfg = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_recency=0.0,
            score_weight_importance=0.0,
            score_weight_strength=0.0,
            usage_influence_weight=0.5,
        )
        records = [_rec("noisy", 0.5), _rec("neutral", 0.5), _rec("used", 0.5)]
        usage_map = {
            "noisy": _usage(use_count=0, noise_count=10),
            "used": _usage(use_count=10, noise_count=0),
            # "neutral" has no row at all -> defaults to a neutral 0.0 signal.
        }
        result = scoring.composite_rescore(
            records, cfg=cfg, now=_NOW, usage_map=usage_map
        )
        scores = {r["id"]: r["composite_score"] for r in result}
        assert scores["used"] > scores["neutral"] > scores["noisy"]
        assert [r["id"] for r in result] == ["used", "neutral", "noisy"]

    def test_blend_formula_is_linear_interpolation(self) -> None:
        # score = base*(1-w) + mapped_usage*w, mapped_usage = (signal+1)/2.
        cfg = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_recency=0.0,
            score_weight_importance=0.0,
            score_weight_strength=0.0,
            usage_influence_weight=0.5,
        )
        records = [_rec("hi", 0.05), _rec("lo", 0.01)]
        usage_map = {
            "hi": _usage(use_count=0, noise_count=10),  # signal -1 -> mapped 0.0
            "lo": _usage(use_count=10, noise_count=0),  # signal +1 -> mapped 1.0
        }
        result = scoring.composite_rescore(
            records, cfg=cfg, now=_NOW, usage_map=usage_map
        )
        scores = {r["id"]: r["composite_score"] for r in result}
        # base composite: hi=1.0 (normalized rel max), lo=0.0 (normalized rel min).
        # hi blended = 1.0*0.5 + 0.0*0.5 = 0.5; lo blended = 0.0*0.5 + 1.0*0.5 = 0.5.
        assert scores["hi"] == 0.5
        assert scores["lo"] == 0.5

    def test_missing_usage_row_defaults_neutral(self) -> None:
        cfg = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_recency=0.0,
            score_weight_importance=0.0,
            score_weight_strength=0.0,
            usage_influence_weight=1.0,
        )
        records = [_rec("solo", 0.5)]
        result = scoring.composite_rescore(records, cfg=cfg, now=_NOW, usage_map={})
        # base composite = 1.0 (single-record normalize), w=1.0 so the blend is
        # entirely the mapped usage signal; a missing row maps to neutral 0.5.
        assert result[0]["composite_score"] == 0.5

    def test_no_usage_map_argument_at_all_is_safe(self) -> None:
        # Positive weight but the caller passed no usage_map (None, the default) —
        # must not raise; behaves as if every row were neutral/absent.
        cfg = _cfg(
            scoring_enabled=True,
            score_weight_relevance=1.0,
            score_weight_strength=0.0,
            usage_influence_weight=0.5,
        )
        records = [_rec("a", 0.05), _rec("b", 0.02)]
        result = scoring.composite_rescore(records, cfg=cfg, now=_NOW)
        assert [r["id"] for r in result] == ["a", "b"]

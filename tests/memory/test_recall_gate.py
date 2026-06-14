"""Tests for the low-confidence rejection (abstention) gate.

Distinct from the per-result ``min_similarity`` floor: this judges the BEST
candidate and suppresses the WHOLE recall when even it is too weak — MemX-style
abstention to cut spurious recalls (LME ``_abs`` / over-retention).
"""
from __future__ import annotations

from simba.memory.scoring import apply_rejection_gate, truncate_to_budget


def _m(sim: float, rid: str = "x"):
    return {"id": rid, "similarity": sim}


class TestRejectionGate:
    def test_disabled_is_passthrough(self) -> None:
        mems = [_m(0.1), _m(0.05)]
        assert apply_rejection_gate(mems, enabled=False, threshold=0.9) == mems

    def test_empty_stays_empty(self) -> None:
        assert apply_rejection_gate([], enabled=True, threshold=0.5) == []

    def test_top_below_threshold_suppresses_whole_batch(self) -> None:
        mems = [_m(0.40), _m(0.38), _m(0.36)]  # all pass a 0.35 floor, top still weak
        assert apply_rejection_gate(mems, enabled=True, threshold=0.50) == []

    def test_top_at_or_above_threshold_keeps_all(self) -> None:
        mems = [_m(0.62), _m(0.41), _m(0.36)]
        assert apply_rejection_gate(mems, enabled=True, threshold=0.50) == mems

    def test_boundary_equal_is_kept(self) -> None:
        mems = [_m(0.50)]
        assert apply_rejection_gate(mems, enabled=True, threshold=0.50) == mems

    def test_missing_score_treated_as_zero(self) -> None:
        mems = [{"id": "x"}]  # no similarity key
        assert apply_rejection_gate(mems, enabled=True, threshold=0.50) == []

    def test_custom_score_key(self) -> None:
        mems = [{"id": "x", "rerank_score": 0.9}]
        assert apply_rejection_gate(
            mems, enabled=True, threshold=0.5, score_key="rerank_score") == mems

    def test_zero_threshold_keeps_everything_when_enabled(self) -> None:
        mems = [_m(0.0)]
        assert apply_rejection_gate(mems, enabled=True, threshold=0.0) == mems


def _r(content: str, rid: str = "x"):
    return {"id": rid, "content": content}


class TestTruncateToBudget:
    def test_budget_off_uses_fixed_max_results(self) -> None:
        recs = [_r("a" * 100, "1"), _r("b" * 100, "2"), _r("c" * 100, "3")]
        assert truncate_to_budget(recs, max_results=2, token_budget=0) == recs[:2]

    def test_budget_fits_a_prefix(self) -> None:
        # ~25 tokens each (100 chars / 4); budget 60 fits 2, not the 3rd
        recs = [_r("a" * 100, "1"), _r("b" * 100, "2"), _r("c" * 100, "3")]
        out = truncate_to_budget(recs, max_results=10, token_budget=60)
        assert [r["id"] for r in out] == ["1", "2"]

    def test_top_always_included_even_if_over_budget(self) -> None:
        recs = [_r("a" * 1000, "1"), _r("b" * 10, "2")]
        out = truncate_to_budget(recs, max_results=10, token_budget=5)
        assert [r["id"] for r in out] == ["1"]

    def test_large_budget_returns_all_above_max_results(self) -> None:
        # budget overrides the fixed-k cap: completeness when there's room
        recs = [_r("x" * 20, str(i)) for i in range(8)]
        out = truncate_to_budget(recs, max_results=3, token_budget=10_000)
        assert len(out) == 8

    def test_context_counts_toward_budget(self) -> None:
        recs = [{"id": "1", "content": "a" * 40, "context": "b" * 40},
                {"id": "2", "content": "c" * 40}]
        # rec1 ~20 tokens (80 chars), budget 15 -> only rec1 (top always in)
        out = truncate_to_budget(recs, max_results=10, token_budget=15)
        assert [r["id"] for r in out] == ["1"]

    def test_empty(self) -> None:
        assert truncate_to_budget([], max_results=3, token_budget=100) == []

"""Tests for eval IR metrics (pure functions)."""

from __future__ import annotations

import math

import pytest

import simba.eval.metrics as m

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "d"}


class TestBridgeRecallAtK:
    """All gold ids must be in top-k (multi-hop needs *every* hop, not any)."""

    def test_all_in_topk_is_one(self) -> None:
        assert m.bridge_recall_at_k(RANKED, RELEVANT, 5) == 1.0  # b,d both in top5

    def test_missing_one_hop_is_zero(self) -> None:
        assert m.bridge_recall_at_k(RANKED, RELEVANT, 3) == 0.0  # d at rank 4

    def test_single_relevant_behaves_like_recall(self) -> None:
        assert m.bridge_recall_at_k(RANKED, {"b"}, 2) == 1.0
        assert m.bridge_recall_at_k(RANKED, {"d"}, 2) == 0.0

    def test_empty_relevant_is_zero(self) -> None:
        assert m.bridge_recall_at_k(RANKED, set(), 5) == 0.0

    def test_k_zero_is_zero(self) -> None:
        assert m.bridge_recall_at_k(RANKED, RELEVANT, 0) == 0.0


class TestRecallAtK:
    def test_recall_at_1(self) -> None:
        assert m.recall_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_recall_at_3(self) -> None:
        assert m.recall_at_k(RANKED, RELEVANT, 3) == 0.5

    def test_recall_at_5(self) -> None:
        assert m.recall_at_k(RANKED, RELEVANT, 5) == 1.0

    def test_empty_relevant_is_zero(self) -> None:
        assert m.recall_at_k(RANKED, set(), 5) == 0.0

    def test_k_zero_is_zero(self) -> None:
        assert m.recall_at_k(RANKED, RELEVANT, 0) == 0.0


class TestPrecisionAtK:
    def test_precision_at_3(self) -> None:
        assert m.precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(1 / 3)

    def test_precision_at_5(self) -> None:
        assert m.precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.4)

    def test_k_zero_is_zero(self) -> None:
        assert m.precision_at_k(RANKED, RELEVANT, 0) == 0.0


class TestHitAtK:
    def test_miss_at_1(self) -> None:
        assert m.hit_at_k(RANKED, RELEVANT, 1) == 0.0

    def test_hit_at_3(self) -> None:
        assert m.hit_at_k(RANKED, RELEVANT, 3) == 1.0


class TestReciprocalRank:
    def test_first_relevant_at_rank_2(self) -> None:
        assert m.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)

    def test_no_relevant_is_zero(self) -> None:
        assert m.reciprocal_rank(RANKED, {"z"}) == 0.0


class TestNdcgAtK:
    def test_ndcg_at_5(self) -> None:
        # relevant at 1-indexed positions 2 and 4
        dcg = 1 / math.log2(2 + 1) + 1 / math.log2(4 + 1)
        idcg = 1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)
        assert m.ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)

    def test_perfect_ranking_is_one(self) -> None:
        assert m.ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0)

    def test_no_relevant_is_zero(self) -> None:
        assert m.ndcg_at_k(RANKED, set(), 5) == 0.0

"""Tests for memory vector_db module — cosine similarity."""

from __future__ import annotations

import math

import pytest

import simba.memory.vector_db


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert simba.memory.vector_db.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert simba.memory.vector_db.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert simba.memory.vector_db.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert simba.memory.vector_db.cosine_similarity([], []) == 0.0

    def test_none_vectors(self):
        assert simba.memory.vector_db.cosine_similarity(None, [1.0]) == 0.0  # type: ignore[arg-type]
        assert simba.memory.vector_db.cosine_similarity([1.0], None) == 0.0  # type: ignore[arg-type]

    def test_mismatched_lengths(self):
        assert simba.memory.vector_db.cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vectors(self):
        assert simba.memory.vector_db.cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_known_similarity(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # Manual computation
        dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
        norm_a = math.sqrt(1 + 4 + 9)  # sqrt(14)
        norm_b = math.sqrt(16 + 25 + 36)  # sqrt(77)
        expected = dot / (norm_a * norm_b)
        assert simba.memory.vector_db.cosine_similarity(a, b) == pytest.approx(
            expected, rel=1e-6
        )

    def test_high_dimensional(self):
        """768-dim vectors (like nomic-embed-text)."""
        import random

        random.seed(42)
        a = [random.random() for _ in range(768)]
        b = [random.random() for _ in range(768)]
        sim = simba.memory.vector_db.cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

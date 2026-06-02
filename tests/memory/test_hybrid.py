"""Tests for hybrid recall fusion + orchestration (src/simba/memory/hybrid.py)."""

from __future__ import annotations

import pathlib

import pytest

import simba.memory.config
import simba.memory.fts as fts
import simba.memory.hybrid as hybrid


def _vec(mid: str, sim: float) -> dict:
    return {
        "id": mid,
        "type": "PATTERN",
        "content": f"vector {mid}",
        "context": "",
        "similarity": sim,
        "confidence": 0.8,
        "createdAt": "t",
        "projectPath": "proj-1",
    }


def _kw(mid: str) -> dict:
    return {
        "memory_id": mid,
        "type": "GOTCHA",
        "content": f"keyword {mid}",
        "context": "",
        "confidence": 0.7,
        "createdAt": "t",
        "projectPath": "proj-1",
    }


class TestRrfFuse:
    def test_fuses_and_orders_by_combined_rank(self) -> None:
        # vector order [a, b]; keyword order [b, c].
        # scores (k=60): a=1/61, b=1/62+1/61, c=1/62  ->  b > a > c.
        fused = hybrid.rrf_fuse([_vec("a", 0.9), _vec("b", 0.8)], [_kw("b"), _kw("c")])
        assert [r["id"] for r in fused] == ["b", "a", "c"]

    def test_dedup_by_id_keeps_vector_record(self) -> None:
        fused = hybrid.rrf_fuse([_vec("x", 0.9)], [_kw("x")])
        assert len(fused) == 1
        # vector record wins -> real similarity preserved, content from vector.
        assert fused[0]["similarity"] == 0.9
        assert fused[0]["content"] == "vector x"

    def test_empty_keyword_is_vector_only(self) -> None:
        fused = hybrid.rrf_fuse([_vec("a", 0.9), _vec("b", 0.8)], [])
        assert [r["id"] for r in fused] == ["a", "b"]

    def test_keyword_weight_zero_ignores_keyword_ranking(self) -> None:
        # keyword_weight 0 -> ordering driven purely by the vector arm.
        fused = hybrid.rrf_fuse(
            [_vec("a", 0.9), _vec("b", 0.8)],
            [_kw("b"), _kw("c")],
            keyword_weight=0.0,
        )
        # c contributes 0 score but still appears (rank recorded); a,b lead.
        assert fused[0]["id"] == "a"
        assert fused[1]["id"] == "b"

    def test_keyword_only_record_shape(self) -> None:
        fused = hybrid.rrf_fuse([], [_kw("only")])
        assert fused[0]["id"] == "only"
        assert fused[0]["similarity"] == 0.0
        assert fused[0]["confidence"] == 0.7


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_merges_vector_and_keyword_arms(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        path = tmp_path / fts.FTS_FILENAME
        with fts.connect(path):
            fts.upsert(
                {
                    "id": "kw1",
                    "type": "GOTCHA",
                    "content": "the unique_zeta keyword token",
                    "context": "",
                    "confidence": 0.7,
                    "createdAt": "t",
                    "projectPath": "proj-1",
                },
            )

        async def fake_vec(table, emb, min_sim, max_res, filters):
            return [_vec("vec1", 0.9)]

        monkeypatch.setattr("simba.memory.vector_db.search_memories", fake_vec)

        cfg = simba.memory.config.MemoryConfig()
        results = await hybrid.hybrid_search(
            None,
            path,
            [0.1] * 768,
            "unique_zeta",
            min_similarity=0.35,
            max_results=5,
            filters={"projectPath": "proj-1"},
            cfg=cfg,
        )
        ids = {r["id"] for r in results}
        assert "vec1" in ids
        assert "kw1" in ids

    @pytest.mark.asyncio
    async def test_no_fts_path_falls_back_to_vector(self, monkeypatch) -> None:
        async def fake_vec(table, emb, min_sim, max_res, filters):
            return [_vec("vec1", 0.9), _vec("vec2", 0.8)]

        monkeypatch.setattr("simba.memory.vector_db.search_memories", fake_vec)
        cfg = simba.memory.config.MemoryConfig()
        results = await hybrid.hybrid_search(
            None,
            None,
            [0.1] * 768,
            "anything",
            min_similarity=0.35,
            max_results=5,
            filters={},
            cfg=cfg,
        )
        assert [r["id"] for r in results] == ["vec1", "vec2"]

"""Tests for folding the KG arm into hybrid recall (the C1 fusion wiring)."""

from __future__ import annotations

import pytest

import simba.memory.hybrid as hybrid
from simba.memory.config import MemoryConfig


def test_rrf_fuse_folds_kg_arm() -> None:
    vec = [{"id": "v1", "content": "seed"}]
    kg = [{"id": "k1", "content": "bridged"}]
    out = hybrid.rrf_fuse(vec, [], kg_results=kg)
    ids = [r["id"] for r in out]
    assert "v1" in ids and "k1" in ids


@pytest.mark.asyncio
async def test_hybrid_search_seeds_from_vectors_and_folds_kg(monkeypatch) -> None:
    async def fake_vec(table, emb, min_sim, pool, filters):
        return [{"id": "v1", "content": "seed", "similarity": 0.9}]

    monkeypatch.setattr("simba.memory.vector_db.search_memories", fake_vec)

    seen: dict[str, list[str]] = {}

    async def kg_arm(seed_ids: list[str]) -> list[dict[str, object]]:
        seen["seeds"] = seed_ids
        return [{"id": "bridged", "content": "multi-hop evidence"}]

    cfg = MemoryConfig(
        kg_recall_enabled=True,
        kg_recall_seed_top_n=5,
        scoring_enabled=False,
        llm_rerank_enabled=False,
    )
    out = await hybrid.hybrid_search(
        None, None, [0.1], "q",
        min_similarity=0.0, max_results=10, filters={}, cfg=cfg, kg_arm=kg_arm,
    )
    ids = [r["id"] for r in out]
    assert seen["seeds"] == ["v1"]  # seeded from the top vector hit
    assert "bridged" in ids  # KG-bridged memory folded into the result


@pytest.mark.asyncio
async def test_kg_arm_skipped_when_disabled(monkeypatch) -> None:
    async def fake_vec(table, emb, min_sim, pool, filters):
        return [{"id": "v1", "content": "seed", "similarity": 0.9}]

    monkeypatch.setattr("simba.memory.vector_db.search_memories", fake_vec)

    called = {"n": 0}

    async def kg_arm(seed_ids: list[str]) -> list[dict[str, object]]:
        called["n"] += 1
        return [{"id": "bridged", "content": "x"}]

    cfg = MemoryConfig(
        kg_recall_enabled=False, scoring_enabled=False, llm_rerank_enabled=False
    )
    out = await hybrid.hybrid_search(
        None, None, [0.1], "q",
        min_similarity=0.0, max_results=10, filters={}, cfg=cfg, kg_arm=kg_arm,
    )
    assert called["n"] == 0  # disabled -> kg_arm never invoked
    assert [r["id"] for r in out] == ["v1"]

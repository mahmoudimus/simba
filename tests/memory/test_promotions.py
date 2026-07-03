"""Promotion-candidate surface (spec 33 Phase 5).

A memory whose ledger shows real consumption (use_count >= min, noise ratio
below the cap, not dormant) graduates toward the rule layer. This surface is
read-only — the promotion itself stays human (`simba memory promote`).
"""

from __future__ import annotations

import pytest

import simba.db
import simba.memory.usage as usage


def _make_memory(memory_id: str, content: str) -> dict:
    return {
        "id": memory_id,
        "type": "GOTCHA",
        "content": content,
        "context": "",
        "tags": "[]",
        "confidence": 0.85,
        "sessionSource": "",
        "projectPath": "/proj",
        "createdAt": "2026-01-01T00:00:00Z",
        "lastAccessedAt": "2026-01-01T00:00:00Z",
        "accessCount": 0,
        "vector": [0.1] * 768,
    }


@pytest.mark.asyncio
async def test_candidates_require_uses_and_low_noise(
    async_client, lance_table, tmp_path
) -> None:
    await lance_table.add(
        [
            _make_memory("mem_hot", "always run the docker test runner"),
            _make_memory("mem_cold", "barely used fact"),
            _make_memory("mem_noisy", "surfaced but never helpful"),
            _make_memory("mem_dormant", "was demoted"),
        ]
    )
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_hot", 100.0, use=4, noise=1)  # ratio 0.25 ✓
        usage.bump_quality("mem_cold", 100.0, use=1)  # below min uses
        usage.bump_quality("mem_noisy", 100.0, use=4, noise=3)  # ratio 0.75 ✗
        usage.bump_quality("mem_dormant", 100.0, use=5)
        usage.set_dormant("mem_dormant", dormant=True)

    resp = await async_client.get("/promotions/candidates")
    assert resp.status_code == 200
    body = resp.json()
    assert [c["id"] for c in body["candidates"]] == ["mem_hot"]
    top = body["candidates"][0]
    assert top["useCount"] == 4
    assert top["content"] == "always run the docker test runner"
    assert top["type"] == "GOTCHA"
    assert body["total"] == 1


@pytest.mark.asyncio
async def test_candidates_sorted_and_limited(
    async_client, lance_table, tmp_path
) -> None:
    await lance_table.add(
        [
            _make_memory("mem_a", "fact a"),
            _make_memory("mem_b", "fact b"),
        ]
    )
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_a", 100.0, use=3)
        usage.bump_quality("mem_b", 100.0, use=9)

    resp = await async_client.get("/promotions/candidates", params={"limit": 1})
    body = resp.json()
    assert [c["id"] for c in body["candidates"]] == ["mem_b"]
    assert body["total"] == 2


@pytest.mark.asyncio
async def test_candidates_empty_ledger(async_client) -> None:
    resp = await async_client.get("/promotions/candidates")
    assert resp.status_code == 200
    assert resp.json()["candidates"] == []

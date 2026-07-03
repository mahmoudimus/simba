"""Daemon-side usage-signal tests (spec 33 Phase 0): match/inject split + ack.

``match`` = returned by search (daemon bumps at recall). ``inject`` = actually
placed into a hook's context — only the hook knows, so it acks via
``POST /recall/ack``. Before the split both were bumped on the same line,
making the ledger degenerate (match == inject always).
"""

from __future__ import annotations

import asyncio

import pytest

import simba.db
import simba.memory.usage as usage


def _make_memory(memory_id: str) -> dict:
    return {
        "id": memory_id,
        "type": "GOTCHA",
        "content": f"memory {memory_id}",
        "context": "",
        "tags": "[]",
        "confidence": 0.85,
        "sessionSource": "",
        "projectPath": "",
        "createdAt": "2025-01-01T00:00:00Z",
        "lastAccessedAt": "2025-01-01T00:00:00Z",
        "accessCount": 0,
        "vector": [0.1] * 768,
    }


async def _drain_background_tasks(timeout: float = 2.0) -> None:
    pending = [t for t in asyncio.all_tasks() if not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=timeout)


@pytest.mark.asyncio
async def test_recall_bumps_match_and_access_but_not_inject(
    async_client, lance_table, tmp_path
) -> None:
    await lance_table.add([_make_memory("mem_m")])
    resp = await async_client.post("/recall", json={"query": "memory"})
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 1
    await _drain_background_tasks()

    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_m"])["mem_m"]
    assert row.access_count == 1
    assert row.match_count == 1
    assert row.inject_count == 0


@pytest.mark.asyncio
async def test_recall_ack_bumps_inject(async_client, tmp_path) -> None:
    resp = await async_client.post(
        "/recall/ack", json={"ids": ["mem_i", "mem_j"]}
    )
    assert resp.status_code == 200
    assert resp.json()["acked"] == 2

    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_i", "mem_j"])
    assert rows["mem_i"].inject_count == 1
    assert rows["mem_j"].inject_count == 1
    assert rows["mem_i"].match_count == 0


@pytest.mark.asyncio
async def test_recall_ack_empty_is_noop(async_client) -> None:
    resp = await async_client.post("/recall/ack", json={"ids": []})
    assert resp.status_code == 200
    assert resp.json()["acked"] == 0


def test_bump_quality_use_sets_last_used(tmp_path) -> None:
    """A ``use`` signal stamps ``last_used`` — the freshness rules (rule TTL
    refresh, spec 33 Phase 2) key off consumption, not retrieval."""
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_u", 123.0, use=1)
        row = usage.get_many(["mem_u"])["mem_u"]
    assert row.last_used == 123.0


def test_bump_quality_noise_leaves_last_used(tmp_path) -> None:
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_n", 123.0, noise=1)
        row = usage.get_many(["mem_n"])["mem_n"]
    assert row.last_used == 0.0

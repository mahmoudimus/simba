"""Tests for re-embedding the LanceDB table (embedder swap → new dim)."""

from __future__ import annotations

import pathlib

import pytest

import simba.memory.vector_db as vdb


async def _make_table(db_path: pathlib.Path):
    import lancedb

    db = await lancedb.connect_async(str(db_path))
    rows = [
        {
            "id": "m1",
            "type": "GOTCHA",
            "content": "gh 401",
            "context": "token",
            "vector": [0.1, 0.2, 0.3],
        },
        {
            "id": "m2",
            "type": "PATTERN",
            "content": "rrf fusion",
            "context": "",
            "vector": [0.4, 0.5, 0.6],
        },
    ]
    await db.create_table("memories", rows)
    return db


@pytest.mark.asyncio
async def test_reembed_rewrites_table_at_new_dim(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "memories.lance"
    await _make_table(db_path)

    seen = []

    async def fake_embed(text: str) -> list[float]:
        seen.append(text)
        return [1.0, 2.0, 3.0, 4.0]  # new 4-dim model

    table, count = await vdb.reembed_table(db_path, fake_embed)
    assert count == 2
    rows = await table.query().to_list()
    by_id = {r["id"]: r for r in rows}
    assert len(by_id["m1"]["vector"]) == 4  # re-embedded at the new dim
    assert by_id["m2"]["content"] == "rrf fusion"  # content preserved
    # content (+context when present) is what gets embedded
    assert "gh 401 token" in seen
    assert "rrf fusion" in seen


@pytest.mark.asyncio
async def test_reembed_keeps_old_vector_on_embed_failure(
    tmp_path: pathlib.Path,
) -> None:
    """A per-row embed failure keeps that row's ORIGINAL vector (2026-07-18:
    the bulk read no longer fetches `vector` up front -- see
    `_SEARCH_RESULT_FIELDS`-style projection in `reembed_table` -- so the
    fallback is now a bounded single-row re-fetch instead of a value already
    sitting in memory; this proves that fallback still recovers the exact
    original vector)."""
    db_path = tmp_path / "memories.lance"
    await _make_table(db_path)  # m1 vector=[0.1,0.2,0.3], m2 vector=[0.4,0.5,0.6]

    async def flaky_embed(text: str) -> list[float]:
        if "401" in text:
            raise RuntimeError("embed backend down")
        return [9.0, 9.0, 9.0]  # same dim (3) -- a refresh, not a dim change

    table, count = await vdb.reembed_table(db_path, flaky_embed)
    assert count == 2
    rows = await table.query().to_list()
    by_id = {r["id"]: r for r in rows}
    assert by_id["m1"]["vector"] == pytest.approx([0.1, 0.2, 0.3])  # kept old
    assert by_id["m2"]["vector"] == pytest.approx([9.0, 9.0, 9.0])  # re-embedded


@pytest.mark.asyncio
async def test_reembed_keeps_old_vector_when_content_and_context_empty(
    tmp_path: pathlib.Path,
) -> None:
    """No content AND no context -> nothing to embed -> the original vector
    survives untouched (and `embed_fn` is never called for that row)."""
    import lancedb

    db_path = tmp_path / "memories.lance"
    db = await lancedb.connect_async(str(db_path))
    await db.create_table(
        "memories",
        [
            {
                "id": "m1",
                "type": "SYSTEM",
                "content": "",
                "context": "",
                "vector": [0.7, 0.8, 0.9],
            }
        ],
    )

    called: list[str] = []

    async def fake_embed(text: str) -> list[float]:
        called.append(text)
        return [1.0, 2.0, 3.0]

    table, count = await vdb.reembed_table(db_path, fake_embed)
    assert count == 1
    rows = await table.query().to_list()
    assert rows[0]["vector"] == pytest.approx([0.7, 0.8, 0.9])
    assert called == []  # never called -- no content/context to embed

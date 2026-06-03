"""Tests for re-embedding the LanceDB table (embedder swap → new dim)."""

from __future__ import annotations

import pathlib

import pytest

import simba.memory.vector_db as vdb


async def _make_table(db_path: pathlib.Path):
    import lancedb

    db = await lancedb.connect_async(str(db_path))
    rows = [
        {"id": "m1", "type": "GOTCHA", "content": "gh 401", "context": "token",
         "vector": [0.1, 0.2, 0.3]},
        {"id": "m2", "type": "PATTERN", "content": "rrf fusion", "context": "",
         "vector": [0.4, 0.5, 0.6]},
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

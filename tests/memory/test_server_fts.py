"""Tests for FTS mirror startup init + reconcile (server.init_fts_mirror)."""

from __future__ import annotations

import pathlib
import time

import pytest

import simba.memory.config
import simba.memory.fts as fts
import simba.memory.server


async def _add(table, mid: str, content: str, mtype: str = "GOTCHA") -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    await table.add(
        [
            {
                "id": mid,
                "type": mtype,
                "content": content,
                "context": "",
                "tags": "[]",
                "confidence": 0.9,
                "sessionSource": "",
                "projectPath": "proj-1",
                "createdAt": now,
                "lastAccessedAt": now,
                "accessCount": 0,
                "vector": [0.1] * 768,
            }
        ]
    )


@pytest.mark.asyncio
async def test_init_fts_mirror_backfills_from_lancedb(
    lance_table, memory_config, tmp_path: pathlib.Path
) -> None:
    # lance_table starts with one SYSTEM row; add two real memories.
    await _add(lance_table, "m1", "ruff lints the python code")
    await _add(lance_table, "m2", "pytest runs the suite")

    app = simba.memory.server.create_app(memory_config)
    app.state.table = lance_table

    await simba.memory.server.init_fts_mirror(app, tmp_path)

    fts_path = tmp_path / fts.FTS_FILENAME
    assert app.state.fts_path == str(fts_path)
    conn = fts.connect(fts_path)
    try:
        # SYSTEM row is excluded; the two memories are indexed and searchable.
        assert fts.count(conn) == 2
        hits = fts.search(conn, "pytest", project_path="proj-1")
        assert [h["memory_id"] for h in hits] == ["m2"]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_init_fts_mirror_is_idempotent_when_in_sync(
    lance_table, memory_config, tmp_path: pathlib.Path
) -> None:
    await _add(lance_table, "m1", "alpha keyword memory")

    app = simba.memory.server.create_app(memory_config)
    app.state.table = lance_table

    await simba.memory.server.init_fts_mirror(app, tmp_path)
    await simba.memory.server.init_fts_mirror(app, tmp_path)  # second run = no-op

    conn = fts.connect(tmp_path / fts.FTS_FILENAME)
    try:
        assert fts.count(conn) == 1
    finally:
        conn.close()

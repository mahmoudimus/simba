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

    table, count, backup = await vdb.reembed_table(db_path, fake_embed)
    assert count == 2
    rows = await table.query().to_list()
    by_id = {r["id"]: r for r in rows}
    assert by_id["m1"]["vector"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert by_id["m2"]["content"] == "rrf fusion"  # content preserved
    assert by_id["m1"]["type"] == "GOTCHA"  # every other column preserved
    # content (+context when present) is what gets embedded
    assert "gh 401 token" in seen
    assert "rrf fusion" in seen
    assert backup is not None and backup.startswith("memories_backup_")


@pytest.mark.asyncio
async def test_reembed_keeps_pre_reembed_rows_in_backup_table(
    tmp_path: pathlib.Path,
) -> None:
    import lancedb

    db_path = tmp_path / "memories.lance"
    await _make_table(db_path)

    async def fake_embed(text: str) -> list[float]:
        return [1.0, 2.0, 3.0, 4.0]

    _, _, backup = await vdb.reembed_table(db_path, fake_embed)
    db = await lancedb.connect_async(str(db_path))
    old = await (await db.open_table(backup)).query().to_list()
    by_id = {r["id"]: r for r in old}
    assert by_id["m1"]["vector"] == pytest.approx([0.1, 0.2, 0.3])  # old model
    assert by_id["m2"]["content"] == "rrf fusion"


@pytest.mark.asyncio
async def test_reembed_without_backup_creates_no_extra_table(
    tmp_path: pathlib.Path,
) -> None:
    import lancedb

    db_path = tmp_path / "memories.lance"
    await _make_table(db_path)

    async def fake_embed(text: str) -> list[float]:
        return [1.0, 2.0, 3.0, 4.0]

    _, _, backup = await vdb.reembed_table(db_path, fake_embed, backup=False)
    db = await lancedb.connect_async(str(db_path))
    assert backup is None
    assert list((await db.list_tables()).tables) == ["memories"]


@pytest.mark.asyncio
async def test_reembed_aborts_untouched_on_any_embed_failure(
    tmp_path: pathlib.Path,
) -> None:
    """One failed row aborts the whole migration with the table untouched.

    Keeping that row's old vector (the previous behaviour) would mix two models'
    vector spaces in one table: silently wrong at the same width, a mixed-width
    table at a different one.
    """
    import lancedb

    db_path = tmp_path / "memories.lance"
    await _make_table(db_path)  # m1 vector=[0.1,0.2,0.3], m2 vector=[0.4,0.5,0.6]

    async def flaky_embed(text: str) -> list[float]:
        if "401" in text:
            raise RuntimeError("embed backend down")
        return [9.0, 9.0, 9.0, 9.0]

    with pytest.raises(vdb.ReembedError) as exc:
        await vdb.reembed_table(db_path, flaky_embed)
    assert exc.value.failed_ids == ["m1"]

    db = await lancedb.connect_async(str(db_path))
    assert list((await db.list_tables()).tables) == ["memories"]  # no backup
    rows = await (await db.open_table("memories")).query().to_list()
    by_id = {r["id"]: r for r in rows}
    assert by_id["m1"]["vector"] == pytest.approx([0.1, 0.2, 0.3])
    assert by_id["m2"]["vector"] == pytest.approx([0.4, 0.5, 0.6])


@pytest.mark.asyncio
async def test_reembed_aborts_on_inconsistent_widths(tmp_path: pathlib.Path) -> None:
    db_path = tmp_path / "memories.lance"
    await _make_table(db_path)

    async def ragged_embed(text: str) -> list[float]:
        return [1.0] * (4 if "401" in text else 5)

    with pytest.raises(vdb.ReembedError):
        await vdb.reembed_table(db_path, ragged_embed)


@pytest.mark.asyncio
async def test_reembed_gives_empty_rows_a_zero_vector_of_the_new_width(
    tmp_path: pathlib.Path,
) -> None:
    """No content AND no context (the SYSTEM init row) -> embed_fn is never
    called for it, and it gets a zero vector of the NEW width (its old
    vector's width would no longer fit the table)."""
    import lancedb

    db_path = tmp_path / "memories.lance"
    db = await lancedb.connect_async(str(db_path))
    await db.create_table(
        "memories",
        [
            {"id": "s", "type": "SYSTEM", "content": "", "context": "",
             "vector": [0.0, 0.0, 0.0]},
            {"id": "m", "type": "GOTCHA", "content": "x", "context": "",
             "vector": [0.7, 0.8, 0.9]},
        ],
    )  # fmt: skip

    called: list[str] = []

    async def fake_embed(text: str) -> list[float]:
        called.append(text)
        return [1.0, 2.0, 3.0, 4.0]

    table, count, _ = await vdb.reembed_table(db_path, fake_embed)
    assert count == 2
    by_id = {r["id"]: r for r in await table.query().to_list()}
    assert by_id["s"]["vector"] == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert by_id["m"]["vector"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert called == ["x"]


@pytest.mark.asyncio
async def test_reembed_only_empty_rows_is_a_noop(tmp_path: pathlib.Path) -> None:
    import lancedb

    db_path = tmp_path / "memories.lance"
    db = await lancedb.connect_async(str(db_path))
    await db.create_table(
        "memories",
        [{"id": "s", "type": "SYSTEM", "content": "", "context": "",
          "vector": [0.7, 0.8, 0.9]}],
    )  # fmt: skip

    async def fake_embed(text: str) -> list[float]:
        raise AssertionError("must not be called")

    table, count, backup = await vdb.reembed_table(db_path, fake_embed)
    assert (count, backup) == (1, None)
    rows = await table.query().to_list()
    assert rows[0]["vector"] == pytest.approx([0.7, 0.8, 0.9])

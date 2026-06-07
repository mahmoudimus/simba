"""The embedding-dimension migration guard (bge-large default = 1024-d)."""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

import simba.memory.vector_db as vdb


def _vector_schema(dim: int) -> pa.Schema:
    """A LanceDB-shaped schema: a fixed-size-list ``vector`` column of ``dim``."""
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


class _SyncTable:
    """Mimics the sync LanceDB ``Table`` — ``schema`` is a plain property."""

    def __init__(self, dim: int) -> None:
        self._schema = _vector_schema(dim)

    @property
    def schema(self) -> pa.Schema:
        return self._schema


class _AsyncTable:
    """Mimics the daemon's ``AsyncTable`` — ``schema`` is an async method."""

    def __init__(self, dim: int) -> None:
        self._schema = _vector_schema(dim)

    async def schema(self) -> pa.Schema:
        return self._schema


def test_resolve_dim_from_async_table() -> None:
    # The daemon opens tables via connect_async -> AsyncTable, whose `schema` is
    # a coroutine. The resolver must await it (the bug: it read it synchronously,
    # always got None, so the guard never fired on the live path).
    assert asyncio.run(vdb._resolve_table_dim(_AsyncTable(768))) == 768


def test_resolve_dim_from_sync_table() -> None:
    assert asyncio.run(vdb._resolve_table_dim(_SyncTable(1024))) == 1024


def test_resolve_dim_no_schema_is_none() -> None:
    assert asyncio.run(vdb._resolve_table_dim(object())) is None


def test_dim_mismatch_raises_actionable_error() -> None:
    with pytest.raises(vdb.EmbeddingDimMismatchError) as exc:
        vdb.check_embedding_dim(query_dim=1024, table_dim=768)
    msg = str(exc.value).lower()
    assert "reembed" in msg  # tells the user how to migrate
    assert "768" in str(exc.value) and "1024" in str(exc.value)


def test_dim_match_is_noop() -> None:
    vdb.check_embedding_dim(query_dim=1024, table_dim=1024)  # no raise


def test_unknown_table_dim_is_noop() -> None:
    # If the store dim can't be determined, don't block recall.
    vdb.check_embedding_dim(query_dim=1024, table_dim=None)

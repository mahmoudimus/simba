"""The embedding-dimension migration guard (bge-large default = 1024-d)."""

from __future__ import annotations

import pytest

import simba.memory.vector_db as vdb


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

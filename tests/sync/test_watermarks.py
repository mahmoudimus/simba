from __future__ import annotations

import sqlite3

import pytest

from simba.sync.watermarks import (
    _init_schema,
    get_all_watermarks,
    get_watermark,
    set_watermark,
)


@pytest.fixture()
def conn():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    _init_schema(db)
    yield db
    db.close()


class TestGetWatermark:
    def test_get_watermark_default(self, conn: sqlite3.Connection) -> None:
        result = get_watermark(conn, "users", "full_sync")
        assert result == "0"

    def test_set_and_get_watermark(self, conn: sqlite3.Connection) -> None:
        set_watermark(conn, "users", "full_sync", "42", rows_processed=10, errors=0)

        result = get_watermark(conn, "users", "full_sync")
        assert result == "42"

    def test_set_watermark_upsert(self, conn: sqlite3.Connection) -> None:
        set_watermark(conn, "users", "full_sync", "10", rows_processed=5, errors=1)
        set_watermark(conn, "users", "full_sync", "20", rows_processed=3, errors=2)

        result = get_watermark(conn, "users", "full_sync")
        assert result == "20"

        rows = get_all_watermarks(conn)
        assert len(rows) == 1
        assert rows[0]["rows_processed"] == 8
        assert rows[0]["errors"] == 3


class TestGetAllWatermarks:
    def test_get_all_watermarks(self, conn: sqlite3.Connection) -> None:
        set_watermark(conn, "orders", "incremental", "100", rows_processed=50, errors=0)
        set_watermark(conn, "users", "full_sync", "42", rows_processed=10, errors=1)

        rows = get_all_watermarks(conn)
        assert len(rows) == 2
        # Ordered by table_name, pipeline
        assert rows[0]["table_name"] == "orders"
        assert rows[0]["pipeline"] == "incremental"
        assert rows[0]["last_cursor"] == "100"
        assert rows[0]["rows_processed"] == 50
        assert rows[0]["errors"] == 0
        assert rows[1]["table_name"] == "users"
        assert rows[1]["pipeline"] == "full_sync"
        assert rows[1]["last_cursor"] == "42"

    def test_get_all_watermarks_empty(self, conn: sqlite3.Connection) -> None:
        rows = get_all_watermarks(conn)
        assert rows == []


class TestSetWatermarkTimestamp:
    def test_set_watermark_records_timestamp(self, conn: sqlite3.Connection) -> None:
        set_watermark(conn, "users", "full_sync", "1", rows_processed=1, errors=0)

        rows = get_all_watermarks(conn)
        assert len(rows) == 1
        assert rows[0]["last_run_at"] is not None
        assert len(rows[0]["last_run_at"]) > 0

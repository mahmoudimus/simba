from __future__ import annotations

import pathlib

import pytest

import simba.db
from simba.sync.watermarks import (
    get_all_watermarks,
    get_watermark,
    set_watermark,
)


@pytest.fixture()
def cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda c=None: db_path)
    return tmp_path


class TestGetWatermark:
    def test_get_watermark_default(self, cwd: pathlib.Path) -> None:
        assert get_watermark("users", "full_sync") == "0"

    def test_set_and_get_watermark(self, cwd: pathlib.Path) -> None:
        set_watermark("users", "full_sync", "42", rows_processed=10, errors=0)
        assert get_watermark("users", "full_sync") == "42"

    def test_set_watermark_upsert(self, cwd: pathlib.Path) -> None:
        set_watermark("users", "full_sync", "10", rows_processed=5, errors=1)
        set_watermark("users", "full_sync", "20", rows_processed=3, errors=2)

        assert get_watermark("users", "full_sync") == "20"

        rows = get_all_watermarks()
        assert len(rows) == 1
        assert rows[0]["rows_processed"] == 8
        assert rows[0]["errors"] == 3


class TestGetAllWatermarks:
    def test_get_all_watermarks(self, cwd: pathlib.Path) -> None:
        set_watermark("orders", "incremental", "100", rows_processed=50, errors=0)
        set_watermark("users", "full_sync", "42", rows_processed=10, errors=1)

        rows = get_all_watermarks()
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

    def test_get_all_watermarks_empty(self, cwd: pathlib.Path) -> None:
        assert get_all_watermarks() == []


class TestSetWatermarkTimestamp:
    def test_set_watermark_records_timestamp(self, cwd: pathlib.Path) -> None:
        set_watermark("users", "full_sync", "1", rows_processed=1, errors=0)

        rows = get_all_watermarks()
        assert len(rows) == 1
        assert rows[0]["last_run_at"] is not None
        assert len(rows[0]["last_run_at"]) > 0

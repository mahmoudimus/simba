from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simba.sync.indexer import IndexResult, run_index
from simba.sync.watermarks import _init_schema, get_watermark


@pytest.fixture()
def db_dir(tmp_path: Path) -> Path:
    """Create a test DB with reflections table and watermarks."""
    simba_dir = tmp_path / ".simba"
    simba_dir.mkdir()
    db_path = simba_dir / "simba.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    conn.execute(
        """CREATE TABLE reflections (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            error_type TEXT NOT NULL,
            snippet TEXT NOT NULL DEFAULT '',
            context TEXT NOT NULL DEFAULT '{}',
            signature TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.execute(
        "INSERT INTO reflections VALUES (?, ?, ?, ?, ?, ?)",
        ("r1", "2025-01-01", "TypeError", "bad arg", "{}", "abc123"),
    )
    conn.execute(
        "INSERT INTO reflections VALUES (?, ?, ?, ?, ?, ?)",
        ("r2", "2025-01-02", "ValueError", "wrong val", "{}", "def456"),
    )
    conn.commit()
    conn.close()
    return tmp_path


def _get_test_conn(db_dir: Path) -> sqlite3.Connection:
    """Open the test DB without running schema initializers."""
    conn = sqlite3.connect(str(db_dir / ".simba" / "simba.db"))
    conn.row_factory = sqlite3.Row
    return conn


class TestRunIndex:
    @patch("simba.sync.indexer.exporter.export_all_tables")
    def test_dry_run_no_http(self, mock_export: MagicMock, db_dir: Path) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            result = run_index(db_dir, dry_run=True)
        assert isinstance(result, IndexResult)
        assert result.rows_indexed == 2
        assert result.errors == 0
        mock_export.assert_not_called()

    @patch("simba.sync.indexer.exporter.export_all_tables")
    def test_dry_run_advances_watermark(
        self, mock_export: MagicMock, db_dir: Path
    ) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            run_index(db_dir, dry_run=True)
        conn = _get_test_conn(db_dir)
        wm = get_watermark(conn, "reflections", "index")
        conn.close()
        assert wm != "0"

    @patch("simba.sync.indexer.exporter.export_all_tables", return_value=[])
    @patch("simba.sync.indexer._post_to_daemon", return_value="ok")
    def test_posts_to_daemon(
        self,
        mock_post: MagicMock,
        mock_export: MagicMock,
        db_dir: Path,
    ) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            result = run_index(db_dir)
        assert result.rows_indexed == 2
        assert mock_post.call_count == 2

    @patch("simba.sync.indexer.exporter.export_all_tables", return_value=[])
    @patch("simba.sync.indexer._post_to_daemon", return_value="duplicate")
    def test_counts_duplicates(
        self,
        mock_post: MagicMock,
        mock_export: MagicMock,
        db_dir: Path,
    ) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            result = run_index(db_dir)
        assert result.duplicates == 2
        assert result.rows_indexed == 0

    @patch("simba.sync.indexer.exporter.export_all_tables", return_value=[])
    @patch("simba.sync.indexer._post_to_daemon", return_value="ok")
    def test_advances_watermark(
        self,
        mock_post: MagicMock,
        mock_export: MagicMock,
        db_dir: Path,
    ) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            run_index(db_dir)
        conn = _get_test_conn(db_dir)
        wm = get_watermark(conn, "reflections", "index")
        conn.close()
        assert wm != "0"

    @patch("simba.sync.indexer.exporter.export_all_tables", return_value=[])
    @patch("simba.sync.indexer._post_to_daemon", return_value="ok")
    def test_second_run_no_new_rows(
        self,
        mock_post: MagicMock,
        mock_export: MagicMock,
        db_dir: Path,
    ) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            run_index(db_dir)
        mock_post.reset_mock()
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            result = run_index(db_dir)
        assert result.rows_indexed == 0
        mock_post.assert_not_called()

    def test_tables_polled_count(self, db_dir: Path) -> None:
        with patch(
            "simba.db.get_connection",
            return_value=_get_test_conn(db_dir),
        ):
            result = run_index(db_dir, dry_run=True)
        assert result.tables_polled == 7

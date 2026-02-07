from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simba.sync.extractor import ExtractResult, run_extract
from simba.sync.watermarks import _init_schema


@pytest.fixture()
def db_dir(tmp_path: Path) -> Path:
    """Create a test DB with watermarks and proven_facts tables."""
    simba_dir = tmp_path / ".simba"
    simba_dir.mkdir()
    db_path = simba_dir / "simba.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS proven_facts
           (subject TEXT, predicate TEXT, object TEXT, proof TEXT,
           UNIQUE(subject, predicate, object))"""
    )
    conn.commit()
    conn.close()
    return tmp_path


@contextlib.contextmanager
def _mock_get_db(db_dir: Path):
    """Context manager that yields a test DB connection."""
    conn = sqlite3.connect(str(db_dir / ".simba" / "simba.db"))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _mock_list_response(memories: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "memories": memories,
        "total": len(memories),
        "limit": 50,
        "offset": 0,
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestRunExtract:
    @patch("httpx.Client")
    def test_extracts_facts_from_memories(
        self, mock_client_cls: MagicMock, db_dir: Path
    ) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        memories = [
            {
                "id": "m1",
                "type": "WORKING_SOLUTION",
                "content": "use ruff for linting",
                "context": "",
                "createdAt": "2025-01-01T00:00:00",
            },
        ]
        client.get.return_value = _mock_list_response(memories)

        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            result = run_extract(db_dir)
        assert isinstance(result, ExtractResult)
        assert result.memories_processed == 1
        assert result.facts_extracted == 1

    @patch("httpx.Client")
    def test_skips_system_memories(
        self, mock_client_cls: MagicMock, db_dir: Path
    ) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        memories = [
            {
                "id": "m1",
                "type": "SYSTEM",
                "content": "indexed row",
                "createdAt": "2025-01-01T00:00:00",
            },
        ]
        client.get.return_value = _mock_list_response(memories)

        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            result = run_extract(db_dir)
        assert result.memories_processed == 0

    @patch("httpx.Client")
    def test_dry_run_no_db_writes(
        self, mock_client_cls: MagicMock, db_dir: Path
    ) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        memories = [
            {
                "id": "m1",
                "type": "WORKING_SOLUTION",
                "content": "use pytest for testing",
                "context": "",
                "createdAt": "2025-01-01T00:00:00",
            },
        ]
        client.get.return_value = _mock_list_response(memories)

        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            result = run_extract(db_dir, dry_run=True)
        assert result.facts_extracted == 1

        # Verify no facts were written to DB
        conn = sqlite3.connect(str(db_dir / ".simba" / "simba.db"))
        count = conn.execute("SELECT COUNT(*) FROM proven_facts").fetchone()[0]
        conn.close()
        assert count == 0

    @patch("httpx.Client")
    def test_duplicate_facts_counted(
        self, mock_client_cls: MagicMock, db_dir: Path
    ) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        memories = [
            {
                "id": "m1",
                "type": "WORKING_SOLUTION",
                "content": "use ruff for linting",
                "context": "",
                "createdAt": "2025-01-01T00:00:00",
            },
        ]
        client.get.return_value = _mock_list_response(memories)

        # First run
        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            run_extract(db_dir)

        # Reset watermark to force reprocessing
        conn = sqlite3.connect(str(db_dir / ".simba" / "simba.db"))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "UPDATE sync_watermarks SET last_cursor = '0' WHERE table_name = 'memories'"
        )
        conn.commit()
        conn.close()

        # Second run - same fact should be duplicate
        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            result = run_extract(db_dir)
        assert result.facts_duplicate >= 1

    @patch("httpx.Client")
    def test_no_match_memories_collected(
        self, mock_client_cls: MagicMock, db_dir: Path
    ) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        memories = [
            {
                "id": "m1",
                "type": "GOTCHA",
                "content": "just a plain note",
                "context": "",
                "createdAt": "2025-01-01T00:00:00",
            },
        ]
        client.get.return_value = _mock_list_response(memories)

        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            result = run_extract(db_dir)
        assert result.memories_processed == 1
        assert result.facts_extracted == 0

    @patch("httpx.Client")
    def test_empty_response(self, mock_client_cls: MagicMock, db_dir: Path) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get.return_value = _mock_list_response([])

        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            result = run_extract(db_dir)
        assert result.memories_processed == 0
        assert result.errors == 0

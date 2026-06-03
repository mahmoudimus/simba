from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import simba.db
import simba.kg.store
from simba.sync.extractor import ExtractResult, run_extract
from simba.sync.watermarks import _init_schema


@pytest.fixture()
def db_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a test DB with watermarks and the kg_edges schema.

    The extractor now stores facts into the temporal knowledge graph
    (``kg_edges``) via :func:`simba.kg.store.kg_add`, so the test DB needs
    the kg schema (table + FTS mirror + sync triggers) installed.
    """
    simba_dir = tmp_path / ".simba"
    simba_dir.mkdir()
    db_path = simba_dir / "simba.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    simba.kg.store._init_schema(conn)
    conn.commit()
    conn.close()
    # kg_add() now uses simba.db.connect() -> get_db_path; redirect it to this
    # test DB so facts land here (not the real repo DB). Freeze the KG clock so
    # re-adding a fact collides on the UNIQUE(..., valid_from) key
    # deterministically (otherwise the two runs can straddle a second boundary).
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    monkeypatch.setattr(simba.kg.store, "_now", lambda: "2025-01-01T00:00:00Z")
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

        # Verify no facts were written to the knowledge graph
        conn = sqlite3.connect(str(db_dir / ".simba" / "simba.db"))
        count = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
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
    def test_occurred_at_populated_from_content(
        self, mock_client_cls: MagicMock, db_dir: Path
    ) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        memories = [
            {
                "id": "m1",
                "type": "WORKING_SOLUTION",
                "content": "use ruff for linting since 2025-03-01",
                "context": "",
                "createdAt": "2025-01-01T00:00:00",
            },
        ]
        client.get.return_value = _mock_list_response(memories)

        with patch(
            "simba.db.get_db",
            side_effect=lambda *a, **kw: _mock_get_db(db_dir),
        ):
            run_extract(db_dir)

        project = simba.db.resolve_project_id(db_dir)
        rows = simba.kg.store.kg_query(project_path=project)
        assert rows, "expected an extracted edge"
        assert rows[0]["occurred_at"] == "2025-03-01"

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

"""Tests for the sync CLI (simba.sync.__main__)."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import simba.db
import simba.sync.__main__
from simba.sync.extractor import ExtractResult
from simba.sync.indexer import IndexResult
from simba.sync.watermarks import set_watermark

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_index_result(**kwargs) -> IndexResult:
    return IndexResult(**kwargs)


def _make_extract_result(**kwargs) -> ExtractResult:
    return ExtractResult(**kwargs)


# ---------------------------------------------------------------------------
# _print_index_result
# ---------------------------------------------------------------------------


class TestPrintIndexResult:
    def test_normal_output(self, capsys) -> None:
        result = _make_index_result(
            tables_polled=3, rows_indexed=5, rows_exported=4,
            duplicates=1, errors=0,
        )
        simba.sync.__main__._print_index_result(result)
        out = capsys.readouterr().out
        assert "5 indexed" in out
        assert "1 duplicates" in out
        assert "0 errors" in out
        assert "3 tables polled" in out
        assert "4 exported" in out

    def test_dry_run_prefix(self, capsys) -> None:
        result = _make_index_result(rows_indexed=1)
        simba.sync.__main__._print_index_result(result, dry_run=True)
        out = capsys.readouterr().out
        assert out.startswith("[dry-run]")


# ---------------------------------------------------------------------------
# _print_extract_result
# ---------------------------------------------------------------------------


class TestPrintExtractResult:
    def test_normal_output(self, capsys) -> None:
        result = _make_extract_result(
            facts_extracted=10, memories_processed=3,
            facts_duplicate=2, errors=0,
        )
        simba.sync.__main__._print_extract_result(result)
        out = capsys.readouterr().out
        assert "10 facts" in out
        assert "3 memories" in out
        assert "2 duplicates" in out
        assert "0 errors" in out

    def test_dry_run_prefix(self, capsys) -> None:
        result = _make_extract_result(facts_extracted=1)
        simba.sync.__main__._print_extract_result(result, dry_run=True)
        out = capsys.readouterr().out
        assert out.startswith("[dry-run]")

    def test_agent_dispatched_message(self, capsys) -> None:
        result = _make_extract_result(agent_dispatched=True)
        simba.sync.__main__._print_extract_result(result)
        out = capsys.readouterr().out
        assert "Claude researcher agent dispatched" in out


# ---------------------------------------------------------------------------
# _cmd_status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    def test_no_database(self, tmp_path, capsys) -> None:
        """When the DB file does not exist, prints a helpful message."""
        args = _make_status_args(cwd=str(tmp_path))
        rc = simba.sync.__main__._cmd_status(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Database not found" in out

    def test_no_watermarks(self, tmp_path, capsys) -> None:
        """With an empty DB, prints 'No sync watermarks'."""
        # Create the DB so get_connection returns a connection
        with simba.db.get_db(tmp_path):
            pass
        args = _make_status_args(cwd=str(tmp_path))
        rc = simba.sync.__main__._cmd_status(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No sync watermarks" in out

    def test_with_watermarks(self, tmp_path, capsys) -> None:
        """With watermarks present, prints a table."""
        with simba.db.get_db(tmp_path) as conn:
            set_watermark(conn, "reflections", "index", "42",
                          rows_processed=10, errors=1)
        args = _make_status_args(cwd=str(tmp_path))
        rc = simba.sync.__main__._cmd_status(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Table" in out  # header
        assert "reflections" in out
        assert "index" in out


# ---------------------------------------------------------------------------
# _cmd_index
# ---------------------------------------------------------------------------


class TestCmdIndex:
    @patch("simba.sync.__main__.run_index")
    def test_calls_run_index(self, mock_run_index, capsys) -> None:
        mock_run_index.return_value = _make_index_result(
            tables_polled=2, rows_indexed=3, errors=0,
        )
        args = _make_index_args()
        rc = simba.sync.__main__._cmd_index(args)
        assert rc == 0
        mock_run_index.assert_called_once()
        out = capsys.readouterr().out
        assert "3 indexed" in out

    @patch("simba.sync.__main__.run_index")
    def test_returns_1_on_errors(self, mock_run_index) -> None:
        mock_run_index.return_value = _make_index_result(errors=1)
        args = _make_index_args()
        rc = simba.sync.__main__._cmd_index(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# _cmd_extract
# ---------------------------------------------------------------------------


class TestCmdExtract:
    @patch("simba.sync.__main__.run_extract")
    def test_calls_run_extract(self, mock_run_extract, capsys) -> None:
        mock_run_extract.return_value = _make_extract_result(
            facts_extracted=5, memories_processed=2, errors=0,
        )
        args = _make_extract_args()
        rc = simba.sync.__main__._cmd_extract(args)
        assert rc == 0
        mock_run_extract.assert_called_once()
        out = capsys.readouterr().out
        assert "5 facts" in out

    @patch("simba.sync.__main__.run_extract")
    def test_returns_1_on_errors(self, mock_run_extract) -> None:
        mock_run_extract.return_value = _make_extract_result(errors=2)
        args = _make_extract_args()
        rc = simba.sync.__main__._cmd_extract(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# _cmd_run
# ---------------------------------------------------------------------------


class TestCmdRun:
    @patch("simba.sync.__main__.run_extract")
    @patch("simba.sync.__main__.run_index")
    def test_calls_both_pipelines(
        self, mock_index, mock_extract, capsys,
    ) -> None:
        mock_index.return_value = _make_index_result(rows_indexed=2)
        mock_extract.return_value = _make_extract_result(facts_extracted=3)
        args = _make_run_args()
        rc = simba.sync.__main__._cmd_run(args)
        assert rc == 0
        mock_index.assert_called_once()
        mock_extract.assert_called_once()
        out = capsys.readouterr().out
        assert "2 indexed" in out
        assert "3 facts" in out

    @patch("simba.sync.__main__.run_extract")
    @patch("simba.sync.__main__.run_index")
    def test_returns_1_when_errors(self, mock_index, mock_extract) -> None:
        mock_index.return_value = _make_index_result(errors=1)
        mock_extract.return_value = _make_extract_result(errors=0)
        args = _make_run_args()
        rc = simba.sync.__main__._cmd_run(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_args_returns_1(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.argv", ["simba-sync"])
        rc = simba.sync.__main__.main()
        assert rc == 1

    @patch("simba.sync.__main__.run_index")
    def test_dispatches_index(self, mock_run_index, monkeypatch) -> None:
        mock_run_index.return_value = _make_index_result()
        monkeypatch.setattr("sys.argv", ["simba-sync", "index", "--dry-run"])
        rc = simba.sync.__main__.main()
        assert rc == 0
        mock_run_index.assert_called_once()

    @patch("simba.sync.__main__.run_extract")
    def test_dispatches_extract(self, mock_run_extract, monkeypatch) -> None:
        mock_run_extract.return_value = _make_extract_result()
        monkeypatch.setattr("sys.argv", ["simba-sync", "extract", "--dry-run"])
        rc = simba.sync.__main__.main()
        assert rc == 0
        mock_run_extract.assert_called_once()

    @patch("simba.sync.__main__.run_extract")
    @patch("simba.sync.__main__.run_index")
    def test_dispatches_run(
        self, mock_index, mock_extract, monkeypatch,
    ) -> None:
        mock_index.return_value = _make_index_result()
        mock_extract.return_value = _make_extract_result()
        monkeypatch.setattr("sys.argv", ["simba-sync", "run"])
        rc = simba.sync.__main__.main()
        assert rc == 0

    def test_dispatches_status(self, tmp_path, monkeypatch, capsys) -> None:
        # Create a DB so status doesn't fail with "not found"
        with simba.db.get_db(tmp_path):
            pass
        monkeypatch.setattr(
            "sys.argv", ["simba-sync", "status", "--cwd", str(tmp_path)],
        )
        rc = simba.sync.__main__.main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "No sync watermarks" in out


# ---------------------------------------------------------------------------
# argparse.Namespace factories
# ---------------------------------------------------------------------------


def _make_status_args(cwd: str = ".") -> argparse.Namespace:
    return argparse.Namespace(cwd=cwd)


def _make_index_args(
    cwd: str = ".",
    daemon_url: str = "http://localhost:8741",
    dry_run: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        cwd=cwd, daemon_url=daemon_url, dry_run=dry_run,
    )


def _make_extract_args(
    cwd: str = ".",
    daemon_url: str = "http://localhost:8741",
    dry_run: bool = False,
    use_claude: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        cwd=cwd,
        daemon_url=daemon_url,
        dry_run=dry_run,
        use_claude=use_claude,
    )


def _make_run_args(
    cwd: str = ".",
    daemon_url: str = "http://localhost:8741",
    dry_run: bool = False,
    use_claude: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        cwd=cwd,
        daemon_url=daemon_url,
        dry_run=dry_run,
        use_claude=use_claude,
    )

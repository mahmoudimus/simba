"""Tests for simba.stats — token economics dashboard."""

from __future__ import annotations

import pathlib
import unittest.mock

import simba.stats


class TestCountActivities:
    def test_tallies_by_tool_name(self) -> None:
        entries = [
            ("2025-01-01 10:00:00", "read", "file1.py"),
            ("2025-01-01 10:00:01", "read", "file2.py"),
            ("2025-01-01 10:00:02", "search", "query"),
        ]
        counts = simba.stats._count_activities(entries)
        assert counts == {"read": 2, "search": 1}

    def test_empty_entries(self) -> None:
        assert simba.stats._count_activities([]) == {}


class TestCodebaseSize:
    def test_returns_file_and_line_counts(self, tmp_path: pathlib.Path) -> None:
        mock_files = unittest.mock.Mock()
        mock_files.stdout = "a.py\nb.py\nc.py\n"

        mock_lines = unittest.mock.Mock()
        mock_lines.stdout = "a.py:10\nb.py:20\nc.py:30\n"

        with unittest.mock.patch(
            "simba.stats.subprocess.run",
            side_effect=[mock_files, mock_lines],
        ):
            files, lines = simba.stats._codebase_size(tmp_path)

        assert files == 3
        assert lines == 60

    def test_returns_zeros_on_error(self, tmp_path: pathlib.Path) -> None:
        import subprocess

        with unittest.mock.patch(
            "simba.stats.subprocess.run",
            side_effect=subprocess.SubprocessError("rg not found"),
        ):
            files, lines = simba.stats._codebase_size(tmp_path)

        assert files == 0
        assert lines == 0


class TestRunStats:
    def test_includes_all_sections(self, tmp_path: pathlib.Path) -> None:
        mock_files = unittest.mock.Mock()
        mock_files.stdout = "a.py\nb.py\n"

        mock_lines = unittest.mock.Mock()
        mock_lines.stdout = "a.py:100\nb.py:200\n"

        entries = [
            ("2025-01-01 10:00:00", "search", "query1"),
            ("2025-01-01 10:00:01", "read", "file1.py"),
        ]

        with (
            unittest.mock.patch(
                "simba.stats.subprocess.run",
                side_effect=[mock_files, mock_lines],
            ),
            unittest.mock.patch(
                "simba.search.activity_tracker.read_activity_log",
                return_value=entries,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=None,
            ),
        ):
            result = simba.stats.run_stats(tmp_path)

        assert "Codebase:" in result
        assert "Files: 2" in result
        assert "Activity" in result
        assert "Searches: 1" in result
        assert "Reads: 1" in result
        assert "Token economics" in result
        assert "not initialized" in result

    def test_includes_project_memory_when_available(
        self, tmp_path: pathlib.Path
    ) -> None:
        mock_files = unittest.mock.Mock()
        mock_files.stdout = ""

        mock_lines = unittest.mock.Mock()
        mock_lines.stdout = ""

        mock_conn = unittest.mock.MagicMock()

        with (
            unittest.mock.patch(
                "simba.stats.subprocess.run",
                side_effect=[mock_files, mock_lines],
            ),
            unittest.mock.patch(
                "simba.search.activity_tracker.read_activity_log",
                return_value=[],
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=mock_conn,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_stats",
                return_value={"sessions": 5, "knowledge": 3, "facts": 10},
            ),
        ):
            result = simba.stats.run_stats(tmp_path)

        assert "Sessions: 5" in result
        assert "Knowledge: 3" in result
        assert "Facts: 10" in result
        mock_conn.close.assert_called_once()

    def test_no_activity_log(self, tmp_path: pathlib.Path) -> None:
        mock_files = unittest.mock.Mock()
        mock_files.stdout = ""

        mock_lines = unittest.mock.Mock()
        mock_lines.stdout = ""

        with (
            unittest.mock.patch(
                "simba.stats.subprocess.run",
                side_effect=[mock_files, mock_lines],
            ),
            unittest.mock.patch(
                "simba.search.activity_tracker.read_activity_log",
                return_value=[],
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=None,
            ),
        ):
            result = simba.stats.run_stats(tmp_path)

        assert "0 logged events" in result
        assert "Searches: 0" in result

    def test_token_savings_calculation(self, tmp_path: pathlib.Path) -> None:
        mock_files = unittest.mock.Mock()
        mock_files.stdout = ""

        mock_lines = unittest.mock.Mock()
        mock_lines.stdout = ""

        # 3 searches + 2 reads = 150 + 2000 = 2150 tokens
        # without = 20000 tokens
        # savings = (1 - 2150/20000) * 100 = ~89%
        entries = [
            ("t", "search", "q1"),
            ("t", "search", "q2"),
            ("t", "search", "q3"),
            ("t", "read", "f1"),
            ("t", "read", "f2"),
        ]

        with (
            unittest.mock.patch(
                "simba.stats.subprocess.run",
                side_effect=[mock_files, mock_lines],
            ),
            unittest.mock.patch(
                "simba.search.activity_tracker.read_activity_log",
                return_value=entries,
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=None,
            ),
        ):
            result = simba.stats.run_stats(tmp_path)

        assert "~2,150 tokens" in result
        assert "~20,000 tokens" in result
        assert "~89%" in result

"""Tests for search.activity_tracker — pipe-separated activity log."""

from __future__ import annotations

import pathlib
import unittest
import unittest.mock

import pytest

import simba.search.activity_tracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_repo_root(tmp_path: pathlib.Path):
    """Patch find_repo_root so every call resolves to tmp_path."""
    with unittest.mock.patch(
        "simba.search.project_memory.find_repo_root",
        return_value=tmp_path,
    ):
        yield


# ---------------------------------------------------------------------------
# TestLogActivity
# ---------------------------------------------------------------------------


class TestLogActivity:
    def test_writes_pipe_separated_entry(self, tmp_path: pathlib.Path) -> None:
        simba.search.activity_tracker.log_activity(tmp_path, "grep", "searched for foo")
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        assert log_path.exists()
        content = log_path.read_text()
        parts = content.strip().split("|")
        assert len(parts) == 3
        assert parts[1] == "grep"
        assert parts[2] == "searched for foo"

    def test_creates_parent_directory(self, tmp_path: pathlib.Path) -> None:
        memory_dir = tmp_path / ".simba" / "search"
        assert not memory_dir.exists()
        simba.search.activity_tracker.log_activity(tmp_path, "read", "file.py")
        assert memory_dir.exists()

    def test_appends_multiple_entries(self, tmp_path: pathlib.Path) -> None:
        simba.search.activity_tracker.log_activity(tmp_path, "grep", "first")
        simba.search.activity_tracker.log_activity(tmp_path, "read", "second")
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert "grep" in lines[0]
        assert "read" in lines[1]


# ---------------------------------------------------------------------------
# TestReadActivityLog
# ---------------------------------------------------------------------------


class TestReadActivityLog:
    def test_parses_pipe_separated_format(self, tmp_path: pathlib.Path) -> None:
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "2024-01-01 10:00:00|grep|searched foo\n2024-01-01 10:01:00|read|file.py\n"
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 2
        assert entries[0] == ("2024-01-01 10:00:00", "grep", "searched foo")
        assert entries[1] == ("2024-01-01 10:01:00", "read", "file.py")

    def test_returns_empty_list_when_file_missing(self, tmp_path: pathlib.Path) -> None:
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert entries == []

    def test_skips_malformed_lines(self, tmp_path: pathlib.Path) -> None:
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "2024-01-01 10:00:00|grep|searched foo\n"
            "malformed line without pipes\n"
            "only|one pipe\n"
            "2024-01-01 10:02:00|edit|bar.py\n"
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 2
        assert entries[0][1] == "grep"
        assert entries[1][1] == "edit"


# ---------------------------------------------------------------------------
# TestClearActivityLog
# ---------------------------------------------------------------------------


class TestClearActivityLog:
    def test_removes_the_file(self, tmp_path: pathlib.Path) -> None:
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("2024-01-01 10:00:00|grep|foo\n")
        assert log_path.exists()
        simba.search.activity_tracker.clear_activity_log(tmp_path)
        assert not log_path.exists()

    def test_no_error_when_file_missing(self, tmp_path: pathlib.Path) -> None:
        # Should not raise
        simba.search.activity_tracker.clear_activity_log(tmp_path)


# ---------------------------------------------------------------------------
# TestRotateLog
# ---------------------------------------------------------------------------


class TestRotateLog:
    def test_keeps_last_100_lines_when_over_200(self, tmp_path: pathlib.Path) -> None:
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"2024-01-01 00:00:{i:02d}|tool{i}|detail{i}\n" for i in range(210)]
        log_path.write_text("".join(lines))
        simba.search.activity_tracker._rotate_log(log_path)
        remaining = log_path.read_text().splitlines()
        assert len(remaining) == 100
        # Should be the last 100 lines (indices 110..209)
        assert "tool110" in remaining[0]
        assert "tool209" in remaining[-1]

    def test_no_rotation_under_200_lines(self, tmp_path: pathlib.Path) -> None:
        log_path = tmp_path / ".simba" / "search" / "activity.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"2024-01-01 00:00:{i:02d}|tool{i}|detail{i}\n" for i in range(150)]
        log_path.write_text("".join(lines))
        simba.search.activity_tracker._rotate_log(log_path)
        remaining = log_path.read_text().splitlines()
        assert len(remaining) == 150

"""Tests for search.activity_tracker -- SQLite-backed activity log."""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.search.activity_tracker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Point simba.db.get_db_path to a tmp_path-based location."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


# ---------------------------------------------------------------------------
# TestLogActivity
# ---------------------------------------------------------------------------


class TestLogActivity:
    def test_inserts_row_into_activities_table(self, tmp_path: pathlib.Path) -> None:
        simba.search.activity_tracker.log_activity(tmp_path, "grep", "searched for foo")
        with simba.db.get_db(tmp_path) as conn:
            rows = conn.execute("SELECT tool_name, detail FROM activities").fetchall()
        assert len(rows) == 1
        assert rows[0]["tool_name"] == "grep"
        assert rows[0]["detail"] == "searched for foo"

    def test_timestamp_is_stored(self, tmp_path: pathlib.Path) -> None:
        simba.search.activity_tracker.log_activity(tmp_path, "read", "file.py")
        with simba.db.get_db(tmp_path) as conn:
            row = conn.execute("SELECT timestamp FROM activities").fetchone()
        assert row["timestamp"] is not None
        # Basic format check: "YYYY-MM-DD HH:MM:SS"
        assert len(row["timestamp"]) == 19

    def test_inserts_multiple_entries(self, tmp_path: pathlib.Path) -> None:
        simba.search.activity_tracker.log_activity(tmp_path, "grep", "first")
        simba.search.activity_tracker.log_activity(tmp_path, "read", "second")
        with simba.db.get_db(tmp_path) as conn:
            rows = conn.execute(
                "SELECT tool_name FROM activities ORDER BY id ASC"
            ).fetchall()
        assert len(rows) == 2
        assert rows[0]["tool_name"] == "grep"
        assert rows[1]["tool_name"] == "read"


# ---------------------------------------------------------------------------
# TestReadActivityLog
# ---------------------------------------------------------------------------


class TestReadActivityLog:
    def test_returns_tuples_in_chronological_order(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Insert rows directly for deterministic timestamps
        with simba.db.get_db(tmp_path) as conn:
            conn.execute(
                "INSERT INTO activities (timestamp, tool_name, detail) "
                "VALUES (?, ?, ?)",
                ("2024-01-01 10:00:00", "grep", "searched foo"),
            )
            conn.execute(
                "INSERT INTO activities (timestamp, tool_name, detail) "
                "VALUES (?, ?, ?)",
                ("2024-01-01 10:01:00", "read", "file.py"),
            )
            conn.commit()

        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 2
        assert entries[0] == ("2024-01-01 10:00:00", "grep", "searched foo")
        assert entries[1] == ("2024-01-01 10:01:00", "read", "file.py")

    def test_returns_empty_list_when_db_missing(self, tmp_path: pathlib.Path) -> None:
        # Point to a path that does not have a DB file
        nonexistent = tmp_path / "nonexistent"
        nonexistent.mkdir()
        entries = simba.search.activity_tracker.read_activity_log(nonexistent)
        assert entries == []


# ---------------------------------------------------------------------------
# TestClearActivityLog
# ---------------------------------------------------------------------------


class TestClearActivityLog:
    def test_deletes_all_rows(self, tmp_path: pathlib.Path) -> None:
        simba.search.activity_tracker.log_activity(tmp_path, "grep", "foo")
        simba.search.activity_tracker.log_activity(tmp_path, "read", "bar")
        simba.search.activity_tracker.clear_activity_log(tmp_path)
        with simba.db.get_db(tmp_path) as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM activities").fetchone()["c"]
        assert count == 0

    def test_no_error_when_db_missing(self, tmp_path: pathlib.Path) -> None:
        # Point to a path that does not have a DB file -- should not raise
        nonexistent = tmp_path / "nonexistent"
        nonexistent.mkdir()
        simba.search.activity_tracker.clear_activity_log(nonexistent)


# ---------------------------------------------------------------------------
# TestRotation
# ---------------------------------------------------------------------------


class TestRotation:
    def test_keeps_only_last_200_rows(self, tmp_path: pathlib.Path) -> None:
        with simba.db.get_db(tmp_path) as conn:
            for i in range(210):
                conn.execute(
                    "INSERT INTO activities (timestamp, tool_name, detail) "
                    "VALUES (?, ?, ?)",
                    (f"2024-01-01 00:00:{i:02d}", f"tool{i}", f"detail{i}"),
                )
            conn.commit()

        # Trigger rotation via log_activity
        simba.search.activity_tracker.log_activity(tmp_path, "trigger", "rotation")

        with simba.db.get_db(tmp_path) as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM activities").fetchone()["c"]
        # 210 existing + 1 new = 211, rotation keeps last 200
        assert count == 200

    def test_no_rotation_under_200_rows(self, tmp_path: pathlib.Path) -> None:
        with simba.db.get_db(tmp_path) as conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO activities (timestamp, tool_name, detail) "
                    "VALUES (?, ?, ?)",
                    (f"2024-01-01 00:00:{i:02d}", f"tool{i}", f"detail{i}"),
                )
            conn.commit()

        # Trigger rotation via log_activity
        simba.search.activity_tracker.log_activity(tmp_path, "trigger", "no-rotation")

        with simba.db.get_db(tmp_path) as conn:
            count = conn.execute("SELECT COUNT(*) AS c FROM activities").fetchone()["c"]
        # 50 + 1 = 51, no rotation needed
        assert count == 51

"""Tests for the PostToolUse hook module."""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.db
import simba.hooks.post_tool_use
import simba.search.activity_tracker


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Point simba.db.get_db_path to a tmp_path-based location."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


class TestPostToolUseHook:
    def test_returns_valid_json(self, tmp_path):
        result = json.loads(
            simba.hooks.post_tool_use.main(
                {"tool_name": "Read", "tool_input": {}, "cwd": str(tmp_path)}
            )
        )
        assert "hookSpecificOutput" in result

    def test_logs_read_tool_with_file_path(self, tmp_path):
        simba.hooks.post_tool_use.main(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "/foo/bar.py"},
                "cwd": str(tmp_path),
            }
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 1
        assert entries[0][1] == "Read"
        assert entries[0][2] == "/foo/bar.py"

    def test_logs_edit_tool_with_file_path(self, tmp_path):
        simba.hooks.post_tool_use.main(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/src/main.py"},
                "cwd": str(tmp_path),
            }
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 1
        assert entries[0][1] == "Edit"
        assert entries[0][2] == "/src/main.py"

    def test_logs_bash_tool_with_truncated_command(self, tmp_path):
        long_cmd = "x" * 200
        simba.hooks.post_tool_use.main(
            {
                "tool_name": "Bash",
                "tool_input": {"command": long_cmd},
                "cwd": str(tmp_path),
            }
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 1
        assert entries[0][1] == "Bash"
        assert len(entries[0][2]) == 100

    def test_logs_grep_tool_with_pattern(self, tmp_path):
        simba.hooks.post_tool_use.main(
            {
                "tool_name": "Grep",
                "tool_input": {"pattern": "TODO"},
                "cwd": str(tmp_path),
            }
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 1
        assert entries[0][1] == "Grep"
        assert entries[0][2] == "TODO"

    def test_logs_task_tool_with_agent_type_and_description(self, tmp_path):
        simba.hooks.post_tool_use.main(
            {
                "tool_name": "Task",
                "tool_input": {
                    "subagent_type": "Explore",
                    "description": "find patterns",
                },
                "cwd": str(tmp_path),
            }
        )
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 1
        assert entries[0][1] == "Task"
        assert entries[0][2] == "Explore: find patterns"

    def test_skips_when_no_tool_name(self, tmp_path):
        result = json.loads(simba.hooks.post_tool_use.main({"cwd": str(tmp_path)}))
        entries = simba.search.activity_tracker.read_activity_log(tmp_path)
        assert len(entries) == 0
        assert "hookSpecificOutput" in result

    def test_no_crash_on_activity_tracker_error(self, tmp_path):
        result = json.loads(
            simba.hooks.post_tool_use.main(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/foo.py"},
                    "cwd": str(tmp_path),
                }
            )
        )
        assert "hookSpecificOutput" in result

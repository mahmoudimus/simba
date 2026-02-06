"""Tests for the PostToolUse hook module."""

from __future__ import annotations

import json
import unittest.mock

import simba.hooks.post_tool_use


class TestPostToolUseHook:
    def test_returns_valid_json(self, tmp_path):
        with unittest.mock.patch("simba.search.activity_tracker.log_activity"):
            result = json.loads(
                simba.hooks.post_tool_use.main(
                    {"tool_name": "Read", "tool_input": {}, "cwd": str(tmp_path)}
                )
            )
        assert "hookSpecificOutput" in result

    def test_logs_read_tool_with_file_path(self, tmp_path):
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity"
        ) as mock_log:
            simba.hooks.post_tool_use.main(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/foo/bar.py"},
                    "cwd": str(tmp_path),
                }
            )
        mock_log.assert_called_once_with(tmp_path, "Read", "/foo/bar.py")

    def test_logs_edit_tool_with_file_path(self, tmp_path):
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity"
        ) as mock_log:
            simba.hooks.post_tool_use.main(
                {
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/src/main.py"},
                    "cwd": str(tmp_path),
                }
            )
        mock_log.assert_called_once_with(tmp_path, "Edit", "/src/main.py")

    def test_logs_bash_tool_with_truncated_command(self, tmp_path):
        long_cmd = "x" * 200
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity"
        ) as mock_log:
            simba.hooks.post_tool_use.main(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": long_cmd},
                    "cwd": str(tmp_path),
                }
            )
        mock_log.assert_called_once()
        _, _, detail = mock_log.call_args[0]
        assert len(detail) == 100

    def test_logs_grep_tool_with_pattern(self, tmp_path):
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity"
        ) as mock_log:
            simba.hooks.post_tool_use.main(
                {
                    "tool_name": "Grep",
                    "tool_input": {"pattern": "TODO"},
                    "cwd": str(tmp_path),
                }
            )
        mock_log.assert_called_once_with(tmp_path, "Grep", "TODO")

    def test_logs_task_tool_with_agent_type_and_description(self, tmp_path):
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity"
        ) as mock_log:
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
        mock_log.assert_called_once_with(tmp_path, "Task", "Explore: find patterns")

    def test_skips_when_no_tool_name(self, tmp_path):
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity"
        ) as mock_log:
            result = json.loads(simba.hooks.post_tool_use.main({"cwd": str(tmp_path)}))
        mock_log.assert_not_called()
        assert "hookSpecificOutput" in result

    def test_no_crash_on_activity_tracker_error(self, tmp_path):
        with unittest.mock.patch(
            "simba.search.activity_tracker.log_activity",
            side_effect=OSError("disk full"),
        ):
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

"""Tests for tailor session_start module — JSON output, git info."""

from __future__ import annotations

import json
import pathlib

import simba.tailor.session_start


class TestGatherGitStatus:
    def test_returns_no_repo_outside_git(self, tmp_path: pathlib.Path):
        status = simba.tailor.session_start.gather_git_status(cwd=tmp_path)
        assert status == "no repo"

    def test_returns_string(self, tmp_path: pathlib.Path):
        status = simba.tailor.session_start.gather_git_status(cwd=tmp_path)
        assert isinstance(status, str)


class TestGatherContext:
    def test_contains_time(self, tmp_path: pathlib.Path):
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        assert "Time:" in ctx

    def test_contains_git(self, tmp_path: pathlib.Path):
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        assert "Git:" in ctx

    def test_no_marks_section(self, tmp_path: pathlib.Path):
        """/mark never survived the JS->Python port; the line must not advertise it."""
        memory_dir = tmp_path / ".simba" / "tailor"
        memory_dir.mkdir(parents=True)
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        assert "Marks:" not in ctx
        assert "/mark" not in ctx


class TestMain:
    def test_returns_valid_json(self, tmp_path: pathlib.Path):
        result = simba.tailor.session_start.main(cwd=tmp_path)
        data = json.loads(result)
        assert "hookSpecificOutput" in data

    def test_hook_event_name(self, tmp_path: pathlib.Path):
        result = simba.tailor.session_start.main(cwd=tmp_path)
        data = json.loads(result)
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_additional_context(self, tmp_path: pathlib.Path):
        result = simba.tailor.session_start.main(cwd=tmp_path)
        data = json.loads(result)
        assert "Time:" in data["hookSpecificOutput"]["additionalContext"]

    def test_system_message(self, tmp_path: pathlib.Path):
        result = simba.tailor.session_start.main(cwd=tmp_path)
        data = json.loads(result)
        assert "systemMessage" in data

"""Tests for tailor session_start module — JSON output, git info, marks display."""

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


class TestFormatTimeAgo:
    def test_minutes(self):
        assert simba.tailor.session_start.format_time_ago(120) == "2m ago"

    def test_hours(self):
        assert simba.tailor.session_start.format_time_ago(7200) == "2h ago"

    def test_days(self):
        assert simba.tailor.session_start.format_time_ago(172800) == "2d ago"

    def test_zero(self):
        assert simba.tailor.session_start.format_time_ago(0) == "0m ago"


class TestGatherCheckpoints:
    def test_no_memory_dir(self, tmp_path: pathlib.Path):
        marks = simba.tailor.session_start.gather_checkpoints(cwd=tmp_path)
        assert marks == []

    def test_no_progress_files(self, tmp_path: pathlib.Path):
        memory_dir = tmp_path / ".claude-tailor" / "memory"
        memory_dir.mkdir(parents=True)
        marks = simba.tailor.session_start.gather_checkpoints(cwd=tmp_path)
        assert marks == []

    def test_finds_progress_files(self, tmp_path: pathlib.Path):
        memory_dir = tmp_path / ".claude-tailor" / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "progress-feature-a.jsonl").write_text("{}\n")
        (memory_dir / "progress-feature-b.jsonl").write_text("{}\n")
        marks = simba.tailor.session_start.gather_checkpoints(cwd=tmp_path)
        assert len(marks) == 2
        names = [m[0] for m in marks]
        assert "feature-a" in names or "feature-b" in names

    def test_limits_to_3(self, tmp_path: pathlib.Path):
        memory_dir = tmp_path / ".claude-tailor" / "memory"
        memory_dir.mkdir(parents=True)
        for i in range(5):
            (memory_dir / f"progress-mark{i}.jsonl").write_text("{}\n")
        marks = simba.tailor.session_start.gather_checkpoints(cwd=tmp_path)
        assert len(marks) <= 3


class TestGatherContext:
    def test_contains_time(self, tmp_path: pathlib.Path):
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        assert "Time:" in ctx

    def test_contains_git(self, tmp_path: pathlib.Path):
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        assert "Git:" in ctx

    def test_marks_none_message(self, tmp_path: pathlib.Path):
        memory_dir = tmp_path / ".claude-tailor" / "memory"
        memory_dir.mkdir(parents=True)
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        assert "Marks:" in ctx

    def test_no_marks_section_without_memory_dir(self, tmp_path: pathlib.Path):
        ctx = simba.tailor.session_start.gather_context(cwd=tmp_path)
        # Without memory dir, no Marks section
        assert "Time:" in ctx


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

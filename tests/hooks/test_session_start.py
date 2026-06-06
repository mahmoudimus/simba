"""Tests for the SessionStart hook module."""

from __future__ import annotations

import json
import unittest.mock

import simba.hooks.session_start


class TestSessionStartHook:
    def test_returns_valid_json(self):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_includes_tailor_context(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        # Tailor context includes time
        assert "Time:" in ctx

    def test_includes_memory_status_when_healthy(self):
        health = {"memoryCount": 42, "embeddingModel": "nomic-embed-text"}
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=health
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "42 memories" in ctx
        assert "nomic-embed-text" in ctx

    def test_no_memory_status_when_unhealthy(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Semantic Memory" not in ctx

    def test_auto_starts_daemon_if_needed(self):
        health = {"memoryCount": 0, "embeddingModel": "nomic-embed-text"}
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health",
                side_effect=[None, health],
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._auto_start_daemon",
                return_value=True,
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Semantic Memory" in ctx

    def test_empty_input_does_not_crash(self):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        assert "hookSpecificOutput" in result

    def test_includes_project_memory_stats(self, tmp_path):
        health = {"memoryCount": 5, "embeddingModel": "nomic-embed-text"}
        db_file = tmp_path / ".simba" / "simba.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_text("")
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=health
            ),
            unittest.mock.patch("simba.db.get_db_path", return_value=db_file),
            unittest.mock.patch(
                "simba.search.project_memory.get_stats",
                return_value={"sessions": 10, "knowledge": 3, "facts": 7},
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "10 sessions" in ctx
        assert "3 knowledge areas" in ctx
        assert "7 facts" in ctx

    def test_project_memory_error_does_not_crash(self, tmp_path):
        db_file = tmp_path / ".simba" / "simba.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_text("")
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("simba.db.get_db_path", return_value=db_file),
            unittest.mock.patch(
                "simba.search.project_memory.get_stats",
                side_effect=OSError("db error"),
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        assert "hookSpecificOutput" in result

    def _write_meta(self, tmp_path, sid, project, transcript):
        # Project-scoped resolution reads <session>/metadata.json, NOT latest.json.
        d = tmp_path / ".claude" / "transcripts" / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "pending_extraction",
                    "session_id": sid,
                    "project_path": project,
                    "transcript_path": transcript,
                    "exported_at": "2026-06-05T01:00:00Z",
                }
            )
        )

    def test_pending_extraction_included(self, tmp_path):
        proj = str(tmp_path / "proj")
        self._write_meta(tmp_path, "sess-A", proj, "/tmp/a.md")

        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("pathlib.Path.home", return_value=tmp_path),
        ):
            result = json.loads(
                simba.hooks.session_start.main({"session_id": "sess-A", "cwd": proj})
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "learning-extraction-required" in ctx
        assert "/tmp/a.md" in ctx  # this project's transcript
        assert proj in ctx  # --project-path is the resolved (correct) project
        # Extraction-quality rules borrowed from agent-oss.
        assert "Preserve proper nouns" in ctx
        assert "Preserve numeric precision" in ctx
        assert "Resolve relative dates" in ctx

    def test_pending_extraction_is_project_scoped(self, tmp_path):
        # A transcript pending for project B must NOT be offered to a session in
        # project A (the cross-wiring bug). A has nothing pending -> no reminder.
        self._write_meta(tmp_path, "sess-B", str(tmp_path / "projB"), "/tmp/b.md")

        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("pathlib.Path.home", return_value=tmp_path),
        ):
            result = json.loads(
                simba.hooks.session_start.main(
                    {"session_id": "x", "cwd": str(tmp_path / "projA")}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "learning-extraction-required" not in ctx
        assert "/tmp/b.md" not in ctx  # never another project's transcript

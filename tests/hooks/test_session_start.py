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
        mock_conn = unittest.mock.MagicMock()
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=health
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                return_value=mock_conn,
            ),
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
        mock_conn.close.assert_called_once()

    def test_project_memory_error_does_not_crash(self, tmp_path):
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch(
                "simba.search.project_memory.get_connection",
                side_effect=OSError("db error"),
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        assert "hookSpecificOutput" in result

    def test_pending_extraction_included(self, tmp_path):
        # Create fake latest.json
        transcripts_dir = tmp_path / ".claude" / "transcripts"
        transcripts_dir.mkdir(parents=True)
        latest = transcripts_dir / "latest.json"
        latest.write_text(
            json.dumps(
                {
                    "status": "pending_extraction",
                    "session_id": "test-123",
                    "transcript_path": "/tmp/test.md",
                }
            )
        )

        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("pathlib.Path.home", return_value=tmp_path),
        ):
            result = json.loads(
                simba.hooks.session_start.main({"session_id": "test-123"})
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "learning-extraction-required" in ctx

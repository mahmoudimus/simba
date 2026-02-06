"""Tests for the Stop hook module."""

from __future__ import annotations

import json

import simba.hooks.stop


class TestStopHook:
    def test_returns_valid_json(self, tmp_path):
        result = json.loads(simba.hooks.stop.main({"cwd": str(tmp_path)}))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["hookEventName"] == "Stop"

    def test_no_warning_when_signal_present(self, tmp_path):
        result = json.loads(
            simba.hooks.stop.main(
                {
                    "response": "Here is my answer [✓ rules]",
                    "cwd": str(tmp_path),
                }
            )
        )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "MEMORY ALERT" not in ctx

    def test_warning_when_signal_missing(self, tmp_path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Project Rules\nFollow these rules.")

        result = json.loads(
            simba.hooks.stop.main(
                {
                    "response": "Here is my answer without the signal",
                    "cwd": str(tmp_path),
                }
            )
        )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "MEMORY ALERT" in ctx
        assert "Project Rules" in ctx

    def test_no_warning_without_response(self, tmp_path):
        result = json.loads(simba.hooks.stop.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert ctx == ""

    def test_runs_tailor_error_capture(self, tmp_path):
        # Verify process_hook is called (should not crash)
        result = json.loads(
            simba.hooks.stop.main(
                {
                    "transcript_path": str(tmp_path / "nonexistent.jsonl"),
                    "cwd": str(tmp_path),
                }
            )
        )
        assert "hookSpecificOutput" in result

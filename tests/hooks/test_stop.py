"""Tests for the Stop hook module."""

from __future__ import annotations

import json

import simba.hooks.stop


def _stop_context(result: dict) -> str:
    """Extract Stop's model-facing text regardless of render shape.

    Claude (default, unset SIMBA_CLIENT) renders non-empty context as
    ``hookSpecificOutput.additionalContext`` (the schema-driven migration);
    Codex still gets the legacy top-level ``stopReason``. Empty either way ->
    "".
    """
    hso = result.get("hookSpecificOutput")
    if isinstance(hso, dict) and "additionalContext" in hso:
        return hso["additionalContext"]
    return result.get("stopReason", "")


class TestStopHook:
    def test_returns_valid_json(self, tmp_path):
        result = json.loads(simba.hooks.stop.main({"cwd": str(tmp_path)}))
        # No response -> no context -> the empty envelope either way (no
        # hookSpecificOutput, no stopReason).
        assert "hookSpecificOutput" not in result
        assert "stopReason" not in result

    def test_no_warning_when_signal_present(self, tmp_path):
        result = json.loads(
            simba.hooks.stop.main(
                {
                    "response": "Here is my answer [✓ rules]",
                    "cwd": str(tmp_path),
                }
            )
        )
        assert "MEMORY ALERT" not in _stop_context(result)

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
        # Default client (unset SIMBA_CLIENT) resolves to claude -> the new
        # hookSpecificOutput.additionalContext shape (see
        # test_claude_adapter.py's Stop migration tests for the codex side).
        reason = result["hookSpecificOutput"]["additionalContext"]
        assert "MEMORY ALERT" in reason
        assert "Project Rules" in reason

    def test_no_warning_without_response(self, tmp_path):
        result = json.loads(simba.hooks.stop.main({"cwd": str(tmp_path)}))
        assert result == {}

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
        # No response -> no context -> empty envelope (no hookSpecificOutput).
        assert "hookSpecificOutput" not in result


class TestStopSignalFlag:
    """Stop records the [✓ rules] signal flag for the session (spec 25)."""

    def test_records_signal_present(self, tmp_path, monkeypatch):
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        simba.hooks.stop.main(
            {
                "response": "Done. [✓ rules]",
                "cwd": str(tmp_path),
                "session_id": "sig-present",
            }
        )
        assert sf.signal_present("sig-present") is True

    def test_records_signal_absent(self, tmp_path, monkeypatch):
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        simba.hooks.stop.main(
            {
                "response": "Done without the marker.",
                "cwd": str(tmp_path),
                "session_id": "sig-absent",
            }
        )
        assert sf.signal_present("sig-absent") is False

    def test_no_session_id_does_not_crash(self, tmp_path, monkeypatch):
        import simba.guardian.signal_flag as sf

        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        # No session_id — nothing to key the flag on; must not raise.
        result = json.loads(
            simba.hooks.stop.main({"response": "Done. [✓ rules]", "cwd": str(tmp_path)})
        )
        assert "hookSpecificOutput" not in result

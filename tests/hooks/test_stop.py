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


class TestStopResponseBackfill:
    """Claude Code's Stop payload carries NO ``response`` field (only Codex
    and tests supply one) -- diagnosed live as three false ⚠️ ENGAGEMENT
    nudges in a row, because ``has_marker("")`` is always False and the
    guardian ``[✓ rules]`` check silently never ran either. Stop must
    backfill ``response`` from the transcript tail so every consumer sees
    the real last-assistant-message text.
    """

    @staticmethod
    def _transcript(tmp_path, text: str, name: str = "t.jsonl"):
        path = tmp_path / name
        path.write_text(
            "\n".join(
                [
                    json.dumps({"message": {"role": "user", "content": "hi"}}),
                    json.dumps(
                        {
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": text}],
                            }
                        }
                    ),
                ]
            )
        )
        return path

    @staticmethod
    def _patch_marker_cfg(monkeypatch) -> None:
        import simba.hooks.config

        cfg = simba.hooks.config.HooksConfig(
            engagement_marker_enabled=True, reasoning_verify_enabled=False
        )
        monkeypatch.setattr(simba.hooks.stop, "_hooks_cfg", lambda: cfg, raising=False)

    def test_no_false_engagement_nudge_when_marker_in_transcript(
        self, tmp_path, monkeypatch
    ) -> None:
        # False-positive repro: simba recorded engagement, the agent's real
        # response opened with the 🦁☑ ledger AND [✓ rules], but Claude
        # Code's payload has no "response" key at all.
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        self._patch_marker_cfg(monkeypatch)
        ef.record_engagement("sess-bf-echo", ledger="🦁☑ recalled 2 (top 0.5)")
        (tmp_path / "CLAUDE.md").write_text("# Project Rules\nFollow these rules.")

        transcript = self._transcript(
            tmp_path, "Done. 🦁☑ recalled 2 (top 0.5) [✓ rules]"
        )
        out = json.loads(
            simba.hooks.stop.main(
                {
                    "transcript_path": str(transcript),
                    "cwd": str(tmp_path),
                    "session_id": "sess-bf-echo",
                }
            )
        )
        reason = _stop_context(out)
        # Engagement echo-verify sees the real text -> no nudge.
        assert "ENGAGEMENT" not in reason
        # Guardian signal check sees the real text too -> no alert.
        assert "MEMORY ALERT" not in reason

    def test_engagement_nudge_when_marker_missing_from_transcript(
        self, tmp_path, monkeypatch
    ) -> None:
        # True positives are preserved: when the backfilled text genuinely
        # lacks the marker, the nudge still fires.
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        self._patch_marker_cfg(monkeypatch)
        ef.record_engagement("sess-bf-miss", ledger="🦁☑ recalled 2 (top 0.5)")

        transcript = self._transcript(tmp_path, "Done, no marker here.")
        out = json.loads(
            simba.hooks.stop.main(
                {
                    "transcript_path": str(transcript),
                    "cwd": str(tmp_path),
                    "session_id": "sess-bf-miss",
                }
            )
        )
        reason = _stop_context(out)
        assert "🦁☑" in reason

    def test_explicit_response_wins_over_transcript(self, tmp_path) -> None:
        # A runtime-provided response (Codex, or an explicit test payload)
        # always wins over the backfill -- even when the transcript
        # disagrees.
        (tmp_path / "CLAUDE.md").write_text("# Project Rules\nFollow these rules.")
        transcript = self._transcript(tmp_path, "Different text [✓ rules]")
        out = json.loads(
            simba.hooks.stop.main(
                {
                    "response": "explicit answer, no signal here",
                    "transcript_path": str(transcript),
                    "cwd": str(tmp_path),
                }
            )
        )
        reason = _stop_context(out)
        assert "MEMORY ALERT" in reason

    def test_missing_transcript_no_crash_response_stays_empty(self, tmp_path) -> None:
        out = json.loads(
            simba.hooks.stop.main(
                {
                    "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
                    "cwd": str(tmp_path),
                }
            )
        )
        assert out == {}

    def test_transcript_path_camelcase_key_accepted(self, tmp_path) -> None:
        (tmp_path / "CLAUDE.md").write_text("# Project Rules\nFollow these rules.")
        transcript = self._transcript(tmp_path, "no signal in this one")
        out = json.loads(
            simba.hooks.stop.main(
                {
                    "transcriptPath": str(transcript),
                    "cwd": str(tmp_path),
                }
            )
        )
        reason = _stop_context(out)
        assert "MEMORY ALERT" in reason

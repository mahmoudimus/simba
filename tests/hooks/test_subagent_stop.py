"""Tests for the SubagentStop hook module."""

from __future__ import annotations

import json

import simba.hooks.subagent_stop


class TestSubagentStopResponseBackfill:
    """SubagentStop's payload has the same empty-``response`` gap as Stop's
    (Claude Code sends no ``response`` field) -- without a backfill,
    reasoning-verify (Tier 2) runs blind on every subagent turn."""

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
    def _patch_cfg(monkeypatch) -> None:
        import simba.hooks.config

        cfg = simba.hooks.config.HooksConfig(reasoning_verify_enabled=True)
        monkeypatch.setattr(
            simba.hooks.subagent_stop, "_hooks_cfg", lambda: cfg, raising=False
        )

    def test_reasoning_verify_sees_backfilled_transcript_text(
        self, tmp_path, monkeypatch
    ) -> None:
        self._patch_cfg(monkeypatch)
        seen: list[str] = []

        def _spy_verify(response, cwd_str, cfg):
            seen.append(response)
            return None

        monkeypatch.setattr(
            "simba.hooks.reasoning_verify.verify", _spy_verify, raising=False
        )

        transcript = self._transcript(tmp_path, "the subagent's real conclusion")
        result = simba.hooks.subagent_stop.run(
            {
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            }
        )
        assert seen == ["the subagent's real conclusion"]
        assert result.block_reason is None

    def test_explicit_response_wins_over_transcript(
        self, tmp_path, monkeypatch
    ) -> None:
        self._patch_cfg(monkeypatch)
        seen: list[str] = []

        def _spy_verify(response, cwd_str, cfg):
            seen.append(response)
            return None

        monkeypatch.setattr(
            "simba.hooks.reasoning_verify.verify", _spy_verify, raising=False
        )

        transcript = self._transcript(tmp_path, "transcript text, ignored")
        simba.hooks.subagent_stop.run(
            {
                "response": "explicit subagent response",
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            }
        )
        assert seen == ["explicit subagent response"]

    def test_missing_transcript_no_crash_reasoning_verify_not_invoked(
        self, tmp_path, monkeypatch
    ) -> None:
        self._patch_cfg(monkeypatch)
        seen: list[str] = []
        monkeypatch.setattr(
            "simba.hooks.reasoning_verify.verify",
            lambda *a, **kw: seen.append(a) or None,
            raising=False,
        )
        result = simba.hooks.subagent_stop.run(
            {
                "transcript_path": str(tmp_path / "does-not-exist.jsonl"),
                "cwd": str(tmp_path),
            }
        )
        assert seen == []  # response stayed "" -> the `and response` guard skips verify
        assert result.block_reason is None

    def test_returns_empty_result_when_lever_off(self, tmp_path) -> None:
        # Default (reasoning_verify_enabled=False): byte-identical to today
        # even with a transcript backfill available.
        transcript = self._transcript(tmp_path, "some text with a marker")
        result = simba.hooks.subagent_stop.run(
            {
                "transcript_path": str(transcript),
                "cwd": str(tmp_path),
            }
        )
        assert result.block_reason is None
        assert result.additional_context == ""

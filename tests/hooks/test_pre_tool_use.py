"""Tests for the PreToolUse hook module."""

from __future__ import annotations

import dataclasses
import json
import unittest.mock

import simba.db
import simba.hooks.config
import simba.hooks.pre_tool_use


class TestPreToolUseHook:
    def test_skips_unsupported_tools(self):
        result = json.loads(simba.hooks.pre_tool_use.main({"tool_name": "Write"}))
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_skips_missing_transcript(self, tmp_path):
        result = json.loads(
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "transcript_path": str(tmp_path / "nonexistent.jsonl"),
                }
            )
        )
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_returns_memories_for_valid_tool(self, tmp_path):
        # Create transcript with thinking block
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I need to check the auth module",
                    }
                ],
            }
        }
        transcript.write_text(json.dumps(entry) + "\n")

        memories = [{"type": "GOTCHA", "content": "auth uses JWT", "similarity": 0.5}]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=memories,
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # Production layer (run()), not the rendered envelope: whether
            # additional_context reaches Claude's PreToolUse envelope is a
            # RENDER-layer concern covered by test_claude_adapter.py's
            # TestPreToolUseAdditionalContextMigration -- this test only
            # pins that the recall pipeline itself still produces context.
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Read",
                    "transcript_path": str(transcript),
                }
            )
        assert "auth uses JWT" in result.additional_context

    def test_dedup_skips_repeated_thinking(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "repeated thought"}],
            }
        }
        transcript.write_text(json.dumps(entry) + "\n")

        with unittest.mock.patch(
            "simba.hooks.pre_tool_use._check_dedup", return_value=True
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {
                        "tool_name": "Grep",
                        "transcript_path": str(transcript),
                    }
                )
            )
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_passes_project_path(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "exploring the codebase"}],
            }
        }
        transcript.write_text(json.dumps(entry) + "\n")

        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[],
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
        ):
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "transcript_path": str(transcript),
                    "cwd": "/my/project",
                }
            )
        mock_recall.assert_called_once()
        _, kwargs = mock_recall.call_args
        assert kwargs["project_path"] == "/my/project"


def _thinking_transcript(tmp_path, text):
    transcript = tmp_path / "transcript.jsonl"
    entry = {
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": text}],
        }
    }
    transcript.write_text(json.dumps(entry) + "\n")
    return transcript


class TestRecallMinQueryChars:
    """MemOS borrow: parity with UserPromptSubmit's ``prompt_min_length`` floor.

    The general thinking-block recall previously had no length floor at all
    (deferred entirely to the daemon's similarity gate) — ``hooks.
    recall_min_query_chars`` adds the same cheap short-circuit UserPromptSubmit
    already has, so a near-empty thinking block never reaches the daemon.
    """

    def test_default_matches_prompt_min_length(self):
        cfg = simba.hooks.config.HooksConfig()
        assert cfg.recall_min_query_chars == 10
        assert cfg.recall_min_query_chars == cfg.prompt_min_length

    def test_short_thinking_skips_recall_entirely(self, tmp_path):
        # "ok" is 2 chars, well under the 10-char default floor.
        transcript = _thinking_transcript(tmp_path, "ok")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories"
            ) as mock_recall,
            unittest.mock.patch("simba.hooks.pre_tool_use._check_dedup") as mock_dedup,
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Read", "transcript_path": str(transcript)}
                )
            )
        mock_recall.assert_not_called()
        mock_dedup.assert_not_called()  # gate short-circuits before dedup work
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_long_enough_thinking_still_recalls(self, tmp_path):
        # Exactly at the floor (10 chars) — inclusive boundary still recalls.
        transcript = _thinking_transcript(tmp_path, "a" * 10)
        memories = [{"type": "GOTCHA", "content": "at the floor", "similarity": 0.5}]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=memories,
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # Production layer (run()) -- see the note in
            # test_returns_memories_for_valid_tool above.
            result = simba.hooks.pre_tool_use.run(
                {"tool_name": "Read", "transcript_path": str(transcript)}
            )
        mock_recall.assert_called_once()
        assert "at the floor" in result.additional_context

    def test_custom_floor_configurable(self, tmp_path, monkeypatch):
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), recall_min_query_chars=50
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        # 22 chars — clears the default floor but not this raised one.
        transcript = _thinking_transcript(tmp_path, "a fairly short thought")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories"
            ) as mock_recall,
            unittest.mock.patch("simba.hooks.pre_tool_use._check_dedup") as mock_dedup,
        ):
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Read", "transcript_path": str(transcript)}
            )
        mock_recall.assert_not_called()
        mock_dedup.assert_not_called()

    def test_zero_floor_disables_gate(self, tmp_path, monkeypatch):
        # 0 = off: even a 1-char thinking block recalls (parity with
        # prompt_min_length's own semantics, where 0 would mean "no floor").
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), recall_min_query_chars=0
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        transcript = _thinking_transcript(tmp_path, "x")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[],
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
        ):
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Read", "transcript_path": str(transcript)}
            )
        mock_recall.assert_called_once()


class TestPreToolTailBound:
    """PreToolUse must not read a multi-GB transcript whole-file (2026-07-20:
    the recurring driver of daemon RSS balloons under concurrent
    ``/hook/pre_tool`` traffic). ``hooks.pre_tool_tail_mb`` bounds
    ``_extract_thinking`` to the last N bytes of the transcript.
    """

    @staticmethod
    def _big_transcript(
        tmp_path, *, filler_bytes: int, thinking_text: str, thinking_first: bool
    ):
        thinking_entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": thinking_text}],
            }
        }
        filler_line = json.dumps({"type": "user", "m": "x" * filler_bytes})
        lines = (
            [json.dumps(thinking_entry), filler_line]
            if thinking_first
            else [filler_line, json.dumps(thinking_entry)]
        )
        path = tmp_path / "big.jsonl"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_thinking_found_within_tail_window(self, tmp_path, monkeypatch):
        # 3MB of filler, thinking block as the LAST line, a tiny cap that still
        # comfortably reaches back far enough -> found (it's in the tail).
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pre_tool_tail_mb=0.1
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        path = self._big_transcript(
            tmp_path,
            filler_bytes=3_000_000,
            thinking_text="deep dive on the auth module",
            thinking_first=False,
        )
        assert path.stat().st_size > 3_000_000  # sanity: file exceeds the tiny cap
        result = simba.hooks.pre_tool_use._extract_thinking(path)
        assert result == "deep dive on the auth module"

    def test_thinking_beyond_tail_window_not_found(self, tmp_path, monkeypatch):
        # Memory-bound proxy: the thinking block sits at the START of a 3MB
        # file and the 1MB cap never reaches back that far -> "" proves we no
        # longer read the whole file (the old unbounded code would find it).
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pre_tool_tail_mb=1.0
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        path = self._big_transcript(
            tmp_path,
            filler_bytes=3_000_000,
            thinking_text="the one at the start",
            thinking_first=True,
        )
        assert path.stat().st_size > 3_000_000
        result = simba.hooks.pre_tool_use._extract_thinking(path)
        assert result == ""


class _ViolatingLLM:
    def complete_json(self, prompt):
        return {
            "violates": True,
            "why": "reverting+xfail violates the no-weakening rule",
        }


class _NonViolatingLLM:
    def complete_json(self, prompt):
        return {"violates": False}


class TestPitfallGate:
    def test_disabled_by_default_no_fire(self, tmp_path, monkeypatch):
        # Default config has the gate OFF — even a strong doctrine match stays silent.
        # Pin the dataclass default so the assertion holds regardless of any ambient
        # .simba/config.toml (e.g. a parent repo enabling the gate); the rest of this
        # class uses the same _hooks_cfg pin.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=False
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        transcript = _thinking_transcript(tmp_path, "I'll revert and xfail the test")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.9,
            }
        ]
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=hit
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Edit", "transcript_path": str(transcript)}
                )
            )
        # Edit is not an _ENABLED_TOOL and the gate is off → no injection at all.
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_violation_fires_for_mutating_tool(self, tmp_path, monkeypatch):
        # violation mode (default) + LLM says the move violates → fires. Patch the LLM
        # client so the test never shells out to a real provider.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        transcript = _thinking_transcript(tmp_path, "I'll revert and xfail the test")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.82,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # "Edit" is NOT in _ENABLED_TOOLS — the gate must still fire on it.
            # Production layer (run()) -- see the note in
            # test_returns_memories_for_valid_tool above.
            result = simba.hooks.pre_tool_use.run(
                {"tool_name": "Edit", "transcript_path": str(transcript)}
            )
        assert "pitfall-warning" in result.additional_context
        assert "No assertion weakening" in result.additional_context

    def test_silent_when_topical_but_no_violation(self, tmp_path, monkeypatch):
        # The key fix: topically-close candidate the LLM judges NON-violating → silent.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use,
            "_pitfall_llm_client",
            lambda _c: _NonViolatingLLM(),
        )
        transcript = _thinking_transcript(tmp_path, "storing concrete memories now")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "store concrete memories",
                "similarity": 0.86,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = json.loads(
                simba.hooks.pre_tool_use.main(
                    {"tool_name": "Edit", "transcript_path": str(transcript)}
                )
            )
        assert result["hookSpecificOutput"] == {"hookEventName": "PreToolUse"}

    def test_fallback_failure_only_when_no_llm(self, tmp_path, monkeypatch):
        # No LLM available → fall back to FAILURE-only similarity gate.
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: None
        )
        transcript = _thinking_transcript(tmp_path, "applying the same fix again")
        hit = [
            {
                "type": "FAILURE",
                "content": "that fix broke the build",
                "similarity": 0.84,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # Production layer (run()) -- see the note in
            # test_returns_memories_for_valid_tool above.
            result = simba.hooks.pre_tool_use.run(
                {"tool_name": "Edit", "transcript_path": str(transcript)}
            )
        assert (
            "pitfall-warning" in result.additional_context
            and "that fix broke the build" in result.additional_context
        )

    def test_recalls_only_doctrine_types(self, tmp_path, monkeypatch):
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: None
        )
        transcript = _thinking_transcript(tmp_path, "thinking about a risky move")
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=[]
            ) as mock_recall,
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            simba.hooks.pre_tool_use.main(
                {"tool_name": "Edit", "transcript_path": str(transcript), "cwd": "/p"}
            )
        _, kwargs = mock_recall.call_args
        assert kwargs["filters"] == {"types": ["FAILURE", "PREFERENCE", "GOTCHA"]}
        assert kwargs["project_path"] == "/p"

    def test_does_not_fire_on_read_tool(self, tmp_path, monkeypatch):
        # The gate runs only before mutating tools (pitfall_gate_tools). On a read tool
        # (Grep) it must NOT fire even though the LLM would say "violates" — the
        # tool-type gate is what stops it (general recall may still inject context).
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), pitfall_gate_enabled=True
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        transcript = _thinking_transcript(tmp_path, "I'll revert and xfail the test")
        hit = [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.9,
            }
        ]
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=hit
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # Production layer (run()) -- see the note in
            # test_returns_memories_for_valid_tool above.
            result = simba.hooks.pre_tool_use.run(
                {"tool_name": "Grep", "transcript_path": str(transcript)}
            )
        # gate suppressed on a read tool
        assert "pitfall-warning" not in result.additional_context

    def test_check_pitfall_disabled_returns_none(self, monkeypatch):
        # Unit: default config (gate off) → None regardless of input. Pin the
        # dataclass default explicitly: _check_pitfall calls the real _hooks_cfg()
        # itself, so an ambient .simba/config.toml with pitfall_gate_enabled=true
        # would otherwise make this "gate off" characterization reach the real
        # recall_memories daemon call instead of short-circuiting immediately.
        monkeypatch.setattr(
            simba.hooks.pre_tool_use,
            "_hooks_cfg",
            lambda: simba.hooks.config.HooksConfig(),
        )
        assert simba.hooks.pre_tool_use._check_pitfall("anything", None) is None


class TestPitfallGatePayloadThinking:
    """The v2.1 pi path: ``thinking`` arrives in the payload (no transcript file).

    A fired pitfall must escalate to ``CanonicalResult.escalated_block`` (so pi's
    block-only tool gate enforces it) while still appearing in ``additional_context``
    (so Claude/Codex see it unchanged).
    """

    def _hit(self):
        return [
            {
                "type": "PREFERENCE",
                "content": "No assertion weakening",
                "similarity": 0.82,
            }
        ]

    @staticmethod
    def _cfg(monkeypatch, **over):
        """Pin ``_hooks_cfg`` for these tests, with ``rule_check_enabled`` forced
        off regardless of ``over``.

        Every test in this class passes a truthy ``tool_input``
        (``{"command": "git revert"}``) with no ``cwd``, so ``run()`` also drives
        the unrelated TOOL_RULE gate (``_recall_tool_rules``) alongside whatever
        the pitfall gate is doing. That gate resolves its project id via
        ``simba.db.resolve_project_id(None)`` -> the *process* cwd when the
        payload has none — worktree-robust by design, ``find_repo_root`` walks
        past a worktree's ``.git`` FILE up to the parent repo's ``.git``
        DIRECTORY — and then asks the real daemon/cache whether that project has
        TOOL_RULE rows. In a checkout with a populated ``.simba/`` (real rows,
        or just a reachable daemon — ``_project_has_tool_rules`` fails OPEN when
        the daemon can't be reached at all), that comes back True and the gate
        consumes THIS file's ``recall_memories`` mock too, leaking an unrelated
        ``escalated_block`` into these pitfall-only assertions. None of these
        tests exercise TOOL_RULE, so it is pinned off here instead of relying on
        cwd isolation alone (see ``test_isolated_from_populated_live_tool_rule_state``
        below for the regression this closes).
        """
        cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), rule_check_enabled=False, **over
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        return cfg

    def test_payload_thinking_escalates_pitfall(self, tmp_path, monkeypatch):
        self._cfg(monkeypatch, pitfall_gate_enabled=True)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=self._hit()
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            # No transcript_path — only payload thinking, as pi sends it. cwd is
            # pinned to the isolated tmp_path so the sibling redirect/check_command
            # path (which reads real config independent of the _cfg pin above)
            # can't consult whatever repo the suite happens to run from either.
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "thinking": "I'll revert and xfail the test",
                    "cwd": str(tmp_path),
                }
            )
        assert result.escalated_block is not None
        assert "pitfall-warning" in result.escalated_block
        # The directive also rides in additional_context (Claude/Codex channel).
        assert "pitfall-warning" in result.additional_context
        assert result.escalated_block in result.additional_context

    def test_no_thinking_no_escalation(self, tmp_path, monkeypatch):
        # No transcript, no payload thinking → the gate never runs → no escalation.
        self._cfg(monkeypatch, pitfall_gate_enabled=True)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=self._hit()
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "cwd": str(tmp_path),
                }
            )
        assert result.escalated_block is None
        assert "pitfall-warning" not in result.additional_context

    def test_disabled_gate_no_escalation_with_thinking(self, tmp_path, monkeypatch):
        # Gate OFF (default) → payload thinking is ignored, no escalation.
        self._cfg(monkeypatch, pitfall_gate_enabled=False)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=self._hit()
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "thinking": "I'll revert and xfail the test",
                    "cwd": str(tmp_path),
                }
            )
        assert result.escalated_block is None

    def test_isolated_from_populated_live_tool_rule_state(self, tmp_path, monkeypatch):
        """Regression fixture for the leak ``_cfg`` documents above.

        Simulates the exact live-checkout condition from the bug report: the
        resolved project genuinely HAS TOOL_RULE rows (a populated
        ``.simba/simba.db``). Forcing that counterfactual explicitly (rather
        than depending on whatever the real machine's daemon/``.simba`` happens
        to contain when the suite runs) makes this test a deterministic tripwire
        — it stays green only because ``_cfg`` pins ``rule_check_enabled=False``;
        if that pin is ever removed, ``_recall_tool_rules`` reaches the
        ``recall_memories`` mock below and leaks its hit into
        ``escalated_block``, and this test goes red.
        """
        self._cfg(monkeypatch, pitfall_gate_enabled=False)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        # The project resolves to a stable id that genuinely "has" TOOL_RULE rows.
        monkeypatch.setattr(
            simba.db, "resolve_project_id", lambda p=None: "live-like-project-id"
        )
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_project_has_tool_rules", lambda *a, **k: True
        )
        with (
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=self._hit()
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "thinking": "I'll revert and xfail the test",
                    "cwd": str(tmp_path),
                }
            )
        assert result.escalated_block is None

    def test_payload_thinking_preferred_over_transcript(self, tmp_path, monkeypatch):
        # When both are present, the payload thinking wins (pi never sends both, but
        # the precedence must be deterministic). Transcript holds a NON-violating
        # thought; the payload holds the violating one → the gate fires on the payload.
        self._cfg(monkeypatch, pitfall_gate_enabled=True)
        monkeypatch.setattr(
            simba.hooks.pre_tool_use, "_pitfall_llm_client", lambda _c: _ViolatingLLM()
        )
        seen = {}

        def _check(thinking, cwd):
            seen["thinking"] = thinking
            return None  # don't matter; we only assert which text was passed

        monkeypatch.setattr(simba.hooks.pre_tool_use, "_check_pitfall", _check)
        transcript = _thinking_transcript(tmp_path, "harmless transcript thought")
        with (
            # _check_pitfall is stubbed above, but the general thinking-block
            # recall (tool_name "Bash" is in _ENABLED_TOOLS) still runs and would
            # otherwise hit the real daemon; this test only cares about
            # seen["thinking"], so an empty recall is enough.
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories", return_value=[]
            ),
            unittest.mock.patch(
                "simba.hooks.pre_tool_use._check_dedup", return_value=False
            ),
            unittest.mock.patch("simba.hooks.pre_tool_use._save_hash"),
        ):
            simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "git revert"},
                    "transcript_path": str(transcript),
                    "thinking": "payload violating thought",
                    "cwd": str(tmp_path),
                }
            )
        assert seen["thinking"] == "payload violating thought"


class TestCheckContextLow:
    def test_below_threshold_returns_none(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("small file\n")
        assert simba.hooks.pre_tool_use._check_context_low(transcript) is None

    def test_above_threshold_returns_warning(self, tmp_path, monkeypatch):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        result = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert result is not None
        assert "context-low-warning" in result
        assert "0.0MB" in result

    def test_warns_only_once(self, tmp_path, monkeypatch):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        first = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert first is not None

        second = simba.hooks.pre_tool_use._check_context_low(transcript)
        assert second is None

    def test_different_transcript_resets(self, tmp_path, monkeypatch):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        t1 = tmp_path / "t1.jsonl"
        t1.write_text("x" * 200 + "\n")
        t2 = tmp_path / "t2.jsonl"
        t2.write_text("x" * 200 + "\n")

        assert simba.hooks.pre_tool_use._check_context_low(t1) is not None
        assert simba.hooks.pre_tool_use._check_context_low(t1) is None
        assert simba.hooks.pre_tool_use._check_context_low(t2) is not None

    def test_nonexistent_transcript(self, tmp_path):
        assert simba.hooks.pre_tool_use._check_context_low(tmp_path / "nope") is None

    def test_fires_for_non_enabled_tool(self, tmp_path, monkeypatch):
        """Context warning fires even for tools not in _ENABLED_TOOLS."""
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        # Production layer (run()) -- see the note in
        # test_returns_memories_for_valid_tool above.
        result = simba.hooks.pre_tool_use.run(
            {"tool_name": "Write", "transcript_path": str(transcript)}
        )
        assert "context-low-warning" in result.additional_context


class TestContextLowSystemMessage:
    """Compact relay leg C: a fired context-low warning also carries a short
    ``systemMessage`` nudge, riding the same warn-once-per-boundary arming as
    the existing ``additional_context`` warning (unchanged)."""

    def _low_cfg(self, monkeypatch, tmp_path):
        low_cfg = dataclasses.replace(
            simba.hooks.config.HooksConfig(), context_low_bytes=100
        )
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: low_cfg)
        flag = tmp_path / "flag.json"
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_CONTEXT_LOW_FLAG", flag)

    def test_fire_sets_system_message_on_canonical_result(self, tmp_path, monkeypatch):
        self._low_cfg(monkeypatch, tmp_path)
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Read",
                "tool_input": {},
                "transcript_path": str(transcript),
            }
        )
        assert result.system_message
        assert "compact" in result.system_message.lower()

    def test_no_fire_leaves_system_message_empty(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("small\n")

        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Read",
                "tool_input": {},
                "transcript_path": str(transcript),
            }
        )
        assert result.system_message == ""

    def test_no_transcript_path_leaves_system_message_empty(self):
        result = simba.hooks.pre_tool_use.run({"tool_name": "Read", "tool_input": {}})
        assert result.system_message == ""

    def test_fire_renders_top_level_system_message(self, tmp_path, monkeypatch):
        self._low_cfg(monkeypatch, tmp_path)
        monkeypatch.delenv("SIMBA_CLIENT", raising=False)
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("x" * 200 + "\n")

        out = json.loads(
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "tool_input": {},
                    "transcript_path": str(transcript),
                }
            )
        )
        assert out.get("systemMessage")
        assert "compact" in out["systemMessage"].lower()

    def test_no_fire_renders_no_top_level_system_message(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("small\n")

        out = json.loads(
            simba.hooks.pre_tool_use.main(
                {
                    "tool_name": "Read",
                    "tool_input": {},
                    "transcript_path": str(transcript),
                }
            )
        )
        assert "systemMessage" not in out


class TestPreflightGate:
    """PreToolUse blocks a mutating tool with no preflight this turn (spec 28)."""

    @staticmethod
    def _cfg(monkeypatch, **over):
        import simba.guardian.preflight_flag as pf

        cfg = dataclasses.replace(simba.hooks.config.HooksConfig(), **over)
        monkeypatch.setattr(simba.hooks.pre_tool_use, "_hooks_cfg", lambda: cfg)
        return cfg, pf

    def test_disabled_by_default_no_block(self, tmp_path, monkeypatch):
        # Gate OFF (default) → a mutating tool with no preflight is NOT blocked
        # (byte-identical to today: no preflight machinery on the path).
        _, pf = self._cfg(monkeypatch)
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/x"},
                "session_id": "s",
            }
        )
        assert result.block_reason is None

    def test_mutating_tool_without_preflight_blocks(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=False,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/x"},
                "session_id": "s",
            }
        )
        assert result.block_reason is not None
        assert "preflight" in result.block_reason.lower()

    def test_read_only_tool_allowed_without_preflight(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=False,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Read",
                    "tool_input": {"file_path": "/x"},
                    "session_id": "s",
                }
            )
        assert result.block_reason is None  # read-only never gated

    def test_preflight_clears_the_gate(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=False,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.set_preflight("s", task="edit the file")  # preflight ran this turn
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/x"},
                    "session_id": "s",
                }
            )
        assert result.block_reason is None

    def test_risk_only_not_armed_without_risk_prime(self, tmp_path, monkeypatch):
        # risk_only=True (default) and no risk-tier prime this turn → gate is NOT
        # armed, so a mutating tool with no preflight is allowed.
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=True,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            result = simba.hooks.pre_tool_use.run(
                {
                    "tool_name": "Edit",
                    "tool_input": {"file_path": "/x"},
                    "session_id": "s",
                }
            )
        assert result.block_reason is None

    def test_risk_only_armed_blocks_without_preflight(self, tmp_path, monkeypatch):
        _, pf = self._cfg(
            monkeypatch,
            preflight_mandate_enabled=True,
            preflight_mandate_risk_only=True,
        )
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.arm_mandate("s")  # a risk-tier doctrine was primed this turn
        result = simba.hooks.pre_tool_use.run(
            {
                "tool_name": "Edit",
                "tool_input": {"file_path": "/x"},
                "session_id": "s",
            }
        )
        assert result.block_reason is not None

"""Pins Claude/Codex hook envelopes so the canonical refactor stays byte-identical."""

from __future__ import annotations

import json

import simba.harness.adapters.claude as claude
import simba.hooks.pre_compact
import simba.hooks.session_start
import simba.hooks.stop
import simba.hooks.user_prompt_submit
from simba.harness.core import CanonicalResult


def test_user_prompt_submit_empty_prompt_envelope():
    out = simba.hooks.user_prompt_submit.main({"prompt": "", "cwd": "/tmp"})
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"


def test_stop_no_response_is_empty_object():
    out = simba.hooks.stop.main({"cwd": "/tmp"})
    assert json.loads(out) == {}


def test_pre_compact_missing_fields_is_bare_empty_object():
    out = simba.hooks.pre_compact.main({})
    assert json.loads(out) == {}


def test_session_start_returns_session_start_envelope():
    out = simba.hooks.session_start.main({"cwd": "/tmp"})
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "SessionStart"


def test_render_user_prompt_submit_with_context():
    out = claude.render("UserPromptSubmit", CanonicalResult(additional_context="hi"))
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["additionalContext"] == "hi"


def test_render_stop_with_context_uses_stop_reason():
    out = claude.render("Stop", CanonicalResult(additional_context="WARN"))
    assert json.loads(out) == {"stopReason": "WARN"}


def test_render_stop_empty_is_empty_object():
    assert json.loads(claude.render("Stop", CanonicalResult())) == {}


def test_render_pre_compact_is_bare_empty_object():
    # Claude Code's tightened hook schema has NO PreCompact variant in the
    # hookSpecificOutput union, and top-level fields (suppressOutput
    # included) are all optional -- the only shape guaranteed to validate is
    # a bare {}. See src/simba/harness/adapters/claude.py's PreCompact branch.
    out = claude.render("PreCompact", CanonicalResult(suppress_output=True))
    assert json.loads(out) == {}


def test_render_pre_compact_never_emits_hookspecificoutput():
    # Even if a future caller populates additional_context, PreCompact
    # cannot inject context under the new schema -- it must still collapse
    # to a bare {}, never a hookSpecificOutput envelope.
    out = claude.render("PreCompact", CanonicalResult(additional_context="hi"))
    assert json.loads(out) == {}


def test_render_pretool_block_reason_uses_pretool_deny():
    # PreToolUse deny uses the permissionDecision shape, NOT the generic block.
    out = claude.render("PreToolUse", CanonicalResult(block_reason="nope"))
    hso = json.loads(out)["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert hso["permissionDecision"] == "deny"
    assert hso["permissionDecisionReason"] == "nope"


def test_render_generic_block_reason_short_circuits():
    # Non-PreToolUse events still use the generic top-level block envelope.
    out = claude.render("Stop", CanonicalResult(block_reason="nope"))
    assert json.loads(out) == {"decision": "block", "reason": "nope"}


def test_native_to_canonical_map():
    assert claude.NATIVE_TO_CANONICAL["UserPromptSubmit"] == "prompt_submit"
    assert claude.NATIVE_TO_CANONICAL["PreCompact"] == "pre_compact"


class TestCompactRelaySystemMessage:
    """Compact relay leg A: PreCompact's terse ``systemMessage`` for the
    human, gated on the resolved SIMBA_CLIENT (Codex's tolerance for the new
    top-level field is unverified -- an unset client is treated as claude,
    the primary caller)."""

    def test_pre_compact_with_message_and_claude_client_emits_system_message(
        self, monkeypatch
    ):
        monkeypatch.delenv("SIMBA_CLIENT", raising=False)
        out = claude.render(
            "PreCompact",
            CanonicalResult(
                suppress_output=True,
                system_message="simba: exported 3 messages -> /tmp/x",
            ),
        )
        assert json.loads(out) == {
            "systemMessage": "simba: exported 3 messages -> /tmp/x"
        }

    def test_pre_compact_with_explicit_claude_code_client_emits_system_message(
        self, monkeypatch
    ):
        monkeypatch.setenv("SIMBA_CLIENT", "claude-code")
        out = claude.render(
            "PreCompact",
            CanonicalResult(system_message="simba: exported 3 messages -> /tmp/x"),
        )
        assert json.loads(out) == {
            "systemMessage": "simba: exported 3 messages -> /tmp/x"
        }

    def test_pre_compact_empty_system_message_stays_bare_object(self, monkeypatch):
        monkeypatch.delenv("SIMBA_CLIENT", raising=False)
        out = claude.render("PreCompact", CanonicalResult(suppress_output=True))
        assert json.loads(out) == {}

    def test_pre_compact_system_message_suppressed_for_codex_client(self, monkeypatch):
        monkeypatch.setenv("SIMBA_CLIENT", "codex")
        out = claude.render(
            "PreCompact",
            CanonicalResult(system_message="simba: exported 3 messages -> /tmp/x"),
        )
        assert json.loads(out) == {}


class TestPreToolContextLowSystemMessage:
    """Compact relay leg C: the context-low nudge rides the same top-level
    ``systemMessage`` channel, merged alongside PreToolUse's existing
    hookSpecificOutput envelope (whichever shape it takes)."""

    def test_context_injection_with_system_message_adds_top_level_field(
        self, monkeypatch
    ):
        monkeypatch.delenv("SIMBA_CLIENT", raising=False)
        out = claude.render(
            "PreToolUse",
            CanonicalResult(
                additional_context="<context-low-warning>...</context-low-warning>",
                system_message="simba: context is filling up -- /compact now",
            ),
        )
        parsed = json.loads(out)
        assert parsed["systemMessage"] == "simba: context is filling up -- /compact now"
        assert (
            parsed["hookSpecificOutput"]["additionalContext"]
            == "<context-low-warning>...</context-low-warning>"
        )

    def test_context_injection_without_system_message_has_no_top_level_field(self):
        out = claude.render("PreToolUse", CanonicalResult(additional_context="hi"))
        assert "systemMessage" not in json.loads(out)

    def test_pretool_system_message_suppressed_for_codex_client(self, monkeypatch):
        monkeypatch.setenv("SIMBA_CLIENT", "codex")
        out = claude.render(
            "PreToolUse",
            CanonicalResult(additional_context="hi", system_message="simba: nudge"),
        )
        assert "systemMessage" not in json.loads(out)

    def test_pretool_deny_shape_still_gets_top_level_system_message(self, monkeypatch):
        # Even the deny (permissionDecision) shape merges the field in --
        # the gating/merge is generic, not tied to the plain-context branch.
        monkeypatch.delenv("SIMBA_CLIENT", raising=False)
        out = claude.render(
            "PreToolUse",
            CanonicalResult(block_reason="nope", system_message="simba: nudge"),
        )
        parsed = json.loads(out)
        assert parsed["systemMessage"] == "simba: nudge"
        assert parsed["hookSpecificOutput"]["permissionDecision"] == "deny"

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


def test_pre_compact_missing_fields_suppresses_output():
    out = simba.hooks.pre_compact.main({})
    assert json.loads(out) == {"suppressOutput": True}


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


def test_render_pre_compact_suppress():
    out = claude.render("PreCompact", CanonicalResult(suppress_output=True))
    assert json.loads(out) == {"suppressOutput": True}


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

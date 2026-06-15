"""Pins Claude/Codex hook envelopes so the canonical refactor stays byte-identical."""

from __future__ import annotations

import json

import simba.hooks.pre_compact
import simba.hooks.session_start
import simba.hooks.stop
import simba.hooks.user_prompt_submit


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

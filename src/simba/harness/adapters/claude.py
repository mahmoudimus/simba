"""Render a CanonicalResult to the Claude Code / Codex stdout envelope.

Claude and Codex share envelope shapes for the four MVP events, so one adapter
serves both.  Output is byte-identical to the pre-refactor hooks.<event>.main().
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import simba.harness.client
import simba.hooks._io

if TYPE_CHECKING:
    from simba.harness.core import CanonicalResult

# Claude/Codex native event name -> canonical event
NATIVE_TO_CANONICAL = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "prompt_submit",
    "Stop": "stop",
    "SubagentStop": "subagent_stop",
    "PreCompact": "pre_compact",
    "PreToolUse": "pre_tool",
    # v2: "PostToolUse": "post_tool",
    #     "PermissionRequest": "permission_request",
}


def _client_accepts_system_message() -> bool:
    """Whether the resolved SIMBA_CLIENT tolerates a top-level ``systemMessage``.

    Compact relay (PreCompact leg A, PreToolUse context-low leg C) adds new
    top-level envelope surface -- Claude Code renders it fine, but Codex's
    tolerance for an unrecognized top-level field on these events hasn't been
    verified, so it stays suppressed there. An unset ``SIMBA_CLIENT`` (e.g. a
    test calling ``render`` directly, or any caller that never went through
    ``simba hook``'s client resolution) is treated as claude, this adapter's
    primary caller -- see ``simba.__main__._cmd_hook``, which always sets
    ``SIMBA_CLIENT`` before dispatch.
    """
    resolved = os.environ.get("SIMBA_CLIENT", simba.harness.client.CLAUDE_CODE)
    return resolved != simba.harness.client.CODEX


def _with_system_message(out: str, system_message: str) -> str:
    """Merge a top-level ``systemMessage`` into an already-rendered envelope.

    No-op (returns ``out`` byte-identical) when there's nothing to say or the
    resolved client doesn't tolerate the field -- so every existing envelope
    shape (context injection, pretool deny/rewrite, block, ...) stays
    unchanged unless a hook actually populated ``system_message``.
    """
    if not system_message or not _client_accepts_system_message():
        return out
    payload = json.loads(out)
    payload["systemMessage"] = system_message
    return json.dumps(payload)


def render(event: str, result: CanonicalResult) -> str:
    """Render ``result`` for Claude/Codex ``event`` as a JSON string."""
    # PreToolUse has its own deny/rewrite shapes (permissionDecision), distinct
    # from the generic top-level block envelope — so it is handled before the
    # generic block_reason short-circuit. ``escalated_block`` is pi-only metadata
    # (the warning is already in additional_context); render ignores it.
    if event == "PreToolUse":
        if result.transform:
            out = simba.hooks._io.pretool_rewrite(
                result.transform["command"], result.transform.get("reason", "")
            )
        elif result.block_reason:
            out = simba.hooks._io.pretool_deny(result.block_reason)
        else:
            out = simba.hooks._io.context("PreToolUse", result.additional_context)
        # Context-low leg C: the nudge rides top-level systemMessage regardless
        # of which PreToolUse shape fired (context injection is the common
        # case, but the merge is generic, not tied to it).
        return _with_system_message(out, result.system_message)
    # A block decision short-circuits event-specific rendering. v2 (tool gating)
    # may refine this to a per-event deny shape; for now the generic block envelope.
    if result.block_reason:
        return simba.hooks._io.block(result.block_reason)
    if event in ("SessionStart", "UserPromptSubmit"):
        return simba.hooks._io.context(event, result.additional_context)
    if event == "PreCompact":
        # Claude Code's hook schema has no PreCompact variant in the
        # hookSpecificOutput union (only PreToolUse / UserPromptSubmit /
        # PostToolUse / PostToolBatch / Stop / SubagentStop) -- a
        # hookSpecificOutput envelope for PreCompact fails schema validation
        # ("Hook JSON output validation failed -- (root): Invalid input").
        # Every top-level field is optional, so the only shape guaranteed to
        # validate is a bare {}. PreCompact cannot inject additionalContext
        # under the new schema either, so result.additional_context is
        # intentionally ignored here -- stderr remains the channel for
        # human-visible notes (see _export_transcript's print()s). The one
        # top-level field it CAN carry is ``systemMessage`` (compact relay leg
        # A): a terse one-liner telling the human what PreCompact actually did
        # (export / distill spawn) -- empty system_message (nothing happened)
        # still collapses to the bare {}.
        if result.system_message and _client_accepts_system_message():
            return json.dumps({"systemMessage": result.system_message})
        return json.dumps({})
    if event in ("Stop", "SubagentStop"):
        # block_reason is handled by the generic short-circuit above (→
        # {"decision":"block","reason":…}). Otherwise stopReason / empty object.
        if result.additional_context:
            return json.dumps({"stopReason": result.additional_context})
        return json.dumps({})
    return simba.hooks._io.context(event, result.additional_context)

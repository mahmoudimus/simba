"""Render a CanonicalResult to the Claude Code / Codex stdout envelope.

Claude and Codex share envelope shapes for most events, so one adapter serves
both -- but Claude Code's hook schema keeps evolving (PostToolBatch, a
tightened PreToolUse, Stop/SubagentStop additionalContext) while Codex's
tolerance for each change is unverified. Every such change is gated on the
resolved ``SIMBA_CLIENT`` (see ``_client_accepts_system_message``) so Codex
keeps receiving byte-identical output until its own tolerance is confirmed.
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
    # Claude-only (no Codex registration -- see .claude-plugin/hooks.json vs
    # .codex/hooks.json): fires once per tool-call round. Default-off.
    "PostToolBatch": "post_tool_batch",
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
        elif not _client_accepts_system_message():
            # Codex: unchanged legacy shape. Its schema tolerance for the
            # removed additionalContext variant (see the claude branch below)
            # is unverified, so it keeps exactly what it has always received.
            out = simba.hooks._io.context("PreToolUse", result.additional_context)
        else:
            # Claude Code's tightened PreToolUse hookSpecificOutput variant has
            # NO additionalContext anymore (only permissionDecision /
            # permissionDecisionReason / updatedInput survive) -- a
            # hookSpecificOutput.additionalContext envelope for PreToolUse now
            # fails schema validation. Any recalled context is silently
            # dropped here; it migrates to the PostToolBatch lane
            # (simba.hooks.post_tool_batch) instead, which renders on its own
            # (valid) hookSpecificOutput variant.
            out = simba.hooks._io.empty("PreToolUse")
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
        # {"decision":"block","reason":…}). Otherwise: Claude Code's hook
        # schema now documents a real hookSpecificOutput.additionalContext
        # variant for Stop/SubagentStop ("feedback for the model; the
        # conversation continues") -- render there for claude (default).
        # Codex's tolerance for this new variant is unverified, so it keeps
        # the legacy top-level ``stopReason`` shape it has always received
        # (same client gate as the PreToolUse migration above / the
        # systemMessage merge's _client_accepts_system_message).
        if result.additional_context:
            if _client_accepts_system_message():
                return simba.hooks._io.context(event, result.additional_context)
            return json.dumps({"stopReason": result.additional_context})
        return json.dumps({})
    return simba.hooks._io.context(event, result.additional_context)

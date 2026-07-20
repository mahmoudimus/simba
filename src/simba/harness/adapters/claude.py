"""Render a CanonicalResult to the Claude Code / Codex stdout envelope.

Claude and Codex share envelope shapes for the four MVP events, so one adapter
serves both.  Output is byte-identical to the pre-refactor hooks.<event>.main().
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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


def render(event: str, result: CanonicalResult) -> str:
    """Render ``result`` for Claude/Codex ``event`` as a JSON string."""
    # PreToolUse has its own deny/rewrite shapes (permissionDecision), distinct
    # from the generic top-level block envelope — so it is handled before the
    # generic block_reason short-circuit. ``escalated_block`` is pi-only metadata
    # (the warning is already in additional_context); render ignores it.
    if event == "PreToolUse":
        if result.transform:
            return simba.hooks._io.pretool_rewrite(
                result.transform["command"], result.transform.get("reason", "")
            )
        if result.block_reason:
            return simba.hooks._io.pretool_deny(result.block_reason)
        return simba.hooks._io.context("PreToolUse", result.additional_context)
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
        # human-visible notes (see _export_transcript's print()s).
        return json.dumps({})
    if event in ("Stop", "SubagentStop"):
        # block_reason is handled by the generic short-circuit above (→
        # {"decision":"block","reason":…}). Otherwise stopReason / empty object.
        if result.additional_context:
            return json.dumps({"stopReason": result.additional_context})
        return json.dumps({})
    return simba.hooks._io.context(event, result.additional_context)

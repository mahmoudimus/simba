"""Shared stdin/stdout helpers for Claude Code + Codex hooks.

Both runtimes use the same wire format on stdout:
``{"hookSpecificOutput": {"hookEventName": ..., "additionalContext": ...}}``.
The ``hookEventName`` field is required even when the hook has nothing
to inject — this module centralizes that invariant so individual hook
modules can't forget it (the bug fixed in 59ffd4f).

Carve-out: ``PreCompact`` has no ``hookSpecificOutput`` variant in Claude
Code's hook schema at all (the union only covers PreToolUse,
UserPromptSubmit, PostToolUse, PostToolBatch, Stop/SubagentStop) — so
PreCompact never calls into this module and always renders a bare ``{}``
directly in ``simba.harness.adapters.claude.render``.

Codex adds a ``PermissionRequest`` event whose decision uses the same
``hookSpecificOutput`` envelope; the ``decision`` shape is encoded here.
"""

from __future__ import annotations

import json
from typing import Any


def empty(event: str) -> str:
    """Return a minimal valid response with only hookEventName."""
    return json.dumps({"hookSpecificOutput": {"hookEventName": event}})


def context(event: str, additional: str) -> str:
    """Return a response that injects ``additional`` as developer context."""
    if not additional:
        return empty(event)
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": additional,
            }
        }
    )


def block(reason: str) -> str:
    """Return a top-level block decision.

    Used by PreToolUse, UserPromptSubmit, PostToolUse, and Stop.
    """
    return json.dumps({"decision": "block", "reason": reason})


def pretool_deny(reason: str) -> str:
    """Claude-shape PreToolUse deny (model sees ``reason`` and retries)."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


def pretool_rewrite(command: str, reason: str = "") -> str:
    """Claude-shape PreToolUse silent rewrite: allow + updatedInput.command."""
    out: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "updatedInput": {"command": command},
    }
    if reason:
        out["permissionDecisionReason"] = reason
    return json.dumps({"hookSpecificOutput": out})


def permission_decision(behavior: str, message: str = "") -> str:
    """Return a Codex PermissionRequest decision (allow/deny)."""
    decision: dict[str, Any] = {"behavior": behavior}
    if message:
        decision["message"] = message
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": decision,
            }
        }
    )

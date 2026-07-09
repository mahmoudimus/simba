"""PermissionRequest hook (Codex-only) — TOOL_RULE-driven denial.

Codex fires PermissionRequest when it's about to ask the user for an
approval (typically a Bash escalation or an MCP call). If a high-confidence
TOOL_RULE memory matches the proposed call, return ``deny`` so Codex blocks
the action without prompting.  Anything below the deny threshold falls
through silently and Codex uses its normal approval flow.

Output shape (Codex):
    {"hookSpecificOutput": {
        "hookEventName": "PermissionRequest",
        "decision": {"behavior": "deny"|"allow", "message": "..."}}
    }
"""

from __future__ import annotations

import contextlib
import json

import simba.config
import simba.hooks._io
import simba.hooks._memory_client


def _hooks_cfg():
    import simba.hooks.config

    return simba.config.load("hooks")


def _build_query(tool_name: str, tool_input: dict) -> str:
    if tool_name == "Bash":
        return tool_input.get("command", "")[:200]
    if tool_name in ("apply_patch", "Edit", "Write"):
        # apply_patch ships a `command` (the patch text) for native edits;
        # MCP-style file tools may use file_path. Try both.
        return (tool_input.get("command") or tool_input.get("file_path") or "")[:200]
    if tool_name.startswith("mcp__"):
        # MCP tools send all args under tool_input; serialize for matching.
        with contextlib.suppress(TypeError, ValueError):
            return json.dumps(tool_input, sort_keys=True)[:200]
    return ""


def strong_rule_deny_message(memory: dict) -> str:
    """Build the deny message for a strong TOOL_RULE match (shared with PreToolUse).

    Uses the rule's ``content`` plus any ``correction`` in its context. Falls back
    to a generic message when neither is present.
    """
    msg = (memory.get("content") or "")[:180]
    correction = ""
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        ctx = json.loads(memory.get("context") or "{}")
        correction = ctx.get("correction", "") if isinstance(ctx, dict) else ""
    if correction:
        msg = f"{msg} (try: {correction})"
    if not msg:
        msg = "Blocked by TOOL_RULE memory."
    return msg


def main(hook_input: dict) -> str:
    cfg = _hooks_cfg()
    if not cfg.permission_check_enabled:
        return simba.hooks._io.empty("PermissionRequest")

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input") or {}
    cwd_str = hook_input.get("cwd")

    query = _build_query(tool_name, tool_input)
    if not query:
        return simba.hooks._io.empty("PermissionRequest")

    memories = simba.hooks._memory_client.recall_memories(
        query,
        project_path=cwd_str,
        min_similarity=cfg.permission_min_similarity,
        max_results=1,
        filters={"types": ["TOOL_RULE"]},
    )
    if not memories:
        return simba.hooks._io.empty("PermissionRequest")

    top = memories[0]
    sim = float(top.get("similarity", 0))
    if sim < cfg.permission_deny_similarity:
        return simba.hooks._io.empty("PermissionRequest")

    return simba.hooks._io.permission_decision("deny", strong_rule_deny_message(top))

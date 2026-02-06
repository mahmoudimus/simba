"""PostToolUse hook -- activity tracking for session memory."""

from __future__ import annotations

import contextlib
import json
import pathlib

import simba.search.activity_tracker


def main(hook_input: dict) -> str:
    """Track tool usage for activity log."""
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else pathlib.Path.cwd()

    if not tool_name:
        return json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                }
            }
        )

    detail = ""
    if tool_name in ("Read", "Edit", "Write"):
        detail = tool_input.get("file_path", "")
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        detail = cmd[:100]
    elif tool_name in ("Glob", "Grep"):
        detail = tool_input.get("pattern", "")
    elif tool_name == "Task":
        agent = tool_input.get("subagent_type", "")
        desc = tool_input.get("description", "")
        detail = f"{agent}: {desc}" if agent else desc

    with contextlib.suppress(Exception):
        simba.search.activity_tracker.log_activity(cwd, tool_name, detail)

    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
            }
        }
    )

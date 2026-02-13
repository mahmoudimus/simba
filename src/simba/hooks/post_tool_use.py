"""PostToolUse hook — activity tracking + auto-learning from failures.

Tracks tool usage in the activity log. When a tool call fails (non-zero
exit code, error patterns in output), automatically generates a TOOL_RULE
memory so PreToolUse can warn before similar mistakes in the future.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import re
import time

import simba.config
import simba.hooks._memory_client
import simba.search.activity_tracker

# Error patterns that indicate a tool failure worth learning from.
_ERROR_PATTERNS = re.compile(
    r"(?:"
    r"ImportError|ModuleNotFoundError|FileNotFoundError"
    r"|PermissionError|Permission denied"
    r"|command not found|No such file or directory"
    r"|SyntaxError|IndentationError"
    r"|ConnectionRefusedError|ConnectionError"
    r"|OSError: \[Errno"
    r")",
    re.IGNORECASE,
)

# Patterns to normalize commands for generalization.
_NORMALIZE_PATTERNS = [
    # Replace absolute paths with /PATH/
    (re.compile(r"/(?:Users|home)/\S+"), "/PATH/"),
    # Replace hex addresses
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    # Replace line:col numbers
    (re.compile(r":\d+:\d+"), ":LINE:COL"),
    # Replace UUIDs
    (re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ), "UUID"),
]

# Session-level dedup to avoid storing duplicate rules.
_RULE_DEDUP_CACHE = pathlib.Path("/tmp/claude-rule-dedup-cache.json")


def _hooks_cfg():
    import simba.hooks.config

    return simba.config.load("hooks")


def _has_error_pattern(text: str) -> bool:
    """Check if text contains a recognizable error pattern."""
    return bool(_ERROR_PATTERNS.search(text))


def _extract_error_line(text: str) -> str:
    """Extract the first meaningful error line from output."""
    for line in text.split("\n"):
        line = line.strip()
        if _ERROR_PATTERNS.search(line):
            return line[:200]
    return text.split("\n")[-1].strip()[:200] if text else ""


def _normalize_command(command: str) -> str:
    """Normalize a command for pattern matching (strip specifics)."""
    result = command
    for pattern, replacement in _NORMALIZE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _check_rule_dedup(error_hash: str) -> bool:
    """Return True if this error was already stored this session."""
    try:
        cache = json.loads(_RULE_DEDUP_CACHE.read_text())
        hashes = cache.get("hashes", [])
        return error_hash in hashes
    except (json.JSONDecodeError, OSError):
        return False


def _save_rule_dedup(error_hash: str) -> None:
    """Record that we stored a rule for this error pattern."""
    hashes: list[str] = []
    with contextlib.suppress(json.JSONDecodeError, OSError):
        cache = json.loads(_RULE_DEDUP_CACHE.read_text())
        hashes = cache.get("hashes", [])

    hashes.append(error_hash)
    # Keep only last N entries (from config)
    max_rules = _hooks_cfg().rule_max_per_session
    hashes = hashes[-max_rules:]

    with contextlib.suppress(OSError):
        _RULE_DEDUP_CACHE.write_text(
            json.dumps({"hashes": hashes, "timestamp": time.time()})
        )


def _detect_failure(
    tool_name: str, tool_input: dict, tool_response: dict
) -> dict | None:
    """Return failure info if the tool call failed, else None."""
    if tool_name == "Bash":
        # tool_response may have stdout/stderr or a single output field
        stdout = tool_response.get("stdout", "")
        stderr = tool_response.get("stderr", "")
        output = tool_response.get("output", "")
        combined = output or f"{stdout}\n{stderr}"

        if _has_error_pattern(combined):
            return {
                "tool": tool_name,
                "command": tool_input.get("command", "")[:200],
                "error": _extract_error_line(combined),
            }
    return None


def _store_failure_rule(failure: dict, cwd: str) -> None:
    """Store a TOOL_RULE memory from a detected failure."""
    tool = failure["tool"]
    command = failure["command"]
    error = failure["error"]

    # Dedup: hash the normalized command + error
    normalized = _normalize_command(command)
    error_hash = hashlib.md5(
        f"{tool}:{normalized}:{error}".encode()
    ).hexdigest()

    if _check_rule_dedup(error_hash):
        return

    content = f"{tool}: {error}"[:200]
    context_data = {
        "tool": tool,
        "pattern": normalized[:200],
        "error_source": error[:200],
        "correction": "",
    }

    simba.hooks._memory_client.store_memory(
        memory_type="TOOL_RULE",
        content=content,
        context=json.dumps(context_data),
        tags=[tool],
        confidence=0.85,
        project_path=cwd,
    )

    _save_rule_dedup(error_hash)


def main(hook_input: dict) -> str:
    """Track tool usage and learn from failures."""
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response", {})
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else pathlib.Path.cwd()

    if not tool_name:
        return json.dumps(
            {"hookSpecificOutput": {"hookEventName": "PostToolUse"}}
        )

    # --- Activity tracking (existing behavior) ---
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

    # --- Auto-learn from failures ---
    cfg = _hooks_cfg()
    if cfg.auto_learn_from_failures and tool_response:
        with contextlib.suppress(Exception):
            failure = _detect_failure(tool_name, tool_input, tool_response)
            if failure:
                _store_failure_rule(failure, str(cwd))

    return json.dumps(
        {"hookSpecificOutput": {"hookEventName": "PostToolUse"}}
    )

"""PreToolUse hook — tool-rule checking, thinking-based memory recall, truth DB.

Reads stdin JSON with tool_name, tool_input, and transcript_path.

Pipeline (in order):
1. Context-low warning (transcript size check, once per session)
2. Tool-rule check (query TOOL_RULE memories matching current tool call)
3. Truth DB check (query proven facts for Bash commands)
4. Memory recall (extract thinking block, query general memories)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import time

import simba.config
import simba.hooks._memory_client
import simba.hooks._truth_client

_HASH_CACHE = pathlib.Path("/tmp/claude-memory-hash-cache.json")
_CONTEXT_LOW_FLAG = pathlib.Path("/tmp/claude-context-low-flag.json")

_ENABLED_TOOLS = frozenset(
    ["Read", "Grep", "Glob", "Task", "WebSearch", "WebFetch", "Bash"]
)


def _hooks_cfg():
    import simba.hooks.config

    return simba.config.load("hooks")


def _extract_thinking(transcript_path: pathlib.Path) -> str:
    """Extract last thinking block from transcript JSONL."""
    if not transcript_path.exists():
        return ""

    try:
        lines = transcript_path.read_text().strip().split("\n")
    except OSError:
        return ""

    # Read from end to find last assistant thinking
    for line in reversed(lines):
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        message = entry.get("message", {})
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        content = message.get("content", [])
        if not isinstance(content, list):
            continue

        for item in reversed(content):
            if isinstance(item, dict) and item.get("type") == "thinking":
                thinking = item.get("thinking", "")
                return thinking[-_hooks_cfg().thinking_chars:]

    return ""


def _check_dedup(text: str) -> bool:
    """Return True if this text was already processed recently."""
    text_hash = hashlib.md5(text.encode()).hexdigest()

    try:
        cache = json.loads(_HASH_CACHE.read_text())
        if (
            cache.get("lastHash") == text_hash
            and (time.time() - cache.get("timestamp", 0)) < _hooks_cfg().dedup_ttl
        ):
            return True
    except (json.JSONDecodeError, OSError):
        pass

    return False


def _save_hash(text: str) -> None:
    """Save hash to cache file."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    with contextlib.suppress(OSError):
        _HASH_CACHE.write_text(
            json.dumps({"lastHash": text_hash, "timestamp": time.time()})
        )


def _check_context_low(transcript_path: pathlib.Path) -> str | None:
    """Return a warning string if transcript size indicates context is near limit.

    Uses ``os.stat()`` (no file reading) and a flag file so the warning
    fires at most once per transcript.
    """
    try:
        size = transcript_path.stat().st_size
    except OSError:
        return None

    if size < _hooks_cfg().context_low_bytes:
        return None

    # Only warn once per transcript.
    try:
        flag = json.loads(_CONTEXT_LOW_FLAG.read_text())
        if flag.get("transcript") == str(transcript_path):
            return None
    except (json.JSONDecodeError, OSError):
        pass

    with contextlib.suppress(OSError):
        _CONTEXT_LOW_FLAG.write_text(
            json.dumps(
                {"transcript": str(transcript_path), "timestamp": time.time()}
            )
        )

    size_mb = size / 1_000_000
    return (
        "<context-low-warning>\n"
        f"Session transcript has reached {size_mb:.1f}MB — "
        "context is approaching the auto-compact threshold.\n\n"
        "RECOMMENDED: Prepare for context compaction now.\n"
        "1. Summarize your current work state "
        "(what's done, what's pending)\n"
        "2. Note the current branch, files being modified, "
        "and any in-progress changes\n"
        "3. If there are pending tasks, document them clearly\n"
        "4. The pre-compact hook will automatically export the "
        "transcript for learning extraction\n"
        "</context-low-warning>"
    )


def _check_tool_rules(
    tool_name: str, tool_input: dict, cwd_str: str | None
) -> str | None:
    """Query TOOL_RULE memories matching this tool call."""
    cfg = _hooks_cfg()
    if not cfg.rule_check_enabled:
        return None

    # Build a query from the tool input
    if tool_name == "Bash":
        query = tool_input.get("command", "")[:200]
    elif tool_name in ("Read", "Write", "Edit"):
        query = tool_input.get("file_path", "")
    else:
        return None

    if not query:
        return None

    memories = simba.hooks._memory_client.recall_memories(
        query,
        project_path=cwd_str,
        min_similarity=cfg.rule_min_similarity,
        max_results=2,
        filters={"types": ["TOOL_RULE"]},
    )
    if not memories:
        return None

    lines = ["<tool-rule-warning>"]
    for m in memories:
        ctx_str = m.get("context", "{}")
        with contextlib.suppress(json.JSONDecodeError):
            ctx = json.loads(ctx_str)
            correction = ctx.get("correction", "")
            lines.append(f"  WARNING: {m.get('content', '')}")
            if correction:
                lines.append(f"  INSTEAD: {correction}")
    lines.append("</tool-rule-warning>")
    return "\n".join(lines) if len(lines) > 2 else None


def _check_truth_constraints(
    tool_name: str, tool_input: dict
) -> str | None:
    """Check truth DB for facts relevant to this tool call."""
    if tool_name != "Bash":
        return None
    command = tool_input.get("command", "")
    if not command:
        return None
    return simba.hooks._truth_client.query_truth_db(command) or None


def main(hook_input: dict) -> str:
    """Run the PreToolUse hook pipeline. Returns JSON output string."""
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    transcript_path_str = hook_input.get("transcript_path", "")
    cwd_str = hook_input.get("cwd")

    parts: list[str] = []

    # --- Context-low check (fires for any tool, once per session) ---
    if transcript_path_str:
        warning = _check_context_low(pathlib.Path(transcript_path_str))
        if warning:
            parts.append(warning)

    # --- Tool-rule check (fires before thinking recall) ---
    if tool_input:
        rule_warning = _check_tool_rules(tool_name, tool_input, cwd_str)
        if rule_warning:
            parts.append(rule_warning)

        truth_warning = _check_truth_constraints(tool_name, tool_input)
        if truth_warning:
            parts.append(truth_warning)

    # --- Memory recall (only for specific tools with thinking) ---
    if tool_name in _ENABLED_TOOLS and transcript_path_str:
        transcript_path = pathlib.Path(transcript_path_str)
        thinking = _extract_thinking(transcript_path)

        if thinking and not _check_dedup(thinking):
            project_path = cwd_str if cwd_str else None
            memories = simba.hooks._memory_client.recall_memories(
                thinking,
                project_path=project_path,
                min_similarity=_hooks_cfg().min_similarity,
            )

            if memories:
                _save_hash(thinking)

            formatted = simba.hooks._memory_client.format_memories(
                memories, source="thinking-block"
            )
            if formatted:
                parts.append(formatted)


    if not parts:
        return json.dumps({"hookSpecificOutput": {}})

    combined = "\n\n".join(parts)
    tokens = len(combined) // 4
    combined += f"\n[simba: ~{tokens} tokens injected]"
    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": combined,
        }
    })

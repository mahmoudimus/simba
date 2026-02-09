"""PreToolUse hook — thinking-based memory recall with dedup.

Reads stdin JSON with tool_name and transcript_path, extracts the last
thinking block, queries memory daemon, deduplicates via hash cache.
Also monitors transcript size to warn when context is approaching the
auto-compact threshold.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import time

import simba.config
import simba.hooks._memory_client

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


def main(hook_input: dict) -> str:
    """Run the PreToolUse hook pipeline. Returns JSON output string."""
    tool_name = hook_input.get("tool_name", "")
    transcript_path_str = hook_input.get("transcript_path", "")
    cwd_str = hook_input.get("cwd")

    parts: list[str] = []

    # --- Context-low check (fires for any tool, once per session) ---
    if transcript_path_str:
        warning = _check_context_low(pathlib.Path(transcript_path_str))
        if warning:
            parts.append(warning)

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

    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": "\n\n".join(parts),
        }
    })

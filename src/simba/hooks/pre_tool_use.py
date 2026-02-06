"""PreToolUse hook — thinking-based memory recall with dedup.

Reads stdin JSON with tool_name and transcript_path, extracts the last
thinking block, queries memory daemon, deduplicates via hash cache.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import sys
import time

import httpx

_DAEMON_PORT = 8741
_MIN_SIMILARITY = 0.35
_MAX_RESULTS = 3
_TIMEOUT = 2.0
_THINKING_CHARS = 1500
_DEDUP_TTL = 60  # seconds
_HASH_CACHE = pathlib.Path("/tmp/claude-memory-hash-cache.json")

_ENABLED_TOOLS = frozenset(
    ["Read", "Grep", "Glob", "Task", "WebSearch", "WebFetch", "Bash"]
)


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
                return thinking[-_THINKING_CHARS:]

    return ""


def _check_dedup(text: str) -> bool:
    """Return True if this text was already processed recently."""
    text_hash = hashlib.md5(text.encode()).hexdigest()

    try:
        cache = json.loads(_HASH_CACHE.read_text())
        if (
            cache.get("lastHash") == text_hash
            and (time.time() - cache.get("timestamp", 0)) < _DEDUP_TTL
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


def _recall_memories(query: str, project_path: str | None = None) -> list[dict]:
    """Query the memory daemon for relevant memories."""
    url = f"http://localhost:{_DAEMON_PORT}/recall"
    payload: dict = {
        "query": query,
        "minSimilarity": _MIN_SIMILARITY,
        "maxResults": _MAX_RESULTS,
    }
    if project_path:
        payload["projectPath"] = project_path

    try:
        resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        pass
    return []


def _format_memories(memories: list[dict]) -> str:
    """Format recalled memories as XML context."""
    if not memories:
        return ""

    similarities = "-".join(f"{m.get('similarity', 0):.2f}" for m in memories)
    lines = [
        f"[Recalled {len(memories)} memories | similarity: {similarities}]",
        '<recalled-memories source="thinking-block">',
    ]
    for m in memories:
        mtype = m.get("type", "UNKNOWN")
        sim = m.get("similarity", 0)
        content = m.get("content", "")
        lines.append(f'  <memory type="{mtype}" similarity="{sim:.2f}">')
        lines.append(f"    {content}")
        lines.append("  </memory>")
    lines.append("</recalled-memories>")
    return "\n".join(lines)


def main(hook_input: dict) -> str:
    """Run the PreToolUse hook pipeline. Returns JSON output string."""
    tool_name = hook_input.get("tool_name", "")
    transcript_path_str = hook_input.get("transcript_path", "")
    cwd_str = hook_input.get("cwd")

    # Only process for specific tools
    if tool_name not in _ENABLED_TOOLS:
        return json.dumps({"hookSpecificOutput": {}})

    if not transcript_path_str:
        return json.dumps({"hookSpecificOutput": {}})

    transcript_path = pathlib.Path(transcript_path_str)
    thinking = _extract_thinking(transcript_path)
    if not thinking:
        return json.dumps({"hookSpecificOutput": {}})

    # Dedup check
    if _check_dedup(thinking):
        return json.dumps({"hookSpecificOutput": {}})

    project_path = cwd_str if cwd_str else None
    memories = _recall_memories(thinking, project_path=project_path)

    if memories:
        _save_hash(thinking)

    formatted = _format_memories(memories)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": formatted,
        }
    }
    return json.dumps(output)


if __name__ == "__main__":
    hook_data: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    print(main(hook_data))

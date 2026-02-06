"""UserPromptSubmit hook — CORE extraction + memory recall.

Reads stdin JSON with user prompt, extracts CORE blocks from CLAUDE.md,
queries memory daemon for relevant memories, outputs combined context.
"""

from __future__ import annotations

import json
import pathlib
import sys

import httpx

import simba.guardian.extract_core
import simba.search.rag_context

_DAEMON_PORT = 8741
_MIN_PROMPT_LENGTH = 10
_MIN_SIMILARITY = 0.45
_MAX_RESULTS = 3
_TIMEOUT = 2.0


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


def _format_memories(memories: list[dict], source: str) -> str:
    """Format recalled memories as XML context."""
    if not memories:
        return ""

    similarities = "-".join(f"{m.get('similarity', 0):.2f}" for m in memories)
    lines = [
        f"[Recalled {len(memories)} memories | similarity: {similarities}]",
        f'<recalled-memories source="{source}">',
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
    """Run the UserPromptSubmit hook pipeline. Returns JSON output string."""

    prompt = hook_input.get("prompt", "")
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else pathlib.Path.cwd()

    parts: list[str] = []

    # 1. Guardian: extract CORE blocks from CLAUDE.md
    core_blocks = simba.guardian.extract_core.main(cwd=cwd)
    if core_blocks:
        parts.append(core_blocks)

    # 2. Memory: recall relevant memories using prompt
    if prompt and len(prompt) >= _MIN_PROMPT_LENGTH:
        project_path = str(cwd) if cwd_str else None
        memories = _recall_memories(prompt, project_path=project_path)
        formatted = _format_memories(memories, source="user-prompt")
        if formatted:
            parts.append(formatted)

    # 3. Search: project memory + QMD context
    if prompt and len(prompt) >= _MIN_PROMPT_LENGTH:
        try:
            search_ctx = simba.search.rag_context.build_context(prompt, cwd)
            if search_ctx:
                parts.append(search_ctx)
        except Exception:
            pass

    combined = "\n\n".join(parts)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": combined,
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

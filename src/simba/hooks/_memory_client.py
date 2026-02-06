"""Shared memory daemon client — constants, recall, formatting.

Used by user_prompt_submit, pre_tool_use, and session_start hooks.
"""

from __future__ import annotations

import httpx

DAEMON_HOST = "localhost"
DAEMON_PORT = 8741
DEFAULT_MAX_RESULTS = 3
DEFAULT_TIMEOUT = 2.0


def daemon_url() -> str:
    """Return the memory daemon base URL."""
    return f"http://{DAEMON_HOST}:{DAEMON_PORT}"


def recall_memories(
    query: str,
    project_path: str | None = None,
    *,
    min_similarity: float = 0.35,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[dict]:
    """Query the memory daemon for relevant memories."""
    url = f"{daemon_url()}/recall"
    payload: dict = {
        "query": query,
        "minSimilarity": min_similarity,
        "maxResults": max_results,
    }
    if project_path:
        payload["projectPath"] = project_path

    try:
        resp = httpx.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        pass
    return []


def format_memories(memories: list[dict], source: str) -> str:
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

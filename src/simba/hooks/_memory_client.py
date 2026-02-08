"""Shared memory daemon client — constants, recall, formatting.

Used by user_prompt_submit, pre_tool_use, and session_start hooks.
"""

from __future__ import annotations

import httpx

# Lazy-loaded config — read at call time, not import time.
_cfg = None


def _get_cfg():
    global _cfg
    if _cfg is None:
        import simba.config
        import simba.hooks.config

        _ = simba.hooks.config  # side-effect: registers "hooks" section
        _cfg = simba.config.load("hooks")
    return _cfg


def daemon_url() -> str:
    """Return the memory daemon base URL."""
    cfg = _get_cfg()
    return f"http://{cfg.daemon_host}:{cfg.daemon_port}"


def recall_memories(
    query: str,
    project_path: str | None = None,
    *,
    min_similarity: float | None = None,
    max_results: int | None = None,
) -> list[dict]:
    """Query the memory daemon for relevant memories."""
    cfg = _get_cfg()
    if min_similarity is None:
        min_similarity = cfg.min_similarity
    if max_results is None:
        max_results = cfg.default_max_results

    url = f"{daemon_url()}/recall"
    payload: dict = {
        "query": query,
        "minSimilarity": min_similarity,
        "maxResults": max_results,
    }
    if project_path:
        payload["projectPath"] = project_path

    try:
        resp = httpx.post(url, json=payload, timeout=cfg.default_timeout)
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

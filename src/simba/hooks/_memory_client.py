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


def project_scope_chain(cwd: str) -> list[str]:
    """Resolved ancestor-prefix scope chain for ``cwd`` (spec 26).

    Returns ``[cwd-resolved, …ancestors…, git-root-resolved]`` — the resolved cwd
    first, then each parent directory up to (and including) the git root found by
    ``simba.db.find_repo_root``. When no git root is found, the chain is just the
    resolved cwd (bounded — never climbs to the filesystem root across repos).
    Both the stored paths and this chain are ``.resolve()``-d so symlinks (proj's
    symlinked MEMORY.md) don't break ancestry.
    """
    import pathlib

    import simba.db

    base = pathlib.Path(cwd).resolve()
    root = simba.db.find_repo_root(base)
    chain: list[str] = [str(base)]
    if root is None:
        return chain
    root = root.resolve()
    current = base
    while current != root:
        parent = current.parent
        if parent == current:  # reached the filesystem root without hitting `root`
            break
        current = parent
        chain.append(str(current))
        if current == root:
            break
    return chain


def _hierarchical_enabled() -> bool:
    """Whether ``memory.hierarchical_recall`` is on (fail-soft → off)."""
    try:
        return bool(getattr(_memory_cfg(), "hierarchical_recall", False))
    except Exception:
        return False


def recall_memories(
    query: str,
    project_path: str | None = None,
    *,
    min_similarity: float | None = None,
    max_results: int | None = None,
    filters: dict | None = None,
) -> list[dict]:
    """Query the memory daemon for relevant memories.

    When ``memory.hierarchical_recall`` is on and ``project_path`` is an existing
    directory (a real cwd, not an opaque project id), the client-computed
    ancestor-prefix chain is attached as ``projectScopes`` so ancestor/root facts
    inherit down (spec 26). Off → the legacy single-path payload, byte-identical.
    """
    cfg = _get_cfg()
    if max_results is None:
        max_results = cfg.default_max_results

    url = f"{daemon_url()}/recall"
    payload: dict = {
        "query": query,
        "maxResults": max_results,
    }
    # Omit the floor unless explicitly set, so the daemon picks it by query
    # intent (memory.min_similarity / min_similarity_broad).
    if min_similarity is not None:
        payload["minSimilarity"] = min_similarity
    if project_path:
        payload["projectPath"] = project_path
        # Hierarchical recall (spec 26): only for a real filesystem cwd — the
        # TOOL_RULE path scopes by an opaque project_id (resolve_project_id),
        # which is not a directory, so no chain is computed there.
        if _hierarchical_enabled():
            import contextlib
            import pathlib

            with contextlib.suppress(OSError, ValueError):
                if pathlib.Path(project_path).is_dir():
                    payload["projectScopes"] = project_scope_chain(project_path)
    if filters:
        payload["filters"] = filters

    try:
        resp = httpx.post(url, json=payload, timeout=cfg.default_timeout)
        if resp.status_code == 200:
            return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        pass
    return []


def count_memories(
    *,
    memory_type: str,
    project_path: str | None = None,
) -> int | None:
    """Return the count of memories of ``memory_type`` for a project.

    Hits ``GET /list`` (project + type scoped) and reads its ``total``. Returns
    ``None`` on any failure so callers can fail-open (treat "unknown" as "exists")
    rather than wrongly suppress behavior. Used by the PreToolUse tool-rule gate
    to skip the per-call embed+recall for projects with no learned rules.
    """
    cfg = _get_cfg()
    url = f"{daemon_url()}/list"
    params: dict = {"type": memory_type, "limit": 1}
    if project_path:
        params["projectPath"] = project_path
    try:
        resp = httpx.get(url, params=params, timeout=cfg.default_timeout)
        if resp.status_code == 200:
            return int(resp.json().get("total", 0))
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    return None


def store_memory(
    *,
    memory_type: str,
    content: str,
    context: str = "",
    tags: list[str] | None = None,
    confidence: float = 0.90,
    project_path: str | None = None,
) -> dict:
    """Store a memory in the daemon. Returns response dict or empty on error."""
    cfg = _get_cfg()
    url = f"{daemon_url()}/store"
    payload: dict = {
        "type": memory_type,
        "content": content,
    }
    if context:
        payload["context"] = context
    if tags:
        payload["tags"] = tags
    if project_path:
        payload["projectPath"] = project_path
    payload["confidence"] = confidence

    try:
        resp = httpx.post(url, json=payload, timeout=cfg.default_timeout)
        if resp.status_code == 200:
            return resp.json()
    except (httpx.HTTPError, ValueError):
        pass
    return {}


def _memory_cfg():
    """Lazy-load the ``memory`` config (registers the section on first use)."""
    import simba.config
    import simba.memory.config

    _ = simba.memory.config  # side-effect: registers "memory" section
    return simba.config.load("memory")


def _conflict_note(memories: list[dict], query: str) -> str:
    """Gated, fail-open answer-time conflict note (default-OFF → "").

    Loads the memory config, and only when ``conflict_surfacing_enabled`` is set
    does it run the one-LLM-call detection. Disabled (the default) ⇒ zero LLM
    cost and zero behavior change. Any failure returns "" (context left intact).
    """
    if not query:
        return ""
    try:
        cfg = _memory_cfg()
        if not getattr(cfg, "conflict_surfacing_enabled", False):
            return ""
        import simba.llm.client
        import simba.memory.conflict

        llm_client = simba.llm.client.get_client()
        if not llm_client.available():
            return ""
        texts = [m.get("content", "") for m in memories]
        return simba.memory.conflict.conflict_note(
            texts, query, cfg=cfg, llm_client=llm_client
        )
    except Exception:
        return ""


def format_memories(
    memories: list[dict],
    source: str,
    *,
    query: str = "",
) -> str:
    """Format recalled memories as XML context.

    When the ``memory`` config has ``conflict_surfacing_enabled`` (default OFF)
    and ``query`` is supplied, a gated, fail-open conflict-surfacing note is
    appended after the recalled-memories block. With the flag off this is a pure
    no-op (no LLM call, identical output).
    """
    if not memories:
        return ""

    similarities = "-".join(f"{m.get('similarity', 0):.2f}" for m in memories)
    # Flag the most-recently-created memory (ISO-8601 sorts chronologically) so
    # the model can prefer fresher info when memories conflict. Relevance order
    # (RRF) is left untouched — we only annotate.
    newest_idx = -1
    if len(memories) >= 2:
        dates = [(i, m.get("createdAt") or "") for i, m in enumerate(memories)]
        if any(d for _, d in dates):
            newest_idx = max(dates, key=lambda t: t[1])[0]
    lines = [
        f"[Recalled {len(memories)} memories | similarity: {similarities}]",
        f'<recalled-memories source="{source}">',
    ]
    for i, m in enumerate(memories):
        mtype = m.get("type", "UNKNOWN")
        sim = m.get("similarity", 0)
        content = m.get("content", "")
        attrs = f'type="{mtype}" similarity="{sim:.2f}"'
        created = (m.get("createdAt") or "")[:10]
        if created:
            attrs += f' created="{created}"'
        if i == newest_idx:
            attrs += ' recency="newest"'
        lines.append(f"  <memory {attrs}>")
        lines.append(f"    {content}")
        lines.append("  </memory>")
    lines.append("</recalled-memories>")
    note = _conflict_note(memories, query)
    if note:
        lines.append(note)
    return "\n".join(lines)

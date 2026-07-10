"""Episodic consolidation orchestrator (L2).

Engine-gated, fire-and-forget: dispatch the configured RLM engine with an
*episode* prompt; the detached agent reads the session's memories (baked into
the prompt), synthesizes ONE coarse ``EPISODE`` memory, stores it, and closes
the ``episode_jobs`` row. Never blocks a hook.

PreCompact ordering: because eligibility requires a session's raw memories to
already exist (``>= min_memories`` and no ``EPISODE`` yet), a just-ended session
whose digest hasn't stored memories yet is simply not eligible — it is picked up
on a later pass. No special-casing needed.
"""

from __future__ import annotations

import logging
import pathlib
import typing

import httpx

import simba.config
import simba.episodes.config
import simba.episodes.jobs
import simba.hooks._memory_client
import simba.memory.config
import simba.rlm.engine

logger = logging.getLogger("simba.episodes")

_SKIP_TYPES = frozenset({"SYSTEM", "EPISODE"})

# Agentic default (claude-cli): the model runs `simba memory store` itself.
_EPISODE_PROMPT = (
    "These memories were captured during coding session '{sid}' "
    "(project '{cwd}'). Synthesize ONE concise episode that summarizes the "
    "session: its goal, the work done, and the outcome. Store exactly one "
    "memory:\n"
    "  simba memory store --type EPISODE --content <={maxlen}-char summary> "
    "--context <key points + member ids> --project-path '{cwd}' "
    "--session-source '{sid}'\n"
    "Keep --content under {maxlen} characters; put detail in --context. When "
    "finished, run: simba episodes complete '{sid}'.\n\nMemories:\n{members}"
)

# Completion default (llm-cli): the model returns a JSON array; the engine
# stores it. Used automatically when rlm.engine = "llm-cli".
_LLM_EPISODE_PROMPT = (
    "These memories were captured during session '{sid}' (project '{cwd}'). "
    "Synthesize ONE concise episode summarizing the session: its goal, the work "
    "done, and the outcome. Return ONLY a JSON array with a single object: "
    '{{"type": "EPISODE", "content": "<={maxlen}-char summary>", '
    '"context": "key points + member ids"}}. Keep "content" at most {maxlen} '
    "characters. Output nothing but the JSON array.\n\nMemories:\n{members}"
)


def _episode_template(ecfg, engine) -> str:
    """Pick the episode prompt: a config override wins; otherwise the built-in
    default appropriate to the engine family (JSON for the completion engine)."""
    if getattr(ecfg, "episode_prompt", ""):
        return ecfg.episode_prompt
    if isinstance(engine, simba.rlm.engine.LlmCliEngine):
        return _LLM_EPISODE_PROMPT
    return _EPISODE_PROMPT


def _episodes_cfg() -> simba.episodes.config.EpisodesConfig:
    return simba.config.load("episodes")


def _rlm_cfg():
    import simba.rlm.config  # registers the "rlm" section

    _ = simba.rlm.config
    return simba.config.load("rlm")


def _list_memories(daemon_url: str, *, limit: int = 100000) -> list[dict]:
    # Session grouping + episode-prompt building only ever reads these fields
    # (never `vector`); an unprojected /list over the whole corpus was the
    # live incident behind routes.py's `_LIST_DEFAULT_FIELDS` comment.
    fields = "id,type,content,context,projectPath,sessionSource"
    try:
        resp = httpx.get(
            f"{daemon_url}/list",
            params={"limit": limit, "fields": fields},
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        return []


def _group_by_session(memories: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for m in memories:
        sid = m.get("sessionSource") or ""
        if sid:
            groups.setdefault(sid, []).append(m)
    return groups


def _members(group: list[dict]) -> list[dict]:
    return [m for m in group if m.get("type") not in _SKIP_TYPES]


def _has_episode(group: list[dict]) -> bool:
    return any(m.get("type") == "EPISODE" for m in group)


def _member_lines(members: list[dict], max_members: int) -> str:
    lines = []
    for m in members[:max_members]:
        content = (m.get("content") or "").strip()
        line = f"- [{m.get('type', '?')}] {content}"
        ctx = (m.get("context") or "").strip()
        if ctx:
            line += f" ({ctx[:160]})"
        mid = m.get("id", "")
        if mid:
            line += f"  <{mid}>"
        lines.append(line)
    return "\n".join(lines)


def _build_episode_prompt(
    sid: str,
    cwd: str,
    members: list[dict],
    max_members: int,
    *,
    template: str = _EPISODE_PROMPT,
    maxlen: int = 200,
) -> str:
    return template.format(
        sid=sid,
        cwd=cwd,
        members=_member_lines(members, max_members),
        maxlen=maxlen,
    )


def consolidate_session(
    sid: str,
    *,
    cwd: str,
    group: list[dict] | None = None,
    ecfg: simba.episodes.config.EpisodesConfig | None = None,
    engine: typing.Any = None,
    daemon_url: str | None = None,
) -> str:
    """Dispatch consolidation for one session. Returns a status string.

    Statuses: ``dispatched`` | ``disabled`` | ``no_engine`` | ``exists`` |
    ``too_few`` | ``in_progress`` | ``error``.
    """
    ecfg = ecfg or _episodes_cfg()
    if not ecfg.enabled:
        return "disabled"
    if engine is None:
        engine = simba.rlm.engine.get_engine(_rlm_cfg())
    if engine is None:
        return "no_engine"
    if daemon_url is None:
        daemon_url = simba.hooks._memory_client.daemon_url()
    if group is None:
        group = _group_by_session(_list_memories(daemon_url)).get(sid, [])

    if _has_episode(group):
        return "exists"
    members = _members(group)
    if len(members) < ecfg.min_memories:
        return "too_few"

    project_path = members[0].get("projectPath") or str(cwd)
    if not simba.episodes.jobs.claim(
        sid,
        project_path,
        cwd=pathlib.Path(project_path) if project_path else None,
        stale_after_seconds=ecfg.job_timeout_hours * 3600,
    ):
        return "in_progress"

    prompt = _build_episode_prompt(
        sid,
        project_path,
        members,
        ecfg.max_members,
        template=_episode_template(ecfg, engine),
        maxlen=simba.memory.config.resolve_max_content_length(),
    )
    try:
        engine.run(prompt, cwd=project_path, session_source=sid)
    except Exception:
        logger.debug("episode dispatch failed for %s", sid, exc_info=True)
        return "error"
    return "dispatched"


def consolidate_eligible(
    cwd: str,
    *,
    all_projects: bool = False,
    ecfg: simba.episodes.config.EpisodesConfig | None = None,
    engine: typing.Any = None,
    daemon_url: str | None = None,
) -> dict:
    """Consolidate every eligible session belonging to ``cwd``'s project.

    Scoped to sessions whose memories are tagged with this project (so a
    PreCompact in project X never consolidates project Y's sessions) unless
    ``all_projects`` is set (each session dispatches to its own project).
    """
    ecfg = ecfg or _episodes_cfg()
    if not ecfg.enabled:
        return {"dispatched": [], "skipped": 0}
    if engine is None:
        engine = simba.rlm.engine.get_engine(_rlm_cfg())
    if engine is None:
        return {"dispatched": [], "skipped": 0, "no_engine": True}
    if daemon_url is None:
        daemon_url = simba.hooks._memory_client.daemon_url()

    groups = _group_by_session(_list_memories(daemon_url))
    target = None if all_projects else str(cwd)
    dispatched: list[str] = []
    skipped = 0
    for sid, group in groups.items():
        members = _members(group)
        proj = members[0].get("projectPath") if members else None
        if target is not None and (proj or "") != target:
            continue
        status = consolidate_session(
            sid,
            cwd=cwd,
            group=group,
            ecfg=ecfg,
            engine=engine,
            daemon_url=daemon_url,
        )
        if status == "dispatched":
            dispatched.append(sid)
        else:
            skipped += 1
    return {"dispatched": dispatched, "skipped": skipped}

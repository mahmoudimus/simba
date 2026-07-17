"""Episodic consolidation orchestrator (L2).

Engine-gated, fire-and-forget: dispatch the configured RLM engine with an
*episode* prompt; the detached agent reads the session's memories (baked into
the prompt), synthesizes ONE coarse ``EPISODE`` memory, stores it, and closes
the ``episode_jobs`` row. Never blocks a hook.

PreCompact ordering: because eligibility requires a session's raw memories to
already exist (``>= min_memories`` and no ``EPISODE`` yet), a just-ended session
whose digest hasn't stored memories yet is simply not eligible — it is picked up
on a later pass. No special-casing needed.

Two-stage discovery/fetch (2026-07-17 RSS-storm fix): a live incident traced
the daemon's RSS watchdog hard-tripping to this module's ``GET /list?
limit=100000`` pulling the ENTIRE corpus WITH content+context just to (a)
group memories into sessions and (b) grab one session's members. ``_discover``
now does a projected (no content/context) corpus scan -- optionally bounded
by ``since=`` (the incremental watermark, ``simba.episodes.watermark``) -- to
find which sessions exist and pre-check cheap eligibility signals; only a
session that survives that pre-check pays for ``_fetch_session``'s full,
server-side ``sessionSource=``-scoped fetch. See ``consolidate_eligible``'s
docstring for why the too-few pre-check is skipped (not applied) on an
incremental (``since``-bounded) scan.
"""

from __future__ import annotations

import logging
import pathlib
import typing

import httpx

import simba.config
import simba.episodes.config
import simba.episodes.jobs
import simba.episodes.watermark
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


def _discover(
    daemon_url: str, *, since: str | None = None, limit: int = 100000
) -> list[dict]:
    """Projected, corpus-wide scan for session discovery/grouping.

    NEVER fetches ``content``/``context`` -- only what's needed to group
    memories into sessions and pre-check cheap eligibility signals (member
    count via type filtering, ``_has_episode`` via ``type == EPISODE``).
    ``since`` (server-side, ``/list?since=``) bounds the scan to memories
    created after the last completed sweep's watermark; a session that
    gained a new memory since then reappears here and gets a full recheck
    via ``_fetch_session`` (see ``consolidate_eligible``).
    """
    fields = "id,type,sessionSource,projectPath,createdAt"
    params: dict[str, typing.Any] = {"limit": limit, "fields": fields}
    if since:
        params["since"] = since
    try:
        resp = httpx.get(f"{daemon_url}/list", params=params, timeout=15.0)
        resp.raise_for_status()
        return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        return []


def _fetch_session(daemon_url: str, sid: str, *, limit: int = 100000) -> list[dict]:
    """Full-fields fetch, server-side scoped to ONE session (``sessionSource=``).

    Called once ``sid`` has already proven eligible on the projected
    discovery rows (or unconditionally when ``consolidate_session`` is asked
    to resolve its own group).
    """
    fields = "id,type,content,context,projectPath,sessionSource"
    params = {"limit": limit, "fields": fields, "sessionSource": sid}
    try:
        resp = httpx.get(f"{daemon_url}/list", params=params, timeout=15.0)
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
        group = _fetch_session(daemon_url, sid)

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


def _max_created_at(rows: list[dict], *, prior: str | None) -> str | None:
    """The chronologically-latest ``createdAt`` among ``rows`` and ``prior``.

    Parses both sides (``simba.memory.scoring.parse_epoch``) rather than
    comparing strings -- the same mixed-precision trap ``/list``'s ``since=``
    guards against (see routes.py's ``/list`` docstring). Returns ``prior``
    unchanged when no row has a later timestamp (including an empty scan).
    """
    import simba.memory.scoring

    best_str = prior
    best_epoch = simba.memory.scoring.parse_epoch(prior) if prior else None
    for m in rows:
        ts = m.get("createdAt")
        if not ts:
            continue
        epoch = simba.memory.scoring.parse_epoch(ts)
        if epoch is None:
            continue
        if best_epoch is None or epoch > best_epoch:
            best_epoch = epoch
            best_str = ts
    return best_str


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

    Two-stage discovery/fetch (see module docstring): a projected corpus
    scan groups memories into sessions and pre-checks ``_has_episode`` (a
    positive-presence check -- always safe to skip on) before paying for a
    per-session ``_fetch_session``. The member-count ("too few") pre-check is
    applied only on an UNBOUNDED scan (``since is None``): under incremental
    discovery, the projected batch only contains memories created *since*
    the watermark, so its count can only ever UNDER-count a session's true
    total -- trusting it as a negative signal would permanently strand a
    session that was too-few before the watermark and just became eligible.
    Every session that reappears in an incremental scan therefore gets the
    full recheck via ``_fetch_session`` regardless of its partial count; the
    authoritative too-few/exists decision happens inside
    ``consolidate_session`` once the full group is in hand.

    Incremental watermark (``ecfg.incremental_discovery``, per-project via
    ``simba.episodes.watermark``): the discovery scan is bounded by
    ``since=<watermark>`` when set. After the sweep, the watermark advances
    to the max ``createdAt`` observed in the discovery scan -- but ONLY if no
    ``consolidate_session`` call returned ``"error"``; an error leaves the
    watermark unchanged so the next sweep retries the same span.
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

    since = None
    if ecfg.incremental_discovery:
        since = simba.episodes.watermark.get(str(cwd), all_projects=all_projects)

    discovered = _discover(daemon_url, since=since)
    groups = _group_by_session(discovered)
    target = None if all_projects else str(cwd)
    dispatched: list[str] = []
    skipped = 0
    had_error = False
    for sid, group in groups.items():
        members = _members(group)
        proj = members[0].get("projectPath") if members else None
        if target is not None and (proj or "") != target:
            continue
        if _has_episode(group):
            skipped += 1
            continue
        if since is None and len(members) < ecfg.min_memories:
            skipped += 1
            continue

        full_group = _fetch_session(daemon_url, sid)
        status = consolidate_session(
            sid,
            cwd=cwd,
            group=full_group,
            ecfg=ecfg,
            engine=engine,
            daemon_url=daemon_url,
        )
        if status == "dispatched":
            dispatched.append(sid)
        else:
            skipped += 1
        if status == "error":
            had_error = True

    if ecfg.incremental_discovery and not had_error:
        max_created = _max_created_at(discovered, prior=since)
        if max_created is not None and max_created != since:
            simba.episodes.watermark.advance(
                str(cwd), max_created, all_projects=all_projects
            )
    return {"dispatched": dispatched, "skipped": skipped}

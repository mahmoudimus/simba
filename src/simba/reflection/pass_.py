"""reflect_pass orchestrator (Phase 5, Task A.4).

Engine-gated, fire-and-forget: fetch memories from the daemon, gate on
``min_source_memories`` / ``interval_cycles``, dedup against existing
reflections (baked into the prompt), and dispatch the configured RLM engine
with a *reflection* prompt. Never blocks, never raises.

Discovery/fetch split (2026-07-17 RSS-storm fix): this module used to pull
the whole corpus WITH content+context (``limit=100000``) just to count
eligible memories for the ``min_source_memories`` gate -- a live incident
traced the daemon's RSS watchdog hard-tripping to exactly this shape.
``_discover`` now does a projected (no content/context) scan -- server-side
bounded to ``cwd`` when ``project_scoped`` -- so the too-few gate never pays
for content; only once a pass is proven eligible does ``_fetch_source`` pay
for the full fetch (also project-bounded, and capped at ``limit=1000`` for a
global, unscoped pass, well above ``max_source_memories``'s default cap --
see routes.py's ``/list`` context-bound gate).
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import httpx

import simba.config
import simba.hooks._memory_client
import simba.memory.config
import simba.reflection.config
import simba.reflection.prompt
import simba.rlm.engine

if typing.TYPE_CHECKING:
    from simba.reflection.config import ReflectionConfig

logger = logging.getLogger("simba.reflection")

_SKIP_TYPES = frozenset({"SYSTEM", "REFLECTION"})


@dataclasses.dataclass
class ReflectResult:
    status: str  # dispatched|disabled|no_engine|too_few|skipped_interval|error
    memories_considered: int = 0
    existing_reflections: int = 0
    dispatched: bool = False
    errors: int = 0


def _reflection_cfg() -> ReflectionConfig:
    return simba.config.load("reflection")


def _rlm_cfg():
    import simba.rlm.config  # registers the "rlm" section

    _ = simba.rlm.config
    return simba.config.load("rlm")


def _discover(daemon_url: str, *, project_path: str | None = None) -> list[dict]:
    """Projected scan (no content/context) -- used only to count eligible
    source memories for the ``min_source_memories`` gate before paying for
    any content fetch. ``project_path`` (server-side ``projectPath=``) scopes
    a project-scoped pass to just that project instead of the whole corpus.
    """
    fields = "id,type,projectPath"
    params: dict[str, typing.Any] = {"limit": 100000, "fields": fields}
    if project_path:
        params["projectPath"] = project_path
    try:
        resp = httpx.get(f"{daemon_url}/list", params=params, timeout=15.0)
        resp.raise_for_status()
        return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        return []


def _fetch_source(daemon_url: str, *, project_path: str | None = None) -> list[dict]:
    """Full-fields (content+context) fetch -- only called once discovery has
    proven the pass eligible. ``limit=1000`` bounds a global
    (``project_scoped=False``) pass, which has no other server-side row
    bound, well above ``max_source_memories``'s default cap of 100.
    """
    fields = "id,type,content,context,projectPath"
    params: dict[str, typing.Any] = {"limit": 1000, "fields": fields}
    if project_path:
        params["projectPath"] = project_path
    try:
        resp = httpx.get(f"{daemon_url}/list", params=params, timeout=15.0)
        resp.raise_for_status()
        return resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        return []


def reflect_pass(
    *,
    cwd: str,
    cycle_count: int = 0,
    rcfg: ReflectionConfig | None = None,
    engine: object | None = None,
    daemon_url: str | None = None,
) -> ReflectResult:
    """Run one reflection synthesis pass. Returns a ReflectResult. Never raises."""
    rcfg = rcfg or _reflection_cfg()
    if not rcfg.enabled:
        return ReflectResult(status="disabled")

    if rcfg.interval_cycles > 0 and cycle_count % rcfg.interval_cycles != 0:
        return ReflectResult(status="skipped_interval")

    if daemon_url is None:
        daemon_url = simba.hooks._memory_client.daemon_url()

    scope = cwd if rcfg.project_scoped else None

    discovered = _discover(daemon_url, project_path=scope)
    source_count = len([m for m in discovered if m.get("type") not in _SKIP_TYPES])
    if source_count < rcfg.min_source_memories:
        return ReflectResult(status="too_few", memories_considered=source_count)

    # Discovery alone proved eligibility -- now pay for the full
    # content+context fetch (still server-side project-bounded via `scope`).
    all_memories = _fetch_source(daemon_url, project_path=scope)
    source = [m for m in all_memories if m.get("type") not in _SKIP_TYPES]
    existing = [m for m in all_memories if m.get("type") == "REFLECTION"]

    if engine is None:
        engine = simba.rlm.engine.get_engine(_rlm_cfg())
    if engine is None:
        return ReflectResult(
            status="no_engine",
            memories_considered=len(source),
            existing_reflections=len(existing),
        )

    prompt = simba.reflection.prompt.build_reflection_prompt(
        source,
        project=cwd,
        existing_reflections=existing,
        max_source_memories=rcfg.max_source_memories,
        max_reflections=rcfg.max_reflections_per_pass,
        importance_threshold=rcfg.importance_threshold,
        max_content_length=simba.memory.config.resolve_max_content_length(),
    )
    try:
        engine.run(prompt, cwd=cwd)
    except Exception:
        logger.debug("reflection dispatch failed for %s", cwd, exc_info=True)
        return ReflectResult(
            status="error",
            memories_considered=len(source),
            existing_reflections=len(existing),
            errors=1,
        )

    return ReflectResult(
        status="dispatched",
        memories_considered=len(source),
        existing_reflections=len(existing),
        dispatched=True,
    )

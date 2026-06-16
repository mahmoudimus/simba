"""reflect_pass orchestrator (Phase 5, Task A.4).

Engine-gated, fire-and-forget: fetch memories from the daemon, gate on
``min_source_memories`` / ``interval_cycles``, dedup against existing
reflections (baked into the prompt), and dispatch the configured RLM engine
with a *reflection* prompt. Never blocks, never raises.
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


def _list_memories(daemon_url: str, *, limit: int = 100000) -> list[dict]:
    try:
        resp = httpx.get(f"{daemon_url}/list", params={"limit": limit}, timeout=15.0)
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

    all_memories = _list_memories(daemon_url)
    source = [m for m in all_memories if m.get("type") not in _SKIP_TYPES]
    if rcfg.project_scoped:
        source = [m for m in source if (m.get("projectPath") or "") == cwd]

    if len(source) < rcfg.min_source_memories:
        return ReflectResult(status="too_few", memories_considered=len(source))

    existing = [m for m in all_memories if m.get("type") == "REFLECTION"]
    if rcfg.project_scoped:
        existing = [m for m in existing if (m.get("projectPath") or "") == cwd]

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

"""FastAPI routes for the memory daemon.

Ported from claude-memory/routes/*.js — all 6 endpoints in a single file.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import pathlib
import sys
import time
import typing
import uuid

import fastapi
import fastapi.concurrency
import pydantic
import starlette.middleware.base
import starlette.requests
import starlette.responses

import simba.harness.client
import simba.harness.core
import simba.memory.anticipated
import simba.memory.background
import simba.memory.config
import simba.memory.conflict
import simba.memory.conflict_store
import simba.memory.dimensions
import simba.memory.fts
import simba.memory.hybrid
import simba.memory.provenance
import simba.memory.query_filters
import simba.memory.recall_cache
import simba.memory.recall_plan
import simba.memory.rss_watchdog
import simba.memory.scoring
import simba.memory.supersession
import simba.memory.vector_db

logger = logging.getLogger("simba.memory")

router = fastapi.APIRouter()

# Shared with simba.memory.background: spawn() tracks tasks created here (and
# hybrid.py's bg_tasks= reranker task, added directly below) in the SAME
# registry so drain() can find every outstanding fire-and-forget task at
# daemon shutdown (handoff item 10). The name stays for hybrid.py's existing
# ``bg_tasks=_background_tasks`` call site --- only the backing set moved.
_background_tasks = simba.memory.background.TASKS


async def _bg_hyde(
    cache: typing.Any,
    key: str,
    query: str,
    llm_client: typing.Any,
) -> None:
    """Generate the HyDE answer off the hot path and warm the cache (best-effort)."""
    import contextlib

    from simba.memory.hyde import hypothetical_answer

    with contextlib.suppress(Exception):
        text = await asyncio.to_thread(hypothetical_answer, query, llm_client)
        if text:
            cache.put(key, text)


# ── FTS keyword-mirror sync helpers (run in a worker thread via to_thread) ──
# Per-call connections sidestep SQLite's thread-affinity. All mirror writes are
# best-effort: failures are logged, never surfaced (startup reconcile heals).


def _fts_upsert(fts_path: str, memory: dict[str, typing.Any]) -> None:
    with simba.memory.fts.connect(fts_path):
        simba.memory.fts.upsert(memory)


def _fts_delete(fts_path: str, memory_id: str) -> None:
    with simba.memory.fts.connect(fts_path):
        simba.memory.fts.delete(memory_id)


def _fts_set_project(fts_path: str, memory_id: str, project_path: str) -> None:
    with simba.memory.fts.connect(fts_path):
        simba.memory.fts.set_project(memory_id, project_path)


def _fts_retarget(fts_path: str, old: str, new: str) -> None:
    with simba.memory.fts.connect(fts_path):
        simba.memory.fts.retarget_project(old, new)


def _directory_size_bytes(path: pathlib.Path) -> int | None:
    """Return recursive byte size, skipping files that disappear mid-scan."""
    if not path.exists():
        return None
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file():
                stat = child.stat()
                blocks = getattr(stat, "st_blocks", None)
                if isinstance(blocks, int) and blocks > 0:
                    total += blocks * 512
                else:
                    total += stat.st_size
        except OSError:
            continue
    return total


def _jsonable(value: typing.Any) -> typing.Any:
    """Best-effort conversion for LanceDB optimize stats."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


async def _lance_storage_snapshot(
    table: typing.Any, db_path: str | None
) -> dict[str, typing.Any]:
    """Report live Lance table size next to actual on-disk size."""
    stats: dict[str, typing.Any] = {}
    try:
        raw_stats = await table.stats()
        if isinstance(raw_stats, dict):
            stats = raw_stats
    except Exception as exc:
        stats = {"error": type(exc).__name__}

    try:
        rows = await table.count_rows()
    except Exception:
        rows = stats.get("num_rows")

    try:
        versions = len(await table.list_versions())
    except Exception:
        versions = None

    on_disk_bytes: int | None = None
    if db_path:
        on_disk_bytes = await asyncio.to_thread(
            _directory_size_bytes, pathlib.Path(db_path)
        )

    fragment_stats = stats.get("fragment_stats") or {}
    return {
        "path": db_path,
        "rows": rows,
        "liveBytes": stats.get("total_bytes"),
        "onDiskBytes": on_disk_bytes,
        "versions": versions,
        "fragments": fragment_stats.get("num_fragments"),
        "smallFragments": fragment_stats.get("num_small_fragments"),
    }


async def _bump_usage(memory_ids: list[str], now: float, cwd: pathlib.Path) -> None:
    """Bump access stats in ``memory_usage`` for recalled ids. Fire-and-forget.

    The sqlite ``memory_usage`` table is the authoritative ranking sidecar; this
    runs alongside the LanceDB fire-and-forget ``update_access_tracking``. Any
    failure is swallowed (recall must never break on a usage write).

    ``match`` only (spec 33): the daemon knows a memory was RETURNED, not that
    it reached the model's context — only the hook knows what survived its
    lane/budget trim, and it acks via ``POST /recall/ack`` (→ ``inject``).
    """
    if not memory_ids:
        return

    import simba.db
    import simba.memory.usage

    def _sync() -> None:
        try:
            with simba.db.connect(cwd):
                for mid in memory_ids:
                    simba.memory.usage.bump_access(mid, now)
                    simba.memory.usage.bump_quality(mid, now, match=1)
        except Exception:
            logger.debug("[recall] usage bump failed", exc_info=True)

    await asyncio.to_thread(_sync)


def _record_demand(
    query: str, result_count: int, best_score: float, cwd: pathlib.Path
) -> None:
    """Fire-and-forget demand-log UPSERT (spec 33 v2, yantrikdb borrow)."""
    import simba.db
    import simba.memory.demand

    try:
        with simba.db.connect(cwd):
            simba.memory.demand.record(query, result_count, best_score, now=time.time())
    except Exception:
        logger.debug("[recall] demand log failed", exc_info=True)


def _maybe_log_demand(
    request: fastapi.Request,
    config: typing.Any,
    body: RecallRequest,
    recall_query: str,
    results: list[dict],
) -> None:
    """Schedule a demand-log write for a USER-FACING recall.

    Skips: lever off, short queries, internal daemon self-calls (client
    ``daemon``/``daemon.daemon`` — dispatched-hook recalls arrive as
    ``<origin>.daemon`` and DO count), and TOOL_RULE gate probes (they would
    swamp the log; the rule ledger already covers them).
    """
    if not getattr(config, "demand_log_enabled", False):
        return
    if len(recall_query) < int(getattr(config, "demand_log_min_query_chars", 10)):
        return
    client = getattr(request.state, "client", "unknown") or "unknown"
    if client in ("daemon", "daemon.daemon"):
        return
    if (body.filters or {}).get("types") == ["TOOL_RULE"]:
        return
    best = float(results[0].get("similarity", 0.0) or 0.0) if results else 0.0
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    simba.memory.background.spawn(
        asyncio.to_thread(_record_demand, recall_query, len(results), best, cwd)
    )


def _last_used_map(memory_ids: list[str], cwd: pathlib.Path) -> dict[str, float]:
    """Consumption freshness (``usage.last_used``) for recalled ids (spec 33).

    Only ids with a non-zero ``last_used`` appear. Fail-soft {} — recall never
    breaks on a sidecar read.
    """
    if not memory_ids:
        return {}

    import simba.db
    import simba.memory.usage

    try:
        with simba.db.connect(cwd):
            rows = simba.memory.usage.get_many(memory_ids)
    except Exception:
        logger.debug("[recall] last_used read failed", exc_info=True)
        return {}
    return {
        mid: float(row.last_used)
        for mid, row in rows.items()
        if float(getattr(row, "last_used", 0.0) or 0.0) > 0
    }


async def _bump_quality(
    memory_id: str,
    now: float,
    cwd: pathlib.Path,
    *,
    use: int = 0,
    noise: int = 0,
    save: int = 0,
) -> None:
    """Bump explicit quality counters. Fire-and-forget safe."""
    import simba.db
    import simba.memory.usage

    def _sync() -> None:
        with simba.db.connect(cwd):
            simba.memory.usage.bump_quality(
                memory_id, now, use=use, noise=noise, save=save
            )

    await asyncio.to_thread(_sync)


async def _append_provenance_event(
    *,
    memory_id: str,
    occurred_at: str,
    observed_at: str,
    source_file: str,
    source_span: str,
    source_url: str,
    extraction_agent: str,
    extraction_version: str,
    source_session: str,
    trust_source: str,
    capture_origin: str,
    trust_score: float,
    cwd: pathlib.Path,
    now: float,
) -> None:
    import simba.db

    def _sync() -> None:
        with simba.db.connect(cwd):
            simba.memory.provenance.append_event(
                memory_id=memory_id,
                occurred_at=occurred_at,
                observed_at=observed_at,
                source_file=source_file,
                source_span=source_span,
                source_url=source_url,
                extraction_agent=extraction_agent,
                extraction_version=extraction_version,
                source_session=source_session,
                trust_source=trust_source,
                capture_origin=capture_origin,
                trust_score=trust_score,
                now=now,
            )

    await asyncio.to_thread(_sync)


async def _append_anticipated_queries(
    *,
    memory_id: str,
    queries: list[str],
    source: str,
    cwd: pathlib.Path,
    now: float,
    limit: int,
) -> None:
    import simba.db

    def _sync() -> None:
        with simba.db.connect(cwd):
            simba.memory.anticipated.append_queries(
                memory_id=memory_id,
                queries=queries,
                source=source,
                now=now,
                limit=limit,
            )

    await asyncio.to_thread(_sync)


async def _append_supersession_event(
    *,
    old_id: str,
    new_id: str,
    project_path: str,
    memory_type: str,
    similarity: float,
    session_source: str,
    status: str,
    old_trust_score: float,
    new_trust_score: float,
    cwd: pathlib.Path,
    now: float,
) -> typing.Any:
    """Append one supersession audit event. Recall remains fail-open."""
    import simba.db

    provenance = json.dumps(
        {
            "source": "store",
            "sessionSource": session_source,
            "oldTrustScore": old_trust_score,
            "newTrustScore": new_trust_score,
        },
        sort_keys=True,
    )

    def _sync() -> typing.Any:
        with simba.db.connect(cwd):
            return simba.memory.supersession.append_event(
                old_id=old_id,
                new_id=new_id,
                project_path=project_path,
                memory_type=memory_type,
                similarity=similarity,
                reason="near_duplicate_same_type",
                provenance=provenance,
                status=status,
                old_trust_score=old_trust_score,
                new_trust_score=new_trust_score,
                now=now,
            )

    return await asyncio.to_thread(_sync)


async def _detect_and_record_write_conflicts(
    *,
    table: typing.Any,
    embedding: list[float],
    memory_id: str,
    memory_text: str,
    project_path: str,
    config: typing.Any,
    request: fastapi.Request,
    cwd: pathlib.Path,
    now: float,
) -> None:
    """Default-off write-time conflict detection with replayable judge logging."""
    if not getattr(config, "conflict_detect_on_write", False):
        return

    llm_client = getattr(request.app.state, "llm_client", None)
    if llm_client is None:
        try:
            from simba.llm.client import get_client as _get_llm_client

            llm_client = _get_llm_client()
        except Exception:
            return
    if llm_client is None:
        return

    max_neighbors = max(1, int(getattr(config, "conflict_write_max_neighbors", 5)))
    try:
        candidates = await simba.memory.vector_db.search_memories(
            table,
            embedding,
            0.0,
            max_neighbors + 5,
            {"projectPath": project_path},
        )
    except Exception:
        logger.debug("[store] write-conflict neighbor search failed", exc_info=True)
        return
    neighbors: list[tuple[str, str]] = []
    for candidate in candidates:
        cid = str(candidate.get("id") or "")
        if not cid or cid == memory_id:
            continue
        text = str(candidate.get("content") or "")
        ctx = str(candidate.get("context") or "")
        if ctx:
            text = f"{text} {ctx}".strip()
        if text:
            neighbors.append((cid, text))
        if len(neighbors) >= max_neighbors:
            break
    if not neighbors:
        return

    def _sync() -> None:
        import simba.db

        with simba.db.connect(cwd):
            conflicts = simba.memory.conflict.detect_conflicts_on_write_logged(
                memory_id,
                memory_text,
                neighbors,
                llm_client=llm_client,
                project_path=project_path,
                max_neighbors=max_neighbors,
                generous=bool(getattr(config, "conflict_recall_recheck", False)),
                now=now,
            )
            for neighbor_id, description in conflicts:
                simba.memory.conflict_store.record_conflict(
                    memory_id,
                    neighbor_id,
                    description,
                    project_path=project_path,
                    now=now,
                )

    try:
        await asyncio.to_thread(_sync)
    except Exception:
        logger.debug("[store] write-conflict detection failed", exc_info=True)


async def _stored_memory_trust_score(
    memory: dict[str, typing.Any],
    cwd: pathlib.Path,
) -> float:
    """Compute current trust score for an existing memory row."""
    import simba.db
    import simba.memory.usage

    mid = str(memory.get("id") or "")
    if not mid:
        return 0.0

    def _sync() -> float:
        with simba.db.connect(cwd):
            prov = simba.memory.provenance.latest_for([mid]).get(mid)
            usage = simba.memory.usage.get_many([mid]).get(mid)
        trust_source = prov.trust_source if prov is not None else "agent_suggested"
        capture_origin = prov.capture_origin if prov is not None else "store"
        return simba.memory.provenance.compute_trust_score(
            trust_source=trust_source,
            capture_origin=capture_origin,
            confidence=float(memory.get("confidence", 0.0) or 0.0),
            memory_type=str(memory.get("type", "")),
            usage=usage,
        )

    return await asyncio.to_thread(_sync)


async def _mark_superseded(
    memories: list[dict[str, typing.Any]], cwd: pathlib.Path
) -> list[dict[str, typing.Any]]:
    """Annotate and demote superseded recall hits using the append-only audit."""
    if not memories:
        return memories
    import simba.db

    ids = [str(m.get("id", "")) for m in memories if m.get("id")]

    def _sync() -> tuple[dict[str, str], dict[str, tuple[str, int]]]:
        with simba.db.connect(cwd):
            active_rows = simba.memory.supersession.latest_successors(ids)
            pending_rows = simba.memory.supersession.latest_pending(ids)
            return (
                {old_id: row.new_id for old_id, row in active_rows.items()},
                {
                    old_id: (row.new_id, int(row.id))
                    for old_id, row in pending_rows.items()
                },
            )

    try:
        successors, pending = await asyncio.to_thread(_sync)
    except Exception:
        logger.debug("[recall] supersession lookup failed", exc_info=True)
        return memories
    if not successors and not pending:
        return memories

    out: list[tuple[bool, int, dict[str, typing.Any]]] = []
    for idx, mem in enumerate(memories):
        mid = str(mem.get("id", ""))
        if mid in successors:
            mem = dict(mem)
            mem["supersededBy"] = successors[mid]
            out.append((True, idx, mem))
        elif mid in pending:
            mem = dict(mem)
            replacement_id, event_id = pending[mid]
            mem["pendingSupersededBy"] = replacement_id
            mem["pendingSupersessionId"] = event_id
            out.append((False, idx, mem))
        else:
            out.append((False, idx, mem))
    out.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in out]


class DiagnosticsMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    """Record every request and emit diagnostics reports at intervals."""

    async def dispatch(
        self,
        request: starlette.requests.Request,
        call_next: typing.Callable[
            [starlette.requests.Request],
            typing.Awaitable[starlette.responses.Response],
        ],
    ) -> starlette.responses.Response:
        diag = getattr(request.app.state, "diagnostics", None)
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        # Client identity (X-Simba-Client): set before call_next so route
        # handlers can read it; "unknown" when a caller omits the header.
        client = request.headers.get("x-simba-client") or "unknown"
        request.state.client = client
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            if diag is not None:
                diag.record_error(request.url.path, exc, request_id)
            raise
        response.headers["x-simba-request-id"] = request_id
        if diag is not None:
            diag.record_request(request.url.path)
            diag.record_client(client)
            diag.record_latency(request.url.path, (time.monotonic() - t0) * 1000)
            if diag.should_report():
                table = getattr(request.app.state, "table", None)
                simba.memory.background.spawn(diag.emit_report(table))
        return response


VALID_TYPES = [
    "GOTCHA",
    "WORKING_SOLUTION",
    "PATTERN",
    "DECISION",
    "FAILURE",
    "PREFERENCE",
    "SYSTEM",
    "TOOL_RULE",
    "EPISODE",
    "REFLECTION",  # cross-session synthesized insight (Phase 5)
]


class StoreRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=True)

    type: str
    content: str
    context: str = ""
    tags: list[str] = pydantic.Field(default_factory=list)
    confidence: float = 0.85
    session_source: str = pydantic.Field(default="", alias="sessionSource")
    project_path: str = pydantic.Field(default="", alias="projectPath")
    occurred_at: str = pydantic.Field(default="", alias="occurredAt")
    observed_at: str = pydantic.Field(default="", alias="observedAt")
    source_file: str = pydantic.Field(default="", alias="sourceFile")
    source_span: str = pydantic.Field(default="", alias="sourceSpan")
    source_url: str = pydantic.Field(default="", alias="sourceUrl")
    extraction_agent: str = pydantic.Field(default="", alias="extractionAgent")
    extraction_version: str = pydantic.Field(default="", alias="extractionVersion")
    trust_source: str = pydantic.Field(default="agent_suggested", alias="trustSource")
    capture_origin: str = pydantic.Field(default="store", alias="captureOrigin")
    anticipated_queries: list[str] = pydantic.Field(
        default_factory=list, alias="anticipatedQueries"
    )


class RecallRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=True)

    query: str
    min_similarity: float | None = pydantic.Field(default=None, alias="minSimilarity")
    max_results: int | None = pydantic.Field(default=None, alias="maxResults")
    project_path: str | None = pydantic.Field(default=None, alias="projectPath")
    # Hierarchical (ancestor-prefix) recall (spec 26): the client-computed scope
    # chain ``[cwd-resolved, …ancestors…, git-root-resolved]``. Honored only when
    # ``memory.hierarchical_recall`` is on; otherwise the daemon ignores it and
    # uses the strict ``project_path`` exact match (byte-identical legacy).
    project_scopes: list[str] | None = pydantic.Field(
        default=None, alias="projectScopes"
    )
    filters: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


@router.post("/store")
async def store_memory(body: StoreRequest, request: fastapi.Request) -> dict:

    table = request.app.state.table
    config = request.app.state.config
    embed = request.app.state.embed

    if body.type not in VALID_TYPES:
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"invalid type, must be one of: {', '.join(VALID_TYPES)}",
        )

    # Per-project cap: `config.max_content_length` is a single value frozen
    # at daemon startup, but resolve_max_content_length(root) already
    # layers a project's own `.simba/config.toml` over it (the same
    # resolver the extraction/digest/episode/reflection prompt guidance
    # uses) -- so enforcement and prompt guidance must resolve the SAME
    # way, not the daemon's frozen boot-time value. An empty/blank
    # project_path (most callers omit projectPath entirely for a
    # global-scope memory) MUST resolve as root=None, never Path("") --
    # Path("") == Path(".") would silently scope the cap to the DAEMON's
    # own cwd instead of "no project override".
    project_root = (
        pathlib.Path(body.project_path) if body.project_path.strip() else None
    )
    max_len = simba.memory.config.resolve_max_content_length(project_root)
    if len(body.content) > max_len:
        got_len = len(body.content)
        # Callers hit this over HTTP (no interactive stderr), so this stays
        # a single terse line -- the CLI's `_memory_store` pre-checks the
        # same (per-project) cap and prints the full two-path guidance
        # before ever reaching this endpoint. Still names the knob + exact
        # command so any non-CLI caller inspecting `detail` gets the fix.
        raise fastapi.HTTPException(
            status_code=400,
            detail=(
                f"content too long (max {max_len} chars, got {got_len}); "
                "raise via: simba config set memory.max_content_length "
                f"{got_len} (project-local; --global raises it everywhere)"
            ),
        )

    # Per-session inflow throttle (spec 33 Phase 2). Over-capture is real
    # (~400 stores/day audited, raw error output stored as rules): when a
    # budget is set, non-EPISODE stores beyond it for one sessionSource are
    # rejected before the embed. EPISODE is exempt — the budget should push
    # toward consolidation, not block it. Counted on successful add below.
    budget = int(getattr(config, "store_budget_per_session", 0) or 0)
    budget_session = (body.session_source or "").strip()
    budget_applies = budget > 0 and bool(budget_session) and body.type != "EPISODE"
    if budget_applies:
        store_counts = getattr(request.app.state, "store_session_counts", None)
        if store_counts is None:
            store_counts = {}
            request.app.state.store_session_counts = store_counts
        if store_counts.get(budget_session, 0) >= budget:
            raise fastapi.HTTPException(
                status_code=429,
                detail=(
                    f"session store budget exhausted ({budget}); consolidate "
                    "into an EPISODE or raise memory.store_budget_per_session"
                ),
            )

    # A store mutates the corpus (new memory, supersession annotation), so drop
    # cached recalls to keep the short-TTL cache from serving pre-store results.
    recall_cache = getattr(request.app.state, "recall_cache", None)
    if recall_cache is not None:
        recall_cache.clear()

    # Normalize the scope path to an absolute, symlink-resolved path (spec 26) so
    # the client's resolved ancestor chain can match it by string membership. An
    # empty (global) path stays empty. With scope_normalize_worktrees (spec 33),
    # a linked worktree additionally folds onto its main repository root.
    project_path = simba.memory.vector_db.normalize_project_path(
        body.project_path,
        resolve_worktrees=getattr(config, "scope_normalize_worktrees", False),
    )

    text_to_embed = body.content + (f" {body.context}" if body.context else "")
    try:
        embedding = await embed(text_to_embed)
    except Exception as e:
        logger.warning("[store] Embedding failed: %s", e)
        raise fastapi.HTTPException(
            status_code=503,
            detail=f"Embedding service error: {e}",
        ) from e

    dup_check = await simba.memory.vector_db.find_duplicates(
        table, embedding, config.duplicate_threshold
    )
    if dup_check["is_duplicate"]:
        logger.info(
            "[store] project=%s type=%s -> duplicate (sim: %.2f)",
            project_path or "(global)",
            body.type,
            round(dup_check["similarity"], 2),
        )
        diag = getattr(request.app.state, "diagnostics", None)
        if diag is not None:
            diag.record_store(body.type, duplicate=True)
        return {
            "status": "duplicate",
            "existing_id": dup_check["existing_id"],
            "similarity": round(dup_check["similarity"], 2),
        }

    # Supersession (opt-in): if a same-type memory sits just below the duplicate
    # threshold, append a lineage audit event and demote the older row at recall.
    superseded_id: str | None = None
    superseded_similarity = 0.0
    supersession_status = simba.memory.supersession.STATUS_ACTIVE
    pending_supersession_id: int | None = None
    old_trust_score = 0.0
    normalized_trust_source = simba.memory.provenance.normalize_trust_source(
        body.trust_source
    )
    normalized_capture_origin = simba.memory.provenance.normalize_capture_origin(
        body.capture_origin
    )
    new_trust_score = simba.memory.provenance.compute_trust_score(
        trust_source=normalized_trust_source,
        capture_origin=normalized_capture_origin,
        confidence=body.confidence,
        memory_type=body.type,
    )
    cwd_path = pathlib.Path(getattr(request.app.state, "cwd", "."))
    if config.supersede_enabled:
        candidates = await simba.memory.vector_db.search_memories(
            table,
            embedding,
            config.supersede_threshold,
            1,
            {"projectPath": project_path, "types": [body.type]},
        )
        if candidates:
            superseded_id = candidates[0]["id"]
            superseded_similarity = float(candidates[0].get("similarity", 0.0) or 0.0)
            if getattr(config, "supersede_trust_gate_enabled", True):
                old_trust_score = await _stored_memory_trust_score(
                    candidates[0],
                    cwd_path,
                )
                margin = float(getattr(config, "supersede_trust_margin", 0.05))
                if new_trust_score + margin < old_trust_score:
                    supersession_status = simba.memory.supersession.STATUS_PENDING

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    now_epoch = time.time()

    # Dimensional tagging (off by default): append a parseable time/keyword blob to
    # context so aggregation can filter by field later. Embedding is computed above
    # from the ORIGINAL text, so the blob never pollutes the vector.
    stored_context = body.context
    if getattr(config, "dimensions_enabled", False):
        dims = simba.memory.dimensions.extract_dimensions(
            body.content + (f" {body.context}" if body.context else "")
        )
        stored_context = (body.context or "") + simba.memory.dimensions.to_blob(dims)

    await table.add(
        [
            {
                "id": memory_id,
                "type": body.type,
                "content": body.content,
                "context": stored_context,
                "tags": json.dumps(body.tags),
                "confidence": body.confidence,
                "sessionSource": body.session_source,
                "projectPath": project_path,
                "createdAt": now,
                "lastAccessedAt": now,
                "accessCount": 0,
                "vector": embedding,
            }
        ]
    )

    if budget_applies:
        store_counts = request.app.state.store_session_counts
        if len(store_counts) > 4096:  # daemon-lifetime bound; sessions churn
            store_counts.clear()
        store_counts[budget_session] = store_counts.get(budget_session, 0) + 1

    logger.info(
        '[store] project=%s type=%s content="%s" -> stored %s',
        project_path or "(global)",
        body.type,
        body.content[:50],
        memory_id,
    )

    fts_path = getattr(request.app.state, "fts_path", None)
    if fts_path:
        try:
            await asyncio.to_thread(
                _fts_upsert,
                fts_path,
                {
                    "id": memory_id,
                    "type": body.type,
                    "content": body.content,
                    "context": stored_context,
                    "confidence": body.confidence,
                    "createdAt": now,
                    "projectPath": project_path,
                },
            )
        except Exception:
            logger.debug("[store] fts mirror upsert failed", exc_info=True)

    if superseded_id:
        try:
            row = await _append_supersession_event(
                old_id=superseded_id,
                new_id=memory_id,
                project_path=project_path,
                memory_type=body.type,
                similarity=superseded_similarity,
                session_source=body.session_source,
                status=supersession_status,
                old_trust_score=old_trust_score,
                new_trust_score=new_trust_score,
                cwd=cwd_path,
                now=now_epoch,
            )
            pending_supersession_id = (
                int(row.id)
                if row.status == simba.memory.supersession.STATUS_PENDING
                else None
            )
        except Exception:
            logger.debug("[store] supersession audit append failed", exc_info=True)

    try:
        await _bump_quality(
            memory_id,
            now_epoch,
            cwd_path,
            save=1,
        )
    except Exception:
        logger.debug("[store] quality counter bump failed", exc_info=True)

    try:
        await _append_provenance_event(
            memory_id=memory_id,
            occurred_at=body.occurred_at,
            observed_at=body.observed_at or now,
            source_file=body.source_file,
            source_span=body.source_span,
            source_url=body.source_url,
            extraction_agent=body.extraction_agent,
            extraction_version=body.extraction_version,
            source_session=body.session_source,
            trust_source=normalized_trust_source,
            capture_origin=normalized_capture_origin,
            trust_score=new_trust_score,
            cwd=cwd_path,
            now=now_epoch,
        )
    except Exception:
        logger.debug("[store] provenance append failed", exc_info=True)

    if body.anticipated_queries:
        try:
            await _append_anticipated_queries(
                memory_id=memory_id,
                queries=body.anticipated_queries,
                source=normalized_capture_origin,
                cwd=cwd_path,
                now=now_epoch,
                limit=int(getattr(config, "anticipated_query_max_per_memory", 5)),
            )
        except Exception:
            logger.debug("[store] anticipated-query append failed", exc_info=True)

    try:
        memory_text = body.content + (f" {stored_context}" if stored_context else "")
        await _detect_and_record_write_conflicts(
            table=table,
            embedding=embedding,
            memory_id=memory_id,
            memory_text=memory_text,
            project_path=project_path,
            config=config,
            request=request,
            cwd=cwd_path,
            now=now_epoch,
        )
    except Exception:
        logger.debug("[store] write-conflict hook failed", exc_info=True)

    diag = getattr(request.app.state, "diagnostics", None)
    if diag is not None:
        diag.record_store(body.type, duplicate=False)

    return {
        "status": (
            "pending_confirmation"
            if pending_supersession_id is not None
            else "superseded"
            if superseded_id
            else "stored"
        ),
        "id": memory_id,
        **(
            {"pendingSupersessionId": pending_supersession_id}
            if pending_supersession_id is not None
            else {}
        ),
        **(
            {"supersededId": superseded_id}
            if superseded_id and pending_supersession_id is None
            else {}
        ),
        **(
            {"supersededCandidateId": superseded_id}
            if superseded_id and pending_supersession_id is not None
            else {}
        ),
        "trustScore": new_trust_score,
        "embedding_dims": len(embedding),
    }


@router.post("/recall")
async def recall_memories(body: RecallRequest, request: fastapi.Request) -> dict:
    """Admission-controlled entry point (see ``_recall_memories_impl``).

    ``memory.max_concurrent_recalls`` (0 = unlimited, default) gates the
    WHOLE handler with an ``asyncio.Semaphore`` built once in
    ``create_app``. The LLAMA lock already serializes native embed/rerank
    compute end to end (see ``simba.memory._llama``), but the surrounding
    pyarrow/RRF/rerank-loop orchestration below is not covered by that lock
    and can stack unboundedly across concurrent requests during a recall
    storm --- each holding its own candidate_pool-sized working set. None
    (the default) preserves the prior byte-identical, ungated behavior.
    Internal callers (e.g. ``/preflight``'s ``_recall`` closure) go through
    this same gate, which is intended: any real recall should count against
    the cap regardless of caller.
    """
    semaphore = getattr(request.app.state, "recall_semaphore", None)
    if semaphore is None:
        return await _recall_memories_impl(body, request)
    async with semaphore:
        return await _recall_memories_impl(body, request)


async def _recall_memories_impl(body: RecallRequest, request: fastapi.Request) -> dict:

    table = request.app.state.table
    config = request.app.state.config
    embed_query = request.app.state.embed_query
    parsed_query = simba.memory.query_filters.parse(body.query)
    recall_query = parsed_query.query

    # Short-TTL recall cache: collapse identical-query storms (multi-runtime
    # hooks + reasoning/conflict loops) before the expensive embed + the
    # LLAMA_LOCK-serialized cross-encoder rerank.
    recall_cache = getattr(request.app.state, "recall_cache", None)
    cache_key = None
    if recall_cache is not None:
        cache_key = simba.memory.recall_cache.RecallCache.key(
            query=body.query,
            project_path=body.project_path,
            min_similarity=body.min_similarity,
            max_results=body.max_results,
            filters=body.filters,
            project_scopes=body.project_scopes,
        )
        cached = recall_cache.get(cache_key, now=time.time())
        if cached is not None:
            logger.info(
                '[recall] client=%s cached project=%s query="%s" -> %d memories',
                getattr(request.state, "client", "unknown"),
                body.project_path or "(global)",
                recall_query[:50],
                len(cached),
            )
            diag = getattr(request.app.state, "diagnostics", None)
            if diag is not None:
                diag.record_recall(body.query, len(cached))
            # A cache-served recall is still a LOGICAL recall (measured live:
            # acks counted while match/access didn't — inject=2, match=1).
            # Only the search is skipped; the sidecar ledger still ticks.
            cached_ids = [m["id"] for m in cached if m.get("id")]
            if cached_ids:
                cached_cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
                simba.memory.background.spawn(
                    _bump_usage(cached_ids, time.time(), cached_cwd)
                )
            _maybe_log_demand(request, config, body, recall_query, cached)
            return {"memories": cached, "queryTimeMs": 0, "cached": True}

    start_time = time.time()
    try:
        embedding = await embed_query(recall_query)
    except Exception as e:
        logger.warning("[recall] Embedding failed: %s", e)
        return {"memories": [], "queryTimeMs": 0, "error": "embedding_failed"}

    # An LLM client is shared by HyDE (2nd-arm answer) and the reranker. Built
    # once, lazily, only when at least one LLM feature is enabled; both uses are
    # fail-open so a missing/failing provider degrades to the non-LLM path.
    hyde_llm_on = getattr(config, "hyde_mode", "keyword") == "llm"
    rerank_on = config.hybrid_enabled and getattr(config, "llm_rerank_enabled", False)
    llm_client = None
    if hyde_llm_on or rerank_on:
        from simba.llm.client import get_client as _get_llm_client

        llm_client = _get_llm_client()

    # HyDE cache (daemon, per-process): serve the keyword fallback now and warm
    # the cache off the hot path so recurring queries get the LLM answer free.
    hyde_cache = None
    if hyde_llm_on and llm_client is not None:
        hyde_cache = getattr(request.app.state, "hyde_cache", None)

    # Intent-aware floor + broad-query widening + HyDE text selection, all
    # derived by the shared planner (the eval harness uses the same logic).
    plan = simba.memory.recall_plan.plan_recall(
        recall_query,
        config,
        min_similarity=body.min_similarity,
        max_results=body.max_results,
        llm_client=llm_client,
        hyde_cache=hyde_cache,
    )
    min_sim = plan.min_similarity
    max_res = plan.max_results
    candidate_pool = plan.candidate_pool
    mode = plan.mode

    filters = dict(body.filters)
    filters.update(parsed_query.route_filters)
    if body.project_path:
        recall_scope = body.project_path
        # Worktree fold (spec 33): recall from a linked worktree matches the
        # main-root scope stores normalize onto. Off ⇒ raw path, byte-identical.
        if getattr(config, "scope_normalize_worktrees", False):
            recall_scope = simba.memory.vector_db.normalize_project_path(
                recall_scope, resolve_worktrees=True
            )
        filters["projectPath"] = recall_scope

    # Hierarchical (ancestor-prefix) recall (spec 26): when the lever is on AND the
    # client supplied a resolved scope chain, hand both arms the scope set so
    # ancestor/root facts inherit down. Off / no chain → strict projectPath match
    # (byte-identical legacy). The flags ride in ``filters`` so search_memories +
    # the FTS arm read a single source of truth.
    if getattr(config, "hierarchical_recall", False) and body.project_scopes:
        filters["hierarchical_recall"] = True
        filters["project_scopes"] = list(body.project_scopes)
        filters["hierarchical_recall_include_global"] = bool(
            getattr(config, "hierarchical_recall_include_global", True)
        )

    # On a HyDE cache miss, warm the cache off the hot path so the next identical
    # query gets the LLM answer for its 2nd arm (this recall serves the fallback).
    if hyde_llm_on and hyde_cache is not None and llm_client is not None:
        hyde_key = hyde_cache.signature(body.query)
        if hyde_cache.get(hyde_key) is None:
            simba.memory.background.spawn(
                _bg_hyde(hyde_cache, hyde_key, body.query, llm_client)
            )

    # Multi-arm HyDE (opt-in): a 2nd vector arm over plan.hyde_text — the focus
    # terms in keyword mode, or the LLM hypothetical answer in llm mode.
    extra_embedding: list[float] | None = None
    if plan.hyde_text:
        try:
            extra_embedding = await embed_query(plan.hyde_text)
        except Exception:
            extra_embedding = None

    # Optional LLM reranker (cross-encoder role): non-blocking via the cache.
    # "async" (default): serve fast order, rerank off the hot path via the cache.
    # "sync": block on the rerank every recall (no cache).
    rerank_cache = None
    if rerank_on and getattr(config, "llm_rerank_mode", "async") != "sync":
        rerank_cache = getattr(request.app.state, "rerank_cache", None)

    fts_path = getattr(request.app.state, "fts_path", None)
    recall_cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    if config.hybrid_enabled:
        memories = await simba.memory.hybrid.hybrid_search(
            table,
            fts_path,
            embedding,
            recall_query,
            min_similarity=min_sim,
            max_results=max_res,
            filters=filters,
            cfg=config,
            candidate_pool=candidate_pool,
            extra_embedding=extra_embedding,
            llm_client=llm_client,
            rerank_cache=rerank_cache,
            bg_tasks=_background_tasks,
            cwd=recall_cwd,
        )
    else:
        memories = await simba.memory.vector_db.search_memories(
            table, embedding, min_sim, max_res, filters
        )
        if getattr(config, "dormant_filter_enabled", True):
            memories = await asyncio.to_thread(
                simba.memory.hybrid._filter_dormant, memories, recall_cwd
            )

    # Low-confidence abstention gate (off by default): suppress the whole recall
    # when even the top candidate is too weak to trust.
    memories = simba.memory.scoring.apply_rejection_gate(
        memories,
        enabled=getattr(config, "recall_reject_enabled", False),
        threshold=getattr(config, "recall_reject_threshold", 0.0),
    )
    memories = simba.memory.query_filters.apply(memories, parsed_query.post_filters)
    memories = await _mark_superseded(memories, recall_cwd)

    # Consumption freshness (spec 33): lets rule-TTL refresh and other clients
    # key off max(createdAt, lastUsedAt) instead of creation alone.
    last_used = await asyncio.to_thread(
        _last_used_map, [m["id"] for m in memories], recall_cwd
    )

    results = [
        {
            "id": m["id"],
            "type": m["type"],
            "content": m["content"],
            **({"context": m["context"]} if m.get("context") else {}),
            "similarity": round(m["similarity"], 2),
            "confidence": m.get("confidence", 0),
            "createdAt": m.get("createdAt"),
            **(
                {
                    "lastUsedAt": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(last_used[m["id"]])
                    )
                }
                if m["id"] in last_used
                else {}
            ),
            **({"projectPath": m["projectPath"]} if m.get("projectPath") else {}),
            **({"sessionSource": m["sessionSource"]} if m.get("sessionSource") else {}),
            **({"supersededBy": m["supersededBy"]} if m.get("supersededBy") else {}),
            **(
                {"pendingSupersededBy": m["pendingSupersededBy"]}
                if m.get("pendingSupersededBy")
                else {}
            ),
            **(
                {"pendingSupersessionId": m["pendingSupersessionId"]}
                if m.get("pendingSupersessionId")
                else {}
            ),
        }
        for m in memories
    ]

    query_time_ms = int((time.time() - start_time) * 1000)
    top_sim = round(results[0]["similarity"], 2) if results else 0.0
    logger.info(
        '[recall] client=%s project=%s mode=%s floor=%.2f query="%s" '
        "-> %d memories (%dms), top: %.2f",
        getattr(request.state, "client", "unknown"),
        body.project_path or "(global)",
        mode,
        min_sim,
        recall_query[:50],
        len(results),
        query_time_ms,
        top_sim,
    )

    diag = getattr(request.app.state, "diagnostics", None)
    if diag is not None:
        diag.record_recall(body.query, len(results))

    # Fire-and-forget: update access tracking for returned memories.
    if results:
        recalled_ids = [r["id"] for r in results]
        now_epoch = time.time()
        cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))

        # LanceDB access tracking (never read for ranking — the sqlite usage
        # sidecar below is authoritative). Only in LEGACY mode (retention == 0):
        # each write adds a LanceDB version, which ballooned a store to 37GB/25k
        # versions. Bounded retention (the default) suppresses it.
        if getattr(config, "lancedb_version_retention_seconds", 86_400) <= 0:
            simba.memory.background.spawn(
                simba.memory.vector_db.update_access_tracking(table, recalled_ids)
            )

        # Authoritative sqlite usage store (drives decay/feedback ranking).
        simba.memory.background.spawn(_bump_usage(recalled_ids, now_epoch, cwd))

    if recall_cache is not None and cache_key is not None:
        recall_cache.put(cache_key, results, now=time.time())

    _maybe_log_demand(request, config, body, recall_query, results)

    return {"memories": results, "queryTimeMs": query_time_ms}


@router.get("/demand/gaps")
async def demand_gaps(
    request: fastapi.Request,
    minAsks: int | None = fastapi.Query(default=None),  # noqa: N803
    maxBest: float | None = fastapi.Query(default=None),  # noqa: N803
    limit: int = fastapi.Query(default=20),
) -> dict:
    """The corpus's known unknowns (spec 33 v2): queries asked repeatedly
    whose best hit never cleared the bar. Read-only over the demand log."""
    config = request.app.state.config
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    min_asks = (
        minAsks
        if minAsks is not None
        else int(getattr(config, "demand_gap_min_asks", 3))
    )
    max_best = (
        maxBest
        if maxBest is not None
        else float(getattr(config, "demand_gap_max_best", 0.5))
    )

    def _sync() -> list[dict]:
        import simba.db
        import simba.memory.demand

        with simba.db.connect(cwd):
            return simba.memory.demand.gaps(
                min_asks=min_asks, max_best=max_best, limit=limit
            )

    try:
        rows = await asyncio.to_thread(_sync)
    except Exception:
        logger.debug("[demand] gaps read failed", exc_info=True)
        rows = []
    return {"gaps": rows, "minAsks": min_asks, "maxBest": max_best}


class RecallAckRequest(pydantic.BaseModel):
    ids: list[str] = pydantic.Field(default_factory=list)


@router.post("/recall/ack")
async def recall_ack(body: RecallAckRequest, request: fastapi.Request) -> dict:
    """Hook-side acknowledgment that recalled ids were actually INJECTED (spec 33).

    Recall bumps ``match`` (returned by search); only the client knows which
    memories survived its lane/budget trim into real context, so it acks them
    here → ``inject``. Bounded and fail-soft: sidecar trouble never breaks the
    calling hook.
    """
    ids = [i for i in body.ids if isinstance(i, str) and i][:100]
    if not ids:
        return {"status": "ok", "acked": 0}

    import simba.db
    import simba.memory.usage

    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    now = time.time()

    def _sync() -> None:
        with simba.db.connect(cwd):
            for mid in ids:
                simba.memory.usage.bump_quality(mid, now, inject=1)

    try:
        await asyncio.to_thread(_sync)
    except Exception:
        logger.debug("[recall/ack] inject bump failed", exc_info=True)
        return {"status": "error", "acked": 0}
    return {"status": "ok", "acked": len(ids)}


class ScopeNormalizeRequest(pydantic.BaseModel):
    # False (default) = dry run: report the fold plan, touch nothing.
    run: bool = False


@router.post("/scopes/normalize")
async def normalize_scopes(
    body: ScopeNormalizeRequest, request: fastapi.Request
) -> dict:
    """Fold linked-worktree scopes onto their main repository root (spec 33).

    The one-time migration for pre-fold rows (the audit found one repo
    sharded 4 ways across its worktrees). Dry-run by default; ``run=true``
    applies one LanceDB update per distinct old path (bounded version churn)
    plus the FTS-mirror retarget, then drops cached recalls.
    """
    table = request.app.state.table

    # Only `projectPath` is read below -- never `vector`/`content` (2026-07-18).
    rows = await table.query().select(["projectPath"]).to_list()
    folds: dict[str, dict] = {}
    for row in rows:
        old = row.get("projectPath") or ""
        # Only absolute filesystem paths are candidates: the corpus also
        # carries opaque project-id hashes (the rules keying), and resolving
        # one relative to the daemon cwd would corrupt the scope — caught by
        # the first live dry-run.
        if not old or not pathlib.PurePath(old).is_absolute():
            continue
        new = simba.memory.vector_db.normalize_project_path(old, resolve_worktrees=True)
        if new != old:
            entry = folds.setdefault(old, {"to": new, "count": 0})
            entry["count"] += 1

    plan = [
        {"from": old, "to": entry["to"], "count": entry["count"]}
        for old, entry in sorted(folds.items())
    ]
    changed = sum(entry["count"] for entry in folds.values())

    if body.run and folds:
        fts_path = getattr(request.app.state, "fts_path", None)
        for old, entry in folds.items():
            escaped = old.replace("'", "''")
            await table.update(
                updates={"projectPath": entry["to"]},
                where=f"projectPath = '{escaped}'",
            )
            if fts_path:
                try:
                    await asyncio.to_thread(_fts_retarget, fts_path, old, entry["to"])
                except Exception:
                    logger.debug("[scopes] fts retarget failed", exc_info=True)
        recall_cache = getattr(request.app.state, "recall_cache", None)
        if recall_cache is not None:
            recall_cache.clear()
        logger.info("[scopes] folded %d memories across %d scopes", changed, len(plan))

    return {"run": bool(body.run), "changed": changed, "folds": plan}


@router.get("/promotions/candidates")
async def promotion_candidates(
    request: fastapi.Request,
    limit: int = fastapi.Query(default=20),
) -> dict:
    """Usage-triggered promotion candidates (spec 33 Phase 5).

    A memory whose ledger shows REAL consumption — ``use_count >=
    memory.promotion_min_uses``, ``noise/use < memory.promotion_max_noise_
    ratio``, not dormant — has earned a look at the rule/CLAUDE.md layer.
    Read-only and stateless (recomputed from the sidecar each call); the
    promotion itself stays human (`simba memory promote`).
    """
    config = request.app.state.config
    table = request.app.state.table
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    min_uses = int(getattr(config, "promotion_min_uses", 3))
    max_ratio = float(getattr(config, "promotion_max_noise_ratio", 0.5))

    min_sessions = int(getattr(config, "promotion_min_sessions", 1))

    def _sync() -> list[dict]:
        import simba.db
        import simba.memory.usage
        import simba.memory.usage_events

        with simba.db.connect(cwd):
            rows = simba.memory.usage.MemoryUsage.select().where(
                (simba.memory.usage.MemoryUsage.use_count >= min_uses)
                & (simba.memory.usage.MemoryUsage.dormant == False)  # noqa: E712
            )
            out: list[dict] = []
            for row in rows:
                if row.use_count > 0 and (row.noise_count / row.use_count) >= max_ratio:
                    continue
                out.append(
                    {
                        "id": row.memory_id,
                        "useCount": int(row.use_count),
                        "noiseCount": int(row.noise_count),
                        "injectCount": int(row.inject_count),
                    }
                )
            # Per-session attribution (spec 33 v2 rule R2): report distinct
            # use sessions; gate on promotion_min_sessions when raised.
            session_counts = simba.memory.usage_events.use_sessions_for(
                [c["id"] for c in out]
            )
            for c in out:
                c["sessions"] = session_counts.get(c["id"], 0)
            if min_sessions > 1:
                out = [c for c in out if c["sessions"] >= min_sessions]
            out.sort(key=lambda c: -c["useCount"])
            return out

    try:
        candidates = await asyncio.to_thread(_sync)
    except Exception:
        logger.debug("[promotions] sidecar read failed", exc_info=True)
        candidates = []

    total = len(candidates)
    candidates = candidates[: max(0, limit)]
    if candidates:
        wanted = {c["id"] for c in candidates}
        meta: dict[str, dict] = {}
        try:
            # Only `id`/`type`/`content`/`projectPath` are read below (see the
            # loop underneath) -- never `vector` (2026-07-18).
            rows = (
                await table.query()
                .select(["id", "type", "content", "projectPath"])
                .to_list()
            )
            meta = {r["id"]: r for r in rows if r.get("id") in wanted}
        except Exception:
            logger.debug("[promotions] content join failed", exc_info=True)
        for c in candidates:
            m = meta.get(c["id"]) or {}
            c["type"] = m.get("type", "")
            c["content"] = m.get("content", "")
            c["projectPath"] = m.get("projectPath", "")

    return {
        "candidates": candidates,
        "total": total,
        "minUses": min_uses,
        "maxNoiseRatio": max_ratio,
    }


@router.get("/digest")
async def boot_digest(request: fastapi.Request) -> dict:
    """Boot digest (spec 33 v2, yantrikdb-server borrow) — one call returning
    the lifecycle state a fresh session should see: latest heartbeat,
    promotion inbox, unadjudicated supersessions, knowledge gaps. The
    consumption surface that keeps the new inboxes from rotting unread (the
    animaworks lesson: surfaced state re-appears until consumed)."""
    config = request.app.state.config
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    heartbeat = getattr(request.app.state, "last_maintenance", None)

    def _sync() -> dict:
        import simba.db
        import simba.memory.demand
        import simba.memory.maintenance
        import simba.memory.supersession
        import simba.memory.usage
        import simba.memory.usage_events

        min_uses = int(getattr(config, "promotion_min_uses", 3))
        max_ratio = float(getattr(config, "promotion_max_noise_ratio", 0.5))
        min_sessions = int(getattr(config, "promotion_min_sessions", 1))
        daemon_url = f"http://127.0.0.1:{getattr(config, 'port', 8741)}"
        with simba.db.connect(cwd):
            rows = simba.memory.usage.MemoryUsage.select().where(
                (simba.memory.usage.MemoryUsage.use_count >= min_uses)
                & (simba.memory.usage.MemoryUsage.dormant == False)  # noqa: E712
            )
            candidates = [
                {"id": row.memory_id, "useCount": int(row.use_count)}
                for row in rows
                if not (
                    row.use_count > 0 and (row.noise_count / row.use_count) >= max_ratio
                )
            ]
            session_counts = simba.memory.usage_events.use_sessions_for(
                [c["id"] for c in candidates]
            )
            for c in candidates:
                c["sessions"] = session_counts.get(c["id"], 0)
            if min_sessions > 1:
                candidates = [c for c in candidates if c["sessions"] >= min_sessions]
            candidates.sort(key=lambda c: -c["useCount"])

            pending_rows = list(
                simba.memory.supersession.MemorySupersession.select().where(
                    simba.memory.supersession.MemorySupersession.status
                    == simba.memory.supersession.STATUS_PENDING
                )
            )
            pending_ids = [p.id for p in pending_rows]
            decided = (
                simba.memory.supersession._decided_pending_ids(pending_ids)
                if pending_ids
                else set()
            )
            pending = len([p for p in pending_ids if p not in decided])

            gap_rows = simba.memory.demand.gaps(
                min_asks=int(getattr(config, "demand_gap_min_asks", 3)),
                max_best=float(getattr(config, "demand_gap_max_best", 0.5)),
                limit=3,
            )

            repeat_failures = simba.memory.maintenance._cluster_repeat_failures(
                cwd, config
            )
            graduation = simba.memory.maintenance._graduation_readiness(
                now=time.time(), cwd=cwd, cfg=config, daemon_url=daemon_url
            )
        return {
            "promotions": {"total": len(candidates), "top": candidates[:3]},
            "supersessions": {"pending": pending},
            "gaps": {
                "total": len(gap_rows),
                "top": [g["query"] for g in gap_rows],
            },
            "repeatFailures": {
                "total": repeat_failures.get("clusters", 0),
                "top": [
                    {
                        "signature": c["signature"],
                        "errorType": c["error_type"],
                        "sessions": c["sessions"],
                        "spanDays": c["span_days"],
                        "occurrences": c["occurrences"],
                    }
                    for c in repeat_failures.get("top", [])
                ],
            },
            # Spec 33 Part 8 rule R1 — informational only; never flips
            # maintenance_apply (a human does, after the manual bench
            # guards). `.get(..., default)` keeps the shape whole even if
            # the read-only pass itself failed (`{"error": True}`).
            "graduation": {
                "signalDays": graduation.get("signalDays", 0.0),
                "usedRatio": graduation.get("usedRatio", 0.0),
                "daysMet": graduation.get("daysMet", False),
                "ratioMet": graduation.get("ratioMet", False),
                "ready": graduation.get("ready", False),
                "benchGuards": graduation.get("benchGuards", "manual"),
            },
        }

    try:
        data = await asyncio.to_thread(_sync)
    except Exception:
        logger.debug("[digest] lifecycle read failed", exc_info=True)
        data = {
            "promotions": {"total": 0, "top": []},
            "supersessions": {"pending": 0},
            "gaps": {"total": 0, "top": []},
            "repeatFailures": {"total": 0, "top": []},
            "graduation": {
                "signalDays": 0.0,
                "usedRatio": 0.0,
                "daysMet": False,
                "ratioMet": False,
                "ready": False,
                "benchGuards": "manual",
            },
        }
    return {"heartbeat": heartbeat, **data}


class MaintenanceRunRequest(pydantic.BaseModel):
    # None → defer to memory.maintenance_apply (shadow by default).
    apply: bool | None = None


@router.post("/maintenance/run")
async def maintenance_run(
    body: MaintenanceRunRequest, request: fastapi.Request
) -> dict:
    """Run one maintenance pass (decay + hygiene) now (spec 33).

    The manual counterpart of the heartbeat — everything the scheduler does
    must be runnable by hand. ``apply`` overrides ``memory.maintenance_apply``
    for this pass only. The result is surfaced in ``GET /stats``.
    """
    import simba.memory.maintenance

    config = request.app.state.config
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))
    daemon_url = f"http://127.0.0.1:{getattr(config, 'port', 8741)}"
    result = await asyncio.to_thread(
        simba.memory.maintenance.run_maintenance,
        now=time.time(),
        cwd=cwd,
        cfg=config,
        daemon_url=daemon_url,
        apply=body.apply,
    )
    request.app.state.last_maintenance = result
    return result


class EmbedRequest(pydantic.BaseModel):
    text: str


@router.post("/embed")
async def embed_text(body: EmbedRequest, request: fastapi.Request) -> dict:
    """Return the query embedding for ``text`` (spec 28 intent classifier).

    Exposes the already-loaded embedder so the cheap, no-LLM intent classifier can
    embed the prompt once on the hot path and cosine-match it against precomputed
    doctrine-trigger embeddings. Fail-soft: an embed error returns an empty vector.
    """
    embed_query = request.app.state.embed_query
    try:
        vector = await embed_query(body.text)
    except Exception as e:
        logger.warning("[embed] failed: %s", e)
        return {"embedding": [], "error": "embedding_failed"}
    return {"embedding": vector}


class PreflightRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=True)

    task: str
    project_path: str | None = pydantic.Field(default=None, alias="projectPath")
    session_id: str = pydantic.Field(default="", alias="sessionId")


@router.post("/preflight")
async def preflight(body: PreflightRequest, request: fastapi.Request) -> dict:
    """Intent-keyed preflight (spec 28): doctrine + applicable rules + a brief.

    The warm-path equivalent of ``simba preflight``: recalls the intent-relevant
    doctrine (project-scoped) + the project's TOOL_RULEs, builds the ``🦁☑`` brief,
    and sets the per-turn preflight flag (host-local tempfile) so the PreToolUse
    gate clears. The flag set is best-effort (a missing session id is a no-op).
    """
    import simba.doctrine.preflight as dpreflight
    import simba.guardian.preflight_flag as preflight_flag

    task = (body.task or "").strip()
    if not task:
        raise fastapi.HTTPException(status_code=400, detail="task is required")

    async def _recall(types: list[str] | None) -> list[dict]:
        filters: dict[str, typing.Any] = {}
        if types:
            filters["types"] = types
        rr = RecallRequest(
            query=task,
            projectPath=body.project_path,
            filters=filters,
        )
        out = await recall_memories(rr, request)
        return out.get("memories", [])

    doctrine_mems = await _recall(None)
    doctrine_lines = [
        (m.get("content") or "").strip()
        for m in doctrine_mems
        if (m.get("content") or "").strip() and m.get("type") != "TOOL_RULE"
    ]
    tool_rule_mems = await _recall(["TOOL_RULE"])
    tool_rules = dpreflight.tool_rule_lines(tool_rule_mems)

    brief = dpreflight.build_brief(
        task=task,
        doctrine_lines=doctrine_lines,
        tool_rules=tool_rules,
        redirects=[],
    )
    preflight_flag.set_preflight(body.session_id, task=task)
    return {
        "brief": brief,
        "doctrine": doctrine_lines,
        "tool_rules": tool_rules,
        "redirects": [],
    }


@router.get("/health")
async def health(request: fastapi.Request) -> dict:
    table = getattr(request.app.state, "table", None)
    config = request.app.state.config
    start_time = request.app.state.start_time
    request_id = getattr(request.state, "request_id", "")
    diag = getattr(request.app.state, "diagnostics", None)

    memory_count = 0
    vector_ready = table is not None
    vector_error = ""
    if vector_ready:
        try:
            memory_count = await simba.memory.vector_db.count_rows(table)
        except Exception as exc:
            vector_ready = False
            vector_error = f"{type(exc).__name__}: {exc}"
            if diag is not None:
                diag.record_error("/health", exc, request_id)

    db_size = "unknown"
    db_path = getattr(request.app.state, "db_path", None)
    if db_path and pathlib.Path(db_path).is_dir():
        total = sum(
            f.stat().st_size for f in pathlib.Path(db_path).rglob("*") if f.is_file()
        )
        db_size = f"{total / 1024 / 1024:.2f}MB"

    fts_path = getattr(request.app.state, "fts_path", None)
    fts_exists = bool(fts_path and pathlib.Path(fts_path).exists())
    fts_count: int | None = None
    if fts_exists:
        try:
            fts_count = await asyncio.to_thread(
                _fts_count, fts_path, config.fts_tokenize
            )
        except Exception as exc:
            fts_exists = False
            if diag is not None:
                diag.record_error("/health", exc, request_id)

    embed_ready = bool(
        getattr(request.app.state, "embed", None)
        or getattr(request.app.state, "embed_query", None)
    )
    sync_scheduler = getattr(request.app.state, "sync_scheduler", None)
    components = {
        "vector": {
            "ready": vector_ready,
            "table": "memories" if table is not None else "",
            "path": str(db_path or ""),
            "count": memory_count,
            "size": db_size,
            "error": vector_error,
        },
        "fts": {
            "ready": fts_exists,
            "path": str(fts_path or ""),
            "count": fts_count,
            "tokenize": config.fts_tokenize,
        },
        "embedder": {
            "ready": embed_ready,
            "model": config.embedding_model,
            "dims": config.embedding_dims,
            "provider": config.embed_provider,
            "mode": "http" if config.embed_url else config.embed_provider,
        },
        "reranker": {
            "mode": (config.reranker_mode if config.llm_rerank_enabled else "none"),
            "intent_gating": config.rerank_intent_gating,
            "async_mode": config.llm_rerank_mode,
        },
        "sync": {
            "enabled": config.sync_interval > 0,
            "running": sync_scheduler is not None,
            "pending": False,
        },
    }
    ready = bool(vector_ready and embed_ready)
    degraded = not all(bool(c.get("ready", True)) for c in components.values())
    last_error = diag.last_error if diag is not None else None

    # RSS watchdog surfacing (config-gated, independent of whether the
    # background poll task happens to be running in THIS process --- e.g.
    # under create_app(use_lifespan=False), the common test path, the task
    # never starts even though the config says "enabled"). Null/absent
    # otherwise so the response shape stays stable for pre-watchdog tests.
    rss_mb: float | None = None
    rss_peak_mb: float | None = None
    if (
        float(getattr(config, "rss_soft_limit_mb", 0) or 0) > 0
        or float(getattr(config, "rss_hard_limit_mb", 0) or 0) > 0
    ):
        rss_mb = await asyncio.to_thread(simba.memory.rss_watchdog.current_rss_mb)
        rss_peak_mb = simba.memory.rss_watchdog.peak_rss_mb()

    # RSS history ring buffer (2026-07-17 follow-up): unlike rssMb/rssPeakMb
    # above (config-gated, computed fresh every call), the history only
    # exists on a running watchdog INSTANCE -- so this reads
    # `app.state.rss_watchdog` directly. `[]` when the watchdog is absent
    # (no lifespan in this process) or history sampling is disabled.
    watchdog = getattr(request.app.state, "rss_watchdog", None)
    rss_history = watchdog.history() if watchdog is not None else []

    return {
        "status": "ok" if ready and not degraded else "degraded",
        "ready": ready,
        "degraded": degraded,
        "requestId": request_id,
        "uptime": int(time.time() - start_time),
        "uptimeSeconds": int(time.time() - start_time),
        "memoryCount": memory_count,
        "embeddingModel": config.embedding_model,
        "embeddingDims": config.embedding_dims,
        "vectorDbSize": db_size,
        "dbPath": str(db_path or ""),
        "ftsPath": str(fts_path or ""),
        "components": components,
        "lastError": last_error,
        # POST /restart's drain/flush/execv sequence runs detached from the
        # request lifecycle (see background.schedule_restart) --- any
        # exception in it is otherwise invisible (the 202 is long gone by
        # the time it could fail). Null on a daemon that has never
        # attempted a restart, or whose last attempt succeeded.
        "lastRestartError": getattr(request.app.state, "last_restart_error", None),
        # Current / peak resident size (rss_watchdog.py) --- null unless
        # memory.rss_soft_limit_mb or memory.rss_hard_limit_mb is set.
        "rssMb": rss_mb,
        "rssPeakMb": rss_peak_mb,
        # RSS history ring buffer (rss_watchdog.py's RssWatchdog.history()) ---
        # `[]` when the watchdog is absent or memory.rss_history_samples=0.
        "rssHistory": rss_history,
    }


def _fts_count(fts_path: str, tokenize: str) -> int:
    with simba.memory.fts.connect(fts_path, tokenize=tokenize):
        return simba.memory.fts.count()


@router.get("/metrics")
async def metrics(request: fastapi.Request) -> dict:
    """Per-endpoint latency (p50/p95) plus uptime and total request count."""
    diag = getattr(request.app.state, "diagnostics", None)
    uptime = int(time.time() - request.app.state.start_time)
    latency = diag.all_latency_stats() if diag else {}
    return {
        "uptime_seconds": uptime,
        "latency": latency,
        "total_requests": diag._total_requests if diag else 0,
    }


@router.get("/stats")
async def stats(request: fastapi.Request) -> dict:
    table = request.app.state.table

    # Only `type`/`confidence`/`createdAt` are ever read below -- never
    # `content`/`context`/`vector` (2026-07-18: this endpoint is hit by every
    # SessionStart hook plus periodic diagnostics, so an unprojected scan was
    # the primary RSS-burst driver; see docs/adr/2026-07-10-internal-api-
    # footguns.md's 2026-07-18 addendum).
    all_memories = (
        await table.query().select(["type", "confidence", "createdAt"]).to_list()
    )

    by_type: dict[str, int] = {}
    total_confidence = 0.0
    oldest = None
    newest = None

    for mem in all_memories:
        t = mem.get("type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1
        total_confidence += mem.get("confidence", 0)

        created = mem.get("createdAt")
        if created:
            if oldest is None or created < oldest:
                oldest = created
            if newest is None or created > newest:
                newest = created

    total = len(all_memories)
    diag = getattr(request.app.state, "diagnostics", None)
    client_hits = diag.client_hits() if diag is not None else {}
    return {
        "total": total,
        "byType": by_type,
        "avgConfidence": round(total_confidence / total, 2) if total > 0 else 0,
        "oldestMemory": oldest,
        "newestMemory": newest,
        # Cumulative per-client request counts (X-Simba-Client attribution).
        "clientHits": client_hits,
        # Latest maintenance pass (spec 33): scheduler heartbeat or manual run.
        "lastMaintenance": getattr(request.app.state, "last_maintenance", None),
    }


# ── GET /list column projection ──
#
# A live incident (2026-07-10) traced a 40+min ~100% CPU / 36GB RSS daemon
# stall to the maintenance heartbeat's decay pass: its id->type join
# (`_fetch_type_map`) hit this endpoint over the WHOLE corpus (9,200+
# memories). `/list` fetched every column via `table.query().to_list()` --
# including each row's 1024-dim `vector` -- so the join alone converted ~9.4M
# Arrow floats to Python objects (the hot frames in the native profile).
# Vectors are opt-in only (`include_vectors`); `fields` lets any caller narrow
# the columns further. Both are pushed down to the Lance query itself
# (`.select()`) so an excluded column is never materialized, not merely
# dropped after the fact.
_LIST_DEFAULT_FIELDS: tuple[str, ...] = (
    "id",
    "type",
    "content",
    "context",
    "confidence",
    "createdAt",
    "accessCount",
    "projectPath",
    "sessionSource",
)
_LIST_ALL_FIELDS: tuple[str, ...] = (*_LIST_DEFAULT_FIELDS, "vector")


def _parse_list_fields(
    fields: str | None, *, include_vectors: bool
) -> tuple[str, ...] | None:
    """Parse ``?fields=a,b,c`` into a validated, order-preserving tuple of known
    top-level ``/list`` output fields.

    ``None`` (param omitted, or nothing valid survived) defers to the caller's
    full default set. Unknown names are dropped silently (fail-soft, matching
    this module's error handling elsewhere); ``vector`` only survives when
    ``include_vectors`` is set, so a stray ``fields=...,vector`` can never
    smuggle the embedding back in without the explicit opt-in.
    """
    if not fields:
        return None
    allowed = set(_LIST_ALL_FIELDS if include_vectors else _LIST_DEFAULT_FIELDS)
    out: list[str] = []
    for raw in fields.split(","):
        name = raw.strip()
        if name and name in allowed and name not in out:
            out.append(name)
    return tuple(out) or None


def _project_memory(
    m: dict[str, typing.Any], fields: tuple[str, ...], *, strict: bool
) -> dict[str, typing.Any]:
    """Build one ``/list`` row restricted to ``fields``.

    ``strict=False`` (no caller-supplied ``fields=``) mirrors the endpoint's
    historical default shape: ``context``/``projectPath``/``sessionSource``
    appear only when truthy, so a no-``fields`` response stays byte-identical
    to before this projection param existed. ``strict=True`` (an explicit
    ``fields=`` request) is a literal projection: every requested key is
    always present, using the same defaults as the legacy shape.
    """
    out: dict[str, typing.Any] = {}
    for f in fields:
        if f == "confidence":
            out[f] = m.get("confidence", 0)
        elif f == "accessCount":
            out[f] = m.get("accessCount", 0)
        elif f in ("context", "projectPath", "sessionSource") and not strict:
            value = m.get(f)
            if value:
                out[f] = value
        else:
            out[f] = m.get(f)
    return out


def _list_filter_where(
    *,
    type: str | None,
    project_path: str | None,
    session_source: str | None,
) -> str | None:
    """Combine ``/list``'s exact-match filters into one LanceDB ``.where()``
    SQL predicate (``None`` when no filter is set).

    Pushed down instead of Python post-filtering: even a one-session
    ``/list?sessionSource=...`` fetch was materializing the projected columns
    for the WHOLE corpus server-side before this (2026-07-17). ``since=`` is
    deliberately never included here -- see ``list_memories``'s docstring:
    ``createdAt`` is mixed-precision ISO text, so it needs a parsed-datetime
    comparison (the existing Python post-filter), not SQL string comparison.

    Every value is caller-supplied and routed through
    ``simba.memory.hybrid._lance_literal`` (single-quote doubling) --- never
    raw f-string interpolation --- so a value containing a quote character
    can't break out of the SQL string literal. LanceDB's own ``.where()``
    OVERWRITES on repeated calls (it is not additive), so all set filters are
    joined into a single ``AND``-ed predicate for one ``.where()`` call.
    """
    clauses: list[str] = []
    if type:
        clauses.append(f"type = {simba.memory.hybrid._lance_literal(type)}")
    if project_path:
        clauses.append(
            f"projectPath = {simba.memory.hybrid._lance_literal(project_path)}"
        )
    if session_source:
        clauses.append(
            f"sessionSource = {simba.memory.hybrid._lance_literal(session_source)}"
        )
    return " AND ".join(clauses) if clauses else None


@router.get("/list")
async def list_memories(
    request: fastapi.Request,
    type: str | None = fastapi.Query(default=None),
    projectPath: str | None = fastapi.Query(default=None),  # noqa: N803
    sessionSource: str | None = fastapi.Query(default=None),  # noqa: N803
    since: str | None = fastapi.Query(default=None),
    limit: int = fastapi.Query(default=20),
    offset: int = fastapi.Query(default=0),
    include_vectors: bool = fastapi.Query(default=False),
    fields: str | None = fastapi.Query(default=None),
) -> dict:
    """List memories with server-side type/project/session/time filtering and
    pagination.

    ``include_vectors`` (default ``False``) and ``fields`` (comma-separated
    output-field allowlist, e.g. ``id,type``) both narrow what leaves LanceDB
    -- see the module-level comment above ``_LIST_DEFAULT_FIELDS`` for why.
    ``sessionSource`` is an exact-match filter (same mechanism as ``type``/
    ``projectPath``). ``since`` is an ISO-8601 UTC timestamp; only rows with
    ``createdAt >= since`` are returned -- compared as parsed datetimes (via
    ``simba.memory.scoring.parse_epoch``), never as raw strings, because
    ``createdAt`` values in the corpus are mixed-precision (``...59.959Z`` vs
    ``...59Z``) and lexicographic comparison gets that pair backwards. An
    unparseable ``since`` is a 400.

    Runtime gates (2026-07-10 + 2026-07-17 incidents, docs/adr/2026-07-10-
    internal-api-footguns.md): a caller self-attributed as the daemon
    (``X-Simba-Client: daemon``, or a nested ``<origin>.daemon`` loopback --
    see harness/client.py's ``detect_client``) MUST pass ``fields=``. An
    unprojected internal self-call is exactly the shape that materialized the
    whole corpus -- including every row's 1024-dim vector -- for a 45GB peak
    footprint (rule 1). Projection alone didn't stop the follow-up incident:
    a daemon-internal caller whose ``fields=`` includes ``context`` MUST ALSO
    pass a row-bounding constraint (``sessionSource=``, ``projectPath=``,
    ``since=``, or ``limit<=1000``) -- an unbounded context-bearing scan is
    the corpus-wide-content shape that tripped the RSS watchdog (rule 2,
    2026-07-17 addendum). External/CLI/plain clients are unaffected by
    either gate; the static counterpart is
    tests/test_internal_list_projection_lint.py.
    """
    client = getattr(request.state, "client", "unknown") or "unknown"
    is_daemon_internal = client == simba.harness.client.DAEMON or client.endswith(
        f".{simba.harness.client.DAEMON}"
    )
    if is_daemon_internal and not fields:
        raise fastapi.HTTPException(
            status_code=400,
            detail="internal /list callers must pass fields= projection",
        )

    requested_fields = _parse_list_fields(fields, include_vectors=include_vectors)
    output_fields = requested_fields or (
        _LIST_ALL_FIELDS if include_vectors else _LIST_DEFAULT_FIELDS
    )

    if is_daemon_internal and "context" in output_fields:
        bounded = (
            bool(sessionSource) or bool(projectPath) or bool(since) or limit <= 1000
        )
        if not bounded:
            raise fastapi.HTTPException(
                status_code=400,
                detail=(
                    "internal /list callers requesting fields=...,context "
                    "must also pass a row-bounding constraint -- "
                    "sessionSource=, projectPath=, since=, or limit<=1000 "
                    "-- an unbounded context-bearing scan is the "
                    "2026-07-17 RSS-storm shape (see "
                    "docs/adr/2026-07-10-internal-api-footguns.md)"
                ),
            )

    since_epoch: float | None = None
    if since:
        since_epoch = simba.memory.scoring.parse_epoch(since)
        if since_epoch is None:
            raise fastapi.HTTPException(
                status_code=400, detail=f"invalid since= timestamp: {since!r}"
            )

    table = request.app.state.table

    # `type`/`projectPath`/`sessionSource` are pushed into LanceDB's
    # `.where()` below (see `_list_filter_where`) rather than fetched and
    # post-filtered in Python -- a bounded fetch (e.g. one sessionSource)
    # must never materialize the projected columns for the whole corpus
    # server-side. `createdAt` (the sort key + `since=` filter) is still
    # always fetched even when the caller's `fields` excludes it from the
    # OUTPUT -- only `_project_memory` below respects `fields` for what
    # actually gets returned.
    fetch_columns = sorted(
        set(output_fields) | {"id", "type", "projectPath", "sessionSource", "createdAt"}
    )
    query = table.query().select(fetch_columns)
    where_clause = _list_filter_where(
        type=type, project_path=projectPath, session_source=sessionSource
    )
    if where_clause:
        query = query.where(where_clause)
    all_memories = await query.to_list()

    if since_epoch is not None:
        all_memories = [
            m
            for m in all_memories
            if (simba.memory.scoring.parse_epoch(m.get("createdAt", "")) or -1.0)
            >= since_epoch
        ]

    all_memories.sort(key=lambda m: m.get("createdAt", ""), reverse=True)
    total = len(all_memories)
    paginated = all_memories[offset : offset + limit]

    memories = [
        _project_memory(m, output_fields, strict=requested_fields is not None)
        for m in paginated
    ]

    return {
        "memories": memories,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/sync")
async def trigger_sync(request: fastapi.Request) -> dict:
    """Trigger a one-off sync cycle (index + extract)."""
    scheduler = getattr(request.app.state, "sync_scheduler", None)
    if scheduler is None:
        return {"status": "not_configured"}

    simba.memory.background.spawn(scheduler.run_once())

    return {"status": "triggered", "cycle": scheduler.cycle_count + 1}


@router.post("/reindex")
async def reindex(request: fastapi.Request) -> dict:
    """Force a full rebuild of the FTS keyword mirror from LanceDB."""
    table = request.app.state.table
    fts_path = getattr(request.app.state, "fts_path", None)
    if not fts_path:
        return {"status": "no_mirror"}

    # Only the fields `fts.rebuild`/`_insert` read (see
    # `REQUIRED_MEMORY_FIELDS`) -- never `vector` (2026-07-18).
    rows = (
        await table.query()
        .select(list(simba.memory.fts.REQUIRED_MEMORY_FIELDS))
        .to_list()
    )
    non_system = [r for r in rows if r.get("type") != "SYSTEM"]

    def _rebuild() -> int:
        with simba.memory.fts.connect(fts_path):
            return simba.memory.fts.rebuild(non_system)

    indexed = await asyncio.to_thread(_rebuild)
    return {"status": "reindexed", "indexed": indexed}


@router.post("/compact")
async def compact(
    request: fastapi.Request,
    dry_run: bool = True,
    older_than_seconds: int = 86_400,
    delete_unverified: bool = False,
) -> dict:
    """Compact LanceDB storage and prune retained versions.

    Dry-run is the default because LanceDB compaction mutates the derived vector
    store.  The memory source rows are unchanged; this only removes old table
    versions and compactable data files after the retention window.
    """
    table = getattr(request.app.state, "table", None)
    if table is None:
        return {"status": "not_ready"}
    if older_than_seconds < 0:
        raise fastapi.HTTPException(
            status_code=400, detail="older_than_seconds must be >= 0"
        )

    db_path = getattr(request.app.state, "db_path", None)
    before = await _lance_storage_snapshot(table, db_path)
    retention = datetime.timedelta(seconds=older_than_seconds)
    if dry_run:
        return {
            "status": "dry_run",
            "retentionSeconds": older_than_seconds,
            "deleteUnverified": delete_unverified,
            "before": before,
        }

    optimize_stats = await simba.memory.vector_db.compact_table(
        table,
        cleanup_older_than=retention,
        delete_unverified=delete_unverified,
    )
    after = await _lance_storage_snapshot(table, db_path)
    if optimize_stats is None:
        return {
            "status": "failed",
            "retentionSeconds": older_than_seconds,
            "deleteUnverified": delete_unverified,
            "before": before,
            "after": after,
        }
    return {
        "status": "compacted",
        "retentionSeconds": older_than_seconds,
        "deleteUnverified": delete_unverified,
        "before": before,
        "after": after,
        "optimize": _jsonable(optimize_stats),
    }


@router.post("/reembed")
async def reembed(request: fastapi.Request) -> dict:
    """Re-embed every memory with the current model and rebuild the table.

    Needed after switching the embedder (a changed dimension requires rebuilding
    the LanceDB table). Uses the daemon's loaded doc embedder, then rebuilds the
    FTS mirror and swaps in the new table handle.
    """
    db_path = getattr(request.app.state, "db_path", None)
    embed = getattr(request.app.state, "embed", None)
    if not db_path or embed is None:
        return {"status": "not_ready"}

    new_table, count = await simba.memory.vector_db.reembed_table(db_path, embed)
    request.app.state.table = new_table

    fts_path = getattr(request.app.state, "fts_path", None)
    if fts_path:
        # Only the fields `fts.rebuild`/`_insert` read (see
        # `REQUIRED_MEMORY_FIELDS`) -- never `vector` (2026-07-18).
        rows = (
            await new_table.query()
            .select(list(simba.memory.fts.REQUIRED_MEMORY_FIELDS))
            .to_list()
        )
        non_system = [r for r in rows if r.get("type") != "SYSTEM"]

        def _rebuild() -> int:
            with simba.memory.fts.connect(fts_path):
                return simba.memory.fts.rebuild(non_system)

        await asyncio.to_thread(_rebuild)

    return {"status": "reembedded", "count": count}


class PatchRequest(pydantic.BaseModel):
    """Partial update to a memory record."""

    project_path: str | None = pydantic.Field(default=None, alias="projectPath")
    session_source: str | None = pydantic.Field(default=None, alias="sessionSource")


@router.patch("/memory/{memory_id}")
async def patch_memory(
    memory_id: str, body: PatchRequest, request: fastapi.Request
) -> dict:
    table = request.app.state.table
    updates: dict[str, str] = {}
    new_project_path: str | None = None
    if body.project_path is not None:
        # Normalize the scope path on patch too (spec 26) so a moved memory keeps
        # the same absolute/resolved form the store path uses.
        new_project_path = simba.memory.vector_db.normalize_project_path(
            body.project_path
        )
        updates["projectPath"] = new_project_path
    if body.session_source is not None:
        updates["sessionSource"] = body.session_source
    if not updates:
        raise fastapi.HTTPException(status_code=400, detail="no fields to update")
    await table.update(updates=updates, where=f"id = '{memory_id}'")

    fts_path = getattr(request.app.state, "fts_path", None)
    if fts_path and new_project_path is not None:
        try:
            await asyncio.to_thread(
                _fts_set_project, fts_path, memory_id, new_project_path
            )
        except Exception:
            logger.debug("[patch] fts mirror project update failed", exc_info=True)

    return {"status": "updated", "id": memory_id, "fields": list(updates.keys())}


class FeedbackRequest(pydantic.BaseModel):
    """Outcome-feedback signal for a recalled memory."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    signal: str  # "good" or "bad"
    weight: float | None = None  # override; None → cfg.feedback_default_weight
    # Session attribution (spec 33 v2 rule R2): when present, an append-only
    # usage event is recorded so "used in ≥N distinct sessions" is answerable.
    session_source: str = pydantic.Field(default="", alias="sessionSource")


@router.post("/memory/{memory_id}/feedback")
async def memory_feedback(
    memory_id: str,
    body: FeedbackRequest,
    request: fastapi.Request,
) -> dict:
    """Adjust ``feedback_score`` for a memory. Never deletes, never touches LanceDB.

    ``good`` adds ``+weight`` and ``bad`` adds ``-weight`` (clamped to
    ``[-1, 1]`` by the usage store). ``weight`` is clamped to ``[0, 1]`` so an
    adversarial value cannot cause an outsized jump. The new score feeds
    ``compute_strength`` on the next decay pass.
    """
    cfg = request.app.state.config
    cwd = pathlib.Path(getattr(request.app.state, "cwd", "."))

    if body.signal not in ("good", "bad"):
        raise fastapi.HTTPException(
            status_code=400, detail="signal must be 'good' or 'bad'"
        )

    weight = (
        body.weight
        if body.weight is not None
        else getattr(cfg, "feedback_default_weight", 0.3)
    )
    weight = max(0.0, min(1.0, float(weight)))
    delta = weight if body.signal == "good" else -weight

    import simba.db
    import simba.memory.usage
    import simba.memory.usage_events

    now = time.time()

    def _apply() -> float:
        with simba.db.connect(cwd):
            simba.memory.usage.apply_feedback(memory_id, delta, now=now)
            simba.memory.usage.bump_quality(
                memory_id,
                now,
                use=1 if body.signal == "good" else 0,
                noise=1 if body.signal == "bad" else 0,
            )
            if body.session_source:
                simba.memory.usage_events.record(
                    memory_id,
                    body.session_source,
                    "use" if body.signal == "good" else "noise",
                    now=now,
                )
            rows = simba.memory.usage.get_many([memory_id])
            return rows[memory_id].feedback_score if memory_id in rows else 0.0

    new_score = await asyncio.to_thread(_apply)
    return {"status": "ok", "id": memory_id, "feedback_score": new_score}


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, request: fastapi.Request) -> dict:
    table = request.app.state.table
    await table.delete(f"id = '{memory_id}'")

    fts_path = getattr(request.app.state, "fts_path", None)
    if fts_path:
        try:
            await asyncio.to_thread(_fts_delete, fts_path, memory_id)
        except Exception:
            logger.debug("[delete] fts mirror delete failed", exc_info=True)

    return {"status": "deleted", "id": memory_id}


@router.post("/hook/{event}")
async def run_hook(event: str, request: fastapi.Request) -> dict:
    """Run a canonical hook and return its CanonicalResult as JSON.

    Fail-open like the CLI: a null/array/malformed body degrades to ``{}`` (the
    CLI path catches ``json.JSONDecodeError`` and continues), never a 422. The
    handler is async only to await the body; dispatch() itself is blocking (it
    may loopback to /recall), so it runs in a threadpool off the event loop.
    """
    raw = await request.body()
    payload: dict = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except (json.JSONDecodeError, ValueError):
            payload = {}
    # Expose the inbound client as the request origin so a loopback /recall
    # (dispatch → recall_memories) nests as "<origin>.daemon". Contextvars are
    # copied into the threadpool, so the sync dispatch path inherits it.
    origin = request.headers.get("x-simba-client")
    token = simba.harness.client.set_origin_client(origin)
    try:
        result = await fastapi.concurrency.run_in_threadpool(
            simba.harness.core.dispatch, event, payload
        )
    except KeyError:
        raise fastapi.HTTPException(
            status_code=404, detail=f"unknown hook event: {event}"
        ) from None
    finally:
        simba.harness.client.reset_origin_client(token)
    return {
        "additional_context": result.additional_context,
        "suppress_output": result.suppress_output,
        "memory_count": result.memory_count,
        "block_reason": result.block_reason,
        "transform": result.transform,  # redirect rewrite (v2 tool gating)
        "escalated_block": result.escalated_block,  # pi-only strong TOOL_RULE block
        "system_message": result.system_message,  # compact relay (legs A/C)
    }


# Belt-and-suspenders margin between the 202 handler returning and tearing the
# process down via os.execv: the detached restart task (see
# simba.memory.background.schedule_restart) is NOT ordered by Starlette's
# "send response, then run background work" contract anymore (that contract
# is exactly what tied the old sequence to the request's cancellation scope
# --- see background.py's module comment for the live 2026-07-10 incident),
# so this sleep is now the ONLY thing giving the ASGI transport a moment to
# actually flush the response bytes before drain/flush/execv proceeds. Not a
# tunable (nothing a user would ever need to adjust), so it stays a plain
# constant rather than a `simba config` knob.
_RESTART_RESPONSE_DELAY_SECONDS = 0.2


async def _run_restart_sequence(argv: list[str], app: fastapi.FastAPI) -> None:
    """Drain, stop, flush, then replace the process image (POST /restart).

    Scheduled as a DETACHED asyncio task (``simba.memory.background.
    schedule_restart``) rather than Starlette ``BackgroundTasks`` --- see
    that function's docstring for the live 2026-07-10 incident this
    replaces: with ``BaseHTTPMiddleware`` in the stack, a client disconnect
    right after the 202 was transmitted cancelled this whole sequence
    before ``os.execv`` ever ran, and the 202 kept coming back regardless
    (uptime never reset). A detached task has no per-request scope for
    anything to cancel.

    Order matters and is covered by tests: the shutdown flag is set before
    ``drain`` runs (so any background pass mid-flight bails on its next
    self-HTTP check instead of hanging once this process disappears), drain
    completes before stdio is flushed, and stdio is flushed before the exec
    seam --- once ``reexec`` returns control to a new process image, nothing
    Python-level here runs again.

    Any exception anywhere in this sequence is logged at CRITICAL (with a
    full traceback) and recorded onto ``app.state.last_restart_error`` for
    ``GET /health`` to surface. Previously an exception here vanished with
    no trace at all: it ran after the 202 was already sent, so nothing
    HTTP-visible could ever show it --- and live, the auto-spawned daemon's
    stdout/stderr went to DEVNULL (see session_start.py), so even the
    traceback on stderr was buried. Success leaves the field null.
    """
    app.state.last_restart_error = None
    try:
        await asyncio.sleep(_RESTART_RESPONSE_DELAY_SECONDS)
        simba.memory.background.mark_shutting_down()
        config = app.state.config
        await simba.memory.background.drain(config.shutdown_timeout)
        scheduler = getattr(app.state, "maintenance_scheduler", None)
        if scheduler is not None:
            scheduler.stop()
        sys.stdout.flush()
        sys.stderr.flush()
        simba.memory.background.reexec(argv)
    except Exception as exc:
        logger.critical("[restart] failed: %s", exc, exc_info=True)
        app.state.last_restart_error = f"{type(exc).__name__}: {exc}"


@router.post("/restart", status_code=202, response_model=None)
async def restart(
    request: fastapi.Request,
) -> dict | starlette.responses.JSONResponse:
    """Self-restart the daemon in place via ``os.execv``.

    The user runs the daemon foreground in a terminal; this replaces the
    running process with a fresh image of the (possibly newer) on-disk code
    --- same PID, same terminal, stdout/stderr piping preserved --- rather
    than requiring an external process manager. fds 1/2 (and every other
    descriptor) survive ``execv`` on POSIX, so the foreground terminal keeps
    streaming output with no pipe to re-plumb; the PID is unchanged, so shell
    job control is unaffected; threads/locks die with the old image, which
    sidesteps the graceful-shutdown-stall class entirely (this path never
    waits on anything past ``drain()``); and the uvicorn listener socket,
    non-inheritable per PEP 446, closes automatically at exec, with the new
    image rebinding it (uvicorn sets ``SO_REUSEADDR``).

    Responds 202 immediately with the pre-restart pid; the drain + exec
    sequence itself runs afterward as a DETACHED task (see
    ``simba.memory.background.schedule_restart`` and
    ``_run_restart_sequence``) --- deliberately NOT Starlette
    ``BackgroundTasks``, which ride the request/response ASGI cycle and can
    be cancelled by a post-response client disconnect (the live 2026-07-10
    incident: the 202 came back, but uptime kept climbing forever after).
    Only the exact argv ``main()`` captured at boot is ever exec'd --- never
    a guessed command --- so an app with no boot argv on record (503) or a
    non-POSIX platform (501) refuses instead of guessing.
    """
    if os.name != "posix":
        return starlette.responses.JSONResponse(
            status_code=501,
            content={
                "error": "restart requires POSIX os.execv (unsupported on this "
                "platform)"
            },
        )
    argv = getattr(request.app.state, "boot_argv", None)
    if not argv:
        return starlette.responses.JSONResponse(
            status_code=503,
            content={"error": "restart unavailable: no boot argv"},
        )
    pid = os.getpid()
    simba.memory.background.schedule_restart(
        _run_restart_sequence(list(argv), request.app)
    )
    return {"restarting": True, "pid": pid}

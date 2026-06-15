"""FastAPI routes for the memory daemon.

Ported from claude-memory/routes/*.js — all 6 endpoints in a single file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
import typing
import uuid

import fastapi
import pydantic
import starlette.middleware.base
import starlette.requests
import starlette.responses

import simba.harness.core
import simba.memory.dimensions
import simba.memory.fts
import simba.memory.hybrid
import simba.memory.recall_plan
import simba.memory.scoring
import simba.memory.vector_db

logger = logging.getLogger("simba.memory")

router = fastapi.APIRouter()

# Background tasks need a strong reference to avoid GC before completion.
_background_tasks: set[asyncio.Task[None]] = set()


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


async def _bump_usage(memory_ids: list[str], now: float, cwd: pathlib.Path) -> None:
    """Bump access stats in ``memory_usage`` for recalled ids. Fire-and-forget.

    The sqlite ``memory_usage`` table is the authoritative ranking sidecar; this
    runs alongside the LanceDB fire-and-forget ``update_access_tracking``. Any
    failure is swallowed (recall must never break on a usage write).
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
        except Exception:
            logger.debug("[recall] usage bump failed", exc_info=True)

    await asyncio.to_thread(_sync)


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
        t0 = time.monotonic()
        response = await call_next(request)
        if diag is not None:
            diag.record_request(request.url.path)
            diag.record_latency(request.url.path, (time.monotonic() - t0) * 1000)
            if diag.should_report():
                table = getattr(request.app.state, "table", None)
                task = asyncio.create_task(diag.emit_report(table))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
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


class RecallRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=True)

    query: str
    min_similarity: float | None = pydantic.Field(default=None, alias="minSimilarity")
    max_results: int | None = pydantic.Field(default=None, alias="maxResults")
    project_path: str | None = pydantic.Field(default=None, alias="projectPath")
    filters: dict[str, typing.Any] = pydantic.Field(default_factory=dict)


class HookPayload(pydantic.BaseModel):
    """Opaque per-harness hook payload.

    Shape varies by event (cwd, prompt, response, transcript_path, …).
    """

    model_config = pydantic.ConfigDict(extra="allow")
    cwd: str = ""


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

    if len(body.content) > config.max_content_length:
        max_len = config.max_content_length
        got_len = len(body.content)
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"content too long (max {max_len} chars, got {got_len})",
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
            body.project_path or "(global)",
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
    # threshold, replace it with this fresher one instead of appending a near-dup.
    superseded_id: str | None = None
    if config.supersede_enabled:
        candidates = await simba.memory.vector_db.search_memories(
            table,
            embedding,
            config.supersede_threshold,
            1,
            {"projectPath": body.project_path, "types": [body.type]},
        )
        if candidates:
            superseded_id = candidates[0]["id"]
            await table.delete(f"id = '{superseded_id}'")

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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
                "projectPath": body.project_path,
                "createdAt": now,
                "lastAccessedAt": now,
                "accessCount": 0,
                "vector": embedding,
            }
        ]
    )

    logger.info(
        '[store] project=%s type=%s content="%s" -> stored %s',
        body.project_path or "(global)",
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
                    "projectPath": body.project_path,
                },
            )
        except Exception:
            logger.debug("[store] fts mirror upsert failed", exc_info=True)
        if superseded_id:
            try:
                await asyncio.to_thread(_fts_delete, fts_path, superseded_id)
            except Exception:
                logger.debug("[store] fts supersede-delete failed", exc_info=True)

    diag = getattr(request.app.state, "diagnostics", None)
    if diag is not None:
        diag.record_store(body.type, duplicate=False)

    return {
        "status": "superseded" if superseded_id else "stored",
        "id": memory_id,
        **({"supersededId": superseded_id} if superseded_id else {}),
        "embedding_dims": len(embedding),
    }


@router.post("/recall")
async def recall_memories(body: RecallRequest, request: fastapi.Request) -> dict:

    table = request.app.state.table
    config = request.app.state.config
    embed_query = request.app.state.embed_query

    start_time = time.time()
    try:
        embedding = await embed_query(body.query)
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
        body.query,
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
    if body.project_path:
        filters["projectPath"] = body.project_path

    # On a HyDE cache miss, warm the cache off the hot path so the next identical
    # query gets the LLM answer for its 2nd arm (this recall serves the fallback).
    if hyde_llm_on and hyde_cache is not None and llm_client is not None:
        hyde_key = hyde_cache.signature(body.query)
        if hyde_cache.get(hyde_key) is None:
            task = asyncio.create_task(
                _bg_hyde(hyde_cache, hyde_key, body.query, llm_client)
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

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
            body.query,
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

    results = [
        {
            "id": m["id"],
            "type": m["type"],
            "content": m["content"],
            **({"context": m["context"]} if m.get("context") else {}),
            "similarity": round(m["similarity"], 2),
            "confidence": m.get("confidence", 0),
            "createdAt": m.get("createdAt"),
            **({"projectPath": m["projectPath"]} if m.get("projectPath") else {}),
            **({"sessionSource": m["sessionSource"]} if m.get("sessionSource") else {}),
        }
        for m in memories
    ]

    query_time_ms = int((time.time() - start_time) * 1000)
    top_sim = round(results[0]["similarity"], 2) if results else 0.0
    logger.info(
        '[recall] project=%s mode=%s floor=%.2f query="%s" '
        "-> %d memories (%dms), top: %.2f",
        body.project_path or "(global)",
        mode,
        min_sim,
        body.query[:50],
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

        # Existing LanceDB access tracking (kept; never read for ranking).
        task1 = asyncio.create_task(
            simba.memory.vector_db.update_access_tracking(table, recalled_ids)
        )
        _background_tasks.add(task1)
        task1.add_done_callback(_background_tasks.discard)

        # Authoritative sqlite usage store (drives decay/feedback ranking).
        task2 = asyncio.create_task(_bump_usage(recalled_ids, now_epoch, cwd))
        _background_tasks.add(task2)
        task2.add_done_callback(_background_tasks.discard)

    return {"memories": results, "queryTimeMs": query_time_ms}


@router.get("/health")
async def health(request: fastapi.Request) -> dict:

    table = request.app.state.table
    config = request.app.state.config
    start_time = request.app.state.start_time

    memory_count = await simba.memory.vector_db.count_rows(table)

    db_size = "unknown"
    db_path = request.app.state.db_path
    if db_path and pathlib.Path(db_path).is_dir():
        total = sum(
            f.stat().st_size for f in pathlib.Path(db_path).iterdir() if f.is_file()
        )
        db_size = f"{total / 1024 / 1024:.2f}MB"

    return {
        "status": "ok",
        "uptime": int(time.time() - start_time),
        "memoryCount": memory_count,
        "embeddingModel": config.embedding_model,
        "vectorDbSize": db_size,
    }


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

    all_memories = await table.query().to_list()

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
    return {
        "total": total,
        "byType": by_type,
        "avgConfidence": round(total_confidence / total, 2) if total > 0 else 0,
        "oldestMemory": oldest,
        "newestMemory": newest,
    }


@router.get("/list")
async def list_memories(
    request: fastapi.Request,
    type: str | None = fastapi.Query(default=None),
    projectPath: str | None = fastapi.Query(default=None),  # noqa: N803
    limit: int = fastapi.Query(default=20),
    offset: int = fastapi.Query(default=0),
) -> dict:
    table = request.app.state.table

    all_memories = await table.query().to_list()

    if type:
        all_memories = [m for m in all_memories if m.get("type") == type]
    if projectPath:
        all_memories = [m for m in all_memories if m.get("projectPath") == projectPath]

    all_memories.sort(key=lambda m: m.get("createdAt", ""), reverse=True)
    total = len(all_memories)
    paginated = all_memories[offset : offset + limit]

    memories = [
        {
            "id": m["id"],
            "type": m["type"],
            "content": m["content"],
            **({"context": m["context"]} if m.get("context") else {}),
            "confidence": m.get("confidence", 0),
            "createdAt": m.get("createdAt"),
            "accessCount": m.get("accessCount", 0),
            **({"projectPath": m["projectPath"]} if m.get("projectPath") else {}),
            **({"sessionSource": m["sessionSource"]} if m.get("sessionSource") else {}),
        }
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

    task = asyncio.create_task(scheduler.run_once())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"status": "triggered", "cycle": scheduler.cycle_count + 1}


@router.post("/reindex")
async def reindex(request: fastapi.Request) -> dict:
    """Force a full rebuild of the FTS keyword mirror from LanceDB."""
    table = request.app.state.table
    fts_path = getattr(request.app.state, "fts_path", None)
    if not fts_path:
        return {"status": "no_mirror"}

    rows = await table.query().to_list()
    non_system = [r for r in rows if r.get("type") != "SYSTEM"]

    def _rebuild() -> int:
        with simba.memory.fts.connect(fts_path):
            return simba.memory.fts.rebuild(non_system)

    indexed = await asyncio.to_thread(_rebuild)
    return {"status": "reindexed", "indexed": indexed}


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
        rows = await new_table.query().to_list()
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
    if body.project_path is not None:
        updates["projectPath"] = body.project_path
    if body.session_source is not None:
        updates["sessionSource"] = body.session_source
    if not updates:
        raise fastapi.HTTPException(status_code=400, detail="no fields to update")
    await table.update(updates=updates, where=f"id = '{memory_id}'")

    fts_path = getattr(request.app.state, "fts_path", None)
    if fts_path and body.project_path is not None:
        try:
            await asyncio.to_thread(
                _fts_set_project, fts_path, memory_id, body.project_path
            )
        except Exception:
            logger.debug("[patch] fts mirror project update failed", exc_info=True)

    return {"status": "updated", "id": memory_id, "fields": list(updates.keys())}


class FeedbackRequest(pydantic.BaseModel):
    """Outcome-feedback signal for a recalled memory."""

    signal: str  # "good" or "bad"
    weight: float | None = None  # override; None → cfg.feedback_default_weight


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

    now = time.time()

    def _apply() -> float:
        with simba.db.connect(cwd):
            simba.memory.usage.apply_feedback(memory_id, delta, now=now)
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
def run_hook(event: str, body: HookPayload) -> dict:
    """Run a canonical hook and return its CanonicalResult as JSON.

    Sync handler: dispatch() may make a blocking loopback to /recall, so FastAPI
    offloads it to a threadpool instead of blocking the event loop.
    """
    try:
        result = simba.harness.core.dispatch(event, body.model_dump())
    except KeyError:
        raise fastapi.HTTPException(
            status_code=404, detail=f"unknown hook event: {event}"
        ) from None
    return {
        "additional_context": result.additional_context,
        "suppress_output": result.suppress_output,
        "block_reason": result.block_reason,
        "transform": result.transform,  # None today; v2 tool gating
    }

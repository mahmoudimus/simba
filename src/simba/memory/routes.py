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

import simba.memory.fts
import simba.memory.hybrid
import simba.memory.intent
import simba.memory.vector_db

logger = logging.getLogger("simba.memory")

router = fastapi.APIRouter()

# Background tasks need a strong reference to avoid GC before completion.
_background_tasks: set[asyncio.Task[None]] = set()


# ── FTS keyword-mirror sync helpers (run in a worker thread via to_thread) ──
# Per-call connections sidestep SQLite's thread-affinity. All mirror writes are
# best-effort: failures are logged, never surfaced (startup reconcile heals).


def _fts_upsert(fts_path: str, memory: dict[str, typing.Any]) -> None:
    conn = simba.memory.fts.connect(fts_path)
    try:
        simba.memory.fts.upsert(conn, memory)
    finally:
        conn.close()


def _fts_delete(fts_path: str, memory_id: str) -> None:
    conn = simba.memory.fts.connect(fts_path)
    try:
        simba.memory.fts.delete(conn, memory_id)
    finally:
        conn.close()


def _fts_set_project(fts_path: str, memory_id: str, project_path: str) -> None:
    conn = simba.memory.fts.connect(fts_path)
    try:
        simba.memory.fts.set_project(conn, memory_id, project_path)
    finally:
        conn.close()


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
        response = await call_next(request)
        if diag is not None:
            diag.record_request(request.url.path)
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

    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    await table.add(
        [
            {
                "id": memory_id,
                "type": body.type,
                "content": body.content,
                "context": body.context,
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
                    "context": body.context,
                    "confidence": body.confidence,
                    "createdAt": now,
                    "projectPath": body.project_path,
                },
            )
        except Exception:
            logger.debug("[store] fts mirror upsert failed", exc_info=True)

    diag = getattr(request.app.state, "diagnostics", None)
    if diag is not None:
        diag.record_store(body.type, duplicate=False)

    return {
        "status": "stored",
        "id": memory_id,
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

    # Cosine floor: an explicit client value always wins (escape hatch);
    # otherwise pick by query intent — broad/aggregation queries widen the net.
    if body.min_similarity is not None:
        min_sim = body.min_similarity
        mode = "explicit"
    elif config.intent_aware:
        mode = simba.memory.intent.classify(body.query)
        min_sim = (
            config.min_similarity_broad if mode == "broad" else config.min_similarity
        )
    else:
        min_sim = config.min_similarity
        mode = "precise"
    max_res = body.max_results if body.max_results is not None else config.max_results

    filters = dict(body.filters)
    if body.project_path:
        filters["projectPath"] = body.project_path

    fts_path = getattr(request.app.state, "fts_path", None)
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
        )
    else:
        memories = await simba.memory.vector_db.search_memories(
            table, embedding, min_sim, max_res, filters
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
        task = asyncio.create_task(
            simba.memory.vector_db.update_access_tracking(table, recalled_ids)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

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
    limit: int = fastapi.Query(default=20),
    offset: int = fastapi.Query(default=0),
) -> dict:
    table = request.app.state.table

    all_memories = await table.query().to_list()

    if type:
        all_memories = [m for m in all_memories if m.get("type") == type]

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
        conn = simba.memory.fts.connect(fts_path)
        try:
            return simba.memory.fts.rebuild(conn, non_system)
        finally:
            conn.close()

    indexed = await asyncio.to_thread(_rebuild)
    return {"status": "reindexed", "indexed": indexed}


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

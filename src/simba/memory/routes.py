"""FastAPI routes for the memory daemon.

Ported from claude-memory/routes/*.js — all 6 endpoints in a single file.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import pathlib
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
import simba.memory.conflict
import simba.memory.conflict_store
import simba.memory.dimensions
import simba.memory.fts
import simba.memory.hybrid
import simba.memory.provenance
import simba.memory.query_filters
import simba.memory.recall_cache
import simba.memory.recall_plan
import simba.memory.scoring
import simba.memory.supersession
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

    if len(body.content) > config.max_content_length:
        max_len = config.max_content_length
        got_len = len(body.content)
        raise fastapi.HTTPException(
            status_code=400,
            detail=f"content too long (max {max_len} chars, got {got_len})",
        )

    # A store mutates the corpus (new memory, supersession annotation), so drop
    # cached recalls to keep the short-TTL cache from serving pre-store results.
    recall_cache = getattr(request.app.state, "recall_cache", None)
    if recall_cache is not None:
        recall_cache.clear()

    # Normalize the scope path to an absolute, symlink-resolved path (spec 26) so
    # the client's resolved ancestor chain can match it by string membership. An
    # empty (global) path stays empty.
    project_path = simba.memory.vector_db.normalize_project_path(body.project_path)

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
        filters["projectPath"] = body.project_path

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
            task1 = asyncio.create_task(
                simba.memory.vector_db.update_access_tracking(table, recalled_ids)
            )
            _background_tasks.add(task1)
            task1.add_done_callback(_background_tasks.discard)

        # Authoritative sqlite usage store (drives decay/feedback ranking).
        task2 = asyncio.create_task(_bump_usage(recalled_ids, now_epoch, cwd))
        _background_tasks.add(task2)
        task2.add_done_callback(_background_tasks.discard)

    if recall_cache is not None and cache_key is not None:
        recall_cache.put(cache_key, results, now=time.time())

    return {"memories": results, "queryTimeMs": query_time_ms}


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
            simba.memory.usage.bump_quality(
                memory_id,
                now,
                use=1 if body.signal == "good" else 0,
                noise=1 if body.signal == "bad" else 0,
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
    }

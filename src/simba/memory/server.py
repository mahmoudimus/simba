"""FastAPI memory daemon server.

Ported from claude-memory/server.js.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import contextlib
import pathlib

import fastapi
import uvicorn

import simba.memory.config
import simba.memory.diagnostics
import simba.memory.embedding_cache
import simba.memory.embeddings
import simba.memory.fts
import simba.memory.hyde_cache
import simba.memory.recall_cache
import simba.memory.rerank_cache
import simba.memory.routes

_DEFAULT_DB_DIR = ".simba/memory"


def _raise_fd_limit(target: int) -> int | None:
    """Raise the process soft RLIMIT_NOFILE toward ``target`` (capped at the hard
    limit). Returns the resulting soft limit, or ``None`` if left unchanged.

    A full LanceDB scan opens many fragment/version files at once; the macOS
    default soft limit (256) is easily exhausted on a large table -> ``os error
    24``. Fail-soft: any platform refusal leaves the OS default in place.
    """
    if target <= 0:
        return None
    import resource

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if new_soft <= soft:
            return soft
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        return new_soft
    except (ValueError, OSError):
        return None


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown for the memory daemon."""
    # Tag this process so loopback /recall calls (run_hook → dispatch → recall)
    # are attributed to "daemon", not conflated with real CLI traffic. Forced
    # (not setdefault): the daemon may be spawned by a SessionStart hook that
    # already exported SIMBA_CLIENT — the daemon's own identity is always
    # "daemon". Only the real server boots the lifespan; tests pass
    # create_app(use_lifespan=False).
    import logging
    import os

    os.environ["SIMBA_CLIENT"] = "daemon"
    # Raise the FD limit before opening the DB / serving scans (fail-soft).
    new_soft = _raise_fd_limit(getattr(app.state.config, "daemon_fd_limit", 65_536))
    if new_soft is not None:
        logging.getLogger("simba.memory").info(
            "[startup] soft FD limit raised to %d", new_soft
        )
    await init_database(app)
    await init_embeddings(app)
    sync_task = await _start_sync_scheduler(app)
    maintenance_task = await _start_maintenance_scheduler(app)
    yield
    # Shutdown: stop schedulers, wait briefly, then force-cancel.
    config: simba.memory.config.MemoryConfig = app.state.config
    sync_timeout = max(1, config.shutdown_timeout // 2)
    if sync_task is not None:
        scheduler = getattr(app.state, "sync_scheduler", None)
        if scheduler is not None:
            scheduler.stop()
        # Give the running cycle a moment to finish, then force-cancel.
        try:
            await asyncio.wait_for(asyncio.shield(sync_task), timeout=sync_timeout)
        except (TimeoutError, asyncio.CancelledError):
            sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sync_task
    if maintenance_task is not None:
        maintenance = getattr(app.state, "maintenance_scheduler", None)
        if maintenance is not None:
            maintenance.stop()
        try:
            await asyncio.wait_for(
                asyncio.shield(maintenance_task), timeout=sync_timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            maintenance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await maintenance_task
    await shutdown_embeddings(app)


async def _start_sync_scheduler(
    app: fastapi.FastAPI,
) -> asyncio.Task | None:  # type: ignore[type-arg]
    """Start a background SyncScheduler if configured."""
    config: simba.memory.config.MemoryConfig = app.state.config
    if config.sync_interval <= 0:
        app.state.sync_scheduler = None
        return None

    from simba.sync.scheduler import SyncScheduler

    logger = logging.getLogger("simba.memory")
    scheduler = SyncScheduler(
        daemon_url=f"http://127.0.0.1:{config.port}",
        interval_seconds=config.sync_interval,
    )
    app.state.sync_scheduler = scheduler
    task = asyncio.create_task(scheduler.run_forever())
    logger.info(
        "[sync] Background scheduler started (interval: %ds)", config.sync_interval
    )
    return task


async def _start_maintenance_scheduler(
    app: fastapi.FastAPI,
) -> asyncio.Task | None:  # type: ignore[type-arg]
    """Start the maintenance heartbeat (spec 33) — independent of sync_interval.

    Decay + hygiene were previously reachable only through the SyncScheduler
    (default-off via ``sync_interval=0``), so they never ran. The heartbeat is
    gated only on ``maintenance_interval_hours`` and runs SHADOW (dry) until
    ``maintenance_apply`` flips after measurement.
    """
    config: simba.memory.config.MemoryConfig = app.state.config
    hours = float(getattr(config, "maintenance_interval_hours", 24.0) or 0.0)
    if hours <= 0:
        app.state.maintenance_scheduler = None
        return None

    from simba.memory.maintenance import MaintenanceScheduler

    scheduler = MaintenanceScheduler(
        cwd=pathlib.Path(getattr(app.state, "cwd", ".")),
        daemon_url=f"http://127.0.0.1:{config.port}",
        interval_seconds=hours * 3600.0,
        startup_delay_seconds=float(
            getattr(config, "maintenance_startup_delay_seconds", 300.0)
        ),
        on_result=lambda r: setattr(app.state, "last_maintenance", r),
    )
    app.state.maintenance_scheduler = scheduler
    task = asyncio.create_task(scheduler.run_forever())
    logging.getLogger("simba.memory").info(
        "[maintenance] heartbeat started (interval: %.1fh, apply=%s)",
        hours,
        bool(getattr(config, "maintenance_apply", False)),
    )
    return task


def create_app(
    config: simba.memory.config.MemoryConfig | None = None,
    use_lifespan: bool = False,
) -> fastapi.FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = simba.memory.config.load_config()

    app = fastapi.FastAPI(
        title="Memory Daemon",
        version="1.0.0",
        lifespan=lifespan if use_lifespan else None,
    )
    app.include_router(simba.memory.routes.router)
    app.add_middleware(simba.memory.routes.DiagnosticsMiddleware)

    app.state.config = config
    app.state.start_time = time.time()
    # Repo cwd for the sqlite usage sidecar (memory_usage lives in .simba/simba.db).
    app.state.cwd = pathlib.Path.cwd()
    app.state.table = None
    app.state.db_path = None
    app.state.fts_path = None
    app.state.embed = None
    app.state.embed_query = None
    app.state.embed_cache = None
    app.state.llm_client = None
    app.state.diagnostics = simba.memory.diagnostics.DiagnosticsTracker(
        report_interval=config.diagnostics_after,
        reservoir_size=config.diagnostics_reservoir_size,
        retention_seconds=config.lancedb_version_retention_seconds,
    )
    # Non-blocking LLM rerank cache (daemon-process lifetime).
    app.state.rerank_cache = simba.memory.rerank_cache.RerankCache(
        max_entries=config.rerank_cache_size,
    )
    # Non-blocking HyDE answer cache (daemon-process lifetime).
    app.state.hyde_cache = simba.memory.hyde_cache.HydeCache(
        max_entries=config.hyde_cache_size,
    )
    # Short-TTL recall result cache: collapses identical-query storms (None when
    # disabled via recall_cache_ttl_seconds=0).
    app.state.recall_cache = (
        simba.memory.recall_cache.RecallCache(
            max_entries=config.recall_cache_size,
            ttl_seconds=config.recall_cache_ttl_seconds,
        )
        if config.recall_cache_ttl_seconds > 0
        else None
    )

    return app


def _resolve_db_path(config: simba.memory.config.MemoryConfig) -> pathlib.Path:
    """Resolve the database directory from config or default."""
    if config.db_path:
        return pathlib.Path(config.db_path)
    return pathlib.Path.cwd() / _DEFAULT_DB_DIR


def _embed_cache_model_id(config: simba.memory.config.MemoryConfig) -> str:
    """Stable model identifier for the embed-cache key (a model change -> new key)."""
    return (
        getattr(config, "embed_url", "")
        or f"{getattr(config, 'model_repo', '')}/{getattr(config, 'model_file', '')}"
        or "default"
    )


def _embed_cache_path(config: simba.memory.config.MemoryConfig) -> pathlib.Path:
    """Persistent query-embed cache path (or ``<db dir>/embed_cache.db``)."""
    if config.embed_cache_path:
        return pathlib.Path(config.embed_cache_path)
    return _resolve_db_path(config) / "embed_cache.db"


def _cached_query_embedder(
    service: simba.memory.embeddings.EmbeddingService,
    cache: simba.memory.embedding_cache.EmbeddingCache,
    model_id: str,
):
    """Async query embed that hits the persistent cache first, dedups identical queries.

    Cache get/put run on the event-loop thread (embed_query is only awaited from the
    async recall handler), so the sqlite connection stays single-threaded.
    """

    async def embed_query(text: str) -> list[float]:
        hit = cache.get(model_id, "search_query", text)
        if hit is not None:
            return hit
        vector = await service.embed(text, task=simba.memory.embeddings.TaskType.QUERY)
        cache.put(model_id, "search_query", text, vector)
        return vector

    return embed_query


async def init_database(
    app: fastapi.FastAPI, data_dir: pathlib.Path | None = None
) -> None:
    """Initialize LanceDB and attach to app state."""
    try:
        import lancedb
    except ImportError as exc:  # core-only install (no `embed` extra)
        raise ImportError(
            "The semantic-memory daemon needs the optional ML dependencies. "
            "Install them with: pip install 'simba-ai[embed]'"
        ) from exc

    config: simba.memory.config.MemoryConfig = app.state.config

    if data_dir is None:
        data_dir = _resolve_db_path(config)
    db_path = data_dir / "memories.lance"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = await lancedb.connect_async(str(db_path))
    app.state.db_path = str(db_path)

    try:
        table = await db.open_table("memories")
    except Exception:
        dims = config.embedding_dims
        zero_vector = [0.0] * dims
        table = await db.create_table(
            "memories",
            [
                {
                    "id": f"init_{int(time.time())}",
                    "type": "SYSTEM",
                    "content": "Memory system initialized",
                    "context": "",
                    "tags": "[]",
                    "confidence": 1.0,
                    "sessionSource": "",
                    "projectPath": "",
                    "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "lastAccessedAt": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "accessCount": 0,
                    "vector": zero_vector,
                }
            ],
        )

    app.state.table = table

    await init_fts_mirror(app, data_dir)


async def init_fts_mirror(app: fastapi.FastAPI, data_dir: pathlib.Path) -> None:
    """Create the FTS5 keyword mirror and reconcile it against LanceDB.

    The mirror lives at ``<data_dir>/memory_fts.db`` so it travels with the
    vectors.  On startup we rebuild it from LanceDB whenever the indexed count
    diverges from the non-SYSTEM memory count — this backfills the existing
    corpus on first run and heals any drift from best-effort writes.
    """
    config: simba.memory.config.MemoryConfig = app.state.config
    logger = logging.getLogger("simba.memory")
    fts_path = data_dir / simba.memory.fts.FTS_FILENAME
    simba.memory.fts.init(fts_path, tokenize=config.fts_tokenize)
    app.state.fts_path = str(fts_path)

    try:
        rows = await app.state.table.query().to_list()
        non_system = [r for r in rows if r.get("type") != "SYSTEM"]

        def _reconcile() -> int | None:
            with simba.memory.fts.connect(fts_path):
                if simba.memory.fts.count() != len(non_system):
                    return simba.memory.fts.rebuild(non_system)
                return None

        rebuilt = await asyncio.to_thread(_reconcile)
        if rebuilt is not None:
            logger.info("[fts] reconciled keyword mirror: %d rows indexed", rebuilt)
    except Exception:
        logger.debug("[fts] mirror reconcile failed", exc_info=True)


async def init_embeddings(
    app: fastapi.FastAPI,
) -> simba.memory.embeddings.EmbeddingService:
    """Initialize the embedding service and attach to app state."""
    logger = logging.getLogger("simba.memory")
    config: simba.memory.config.MemoryConfig = app.state.config
    service = simba.memory.embeddings.EmbeddingService(config)
    if config.embed_url:
        logger.info("[embed] Connecting to %s ...", config.embed_url)
    else:
        logger.info("[embed] Loading model...")
    await service.start()
    logger.info("[embed] Ready")
    app.state.embed = service.embed
    if config.embed_cache_enabled:
        cache = simba.memory.embedding_cache.EmbeddingCache(_embed_cache_path(config))
        app.state.embed_cache = cache
        app.state.embed_query = _cached_query_embedder(
            service, cache, _embed_cache_model_id(config)
        )
        logger.info("[embed] query cache: %s", _embed_cache_path(config))
    else:
        app.state.embed_query = lambda text: service.embed(
            text, task=simba.memory.embeddings.TaskType.QUERY
        )
    app.state._embedding_service = service
    return service


async def shutdown_embeddings(app: fastapi.FastAPI) -> None:
    """Stop the embedding service."""
    service = getattr(app.state, "_embedding_service", None)
    if service:
        await service.stop()
    cache = getattr(app.state, "embed_cache", None)
    if cache is not None:
        cache.close()


def main() -> None:
    """Start the memory daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    parser = argparse.ArgumentParser(description="Simba memory daemon")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on (default: 8741)"
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=f"Database directory (default: cwd/{_DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to GGUF model file (default: auto-download from HuggingFace)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=None,
        help="GPU layers to offload (-1=all, 0=CPU only, default: -1)",
    )
    parser.add_argument(
        "--embed-url",
        default=None,
        help="URL of an OpenAI-compatible embedding server "
        "(e.g. http://localhost:8080). When set, uses HTTP "
        "instead of loading the model in-process.",
    )
    parser.add_argument(
        "--diagnostics-after",
        type=int,
        default=None,
        help="Print diagnostics summary every N requests (0=disabled, default: 50)",
    )
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=None,
        help="Seconds between sync cycles (0=disabled, default: 0)",
    )
    args = parser.parse_args()

    config = simba.memory.config.load_config(
        port=args.port,
        db_path=args.db_path,
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        embed_url=args.embed_url,
        diagnostics_after=args.diagnostics_after,
        sync_interval=args.sync_interval,
    )
    app = create_app(config, use_lifespan=True)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config.port,
        timeout_graceful_shutdown=config.shutdown_timeout,
    )


if __name__ == "__main__":
    main()

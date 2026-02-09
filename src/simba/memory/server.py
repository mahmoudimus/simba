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
import simba.memory.embeddings
import simba.memory.routes

_DEFAULT_DB_DIR = ".simba/memory"


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
    """Manage startup/shutdown for the memory daemon."""
    await init_database(app)
    await init_embeddings(app)
    sync_task = await _start_sync_scheduler(app)
    yield
    # Shutdown: stop scheduler, wait briefly, then force-cancel.
    config: simba.memory.config.MemoryConfig = app.state.config
    sync_timeout = max(1, config.shutdown_timeout // 2)
    if sync_task is not None:
        scheduler = getattr(app.state, "sync_scheduler", None)
        if scheduler is not None:
            scheduler.stop()
        # Give the running cycle a moment to finish, then force-cancel.
        try:
            await asyncio.wait_for(
                asyncio.shield(sync_task), timeout=sync_timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sync_task
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
    app.state.table = None
    app.state.db_path = None
    app.state.embed = None
    app.state.embed_query = None
    app.state.diagnostics = simba.memory.diagnostics.DiagnosticsTracker(
        report_interval=config.diagnostics_after,
    )

    return app


def _resolve_db_path(config: simba.memory.config.MemoryConfig) -> pathlib.Path:
    """Resolve the database directory from config or default."""
    if config.db_path:
        return pathlib.Path(config.db_path)
    return pathlib.Path.cwd() / _DEFAULT_DB_DIR


async def init_database(
    app: fastapi.FastAPI, data_dir: pathlib.Path | None = None
) -> None:
    """Initialize LanceDB and attach to app state."""
    import lancedb

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

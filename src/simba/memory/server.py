"""FastAPI memory daemon server.

Ported from claude-memory/server.js.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import contextlib
import pathlib

import fastapi
import httpx
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
# The daemon only ever binds loopback; shared by the bind probe and the real
# uvicorn.run() call below so the two can never drift apart.
_HOST = "127.0.0.1"
# How long the bind probe waits for GET /health before treating a port
# occupant as a non-answering squatter rather than a healthy daemon.
_PROBE_HEALTH_TIMEOUT = 2.0


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
    rss_watchdog_task = await _start_rss_watchdog(app)
    yield
    await _shutdown_daemon(
        app,
        sync_task=sync_task,
        maintenance_task=maintenance_task,
        rss_watchdog_task=rss_watchdog_task,
    )


async def _shutdown_daemon(
    app: fastapi.FastAPI,
    *,
    sync_task: asyncio.Task | None,  # type: ignore[type-arg]
    maintenance_task: asyncio.Task | None,  # type: ignore[type-arg]
    rss_watchdog_task: asyncio.Task | None = None,  # type: ignore[type-arg]
) -> None:
    """Stop schedulers, drain tracked background tasks, then force-cancel.

    Handoff item 10 (2026-07-08 SIGTERM breach: uvicorn's graceful window
    exceeded --- "Cancel 54 running task(s)"). Root causes and how this
    function addresses them, in order:

    (b) Marks the process-level shutdown flag FIRST, before anything else,
        so a self-HTTP helper already mid-flight in a background pass
        (maintenance/hygiene fetches --- see background.py's module
        docstring) bails out on its very next check instead of hanging on a
        loopback request that can never complete once uvicorn stops serving.
    (existing) Stops the sync/maintenance schedulers and gives each a brief
        window to finish before force-cancelling --- unchanged from before
        this refactor (moved out of ``lifespan`` so it's testable without
        a real DB/embedding-model startup).
    (a) Drains simba.memory.background's registry --- the fire-and-forget
        routes.py tasks (usage bumps, demand logging, HyDE/rerank cache
        warms, ...) that were previously never tracked or awaited at
        shutdown at all.

    ``config.shutdown_timeout`` is the one knob governing both uvicorn's own
    ``timeout_graceful_shutdown`` (see ``main()``) and this drain budget.
    """
    import simba.memory.background

    simba.memory.background.mark_shutting_down()
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
    if rss_watchdog_task is not None:
        watchdog = getattr(app.state, "rss_watchdog", None)
        if watchdog is not None:
            watchdog.stop()
        try:
            await asyncio.wait_for(
                asyncio.shield(rss_watchdog_task), timeout=sync_timeout
            )
        except (TimeoutError, asyncio.CancelledError):
            rss_watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await rss_watchdog_task
    await simba.memory.background.drain(config.shutdown_timeout)
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


async def _start_rss_watchdog(
    app: fastapi.FastAPI,
) -> asyncio.Task | None:  # type: ignore[type-arg]
    """Start the RSS watchdog if either limit is configured (default-off).

    macOS has no enforceable RSS rlimit (RLIMIT_RSS is a no-op; RLIMIT_AS
    would abort LanceDB's mmap-based table reads), so this is a
    self-watchdog: poll RSS on an interval, relieve allocator pressure past
    the soft limit, self-restart --- reusing the existing PR #89/#91
    os.execv exec seam (``routes._run_restart_sequence``) --- past the hard
    limit once the process is old enough to rule out a startup transient.
    See ``rss_watchdog.py``. Held OUTSIDE ``simba.memory.background.TASKS``
    (mirrors sync_task/maintenance_task above): it must survive
    ``background.drain()``, and is stopped explicitly in
    ``_shutdown_daemon`` instead.
    """
    config: simba.memory.config.MemoryConfig = app.state.config
    soft = float(getattr(config, "rss_soft_limit_mb", 0) or 0)
    hard = float(getattr(config, "rss_hard_limit_mb", 0) or 0)
    if soft <= 0 and hard <= 0:
        app.state.rss_watchdog = None
        return None

    from simba.memory.rss_watchdog import RssWatchdog

    async def _restart(argv: list[str]) -> None:
        await simba.memory.routes._run_restart_sequence(argv, app)

    watchdog = RssWatchdog(
        soft_limit_mb=soft,
        hard_limit_mb=hard,
        interval_seconds=float(getattr(config, "rss_check_interval_seconds", 30.0)),
        min_uptime_seconds=float(
            getattr(config, "rss_restart_min_uptime_seconds", 300.0)
        ),
        start_time=float(getattr(app.state, "start_time", time.time())),
        boot_argv=getattr(app.state, "boot_argv", None),
        restart=_restart,
    )
    app.state.rss_watchdog = watchdog
    task = asyncio.create_task(watchdog.run_forever())
    logging.getLogger("simba.memory").info(
        "[rss-watchdog] started (soft=%s hard=%s interval=%.0fs)",
        soft or "off",
        hard or "off",
        watchdog.interval_seconds,
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
    # Faithful re-exec argv for POST /restart, captured by main() at boot
    # (module invocation `python -m simba.memory.server ...` is the canonical
    # launch --- see main() below). None here means "no boot argv on
    # record", which is exactly true for every app built via create_app()
    # outside that __main__ path (every test in this file included); the
    # route returns 503 rather than ever guessing a launch command.
    app.state.boot_argv = None
    # Set by routes.py's `_run_restart_sequence` on failure, surfaced by
    # GET /health as `lastRestartError`; null until (unless) a restart
    # attempt fails.
    app.state.last_restart_error = None
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
    # Recall admission control: bounds concurrent /recall handlers in flight
    # (None when disabled via max_concurrent_recalls=0 --- unlimited, the
    # default, byte-identical to pre-admission-control behavior). See
    # routes.py's recall_memories for why this is needed even though the
    # LLAMA lock already serializes native embed/rerank compute.
    app.state.recall_semaphore = (
        asyncio.Semaphore(config.max_concurrent_recalls)
        if config.max_concurrent_recalls > 0
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


# --- bind-first probe (portless-zombie defense) -----------------------------
#
# Live 2026-07-10: multiple daemon processes racing to bind :8741 (session
# auto-start on every health-check failure, plus the user's own manual
# starts) were each paying the FULL ~2.5GB model-load cost (bge-large embed +
# llama-cpp, loaded from `lifespan()` above) BEFORE uvicorn ever attempted
# the socket bind --- confirmed by reading uvicorn.server.Server.startup():
# ``await self.lifespan.startup()`` is its first line; ``loop.create_server``
# (the actual bind) runs only after that returns. Every loser of the race
# therefore finished loading, THEN discovered the port was taken, and then
# --- per the module comment on ``_run_server`` below --- frequently never
# exited at all: a portless zombie burning CPU forever.
#
# This probe moves the port check to the earliest possible point in main(),
# before create_app() or any model/DB work, so a loser bails in milliseconds.


def _probe_health(host: str, port: int, *, timeout: float) -> bool:
    """Best-effort GET /health within ``timeout`` seconds.

    True on ANY HTTP response (status code is irrelevant --- something is
    alive enough to answer HTTP at all, which is what distinguishes a benign
    startup race from a stuck squatter). False on connection refusal, reset,
    or timeout --- the live incident's signature: a process that LISTENS
    (so bind() fails) but never SERVES.
    """
    try:
        httpx.get(f"http://{host}:{port}/health", timeout=timeout)
    except httpx.HTTPError:
        return False
    return True


def _bind_probe_or_exit(host: str, port: int) -> None:
    """Cheap exclusive bind probe. Exits the process if the port is taken.

    A plain ``bind()`` --- deliberately WITHOUT ``SO_REUSEADDR`` --- is the
    cheapest possible occupancy check: it raises ``OSError`` (EADDRINUSE)
    if, and only if, something else is already bound to ``(host, port)``,
    regardless of what socket options THAT occupant set. (``SO_REUSEADDR``
    lets a new bind reclaim an address stuck in TIME_WAIT; it does not let
    two sockets both actively occupy the same port.) The probe socket is
    closed immediately either way, so uvicorn's own bind moments later (deep
    inside ``_run_server``) remains the sole real, authoritative attempt.

    That leaves a small TOCTOU window between this probe's close and
    uvicorn's bind --- accepted: this probe's job is only to eliminate the
    "load 2.5GB, then discover the port is taken" class; a genuine dead-heat
    in that narrow window still ends in a bind failure inside uvicorn, which
    ``_run_server``'s hard-exit guarantee turns into a clean process exit
    rather than a zombie.

    On EADDRINUSE, a quick GET /health tells a benign race (a healthy daemon
    already owns the port --- exit 0, quietly) apart from a squatter (bound
    but not serving --- exit 1, loudly). Both exits here are plain
    ``sys.exit()``, not the ``os._exit`` guarantee below: at this point in
    main() no model has loaded and no thread has started, so there is
    nothing that could keep the interpreter alive after a normal SystemExit.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
    except OSError:
        probe.close()
    else:
        probe.close()
        return

    if _probe_health(host, port, timeout=_PROBE_HEALTH_TIMEOUT):
        print(
            f"simba-memory: port {port} already serving (healthy daemon "
            "detected) -- exiting",
            file=sys.stderr,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(0)

    print(
        f"simba-memory: port {port} is held by a process that did not "
        f"answer /health within {_PROBE_HEALTH_TIMEOUT:.0f}s (stuck/zombie "
        "daemon?) -- exiting",
        file=sys.stderr,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)


# --- hard-exit guarantee (portless-zombie defense, part 2) ------------------
#
# Live 2026-07-10: every observed zombie ignored SIGTERM (all four required
# `kill -9`). uvicorn's ``capture_signals()`` installs a SIGTERM/SIGINT
# handler for the ENTIRE ``Server.serve()`` call, but that handler only ever
# flips a flag (``should_exit``/``force_exit``) that ``main_loop``'s tick
# polls once per iteration --- and that tick never runs while startup is
# stuck loading models (before a bind is even attempted) or while our own
# ``lifespan`` shutdown is awaiting something unbounded (e.g. cancelling a
# task parked in ``asyncio.to_thread`` on a live native llama.cpp call ---
# cancelling the asyncio side does not stop the underlying OS thread). The
# handler is "installed but its loop is dead": a signal arrives, gets
# recorded, and nothing ever acts on it.
#
# Even when uvicorn DOES get as far as raising (its ``Server.startup``
# catches a real bind ``OSError`` and calls ``sys.exit(1)``), a plain
# ``SystemExit`` only *starts* interpreter shutdown --- CPython then waits
# for every non-daemon thread it knows about to finish
# (``threading._shutdown()``) before the process can actually end. The
# default executor behind every ``asyncio.to_thread`` call (the embedding
# queue, self-HTTP background passes, ...) uses non-daemon worker threads
# by design, specifically so ``atexit`` can join them --- so one stuck
# mid-call blocks that join forever, and the "exited" process never
# actually goes away.
#
# ``os._exit()`` is the only way out: an immediate, unconditional process
# exit at the OS level that skips atexit handlers and every non-daemon
# thread join. A thin indirection (mirrors ``background.py``'s
# ``_execv``/``reexec`` seam) so tests can assert the exit code without
# ending the test process.
_os_exit = os._exit


def _run_server(app: fastapi.FastAPI, config: simba.memory.config.MemoryConfig) -> None:
    """Run uvicorn and GUARANTEE the process ends when it stops.

    Wraps the one call that can otherwise leave a portless zombie behind:
    every way ``uvicorn.run`` can finish --- an ordinary clean return, its
    own ``SystemExit`` (a true-tie bind failure racing past
    ``_bind_probe_or_exit``), or any other exception --- funnels into
    ``_os_exit``. See the module comment above for why a plain return/raise
    is not a strong enough guarantee on its own. Normal cleanup (uvicorn's
    graceful shutdown window, our ``lifespan`` shutdown handler) still runs
    first; this is the guarantee of last resort once no cleanup will ever
    finish, or once cleanup already has.
    """
    exit_code = 0
    try:
        uvicorn.run(
            app,
            host=_HOST,
            port=config.port,
            timeout_graceful_shutdown=config.shutdown_timeout,
        )
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            exit_code = code
        elif code is not None:
            exit_code = 1
    except BaseException as exc:  # must exit regardless of cause (see docstring)
        print(f"simba-memory: server exited abnormally: {exc}", file=sys.stderr)
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    _os_exit(exit_code)


def main() -> None:
    """Start the memory daemon."""
    # Captured before argparse touches anything (argparse never mutates
    # sys.argv, but the boot argv must reflect exactly what this process was
    # invoked with, in case that ever changes). `python -m
    # simba.memory.server ...` is the canonical launch (the `if __name__ ==
    # "__main__"` block below); `simba server ...` rewrites sys.argv to this
    # same `["simba server", *args]` shape before calling this function (see
    # `_cmd_server` in __main__.py), so re-invoking via `-m` reproduces
    # identical argparse behavior regardless of which entrypoint actually
    # started this process --- only sys.argv[1:] (the flags) ever matter to
    # argparse, never argv[0]. POST /restart execs this back later.
    boot_argv = [sys.executable, "-m", "simba.memory.server", *sys.argv[1:]]
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

    # Bind-first: before ANY model/DB work, refuse to load 2.5GB just to
    # discover the port is taken (see _bind_probe_or_exit's docstring).
    _bind_probe_or_exit(_HOST, config.port)

    app = create_app(config, use_lifespan=True)
    app.state.boot_argv = boot_argv
    _run_server(app, config)


if __name__ == "__main__":
    main()

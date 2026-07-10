"""Tests for POST /restart --- os.execv self-restart.

The daemon runs foreground in the user's terminal (``.venv/bin/python -m
simba.memory.server --port 8741 ...``); POST /restart must reply 202
immediately with the pre-restart pid, then --- strictly AFTER that response
has been transmitted --- drain tracked background work, stop the maintenance
scheduler if one is running, flush stdio, and replace the process image via
``os.execv`` with the exact argv ``main()`` captured at boot.

Live 2026-07-10: the OLD implementation scheduled that sequence via
Starlette ``BackgroundTasks``, tied to the request/response ASGI cycle. With
``BaseHTTPMiddleware`` in the app's middleware stack (routes.py's
``DiagnosticsMiddleware``), the downstream response --- background tasks
included --- runs as a child of the SAME anyio task group scoping the
request; a client disconnect right after the 202 cancelled that whole group,
killing the sequence before ``os.execv`` ever ran. The 202 kept coming back;
uptime kept climbing. Our ASGI-transport tests never caught it because that
transport never disconnects.

The fix schedules the sequence as a DETACHED ``asyncio.create_task``
(``simba.memory.background.schedule_restart`` / ``restart_task``), held in a
dedicated slot outside the drained ``background.TASKS`` registry --- nothing
tied to the request/response lifecycle can reach it. Because the sequence no
longer runs synchronously inside the ASGI call, every test that depends on
its side effects (drain completing, the scheduler stopping, ``os.execv``
firing) must explicitly ``await background.restart_task()`` before asserting.

No test here ever execs the real test process: the seam
(``simba.memory.background._execv``) is monkeypatched everywhere via the
autouse ``_patch_execv`` fixture below.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

import fastapi
import httpx
import pytest
import pytest_asyncio

import simba.memory.background as background
import simba.memory.config
import simba.memory.server as server

_ARGV = [sys.executable, "-m", "simba.memory.server", "--port", "18741"]


def _make_app(**config_kwargs: object) -> fastapi.FastAPI:
    return server.create_app(simba.memory.config.MemoryConfig(**config_kwargs))


class _FakeMaintenanceScheduler:
    def __init__(self) -> None:
        self.stop_called = False

    def stop(self) -> None:
        self.stop_called = True


async def _post_restart(app: fastapi.FastAPI) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/restart")


@pytest.fixture(autouse=True)
def _patch_execv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Never exec the real test process --- record calls instead."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        background, "_execv", lambda path, argv: calls.append(list(argv))
    )
    return calls


@pytest_asyncio.fixture(autouse=True)
async def _drain_restart_task():
    """Hygiene: a detached restart task left pending when a test ends would
    otherwise leak into the next event loop (pytest-asyncio tears the loop
    down per test). Runs AFTER each test, in the same async context."""
    yield
    task = background.restart_task()
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


@pytest.fixture
def restart_app() -> fastapi.FastAPI:
    return _make_app(port=18741, shutdown_timeout=1.0)


# ---------------------------------------------------------------------------
# create_app() / main() wiring --- capturing the boot argv
# ---------------------------------------------------------------------------


def test_create_app_defaults_boot_argv_to_none() -> None:
    """An app built outside the __main__ path (every other test in this
    file, and every existing route test) has no faithful launch argv to
    re-exec --- the route must see this as unavailable, never guess one."""
    app = _make_app(port=18742)
    assert app.state.boot_argv is None


def test_create_app_defaults_last_restart_error_to_none() -> None:
    """GET /health's `lastRestartError` must start null on every fresh app."""
    app = _make_app(port=18743)
    assert app.state.last_restart_error is None


def test_main_captures_boot_argv_under_module_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() must reconstruct argv as `[sys.executable, -m,
    simba.memory.server, *sys.argv[1:]]` --- module invocation is the
    canonical launch, and `simba server ...` (_cmd_server in __main__.py)
    rewrites sys.argv to this same shape before calling main(), so this
    reconstruction is faithful regardless of which entrypoint started the
    process."""
    import uvicorn

    captured: dict[str, fastapi.FastAPI] = {}

    def _fake_run(app: fastapi.FastAPI, **kwargs: object) -> None:
        captured["app"] = app

    fixed_config = simba.memory.config.MemoryConfig(port=18741)
    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(
        simba.memory.config, "load_config", lambda **overrides: fixed_config
    )
    monkeypatch.setattr(
        sys, "argv", ["simba-memory-daemon", "--port", "18741", "--db-path", "/tmp/x"]
    )
    # main() now funnels a clean uvicorn.run() return through the hard-exit
    # guarantee (server._run_server); stub the os._exit seam so this test
    # can make its assertions instead of ending the test process.
    monkeypatch.setattr(server, "_os_exit", lambda code: None)

    server.main()

    assert captured["app"].state.boot_argv == [
        sys.executable,
        "-m",
        "simba.memory.server",
        "--port",
        "18741",
        "--db-path",
        "/tmp/x",
    ]


# ---------------------------------------------------------------------------
# Route behavior via a real ASGI transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_returns_202_with_pid(restart_app: fastapi.FastAPI) -> None:
    restart_app.state.boot_argv = list(_ARGV)
    resp = await _post_restart(restart_app)
    assert resp.status_code == 202
    assert resp.json() == {"restarting": True, "pid": os.getpid()}


@pytest.mark.asyncio
async def test_restart_execs_with_exact_captured_argv(
    restart_app: fastapi.FastAPI, _patch_execv: list[list[str]]
) -> None:
    restart_app.state.boot_argv = list(_ARGV)
    await _post_restart(restart_app)
    task = background.restart_task()
    assert task is not None
    await task
    assert _patch_execv == [_ARGV]


@pytest.mark.asyncio
async def test_restart_503_when_no_boot_argv(
    restart_app: fastapi.FastAPI, _patch_execv: list[list[str]]
) -> None:
    assert restart_app.state.boot_argv is None
    resp = await _post_restart(restart_app)
    assert resp.status_code == 503
    assert resp.json() == {"error": "restart unavailable: no boot argv"}
    assert _patch_execv == []
    assert background.restart_task() is None  # never scheduled


@pytest.mark.asyncio
async def test_restart_501_on_non_posix(
    monkeypatch: pytest.MonkeyPatch,
    restart_app: fastapi.FastAPI,
    _patch_execv: list[list[str]],
) -> None:
    restart_app.state.boot_argv = list(_ARGV)
    monkeypatch.setattr(os, "name", "nt")
    resp = await _post_restart(restart_app)
    assert resp.status_code == 501
    assert "error" in resp.json()
    assert _patch_execv == []
    assert background.restart_task() is None  # never scheduled


@pytest.mark.asyncio
async def test_restart_stops_maintenance_scheduler_when_present(
    restart_app: fastapi.FastAPI,
) -> None:
    restart_app.state.boot_argv = list(_ARGV)
    fake_scheduler = _FakeMaintenanceScheduler()
    restart_app.state.maintenance_scheduler = fake_scheduler
    await _post_restart(restart_app)
    task = background.restart_task()
    assert task is not None
    await task
    assert fake_scheduler.stop_called is True


@pytest.mark.asyncio
async def test_restart_missing_maintenance_scheduler_is_harmless(
    restart_app: fastapi.FastAPI,
) -> None:
    """create_app() (no lifespan) never sets maintenance_scheduler at all ---
    the route must tolerate its absence, not just a None value."""
    restart_app.state.boot_argv = list(_ARGV)
    assert not hasattr(restart_app.state, "maintenance_scheduler")
    resp = await _post_restart(restart_app)
    assert resp.status_code == 202
    task = background.restart_task()
    assert task is not None
    await task  # must not raise (AttributeError) on the missing attribute


@pytest.mark.asyncio
async def test_restart_drains_background_tasks_before_exec(
    restart_app: fastapi.FastAPI,
) -> None:
    restart_app.state.boot_argv = list(_ARGV)
    completed = []

    async def _quick() -> None:
        await asyncio.sleep(0.01)
        completed.append(True)

    background.spawn(_quick())
    await _post_restart(restart_app)

    task = background.restart_task()
    assert task is not None
    await task

    assert completed == [True]
    assert len(background.TASKS) == 0


@pytest.mark.asyncio
async def test_restart_sequence_ordering(
    monkeypatch: pytest.MonkeyPatch, restart_app: fastapi.FastAPI
) -> None:
    """flag set -> drain called -> flush -> execv, in that exact order."""
    restart_app.state.boot_argv = list(_ARGV)
    order: list[str] = []

    async def _fake_drain(
        grace_seconds: float, *, tail_seconds: float | None = None
    ) -> None:
        order.append(f"drain(flag={background.is_shutting_down()})")

    def _fake_execv(path: str, argv: list[str]) -> None:
        order.append("execv")

    real_stdout_flush = sys.stdout.flush
    real_stderr_flush = sys.stderr.flush

    def _fake_stdout_flush() -> None:
        order.append("flush_stdout")
        real_stdout_flush()

    def _fake_stderr_flush() -> None:
        order.append("flush_stderr")
        real_stderr_flush()

    monkeypatch.setattr(background, "drain", _fake_drain)
    monkeypatch.setattr(background, "_execv", _fake_execv)
    monkeypatch.setattr(sys.stdout, "flush", _fake_stdout_flush)
    monkeypatch.setattr(sys.stderr, "flush", _fake_stderr_flush)

    resp = await _post_restart(restart_app)
    task = background.restart_task()
    assert task is not None
    await task

    assert resp.status_code == 202
    assert order == ["drain(flag=True)", "flush_stdout", "flush_stderr", "execv"]
    assert restart_app.state.last_restart_error is None


@pytest.mark.asyncio
async def test_restart_response_ready_before_background_sequence_runs(
    restart_app: fastapi.FastAPI, _patch_execv: list[list[str]]
) -> None:
    """The 202 response is handed back WITHOUT waiting for the drain/flush/
    execv sequence: that sequence is a DETACHED task (see
    ``background.schedule_restart``), not Starlette BackgroundTasks tied to
    the response --- so a client disconnect (or anything else scoped to the
    request) can never reach it (the live 2026-07-10 bug this replaces)."""
    restart_app.state.boot_argv = list(_ARGV)

    resp = await _post_restart(restart_app)

    assert resp.status_code == 202
    assert resp.json() == {"restarting": True, "pid": os.getpid()}
    assert _patch_execv == []  # not yet --- the detached task is still sleeping

    task = background.restart_task()
    assert task is not None
    assert not task.done()

    await task  # let it run to completion

    assert _patch_execv == [list(restart_app.state.boot_argv)]


@pytest.mark.asyncio
async def test_restart_task_not_registered_in_background_tasks(
    restart_app: fastapi.FastAPI,
) -> None:
    """The restart task must NOT ride ``background.TASKS`` --- ``drain()``
    (called from inside the sequence itself) awaits every entry there with a
    bounded grace/tail; registering the restart task would make drain()
    wait on its own caller and deadlock every restart."""
    restart_app.state.boot_argv = list(_ARGV)
    resp = await _post_restart(restart_app)
    assert resp.status_code == 202

    task = background.restart_task()
    assert task is not None
    assert task not in background.TASKS

    await task
    assert task not in background.TASKS


@pytest.mark.asyncio
async def test_restart_task_survives_cancellation_of_other_tasks(
    restart_app: fastapi.FastAPI, _patch_execv: list[list[str]]
) -> None:
    """Models the live incident's disconnect: the restart task is not a
    child of the request's task group (or anything else), so even
    cancelling every OTHER task in the process --- the blast radius a real
    client disconnect could plausibly cause --- must not stop it from
    completing and firing the (monkeypatched) exec seam."""
    restart_app.state.boot_argv = list(_ARGV)

    resp = await _post_restart(restart_app)
    assert resp.status_code == 202

    task = background.restart_task()
    assert task is not None
    assert not task.done()

    others = [t for t in asyncio.all_tasks() if t is not task and not t.done()]
    for other in others:
        other.cancel()
    for other in others:
        with contextlib.suppress(BaseException):
            await other

    await task  # unaffected by the cancellations above

    assert _patch_execv == [list(restart_app.state.boot_argv)]
    assert restart_app.state.last_restart_error is None


@pytest.mark.asyncio
async def test_restart_failure_surfaces_on_health(
    monkeypatch: pytest.MonkeyPatch, restart_app: fastapi.FastAPI, caplog
) -> None:
    """An exception anywhere in the sequence (here: the exec seam itself)
    must not vanish --- it's logged at CRITICAL and surfaced by GET /health
    so an operator (or monitoring) can see a restart never actually
    happened, instead of only noticing uptime never reset."""
    restart_app.state.boot_argv = list(_ARGV)

    def _boom(path: str, argv: list[str]) -> None:
        raise RuntimeError("exec seam exploded")

    monkeypatch.setattr(background, "_execv", _boom)

    with caplog.at_level("CRITICAL", logger="simba.memory"):
        resp = await _post_restart(restart_app)
        assert resp.status_code == 202

        task = background.restart_task()
        assert task is not None
        await task  # the sequence catches the exception internally

    assert any("exec seam exploded" in r.getMessage() for r in caplog.records)

    assert restart_app.state.last_restart_error is not None
    assert "exec seam exploded" in restart_app.state.last_restart_error

    transport = httpx.ASGITransport(app=restart_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health_resp = await client.get("/health")
    body = health_resp.json()
    assert body["lastRestartError"] is not None
    assert "exec seam exploded" in body["lastRestartError"]

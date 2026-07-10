"""Tests for POST /restart --- os.execv self-restart.

The daemon runs foreground in the user's terminal (``.venv/bin/python -m
simba.memory.server --port 8741 ...``); POST /restart must reply 202
immediately with the pre-restart pid, then --- strictly AFTER that response
has been transmitted --- drain tracked background work, stop the maintenance
scheduler if one is running, flush stdio, and replace the process image via
``os.execv`` with the exact argv ``main()`` captured at boot.

No test here ever execs the real test process: the seam
(``simba.memory.background._execv``) is monkeypatched everywhere via the
autouse ``_patch_execv`` fixture below.
"""

from __future__ import annotations

import asyncio
import os
import sys

import fastapi
import httpx
import pytest

import simba.memory.background as background
import simba.memory.config
import simba.memory.routes as routes
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


@pytest.mark.asyncio
async def test_restart_stops_maintenance_scheduler_when_present(
    restart_app: fastapi.FastAPI,
) -> None:
    restart_app.state.boot_argv = list(_ARGV)
    fake_scheduler = _FakeMaintenanceScheduler()
    restart_app.state.maintenance_scheduler = fake_scheduler
    await _post_restart(restart_app)
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

    assert resp.status_code == 202
    assert order == ["drain(flag=True)", "flush_stdout", "flush_stderr", "execv"]


@pytest.mark.asyncio
async def test_restart_response_ready_before_background_sequence_runs(
    restart_app: fastapi.FastAPI, _patch_execv: list[list[str]]
) -> None:
    """The 202 response is fully formed and handed back WITHOUT running the
    drain/flush/execv sequence --- that sequence is merely SCHEDULED via
    Starlette's BackgroundTasks, which runs registered tasks only after the
    response has been transmitted. Proven directly here (no ASGI transport
    timing ambiguity, since httpx's in-process ASGITransport happens to await
    the whole background sequence before ``client.post()`` returns, which
    would otherwise make "before" unobservable from the outside): call the
    route function with a bare BackgroundTasks instance and assert the
    sequence has NOT run by the time the response dict comes back, then run
    the scheduled task by hand and assert it now has."""
    restart_app.state.boot_argv = list(_ARGV)
    request = fastapi.Request(scope={"type": "http", "app": restart_app})
    background_tasks = fastapi.BackgroundTasks()

    result = await routes.restart(request, background_tasks)

    assert result == {"restarting": True, "pid": os.getpid()}
    assert _patch_execv == []  # not yet --- only scheduled
    assert len(background_tasks.tasks) == 1
    assert background_tasks.tasks[0].func is routes._run_restart_sequence

    await background_tasks()  # simulate Starlette running it after the response

    assert _patch_execv == [list(restart_app.state.boot_argv)]

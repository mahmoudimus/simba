"""Tests for the bind-first probe + hard-exit guarantee (portless zombies).

Live 2026-07-10: multiple daemon processes raced to bind :8741 (session
auto-start firing on every health-check failure, stacked with the user's own
manual starts). Reading uvicorn.server.Server.startup() confirmed the root
cause: ``await self.lifespan.startup()`` --- where our own ``lifespan()``
loads the ~2.5GB embed/rerank models and opens the DB --- is its FIRST line;
the actual socket bind (``loop.create_server(...)``) only happens after that
returns. So every loser of the race paid the full model-load cost BEFORE
ever discovering the port was taken, and then --- because uvicorn's SIGTERM
handler only ever flips a flag that its own main-loop tick polls, and that
tick never runs during startup or during an unbounded shutdown await ---
never exited at all: a portless zombie burning CPU forever (all four
observed processes needed `kill -9`).

Two independent defenses, tested here:
  1. ``_bind_probe_or_exit`` --- a cheap bind() probe at the very top of
     main(), before create_app() or any model/DB work, so a loser bails in
     milliseconds instead of after a full model load.
  2. ``_run_server`` --- wraps uvicorn.run() so EVERY way it can finish
     (clean return, its own SystemExit on a true-tie bind failure, or any
     other exception) funnels into ``os._exit`` --- the only way to
     guarantee the process ends regardless of lingering non-daemon threads.

Never binds the real 8741: every test here uses an OS-assigned ephemeral
port on 127.0.0.1.
"""

from __future__ import annotations

import contextlib
import socket
import sys
import threading

import pytest
import uvicorn

import simba.memory.server as server


class _StubListener:
    """A background TCP listener occupying an ephemeral port.

    ``respond="health_ok"`` answers every connection with a minimal valid
    HTTP 200 (the "healthy daemon" case --- a benign bind race).
    ``respond="silent"`` accepts the connection and then never writes
    anything back (the "squatter" case --- listens but never serves, the
    live incident's exact signature).
    """

    def __init__(self, respond: str) -> None:
        self._respond = respond
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port: int = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._held: list[socket.socket] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if self._respond == "health_ok":
                with contextlib.suppress(OSError):
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: 2\r\n\r\n{}"
                    )
                with contextlib.suppress(OSError):
                    conn.close()
            else:  # "silent" --- accept and hold, never respond
                self._held.append(conn)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        for conn in self._held:
            with contextlib.suppress(OSError):
                conn.close()
        with contextlib.suppress(OSError):
            self._sock.close()


def _free_port() -> int:
    """Ask the OS for an unused ephemeral port, then release it.

    Reserve-then-release: the window before the caller rebinds it is
    microseconds, not a realistic collision risk within a single test
    process.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _patch_bind_probe_grace(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """``main()`` derives its grace window from
    ``config.bind_probe_grace_seconds`` (no CLI flag for it) --- pin it here
    for tests that go through ``main()`` but aren't exercising the grace
    window itself (they test the single-check/zombie semantics), so they
    stay fast regardless of the real default (45s)."""
    import dataclasses

    import simba.memory.config as memory_config

    real_load_config = memory_config.load_config

    def _load_config(**kwargs: object) -> memory_config.MemoryConfig:
        cfg = real_load_config(**kwargs)
        return dataclasses.replace(cfg, bind_probe_grace_seconds=seconds)

    monkeypatch.setattr(memory_config, "load_config", _load_config)


class _FakeClock:
    """A ``clock``/``sleep`` pair that models time advancing only through
    ``sleep()`` calls --- no real waiting, fully deterministic."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start
        self.sleep_calls: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# _bind_probe_or_exit
# ---------------------------------------------------------------------------


def test_probe_exits_zero_on_healthy_daemon_before_any_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _StubListener(respond="health_ok")
    try:
        create_app_calls: list[object] = []

        def _forbidden_create_app(*args: object, **kwargs: object) -> None:
            create_app_calls.append((args, kwargs))
            raise AssertionError("create_app must not run when the probe exits early")

        monkeypatch.setattr(server, "create_app", _forbidden_create_app)
        monkeypatch.setattr(
            sys, "argv", ["simba-memory-daemon", "--port", str(listener.port)]
        )

        with pytest.raises(SystemExit) as exc_info:
            server.main()

        assert exc_info.value.code == 0
        assert create_app_calls == []
    finally:
        listener.close()


def test_probe_exits_nonzero_on_squatter_before_any_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _StubListener(respond="silent")
    try:
        # Keep the test fast: the real default (2s) is a UX/production
        # concern, not something this test needs to wait out.
        monkeypatch.setattr(server, "_PROBE_HEALTH_TIMEOUT", 0.2)
        # This test is about the OLD single-check zombie semantics, not the
        # grace-window polling below -- pin grace to 0 so it doesn't wait out
        # the real 45s default (see TestBindProbeGraceWindow for that).
        _patch_bind_probe_grace(monkeypatch, 0.0)

        create_app_calls: list[object] = []

        def _forbidden_create_app(*args: object, **kwargs: object) -> None:
            create_app_calls.append((args, kwargs))
            raise AssertionError("create_app must not run when the probe exits early")

        monkeypatch.setattr(server, "create_app", _forbidden_create_app)
        monkeypatch.setattr(
            sys, "argv", ["simba-memory-daemon", "--port", str(listener.port)]
        )

        with pytest.raises(SystemExit) as exc_info:
            server.main()

        assert exc_info.value.code != 0
        assert create_app_calls == []
    finally:
        listener.close()


def test_probe_passes_through_on_free_port_and_startup_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No occupant on the port: the probe must be a no-op, letting main()
    reach create_app()/uvicorn.run() exactly as before this change. The
    heavy parts (uvicorn.run, the hard-exit seam) are stubbed --- this test
    is about the probe getting out of the way, not about a real server
    booting."""
    port = _free_port()

    run_calls: list[dict[str, object]] = []

    def _fake_run(app: object, **kwargs: object) -> None:
        run_calls.append(kwargs)

    exit_calls: list[int] = []

    def _fake_os_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(server, "_os_exit", _fake_os_exit)
    monkeypatch.setattr(sys, "argv", ["simba-memory-daemon", "--port", str(port)])

    server.main()

    assert len(run_calls) == 1
    assert run_calls[0]["port"] == port
    # A clean uvicorn.run() return still funnels through the hard-exit
    # guarantee (see test_hard_exit_seam_invoked_* below for the raise path).
    assert exit_calls == [0]


# ---------------------------------------------------------------------------
# _bind_probe_or_exit --- booting-vs-zombie grace window (2026-07-17)
# ---------------------------------------------------------------------------
#
# A BOOTING daemon (model load takes 10-60s) used to be indistinguishable
# from a genuine zombie: ONE /health attempt, no answer, exit non-zero ---
# every bind-race loser died instantly. These tests drive the real
# ``_bind_probe_or_exit`` directly (not through ``main()``) with the
# injected ``probe_health``/``sleep``/``clock`` seams, so "waiting up to
# 45s" never involves a real wait.


class TestBindProbeGraceWindow:
    def test_grace_zero_healthy_matches_old_single_check(self) -> None:
        """grace=0 must reproduce the OLD behavior exactly: one probe_health
        call, immediate exit 0, no sleep."""
        listener = _StubListener(respond="silent")
        try:
            calls: list[int] = []

            def _health(host: str, port: int, *, timeout: float) -> bool:
                calls.append(1)
                return True

            sleep_calls: list[float] = []
            with pytest.raises(SystemExit) as exc_info:
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=0.0,
                    probe_health=_health,
                    sleep=lambda s: sleep_calls.append(s),
                    clock=lambda: 0.0,
                )
            assert exc_info.value.code == 0
            assert len(calls) == 1
            assert sleep_calls == []
        finally:
            listener.close()

    def test_grace_zero_zombie_matches_old_single_check(self) -> None:
        listener = _StubListener(respond="silent")
        try:
            calls: list[int] = []

            def _health(host: str, port: int, *, timeout: float) -> bool:
                calls.append(1)
                return False

            sleep_calls: list[float] = []
            with pytest.raises(SystemExit) as exc_info:
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=0.0,
                    probe_health=_health,
                    sleep=lambda s: sleep_calls.append(s),
                    clock=lambda: 0.0,
                )
            assert exc_info.value.code != 0
            assert len(calls) == 1
            assert sleep_calls == []
        finally:
            listener.close()

    def test_healthy_duplicate_after_polling_third_time(self) -> None:
        """A holder that answers /health on the THIRD poll (simulating a
        daemon that finishes booting mid-grace-window) exits 0 -- a healthy
        duplicate is not an error."""
        listener = _StubListener(respond="silent")
        try:
            results = iter([False, False, True])

            def _health(host: str, port: int, *, timeout: float) -> bool:
                return next(results)

            fake = _FakeClock()
            with pytest.raises(SystemExit) as exc_info:
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=10.0,
                    probe_health=_health,
                    sleep=fake.sleep,
                    clock=fake.clock,
                )
            assert exc_info.value.code == 0
            # Slept between polls 1->2 and 2->3, never after the successful
            # 3rd poll.
            assert fake.sleep_calls == [2.0, 2.0]
        finally:
            listener.close()

    def test_zombie_after_grace_window_expires(self) -> None:
        listener = _StubListener(respond="silent")
        try:
            fake = _FakeClock()

            def _health(host: str, port: int, *, timeout: float) -> bool:
                return False

            with pytest.raises(SystemExit) as exc_info:
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=5.0,
                    probe_health=_health,
                    sleep=fake.sleep,
                    clock=fake.clock,
                )
            assert exc_info.value.code != 0
            assert fake.now >= 5.0
        finally:
            listener.close()

    def test_waiting_log_line_printed_once_up_front(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        listener = _StubListener(respond="silent")
        try:
            with pytest.raises(SystemExit):
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=5.0,
                    probe_health=lambda host, port, *, timeout: True,
                    sleep=lambda s: None,
                    clock=lambda: 0.0,
                )
            captured = capsys.readouterr()
            assert captured.err.count("waiting up to") == 1
            assert "finish booting" in captured.err
        finally:
            listener.close()

    def test_healthy_duplicate_log_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        listener = _StubListener(respond="silent")
        try:
            with pytest.raises(SystemExit) as exc_info:
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=5.0,
                    probe_health=lambda host, port, *, timeout: True,
                    sleep=lambda s: None,
                    clock=lambda: 0.0,
                )
            assert exc_info.value.code == 0
            captured = capsys.readouterr()
            assert "healthy duplicate" in captured.err
        finally:
            listener.close()

    def test_zombie_log_line_after_grace_expires(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        listener = _StubListener(respond="silent")
        try:
            fake = _FakeClock()
            with pytest.raises(SystemExit) as exc_info:
                server._bind_probe_or_exit(
                    "127.0.0.1",
                    listener.port,
                    grace_seconds=3.0,
                    probe_health=lambda host, port, *, timeout: False,
                    sleep=fake.sleep,
                    clock=fake.clock,
                )
            assert exc_info.value.code != 0
            captured = capsys.readouterr()
            assert "stuck/zombie" in captured.err
        finally:
            listener.close()

    def test_main_wires_grace_seconds_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``main()`` must pass ``config.bind_probe_grace_seconds`` through
        to ``_bind_probe_or_exit`` -- proves the config field is actually
        wired, not just present on the dataclass. `_bind_probe_or_exit`
        itself is fully stubbed (never runs real bind/poll logic), so this
        needs no occupied port and no real waiting."""
        port = _free_port()
        _patch_bind_probe_grace(monkeypatch, 12.5)
        captured_kwargs: dict[str, object] = {}

        def _stub(host: str, port: int, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(server, "_bind_probe_or_exit", _stub)
        monkeypatch.setattr(sys, "argv", ["simba-memory-daemon", "--port", str(port)])
        with pytest.raises(SystemExit):
            server.main()
        assert captured_kwargs.get("grace_seconds") == 12.5


# ---------------------------------------------------------------------------
# _run_server hard-exit guarantee
# ---------------------------------------------------------------------------


def test_hard_exit_seam_invoked_with_nonzero_code_on_bind_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates the true-tie window the probe can't close: the port was
    free when _bind_probe_or_exit checked, but uvicorn's OWN bind fails by
    the time it actually runs. Must still guarantee process exit."""
    port = _free_port()

    def _fake_run(app: object, **kwargs: object) -> None:
        raise OSError(48, "Address already in use")

    exit_calls: list[int] = []

    def _fake_os_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(server, "_os_exit", _fake_os_exit)
    monkeypatch.setattr(sys, "argv", ["simba-memory-daemon", "--port", str(port)])

    server.main()

    assert exit_calls
    assert exit_calls[0] != 0


def test_hard_exit_seam_preserves_uvicorns_own_systemexit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """uvicorn.Server.startup() itself calls sys.exit(1) on a real bind
    OSError (see uvicorn/server.py) --- the wrapper must forward that exact
    code to the hard-exit seam rather than swallowing it as a generic 1."""
    port = _free_port()

    def _fake_run(app: object, **kwargs: object) -> None:
        sys.exit(7)

    exit_calls: list[int] = []

    def _fake_os_exit(code: int) -> None:
        exit_calls.append(code)

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    monkeypatch.setattr(server, "_os_exit", _fake_os_exit)
    monkeypatch.setattr(sys, "argv", ["simba-memory-daemon", "--port", str(port)])

    server.main()

    assert exit_calls == [7]

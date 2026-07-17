"""Tests for daemon log-line timestamps (src/simba/memory/server.py).

2026-07-17 forensics: watchdog/bind-probe/store/recall lines in daemon.log
carried no timestamp -- `logging.basicConfig`'s prior ``format="%(message)s"``
never included one, and every daemon module logs through ``logging.getLogger(
"simba.memory")``, which has no handler of its own and propagates to root.
``_configure_logging`` installs (or updates) an ISO-8601-timestamped
``Formatter`` -- these tests drive it directly against throwaway logger
names so they never touch the real root logger or interfere with pytest's
own log capturing.
"""

from __future__ import annotations

import io
import logging
import re
import socket
import sys

import pytest
import uvicorn

import simba.memory.server as server

_ISO_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _fresh_logger(name: str) -> logging.Logger:
    """A uniquely-named, handler-free, non-propagating logger for isolated
    assertions (never the real "simba.memory" logger or root)."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    return logger


class TestConfigureLogging:
    def test_adds_iso_timestamp_prefix_to_formatted_records(self) -> None:
        test_logger = _fresh_logger("test.daemon_logging.iso_prefix")
        server._configure_logging(test_logger)
        stream = io.StringIO()
        test_logger.handlers[0].stream = stream

        test_logger.warning("hello world")

        output = stream.getvalue()
        assert _ISO_PREFIX.match(output)
        assert "hello world" in output

    def test_installs_exactly_one_handler_when_none_exists(self) -> None:
        test_logger = _fresh_logger("test.daemon_logging.install_handler")
        assert test_logger.handlers == []

        server._configure_logging(test_logger)

        assert len(test_logger.handlers) == 1

    def test_second_call_does_not_stack_a_duplicate_handler(self) -> None:
        """Do-not-double-configure: a repeat call must update the existing
        handler's formatter, never add a second one (which would double-
        print every subsequent line)."""
        test_logger = _fresh_logger("test.daemon_logging.idempotent")

        server._configure_logging(test_logger)
        server._configure_logging(test_logger)

        assert len(test_logger.handlers) == 1

    def test_reuses_and_reformats_a_preexisting_handler(self) -> None:
        """A logger that already has a handler (e.g. set up by something
        else first) keeps that handler -- only its Formatter is replaced."""
        test_logger = _fresh_logger("test.daemon_logging.preexisting_handler")
        stream = io.StringIO()
        existing_handler = logging.StreamHandler(stream)
        existing_handler.setFormatter(logging.Formatter("%(message)s"))
        test_logger.addHandler(existing_handler)

        server._configure_logging(test_logger)

        assert len(test_logger.handlers) == 1
        assert test_logger.handlers[0] is existing_handler
        test_logger.warning("reused handler")
        output = stream.getvalue()
        assert _ISO_PREFIX.match(output)
        assert "reused handler" in output

    def test_defaults_to_the_root_logger(self) -> None:
        """No explicit target -- the real call site in main() -- configures
        the root logger, which every daemon logger (including
        "simba.memory") propagates to without its own handler."""
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        try:
            root.handlers.clear()
            returned = server._configure_logging()
            assert returned is root
            assert len(root.handlers) == 1
        finally:
            root.handlers.clear()
            root.handlers.extend(saved_handlers)
            root.setLevel(saved_level)

    def test_simba_memory_logger_lines_get_the_prefix_via_propagation(
        self,
    ) -> None:
        """End-to-end: a real `simba.memory`-style child logger with no
        handler of its own still gets the timestamp, via propagation to the
        configured root."""
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        child = logging.getLogger("test.daemon_logging.simba_memory_child")
        child.handlers.clear()
        child.propagate = True
        try:
            root.handlers.clear()
            server._configure_logging()
            stream = io.StringIO()
            root.handlers[0].stream = stream

            child.warning("daemon line")

            output = stream.getvalue()
            assert _ISO_PREFIX.match(output)
            assert "daemon line" in output
        finally:
            root.handlers.clear()
            root.handlers.extend(saved_handlers)
            root.setLevel(saved_level)


class TestMainWiresLoggingSetup:
    def test_main_calls_configure_logging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main() must call the setup function (not the old bare
        `logging.basicConfig`) so real daemon boots get timestamped lines."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        calls: list[object] = []
        real_configure = server._configure_logging

        def _spy(target: logging.Logger | None = None) -> logging.Logger:
            calls.append(target)
            return real_configure(target)

        def _fake_run(app: object, **kwargs: object) -> None:
            return None

        def _fake_os_exit(code: int) -> None:
            return None

        monkeypatch.setattr(server, "_configure_logging", _spy)
        monkeypatch.setattr(uvicorn, "run", _fake_run)
        monkeypatch.setattr(server, "_os_exit", _fake_os_exit)
        monkeypatch.setattr(sys, "argv", ["simba-memory-daemon", "--port", str(port)])

        server.main()

        assert len(calls) == 1

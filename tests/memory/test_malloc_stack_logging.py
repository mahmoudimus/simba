"""Boot-time visibility for the `malloc_stack_logging` diagnostic lever.

2026-07-19: a 16.7GB daemon RSS burst had no attributable allocator stacks
because the daemon that ended up serving was an unarmed hook auto-start (or a
watchdog execv descendant of one) --- `MallocStackLogging` can only be armed
at process SPAWN, so daemon.log must always answer "was THIS daemon armed?".

The spawn-side env injection is covered in tests/hooks/test_session_start.py;
this file covers the boot-time logging half, common to every spawn path
(hook auto-start, `simba server`, and a restart execv --- which preserves
whatever env the process already had).
"""

from __future__ import annotations

import logging

import simba.memory.config as config
import simba.memory.server as server


class TestMallocStackLoggingBootLog:
    def test_logs_active_when_env_present(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("MallocStackLogging", "lite")
        cfg = config.MemoryConfig(malloc_stack_logging=True)
        with caplog.at_level(logging.INFO, logger="simba.memory"):
            server._log_malloc_stack_logging_status(cfg)
        infos = [r for r in caplog.records if r.levelname == "INFO"]
        assert any(
            "MallocStackLogging" in r.getMessage() and "ACTIVE" in r.getMessage()
            for r in infos
        )

    def test_logs_inactive_when_env_absent(self, monkeypatch, caplog) -> None:
        monkeypatch.delenv("MallocStackLogging", raising=False)
        cfg = config.MemoryConfig(malloc_stack_logging=False)
        with caplog.at_level(logging.INFO, logger="simba.memory"):
            server._log_malloc_stack_logging_status(cfg)
        infos = [r for r in caplog.records if r.levelname == "INFO"]
        assert any(
            "MallocStackLogging" in r.getMessage() and "inactive" in r.getMessage()
            for r in infos
        )

    def test_warns_when_flag_true_but_env_absent(self, monkeypatch, caplog) -> None:
        """Env can't be injected post-spawn into self --- warn so the
        operator knows to relaunch (a restart's execv PRESERVES env, it
        never adds env that was never set in the first place)."""
        monkeypatch.delenv("MallocStackLogging", raising=False)
        cfg = config.MemoryConfig(malloc_stack_logging=True)
        with caplog.at_level(logging.INFO, logger="simba.memory"):
            server._log_malloc_stack_logging_status(cfg)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "malloc_stack_logging" in warnings[0].getMessage()

    def test_no_warning_when_flag_true_and_env_present(
        self, monkeypatch, caplog
    ) -> None:
        monkeypatch.setenv("MallocStackLogging", "lite")
        cfg = config.MemoryConfig(malloc_stack_logging=True)
        with caplog.at_level(logging.INFO, logger="simba.memory"):
            server._log_malloc_stack_logging_status(cfg)
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_no_warning_when_flag_false(self, monkeypatch, caplog) -> None:
        monkeypatch.delenv("MallocStackLogging", raising=False)
        cfg = config.MemoryConfig(malloc_stack_logging=False)
        with caplog.at_level(logging.INFO, logger="simba.memory"):
            server._log_malloc_stack_logging_status(cfg)
        assert not any(r.levelname == "WARNING" for r in caplog.records)

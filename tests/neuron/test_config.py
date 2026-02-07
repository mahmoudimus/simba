"""Tests for neuron config module -- enums, constants, ServerConfig, and helpers."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import simba.neuron.config


class TestStatusEnum:
    def test_pending_value(self) -> None:
        assert simba.neuron.config.Status.PENDING == 1

    def test_started_value(self) -> None:
        assert simba.neuron.config.Status.STARTED == 2

    def test_running_value(self) -> None:
        assert simba.neuron.config.Status.RUNNING == 3

    def test_completed_value(self) -> None:
        assert simba.neuron.config.Status.COMPLETED == 4

    def test_failed_value(self) -> None:
        assert simba.neuron.config.Status.FAILED == 5

    def test_all_members(self) -> None:
        names = [s.name for s in simba.neuron.config.Status]
        assert names == ["PENDING", "STARTED", "RUNNING", "COMPLETED", "FAILED"]

    def test_is_int_enum(self) -> None:
        assert isinstance(simba.neuron.config.Status.PENDING, int)


class TestLogLevelEnum:
    def test_debug_value(self) -> None:
        assert simba.neuron.config.LogLevel.DEBUG == 1

    def test_info_value(self) -> None:
        assert simba.neuron.config.LogLevel.INFO == 2

    def test_warning_value(self) -> None:
        assert simba.neuron.config.LogLevel.WARNING == 3

    def test_error_value(self) -> None:
        assert simba.neuron.config.LogLevel.ERROR == 4

    def test_all_members(self) -> None:
        names = [lvl.name for lvl in simba.neuron.config.LogLevel]
        assert names == ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_is_int_enum(self) -> None:
        assert isinstance(simba.neuron.config.LogLevel.DEBUG, int)


class TestStatusNameMap:
    def test_all_keys_present(self) -> None:
        expected_keys = {"pending", "started", "running", "completed", "failed"}
        assert set(simba.neuron.config.STATUS_NAME_MAP.keys()) == expected_keys

    def test_pending_maps_to_status(self) -> None:
        result = simba.neuron.config.STATUS_NAME_MAP["pending"]
        assert result is simba.neuron.config.Status.PENDING

    def test_started_maps_to_status(self) -> None:
        result = simba.neuron.config.STATUS_NAME_MAP["started"]
        assert result is simba.neuron.config.Status.STARTED

    def test_running_maps_to_status(self) -> None:
        result = simba.neuron.config.STATUS_NAME_MAP["running"]
        assert result is simba.neuron.config.Status.RUNNING

    def test_completed_maps_to_status(self) -> None:
        assert (
            simba.neuron.config.STATUS_NAME_MAP["completed"]
            is simba.neuron.config.Status.COMPLETED
        )

    def test_failed_maps_to_status(self) -> None:
        result = simba.neuron.config.STATUS_NAME_MAP["failed"]
        assert result is simba.neuron.config.Status.FAILED


class TestLoggingLevelMap:
    def test_debug_maps(self) -> None:
        assert (
            simba.neuron.config.LOGGING_LEVEL_MAP[logging.DEBUG]
            is simba.neuron.config.LogLevel.DEBUG
        )

    def test_info_maps(self) -> None:
        assert (
            simba.neuron.config.LOGGING_LEVEL_MAP[logging.INFO]
            is simba.neuron.config.LogLevel.INFO
        )

    def test_warning_maps(self) -> None:
        assert (
            simba.neuron.config.LOGGING_LEVEL_MAP[logging.WARNING]
            is simba.neuron.config.LogLevel.WARNING
        )

    def test_error_maps(self) -> None:
        assert (
            simba.neuron.config.LOGGING_LEVEL_MAP[logging.ERROR]
            is simba.neuron.config.LogLevel.ERROR
        )

    def test_critical_maps_to_error(self) -> None:
        assert (
            simba.neuron.config.LOGGING_LEVEL_MAP[logging.CRITICAL]
            is simba.neuron.config.LogLevel.ERROR
        )

    def test_all_standard_levels_covered(self) -> None:
        expected_keys = {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }
        assert set(simba.neuron.config.LOGGING_LEVEL_MAP.keys()) == expected_keys


class TestServerConfig:
    def test_default_db_path(self) -> None:
        assert simba.neuron.config.CONFIG.db_path == Path(".simba/neuron/truth.db")

    def test_default_python_cmd_is_string(self) -> None:
        assert isinstance(simba.neuron.config.CONFIG.python_cmd, str)
        assert len(simba.neuron.config.CONFIG.python_cmd) > 0

    def test_souffle_cmd_is_string_or_none(self) -> None:
        cmd = simba.neuron.config.CONFIG.souffle_cmd
        assert cmd is None or isinstance(cmd, str)

    def test_resolved_db_path_is_absolute(self) -> None:
        assert simba.neuron.config.CONFIG.resolved_db_path.is_absolute()

    def test_resolved_db_path_ends_with_db_path(self) -> None:
        resolved = simba.neuron.config.CONFIG.resolved_db_path
        assert resolved.parts[-2:] == ("neuron", "truth.db")

    def test_custom_config(self) -> None:
        cfg = simba.neuron.config.ServerConfig(
            db_path=Path("/tmp/test.db"),
            python_cmd="/usr/bin/python3",
            souffle_cmd=None,
        )
        assert cfg.db_path == Path("/tmp/test.db")
        assert cfg.python_cmd == "/usr/bin/python3"
        assert cfg.souffle_cmd is None


class TestAgentDbPath:
    def test_value(self) -> None:
        expected = Path(".simba/neuron/agents.db")
        assert expected == simba.neuron.config.AGENT_DB_PATH


class TestUtcNow:
    def test_returns_int(self) -> None:
        result = simba.neuron.config.utc_now()
        assert isinstance(result, int)

    def test_returns_recent_timestamp(self) -> None:
        before = int(time.time())
        result = simba.neuron.config.utc_now()
        after = int(time.time())
        assert before <= result <= after

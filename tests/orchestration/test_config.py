"""Tests for orchestration config module -- enums, constants, and helpers."""

from __future__ import annotations

import logging
import time

import simba.orchestration.config


class TestStatusEnum:
    def test_pending_value(self) -> None:
        assert simba.orchestration.config.Status.PENDING == 1

    def test_started_value(self) -> None:
        assert simba.orchestration.config.Status.STARTED == 2

    def test_running_value(self) -> None:
        assert simba.orchestration.config.Status.RUNNING == 3

    def test_completed_value(self) -> None:
        assert simba.orchestration.config.Status.COMPLETED == 4

    def test_failed_value(self) -> None:
        assert simba.orchestration.config.Status.FAILED == 5

    def test_all_members(self) -> None:
        names = [s.name for s in simba.orchestration.config.Status]
        assert names == ["PENDING", "STARTED", "RUNNING", "COMPLETED", "FAILED"]

    def test_is_int_enum(self) -> None:
        assert isinstance(simba.orchestration.config.Status.PENDING, int)


class TestLogLevelEnum:
    def test_debug_value(self) -> None:
        assert simba.orchestration.config.LogLevel.DEBUG == 1

    def test_info_value(self) -> None:
        assert simba.orchestration.config.LogLevel.INFO == 2

    def test_warning_value(self) -> None:
        assert simba.orchestration.config.LogLevel.WARNING == 3

    def test_error_value(self) -> None:
        assert simba.orchestration.config.LogLevel.ERROR == 4

    def test_all_members(self) -> None:
        names = [lvl.name for lvl in simba.orchestration.config.LogLevel]
        assert names == ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_is_int_enum(self) -> None:
        assert isinstance(simba.orchestration.config.LogLevel.DEBUG, int)


class TestStatusNameMap:
    def test_all_keys_present(self) -> None:
        expected_keys = {"pending", "started", "running", "completed", "failed"}
        assert set(simba.orchestration.config.STATUS_NAME_MAP.keys()) == expected_keys

    def test_pending_maps_to_status(self) -> None:
        result = simba.orchestration.config.STATUS_NAME_MAP["pending"]
        assert result is simba.orchestration.config.Status.PENDING

    def test_started_maps_to_status(self) -> None:
        result = simba.orchestration.config.STATUS_NAME_MAP["started"]
        assert result is simba.orchestration.config.Status.STARTED

    def test_running_maps_to_status(self) -> None:
        result = simba.orchestration.config.STATUS_NAME_MAP["running"]
        assert result is simba.orchestration.config.Status.RUNNING

    def test_completed_maps_to_status(self) -> None:
        assert (
            simba.orchestration.config.STATUS_NAME_MAP["completed"]
            is simba.orchestration.config.Status.COMPLETED
        )

    def test_failed_maps_to_status(self) -> None:
        result = simba.orchestration.config.STATUS_NAME_MAP["failed"]
        assert result is simba.orchestration.config.Status.FAILED


class TestLoggingLevelMap:
    def test_debug_maps(self) -> None:
        assert (
            simba.orchestration.config.LOGGING_LEVEL_MAP[logging.DEBUG]
            is simba.orchestration.config.LogLevel.DEBUG
        )

    def test_info_maps(self) -> None:
        assert (
            simba.orchestration.config.LOGGING_LEVEL_MAP[logging.INFO]
            is simba.orchestration.config.LogLevel.INFO
        )

    def test_warning_maps(self) -> None:
        assert (
            simba.orchestration.config.LOGGING_LEVEL_MAP[logging.WARNING]
            is simba.orchestration.config.LogLevel.WARNING
        )

    def test_error_maps(self) -> None:
        assert (
            simba.orchestration.config.LOGGING_LEVEL_MAP[logging.ERROR]
            is simba.orchestration.config.LogLevel.ERROR
        )

    def test_critical_maps_to_error(self) -> None:
        assert (
            simba.orchestration.config.LOGGING_LEVEL_MAP[logging.CRITICAL]
            is simba.orchestration.config.LogLevel.ERROR
        )

    def test_all_standard_levels_covered(self) -> None:
        expected_keys = {
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }
        assert set(simba.orchestration.config.LOGGING_LEVEL_MAP.keys()) == expected_keys


class TestUtcNow:
    def test_returns_int(self) -> None:
        result = simba.orchestration.config.utc_now()
        assert isinstance(result, int)

    def test_returns_recent_timestamp(self) -> None:
        before = int(time.time())
        result = simba.orchestration.config.utc_now()
        after = int(time.time())
        assert before <= result <= after

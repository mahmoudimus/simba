"""Tests for simba.config_cli — CLI commands for unified configuration."""

from __future__ import annotations

import pathlib

import pytest

import simba.config
import simba.config_cli


class TestCmdList:
    def test_lists_sections(self, capsys: pytest.CaptureFixture[str]) -> None:
        simba.config_cli.cmd_list()
        captured = capsys.readouterr()
        # At least the "memory" section should appear (registered by import).
        assert "[memory]" in captured.out or "[test_section]" in captured.out


class TestCmdGet:
    def test_get_existing_key(
        self,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Import to ensure registration
        import simba.memory.config

        rc = simba.config_cli.cmd_get("memory.port", tmp_path)
        assert rc == 0
        captured = capsys.readouterr()
        assert "8741" in captured.out

    def test_get_invalid_format(
        self,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = simba.config_cli.cmd_get("no_dot", tmp_path)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid key format" in captured.err


class TestCmdSet:
    def test_set_and_get(
        self,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import simba.memory.config

        rc = simba.config_cli.cmd_set(
            "memory.port", "9000", global_flag=False, root=tmp_path
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "Set memory.port = 9000" in captured.out

        rc = simba.config_cli.cmd_get("memory.port", tmp_path)
        captured = capsys.readouterr()
        assert "9000" in captured.out


class TestCmdReset:
    def test_reset(
        self,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import simba.memory.config

        simba.config.set_value(
            "memory", "port", 1234, scope="local", root=tmp_path
        )
        rc = simba.config_cli.cmd_reset(
            "memory.port", global_flag=False, root=tmp_path
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "Reset memory.port" in captured.out


class TestCmdShow:
    def test_show_prints_sections(
        self,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        simba.config_cli.cmd_show(tmp_path)
        captured = capsys.readouterr()
        assert "[" in captured.out  # at least one section header


class TestMain:
    def test_no_subcmd(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = simba.config_cli.main([])
        assert rc == 1

    def test_list_via_main(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = simba.config_cli.main(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out  # non-empty

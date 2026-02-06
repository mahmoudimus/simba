"""Tests for simba.neuron.verify module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from simba.neuron.verify import TempFileCleanup, analyze_datalog, verify_z3

# ---------------------------------------------------------------------------
# verify_z3 tests
# ---------------------------------------------------------------------------


class TestVerifyZ3:
    """Tests for the verify_z3 function."""

    @patch("simba.neuron.verify.subprocess.run")
    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_proven_result(self, mock_config: MagicMock, mock_run: MagicMock) -> None:
        """verify_z3 returns execution result when script prints PROVEN."""
        mock_config.python_cmd = "python3"
        mock_run.return_value = MagicMock(
            stdout="PROVEN\n",
            stderr="",
            returncode=0,
        )

        result = verify_z3("s = Solver(); print('PROVEN')")

        assert "Execution Result:" in result
        assert "PROVEN" in result
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args.kwargs["capture_output"] is True
        assert call_args.kwargs["text"] is True
        assert call_args.kwargs["timeout"] == 30

    @patch("simba.neuron.verify.subprocess.run")
    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_counterexample_result(
        self, mock_config: MagicMock, mock_run: MagicMock
    ) -> None:
        """verify_z3 returns execution result when script prints COUNTEREXAMPLE."""
        mock_config.python_cmd = "python3"
        mock_run.return_value = MagicMock(
            stdout="COUNTEREXAMPLE\nx = 42\n",
            stderr="",
            returncode=0,
        )

        result = verify_z3("s = Solver(); print('COUNTEREXAMPLE')")

        assert "Execution Result:" in result
        assert "COUNTEREXAMPLE" in result
        assert "x = 42" in result

    @patch("simba.neuron.verify.subprocess.run")
    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_timeout(self, mock_config: MagicMock, mock_run: MagicMock) -> None:
        """verify_z3 returns timeout error when subprocess exceeds time limit."""
        mock_config.python_cmd = "python3"
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=["python3", "/tmp/test.py"], timeout=30
        )

        result = verify_z3("import time; time.sleep(999)")

        assert "timed out" in result
        assert "30s" in result

    @patch("simba.neuron.verify.subprocess.run")
    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_runtime_error(self, mock_config: MagicMock, mock_run: MagicMock) -> None:
        """verify_z3 returns script error when subprocess exits with non-zero code."""
        mock_config.python_cmd = "python3"
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="NameError: name 'foo' is not defined\n",
            returncode=1,
        )

        result = verify_z3("foo")

        assert "Script Error" in result
        assert "Exit Code 1" in result
        assert "NameError" in result


# ---------------------------------------------------------------------------
# analyze_datalog tests
# ---------------------------------------------------------------------------


class TestAnalyzeDatalog:
    """Tests for the analyze_datalog function."""

    @patch("simba.neuron.verify.subprocess.run")
    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_success(self, mock_config: MagicMock, mock_run: MagicMock) -> None:
        """analyze_datalog returns analysis output on successful execution."""
        mock_config.souffle_cmd = "/usr/bin/souffle"
        mock_run.return_value = MagicMock(
            stdout="edge(1,2)\nedge(2,3)\n",
            stderr="",
            returncode=0,
        )

        result = analyze_datalog(".decl edge(a:number, b:number)\n", facts_dir="/tmp")

        assert "Analysis Output:" in result
        assert "edge(1,2)" in result
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "/usr/bin/souffle"
        assert "-F" in cmd
        assert "/tmp" in cmd
        assert "-D" in cmd
        assert "-" in cmd

    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_souffle_not_found(self, mock_config: MagicMock) -> None:
        """analyze_datalog returns error when souffle binary is not available."""
        mock_config.souffle_cmd = None

        result = analyze_datalog(".decl edge(a:number, b:number)\n")

        assert "Error" in result
        assert "souffle" in result
        assert "not found" in result

    @patch("simba.neuron.verify.subprocess.run")
    @patch("simba.neuron.verify.simba.neuron.config.CONFIG")
    def test_souffle_error(self, mock_config: MagicMock, mock_run: MagicMock) -> None:
        """analyze_datalog returns error output when souffle reports an error."""
        mock_config.souffle_cmd = "/usr/bin/souffle"
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="Error: syntax error at line 1\n",
            returncode=1,
        )

        result = analyze_datalog("bad syntax")

        assert "Souffle Logic Error:" in result
        assert "syntax error" in result


# ---------------------------------------------------------------------------
# TempFileCleanup tests
# ---------------------------------------------------------------------------


class TestTempFileCleanup:
    """Tests for the TempFileCleanup context manager."""

    def test_creates_and_cleans_up_file(self, tmp_path: Path) -> None:
        """TempFileCleanup deletes the file on context exit."""
        temp_file = tmp_path / "test_cleanup.txt"
        temp_file.write_text("temporary content")
        assert temp_file.exists()

        with TempFileCleanup(temp_file) as path:
            assert path == temp_file
            assert path.exists()

        assert not temp_file.exists()

    def test_no_error_if_file_missing(self, tmp_path: Path) -> None:
        """TempFileCleanup does not raise if file was already deleted."""
        nonexistent = tmp_path / "does_not_exist.txt"

        with TempFileCleanup(nonexistent) as path:
            assert path == nonexistent

        # Should not raise -- file never existed
        assert not nonexistent.exists()

    def test_returns_path_object(self, tmp_path: Path) -> None:
        """TempFileCleanup __enter__ returns a Path instance."""
        temp_file = tmp_path / "pathcheck.txt"
        temp_file.write_text("check")

        with TempFileCleanup(str(temp_file)) as path:
            assert isinstance(path, Path)

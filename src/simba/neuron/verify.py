"""Formal verification tools — Z3 theorem prover and Soufflé Datalog."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import simba.neuron.config


class TempFileCleanup:
    """Context manager that deletes a temporary file on exit."""

    def __init__(self, filepath: str | Path) -> None:
        self.path = Path(filepath)

    def __enter__(self) -> Path:
        return self.path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self.path.exists():
            self.path.unlink()


def verify_z3(python_script: str) -> str:
    """Execute a Z3 proof script in an isolated process.

    The script MUST print 'PROVEN' or 'COUNTEREXAMPLE' to stdout.
    The environment already has 'from z3 import *'.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("from z3 import *\n")
        f.write(python_script)
        temp_name = f.name

    with TempFileCleanup(temp_name):
        try:
            result = subprocess.run(
                [simba.neuron.config.CONFIG.python_cmd, temp_name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout + result.stderr
            if result.returncode != 0:
                return f"Script Error (Exit Code {result.returncode}):\n{output}"

            return f"Execution Result:\n{output}"
        except subprocess.TimeoutExpired:
            return "Error: Z3 verification timed out (limit: 30s)."
        except Exception as exc:
            return f"Logic Execution Failed: {exc}"


def analyze_datalog(datalog_code: str, facts_dir: str = ".") -> str:
    """Run a Soufflé Datalog analysis.

    Writes code to a temporary file and executes against the specified
    fact directory.
    """
    souffle_cmd = simba.neuron.config.CONFIG.souffle_cmd
    if not souffle_cmd:
        return "Error: 'souffle' binary not found in system PATH."

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dl", delete=False) as f:
        f.write(datalog_code)
        temp_name = f.name

    with TempFileCleanup(temp_name):
        cmd = [souffle_cmd, "-F", facts_dir, "-D", "-", temp_name]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return f"Souffle Logic Error:\n{result.stderr}"
        return f"Analysis Output:\n{result.stdout}"

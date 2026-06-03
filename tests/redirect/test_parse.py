"""Tests for shell-command parsing used by the tool-redirect feature."""

from __future__ import annotations

import simba.redirect.parse as p


class TestProgramName:
    def test_basename_and_lowercase(self) -> None:
        assert p.program_name("/usr/bin/Cargo") == "cargo"
        assert p.program_name("./scripts/run") == "run"

    def test_strips_windows_suffix(self) -> None:
        assert p.program_name("python.EXE") == "python"


class TestInvokedPrograms:
    def _progs(self, cmd):
        return [inv.program for inv in p.invoked_programs(cmd)]

    def test_simple(self) -> None:
        assert self._progs("cargo build --release") == ["cargo"]

    def test_env_prefix_stripped(self) -> None:
        assert self._progs("env FOO=1 RUST_LOG=debug cargo test") == ["cargo"]
        assert self._progs("FOO=1 python x.py") == ["python"]

    def test_segments_split(self) -> None:
        assert self._progs("cargo fmt && python lint.py ; rustc x.rs") == [
            "cargo",
            "python",
            "rustc",
        ]

    def test_pipe_split(self) -> None:
        assert self._progs("cat f | python parse.py") == ["cat", "python"]

    def test_uv_run_resolves_inner_tool(self) -> None:
        # `uv run cargo` should resolve to the effective tool, cargo
        assert self._progs("uv run cargo build") == ["cargo"]
        assert self._progs("uv run --frozen cargo test") == ["cargo"]

    def test_nested_bash_c(self) -> None:
        assert "cargo" in self._progs('bash -c "cargo build"')

    def test_args_preserved(self) -> None:
        inv = p.invoked_programs("cargo build --release")[0]
        assert inv.words[0].endswith("cargo")
        assert "build" in inv.words

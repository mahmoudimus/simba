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


class TestShlexTokenization:
    def test_quoted_arg_is_one_token(self) -> None:
        # shlex keeps a quoted run together (hand-rolled split would too, but
        # this locks the shlex behavior)
        assert p.tokenize('cargo build "--message-format json"') == [
            "cargo",
            "build",
            "--message-format json",
        ]

    def test_escaped_space(self) -> None:
        assert p.tokenize(r"python my\ script.py") == ["python", "my script.py"]

    def test_single_quotes(self) -> None:
        assert p.tokenize("python -c 'import sys; print(1)'") == [
            "python",
            "-c",
            "import sys; print(1)",
        ]

    def test_malformed_quotes_falls_back_not_raises(self) -> None:
        # unbalanced quote must NOT raise (the hook can't crash) — best-effort
        out = p.tokenize('cargo build "--unterminated')
        assert isinstance(out, list)
        assert out and out[0] == "cargo"

    def test_invoked_programs_with_quoted_args(self) -> None:
        progs = [i.program for i in p.invoked_programs('cargo run -- "a b c"')]
        assert progs == ["cargo"]

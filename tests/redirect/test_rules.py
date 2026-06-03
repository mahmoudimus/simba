"""Tests for redirect rule matching + the deny/rewrite decision."""

from __future__ import annotations

import simba.redirect.rules as r

CARGO = r.RedirectRule(
    program="cargo", replacement="soldr cargo", reason="use the pinned toolchain"
)
PY = r.RedirectRule(program="python", replacement="uv run python", reason="")
RULES = [CARGO, PY]


def test_no_match_returns_none() -> None:
    assert r.evaluate("ls -la", RULES, mode="deny") is None
    # already wrapped in soldr -> no match (soldr isn't a rule program)
    assert r.evaluate("soldr cargo build", RULES, mode="deny") is None


class TestDenyMode:
    def test_deny_message_names_replacement(self) -> None:
        d = r.evaluate("cargo build --release", RULES, mode="deny")
        assert d.action == "deny"
        assert "soldr cargo" in d.reason
        assert "cargo" in d.reason
        assert "pinned toolchain" in d.reason

    def test_matches_inside_uv_run(self) -> None:
        d = r.evaluate("uv run cargo test", RULES, mode="deny")
        assert d is not None and d.action == "deny"

    def test_matches_inside_nested_shell(self) -> None:
        d = r.evaluate('bash -c "cargo build"', RULES, mode="deny")
        assert d is not None and d.action == "deny"


class TestRewriteMode:
    def test_simple_leading_command_is_rewritten(self) -> None:
        d = r.evaluate("cargo build --release", RULES, mode="rewrite")
        assert d.action == "rewrite"
        assert d.command == "soldr cargo build --release"

    def test_python_rewrite(self) -> None:
        d = r.evaluate("python script.py --flag x", RULES, mode="rewrite")
        assert d.action == "rewrite"
        assert d.command == "uv run python script.py --flag x"

    def test_env_prefixed_falls_back_to_deny(self) -> None:
        # program isn't the leading token -> can't safely rewrite -> deny
        d = r.evaluate("FOO=1 cargo build", RULES, mode="rewrite")
        assert d.action == "deny"

    def test_multi_segment_falls_back_to_deny(self) -> None:
        d = r.evaluate("cargo fmt && cargo test", RULES, mode="rewrite")
        assert d.action == "deny"

    def test_uv_run_falls_back_to_deny(self) -> None:
        d = r.evaluate("uv run cargo build", RULES, mode="rewrite")
        assert d.action == "deny"

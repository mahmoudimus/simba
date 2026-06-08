"""Integration: redirect decision flows through the PreToolUse hook."""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.config
import simba.db
import simba.hooks.config as hcfg
import simba.hooks.pre_tool_use as hook
import simba.redirect.store as store


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _force_mode(monkeypatch, mode: str) -> None:
    real = simba.config.load

    def fake(section, *a, **k):
        if section == "hooks":
            return hcfg.HooksConfig(redirect_enabled=True, redirect_mode=mode)
        return real(section, *a, **k)

    monkeypatch.setattr(simba.config, "load", fake)


def _add_cargo_rule(cwd: pathlib.Path) -> None:
    pid = simba.db.resolve_project_id(cwd)
    store.add("cargo", "soldr cargo", reason="pinned toolchain", project_path=pid)


def test_deny_mode_blocks_with_correction(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "deny")
    _add_cargo_rule(tmp_path)
    out = json.loads(
        hook.main(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "cargo build --release"},
                "cwd": str(tmp_path),
            }
        )
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "soldr cargo" in hso["permissionDecisionReason"]


def test_rewrite_mode_substitutes_command(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "rewrite")
    _add_cargo_rule(tmp_path)
    out = json.loads(
        hook.main(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "cargo build --release"},
                "cwd": str(tmp_path),
            }
        )
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    assert hso["updatedInput"]["command"] == "soldr cargo build --release"


def test_no_rule_does_not_deny(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "deny")  # no rules added
    out = json.loads(
        hook.main(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "cargo build"},
                "cwd": str(tmp_path),
            }
        )
    )
    # falls through to the normal context/empty path — no permissionDecision
    assert "permissionDecision" not in out.get("hookSpecificOutput", {})


def test_builtin_rg_trap_fires_with_no_project_rules(tmp_path, monkeypatch) -> None:
    # Zero project config + default "deny" mode: the built-in rg `-r` rule still
    # fires AND auto-rewrites (it carries mode="rewrite"), so the corrupting
    # `rg -rn` never runs.
    _force_mode(monkeypatch, "deny")  # no rules added
    out = json.loads(
        hook.main(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "rg -rn 'TODO' src/ | head"},
                "cwd": str(tmp_path),
            }
        )
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    assert hso["updatedInput"]["command"] == "rg -n 'TODO' src/ | head"


def test_non_bash_skips_redirect(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "deny")
    _add_cargo_rule(tmp_path)
    out = json.loads(
        hook.main(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "cargo.toml"},
                "cwd": str(tmp_path),
            }
        )
    )
    assert "permissionDecision" not in out.get("hookSpecificOutput", {})

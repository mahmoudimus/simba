"""Byte-identical characterization gate for the PreToolUse canonical refactor.

These pin ``pre_tool_use.main()`` output (Claude/Codex envelope) across the three
output shapes — redirect deny, redirect rewrite, and the empty context path — so
the spec-24 canonicalization cannot change what Claude/Codex see. They also pin
the ``claude.render("PreToolUse", …)`` equivalences and that a strong TOOL_RULE
match populates ``escalated_block`` (pi-only metadata) without touching the
Claude/Codex bytes.
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest

import simba.config
import simba.db
import simba.harness.adapters.claude as claude
import simba.hooks._io
import simba.hooks.config as hcfg
import simba.hooks.pre_tool_use as hook
import simba.redirect.store as store
from simba.harness.core import CanonicalResult


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


# ── main() byte-identical: the three output shapes ──────────────────────────


def test_main_deny_is_byte_identical(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "deny")
    _add_cargo_rule(tmp_path)
    out = hook.main(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "cargo build --release"},
            "cwd": str(tmp_path),
        }
    )
    expected = simba.hooks._io.pretool_deny(
        "Use `soldr cargo ...` instead of `cargo ...`. pinned toolchain"
    )
    assert out == expected


def test_main_rewrite_is_byte_identical(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "rewrite")
    _add_cargo_rule(tmp_path)
    out = hook.main(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "cargo build --release"},
            "cwd": str(tmp_path),
        }
    )
    expected = simba.hooks._io.pretool_rewrite("soldr cargo build --release")
    assert out == expected


def test_main_builtin_rg_rewrite_is_byte_identical(tmp_path, monkeypatch) -> None:
    _force_mode(monkeypatch, "deny")  # no project rules; builtin carries rewrite
    out = hook.main(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "rg -rn 'TODO' src/ | head"},
            "cwd": str(tmp_path),
        }
    )
    reason = (
        "rg -r is --replace (consumes the next token); recursion + line "
        "numbers are default. Dropped the bundled -r."
    )
    expected = simba.hooks._io.pretool_rewrite("rg -n 'TODO' src/ | head", reason)
    assert out == expected


def test_main_empty_context_path_is_byte_identical(monkeypatch) -> None:
    # No transcript, no tool_input → no parts fire → empty envelope.
    out = hook.main({"tool_name": "Read", "tool_input": {}, "cwd": "/tmp"})
    assert out == simba.hooks._io.empty("PreToolUse")
    assert json.loads(out) == {"hookSpecificOutput": {"hookEventName": "PreToolUse"}}


def test_main_no_thinking_field_is_unaffected(tmp_path, monkeypatch) -> None:
    # v2.1 added a payload ``thinking`` channel (pi). A Claude/Codex-style payload
    # has NO ``thinking`` field — it must produce the SAME envelope as before. Here
    # a redirect rule fires deterministically and the absence of ``thinking`` leaves
    # the output byte-identical to the canonical rewrite.
    _force_mode(monkeypatch, "rewrite")
    _add_cargo_rule(tmp_path)
    out = hook.main(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "cargo build --release"},
            "cwd": str(tmp_path),
        }
    )
    assert out == simba.hooks._io.pretool_rewrite("soldr cargo build --release")


# ── render() equivalences ───────────────────────────────────────────────────


def test_render_pretool_rewrite_equals_io() -> None:
    out = claude.render(
        "PreToolUse", CanonicalResult(transform={"command": "rg -n x", "reason": "r"})
    )
    assert out == simba.hooks._io.pretool_rewrite("rg -n x", "r")


def test_render_pretool_deny_equals_io() -> None:
    out = claude.render("PreToolUse", CanonicalResult(block_reason="nope"))
    assert out == simba.hooks._io.pretool_deny("nope")


def test_render_pretool_empty_context() -> None:
    out = claude.render("PreToolUse", CanonicalResult())
    assert out == simba.hooks._io.empty("PreToolUse")


def test_render_pretool_context_injection() -> None:
    out = claude.render("PreToolUse", CanonicalResult(additional_context="warn"))
    assert out == simba.hooks._io.context("PreToolUse", "warn")


def test_render_ignores_escalated_block() -> None:
    # escalated_block is pi-only metadata; Claude/Codex render must ignore it.
    # With only a TOOL_RULE warning in additional_context, render injects context.
    out = claude.render(
        "PreToolUse",
        CanonicalResult(
            additional_context="<tool-rule-warning>...",
            escalated_block="blocked by rule",
        ),
    )
    assert out == simba.hooks._io.context("PreToolUse", "<tool-rule-warning>...")


# ── escalated_block population (pi-only metadata) ───────────────────────────


def _strong_rule_memories(sim: float) -> list[dict]:
    # createdAt must be generated fresh: a hardcoded date silently ages past
    # hooks.rule_max_age_days and the recency gate drops the rule (this test
    # time-bombed exactly that way once).
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return [
        {
            "content": "never run rm -rf /",
            "context": json.dumps({"correction": "use a scoped path"}),
            "similarity": sim,
            "createdAt": created,
        }
    ]


def test_escalated_block_set_on_strong_tool_rule(tmp_path, monkeypatch) -> None:
    real = simba.config.load

    def fake(section, *a, **k):
        if section == "hooks":
            return hcfg.HooksConfig(
                redirect_enabled=False,
                rule_check_enabled=True,
                rule_count_ttl=0,
                permission_deny_similarity=0.78,
                rule_min_similarity=0.3,
            )
        return real(section, *a, **k)

    monkeypatch.setattr(simba.config, "load", fake)
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: _strong_rule_memories(0.91),
    )
    result = hook.run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "cwd": str(tmp_path),
        }
    )
    assert result.escalated_block is not None
    # The TOOL_RULE warning still goes into additional_context as today.
    assert "<tool-rule-warning>" in result.additional_context


def test_escalated_block_none_on_weak_tool_rule(tmp_path, monkeypatch) -> None:
    real = simba.config.load

    def fake(section, *a, **k):
        if section == "hooks":
            return hcfg.HooksConfig(
                redirect_enabled=False,
                rule_check_enabled=True,
                rule_count_ttl=0,
                permission_deny_similarity=0.78,
                rule_min_similarity=0.3,
            )
        return real(section, *a, **k)

    monkeypatch.setattr(simba.config, "load", fake)
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories",
        lambda *a, **k: _strong_rule_memories(0.50),
    )
    result = hook.run(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf /"},
            "cwd": str(tmp_path),
        }
    )
    assert result.escalated_block is None
    # The weak match still surfaces as a context warning.
    assert "<tool-rule-warning>" in result.additional_context

"""PostToolBatch lane (default-OFF): payload cap + recall-on-batch.

``PostToolBatch`` fires once per tool-call round with a ``tool_calls`` array,
each carrying a (potentially huge) serialized ``tool_response``. This lane is
default-OFF (UNMEASURED lever); when enabled it builds a compact recall query
from the batch and injects formatted memories the same way ``pre_tool_use.py``
does. The payload trim is MANDATORY and unconditional (runs before any query
is built, whether or not recall ends up firing) — the daemon must never see
an unbounded batch.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.harness.adapters.claude as claude
import simba.hooks._io
import simba.hooks.config as hcfg
import simba.hooks.post_tool_batch as ptb
from simba.harness.core import CanonicalResult

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _cfg(**over):
    return hcfg.HooksConfig(**over)


# ── Registration: Claude only, never Codex ──────────────────────────────────


class TestRegistration:
    def test_claude_plugin_hooks_registers_post_tool_batch(self) -> None:
        data = json.loads((_REPO_ROOT / ".claude-plugin" / "hooks.json").read_text())
        entries = data["hooks"]["PostToolBatch"]
        # Mirrors the flat command shape of the other events in this file.
        commands = [e.get("command") for e in entries]
        assert "simba hook PostToolBatch" in commands

    def test_codex_hooks_json_has_no_post_tool_batch(self) -> None:
        # Codex has no PostToolBatch event; this lane must never be registered
        # there (see the doc-verified facts this migration is built on).
        data = json.loads((_REPO_ROOT / ".codex" / "hooks.json").read_text())
        assert "PostToolBatch" not in data["hooks"]


# ── Config defaults ──────────────────────────────────────────────────────


class TestConfigDefaults:
    def test_post_tool_batch_off_by_default(self) -> None:
        assert hcfg.HooksConfig().post_tool_batch_enabled is False

    def test_default_payload_cap_kb(self) -> None:
        assert hcfg.HooksConfig().post_tool_batch_max_payload_kb == 256.0


# ── Wiring: canonical event table + native->canonical map ──────────────────


class TestWiring:
    def test_native_to_canonical_has_post_tool_batch(self) -> None:
        assert claude.NATIVE_TO_CANONICAL["PostToolBatch"] == "post_tool_batch"

    def test_dispatch_resolves_post_tool_batch(self, monkeypatch) -> None:
        # The daemon's generic POST /hook/{event} calls simba.harness.core.dispatch
        # by canonical name -- it must resolve "post_tool_batch" to this module
        # rather than raising KeyError (which the route maps to a 404).
        import simba.harness.core as core

        monkeypatch.setattr(
            hcfg.HooksConfig, "post_tool_batch_enabled", False, raising=False
        )
        result = core.dispatch("post_tool_batch", {"tool_calls": []})
        assert isinstance(result, CanonicalResult)


# ── Payload trim algorithm ──────────────────────────────────────────────────


class TestPayloadTrim:
    def test_under_cap_batch_is_untouched(self) -> None:
        calls = [
            {
                "tool_name": "Bash",
                "tool_input": {"command": "pytest -q"},
                "tool_use_id": "1",
                "tool_response": "ok, 3 passed",
            }
        ]
        trimmed, was_trimmed = ptb._trim_batch_payload(calls, max_kb=256.0)
        assert was_trimmed is False
        assert trimmed[0]["tool_response"] == "ok, 3 passed"
        assert trimmed[0]["tool_input"] == {"command": "pytest -q"}
        # Metadata fields survive untouched.
        assert trimmed[0]["tool_name"] == "Bash"
        assert trimmed[0]["tool_use_id"] == "1"

    def test_over_cap_batch_is_trimmed_and_marked(self) -> None:
        # 3 calls, each with a 1MB response -> way over a small cap.
        calls = [
            {
                "tool_name": "Bash",
                "tool_input": {"command": f"cmd{i}"},
                "tool_use_id": str(i),
                "tool_response": "x" * 1_000_000,
            }
            for i in range(3)
        ]
        trimmed, was_trimmed = ptb._trim_batch_payload(calls, max_kb=12.0)
        assert was_trimmed is True
        total_bytes = len(json.dumps(trimmed).encode("utf-8"))
        assert total_bytes <= 12.0 * 1024
        for item in trimmed:
            assert len(item["tool_response"]) < 1_000_000
            assert ptb._TRUNCATION_MARKER in item["tool_response"]
            # Metadata untouched even when content is trimmed.
            assert item["tool_name"] == "Bash"

    def test_empty_batch_is_untouched(self) -> None:
        trimmed, was_trimmed = ptb._trim_batch_payload([], max_kb=256.0)
        assert trimmed == []
        assert was_trimmed is False

    def test_trim_never_exceeds_cap_across_many_items(self) -> None:
        # A wider batch still respects the cap -- the per-item share shrinks
        # instead of the total budget being exceeded.
        calls = [
            {
                "tool_name": "Read",
                "tool_input": {"file_path": f"/f{i}"},
                "tool_use_id": str(i),
                "tool_response": "y" * 500_000,
            }
            for i in range(20)
        ]
        trimmed, was_trimmed = ptb._trim_batch_payload(calls, max_kb=64.0)
        assert was_trimmed is True
        total_bytes = len(json.dumps(trimmed).encode("utf-8"))
        assert total_bytes <= 64.0 * 1024


# ── Disabled lane: zero cost, no daemon calls ───────────────────────────────


class TestDisabledLane:
    def test_run_disabled_never_calls_recall(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ptb, "_hooks_cfg", lambda: _cfg(post_tool_batch_enabled=False)
        )

        def _boom(*a, **k):
            raise AssertionError("recall_memories must not be called when disabled")

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", _boom)
        result = ptb.run(
            {
                "tool_calls": [
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "pytest"},
                        "tool_use_id": "1",
                        "tool_response": "boom: traceback error",
                    }
                ],
                "cwd": "/tmp",
            }
        )
        assert result.additional_context == ""

    def test_main_disabled_is_valid_empty_envelope(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ptb, "_hooks_cfg", lambda: _cfg(post_tool_batch_enabled=False)
        )
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no daemon call")),
        )
        out = ptb.main({"tool_calls": [{"tool_name": "Bash"}], "cwd": "/tmp"})
        assert json.loads(out) == {
            "hookSpecificOutput": {"hookEventName": "PostToolBatch"}
        }


# ── Enabled lane: recall stub -> additionalContext in the rendered envelope ─


class TestEnabledLane:
    def _enabled_cfg(self, **over):
        return _cfg(post_tool_batch_enabled=True, **over)

    def test_run_builds_query_and_injects_formatted_memories(self, monkeypatch) -> None:
        monkeypatch.setattr(ptb, "_hooks_cfg", lambda: self._enabled_cfg())
        memories = [
            {"type": "GOTCHA", "content": "flaky test needs retry", "similarity": 0.6}
        ]
        captured = {}

        def fake_recall(query, project_path=None, **kw):
            captured["query"] = query
            captured["project_path"] = project_path
            return memories

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", fake_recall)
        result = ptb.run(
            {
                "tool_calls": [
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "pytest -q tests/test_x.py"},
                        "tool_use_id": "1",
                        "tool_response": "Traceback: Error: boom",
                    }
                ],
                "cwd": "/proj",
            }
        )
        assert "flaky test needs retry" in result.additional_context
        assert captured["project_path"] == "/proj"
        assert "Bash" in captured["query"]

    def test_main_enabled_renders_additional_context(self, monkeypatch) -> None:
        monkeypatch.setattr(ptb, "_hooks_cfg", lambda: self._enabled_cfg())
        memories = [{"type": "GOTCHA", "content": "known flake", "similarity": 0.55}]
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories", lambda *a, **k: memories
        )
        out = ptb.main(
            {
                "tool_calls": [
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "pytest -q"},
                        "tool_use_id": "1",
                        "tool_response": "error: something failed",
                    }
                ],
                "cwd": "/proj",
            }
        )
        parsed = json.loads(out)
        hso = parsed["hookSpecificOutput"]
        assert hso["hookEventName"] == "PostToolBatch"
        assert "known flake" in hso["additionalContext"]

    def test_no_memories_renders_valid_empty_envelope(self, monkeypatch) -> None:
        monkeypatch.setattr(ptb, "_hooks_cfg", lambda: self._enabled_cfg())
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories", lambda *a, **k: []
        )
        out = ptb.main(
            {
                "tool_calls": [
                    {
                        "tool_name": "Read",
                        "tool_input": {"file_path": "/x"},
                        "tool_use_id": "1",
                        "tool_response": "file contents here",
                    }
                ],
                "cwd": "/proj",
            }
        )
        assert json.loads(out) == {
            "hookSpecificOutput": {"hookEventName": "PostToolBatch"}
        }

    def test_empty_tool_calls_short_circuits_without_recall(self, monkeypatch) -> None:
        monkeypatch.setattr(ptb, "_hooks_cfg", lambda: self._enabled_cfg())

        def _boom(*a, **k):
            raise AssertionError(
                "recall_memories must not be called for an empty batch"
            )

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", _boom)
        result = ptb.run({"tool_calls": [], "cwd": "/proj"})
        assert result.additional_context == ""

    def test_trim_runs_before_query_build(self, monkeypatch) -> None:
        # A batch that would be over the cap still produces a bounded query --
        # proves the trim runs unconditionally before recall, not just when a
        # caller happens to forward the raw batch onward.
        monkeypatch.setattr(
            ptb,
            "_hooks_cfg",
            lambda: self._enabled_cfg(post_tool_batch_max_payload_kb=4.0),
        )
        captured = {}

        def fake_recall(query, project_path=None, **kw):
            captured["query"] = query
            return []

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", fake_recall)
        ptb.run(
            {
                "tool_calls": [
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "cmd"},
                        "tool_use_id": "1",
                        "tool_response": "z" * 2_000_000,
                    }
                ],
                "cwd": "/proj",
            }
        )
        # The query built from the trimmed batch stays small (nowhere near the
        # 2MB raw response) -- a bug that skipped trimming before query-build
        # would blow this well past a few KB.
        assert len(captured["query"]) < 5000


# ── Render fallthrough: PostToolBatch is a VALID hookSpecificOutput variant ─


class TestRenderFallthrough:
    def test_render_with_context_uses_generic_context_shape(self) -> None:
        out = claude.render("PostToolBatch", CanonicalResult(additional_context="hi"))
        assert out == simba.hooks._io.context("PostToolBatch", "hi")
        parsed = json.loads(out)
        assert parsed == {
            "hookSpecificOutput": {
                "hookEventName": "PostToolBatch",
                "additionalContext": "hi",
            }
        }

    def test_render_empty_is_valid_empty_envelope(self) -> None:
        out = claude.render("PostToolBatch", CanonicalResult())
        assert out == simba.hooks._io.empty("PostToolBatch")


@pytest.mark.parametrize("client", ["claude-code", "codex", None])
def test_render_fallthrough_stable_across_clients(monkeypatch, client) -> None:
    # PostToolBatch has no client-gated shape (unlike the PreToolUse/Stop
    # migrations) -- it is a brand-new lane, so both runtimes get the same
    # generic hookSpecificOutput envelope.
    if client is None:
        monkeypatch.delenv("SIMBA_CLIENT", raising=False)
    else:
        monkeypatch.setenv("SIMBA_CLIENT", client)
    out = claude.render("PostToolBatch", CanonicalResult(additional_context="hi"))
    assert json.loads(out) == {
        "hookSpecificOutput": {
            "hookEventName": "PostToolBatch",
            "additionalContext": "hi",
        }
    }

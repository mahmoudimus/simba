"""Tests for the PermissionRequest hook (Codex-only)."""

from __future__ import annotations

import dataclasses
import json
import unittest.mock

import simba.hooks.config
import simba.hooks.permission_request


def _cfg(**overrides):
    base = simba.hooks.config.HooksConfig()
    return dataclasses.replace(base, **overrides)


class TestPermissionRequestHook:
    def test_disabled_returns_empty(self):
        with unittest.mock.patch(
            "simba.hooks.permission_request._hooks_cfg",
            return_value=_cfg(permission_check_enabled=False),
        ):
            out = json.loads(
                simba.hooks.permission_request.main(
                    {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}
                )
            )
        assert out == {"hookSpecificOutput": {"hookEventName": "PermissionRequest"}}

    def test_unknown_tool_returns_empty(self):
        out = json.loads(
            simba.hooks.permission_request.main(
                {"tool_name": "WebSearch", "tool_input": {"query": "x"}}
            )
        )
        assert out["hookSpecificOutput"] == {"hookEventName": "PermissionRequest"}

    def test_no_matching_memories_returns_empty(self):
        with unittest.mock.patch(
            "simba.hooks._memory_client.recall_memories", return_value=[]
        ):
            out = json.loads(
                simba.hooks.permission_request.main(
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "ls"},
                    }
                )
            )
        assert out["hookSpecificOutput"] == {"hookEventName": "PermissionRequest"}

    def test_weak_match_falls_through(self):
        with (
            unittest.mock.patch(
                "simba.hooks.permission_request._hooks_cfg",
                return_value=_cfg(permission_deny_similarity=0.78),
            ),
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[{"content": "don't run rm -rf", "similarity": 0.55}],
            ),
        ):
            out = json.loads(
                simba.hooks.permission_request.main(
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "rm temp"},
                    }
                )
            )
        assert out["hookSpecificOutput"] == {"hookEventName": "PermissionRequest"}

    def test_strong_match_denies(self):
        with (
            unittest.mock.patch(
                "simba.hooks.permission_request._hooks_cfg",
                return_value=_cfg(permission_deny_similarity=0.78),
            ),
            unittest.mock.patch(
                "simba.hooks._memory_client.recall_memories",
                return_value=[
                    {
                        "content": "Bash: never use rm -rf /",
                        "similarity": 0.91,
                        "context": json.dumps({"correction": "use trash CLI"}),
                    }
                ],
            ),
        ):
            out = json.loads(
                simba.hooks.permission_request.main(
                    {
                        "tool_name": "Bash",
                        "tool_input": {"command": "rm -rf /"},
                    }
                )
            )
        decision = out["hookSpecificOutput"]["decision"]
        assert decision["behavior"] == "deny"
        assert "rm -rf" in decision["message"]
        assert "trash CLI" in decision["message"]

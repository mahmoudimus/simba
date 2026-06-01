"""Tests that the tool-rule matcher scopes by resolved project id (no leak)."""

from __future__ import annotations

import simba.db
import simba.hooks.config
import simba.hooks.pre_tool_use as ptu


def _default_cfg():
    return simba.hooks.config.HooksConfig()


class TestCheckToolRulesScoping:
    def test_recall_scoped_to_resolved_project_id(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_recall(query, project_path=None, **kw):
            captured["project_path"] = project_path
            captured["filters"] = kw.get("filters")
            return []

        monkeypatch.setattr(ptu, "_hooks_cfg", _default_cfg)
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories", fake_recall
        )
        monkeypatch.setattr(
            simba.db, "resolve_project_id", lambda p=None: "RESOLVED-ID"
        )

        out = ptu._check_tool_rules("Bash", {"command": "pytest -q"}, "/some/cwd")
        assert out is None  # no memories returned
        # The recall was scoped to the resolved id, not the raw cwd path.
        assert captured["project_path"] == "RESOLVED-ID"
        assert captured["filters"] == {"types": ["TOOL_RULE"]}

    def test_other_project_rule_not_surfaced(self, monkeypatch) -> None:
        # The daemon (faked) scopes strictly: a rule tagged with another
        # project's id is invisible when recalling under this project's id.
        monkeypatch.setattr(ptu, "_hooks_cfg", _default_cfg)
        monkeypatch.setattr(
            simba.db, "resolve_project_id", lambda p=None: "simba-id"
        )

        def fake_recall(query, project_path=None, **kw):
            d810_rule = {
                "content": "Bash: ls: src/d810/foo: No such file",
                "context": "{}",
                "createdAt": "2026-05-30T00:00:00Z",
                "projectPath": "d810-id",
            }
            return [d810_rule] if project_path == "d810-id" else []

        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories", fake_recall
        )

        out = ptu._check_tool_rules("Bash", {"command": "ls src/simba"}, "/simba")
        assert out is None  # d810 rule never leaks into the simba-scoped recall

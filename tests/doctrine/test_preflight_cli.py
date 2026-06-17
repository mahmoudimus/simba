"""Tests for the `simba preflight <task>` CLI (spec 28 Phase C)."""

from __future__ import annotations

import pytest

import simba.__main__ as cli
import simba.guardian.preflight_flag as pf


@pytest.fixture
def _flag_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
    return tmp_path


class TestPreflightCli:
    def test_prints_brief_and_sets_flag(
        self, _flag_in_tmp, monkeypatch, capsys, tmp_path
    ):
        # Recall returns a TOOL_RULE for the project; doctrine recall returns one.
        def fake_recall(query, project_path=None, **kw):
            filters = kw.get("filters") or {}
            types = filters.get("types") or []
            if "TOOL_RULE" in types:
                return [{"type": "TOOL_RULE", "content": "never hand-edit schema"}]
            return [{"type": "PREFERENCE", "content": "use the regen script"}]

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", fake_recall)
        monkeypatch.setattr(
            "simba.redirect.store.load_rules", lambda cwd, *, project_path: []
        )

        rc = cli._cmd_preflight(["--session", "sess-A", "regenerate the init-schema"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "🦁☑" in out
        assert "regenerate the init-schema" in out
        assert "never hand-edit schema" in out
        # The per-turn flag is set so the PreToolUse gate sees the preflight.
        assert pf.preflight_ran("sess-A") is True

    def test_requires_task(self, _flag_in_tmp, capsys):
        rc = cli._cmd_preflight(["--session", "s"])
        assert rc == 1
        assert "task" in capsys.readouterr().err.lower()

    def test_no_session_still_prints(self, _flag_in_tmp, monkeypatch, capsys):
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **k: [],
        )
        monkeypatch.setattr(
            "simba.redirect.store.load_rules", lambda cwd, *, project_path: []
        )
        rc = cli._cmd_preflight(["do a thing"])
        assert rc == 0
        assert "🦁☑" in capsys.readouterr().out

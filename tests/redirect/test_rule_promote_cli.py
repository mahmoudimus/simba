"""CLI tests for `simba rule scan-arcs` and `simba rule promote`
(src/simba/__main__.py) -- the review surface for arc-derived redirect-rule
candidates (redirect/arc_promotion.py, redirect/candidates.py).

Mirrors how `simba memory supersession confirm/reject <id>` is tested
(tests/test_main_cli.py): DB-backed via `simba.db.connect(tmp_path)` plus
`monkeypatch.chdir`, not daemon HTTP.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.__main__ as cli
import simba.db
import simba.redirect.candidates as candidates
import simba.redirect.store as store
import simba.transcripts.arcs as arcs


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    monkeypatch.chdir(tmp_path)


def _seed_pattern_candidate(cwd: pathlib.Path) -> int:
    arcs.upsert_arc(
        session_source="sess-1",
        harness="claude-code",
        tool="Bash",
        signature="sig-rg-rn",
        error_head="error",
        failed_args_head="rg -rn pattern",
        fix_args_head="rg -n pattern",
        resolved=True,
        repeat_count=5,
        project_path=str(cwd),
        cwd=cwd,
    )
    cli._cmd_rule_scan_arcs([])
    return candidates.list_pending(cwd=cwd)[0].id


def _seed_program_candidate(cwd: pathlib.Path) -> int:
    arcs.upsert_arc(
        session_source="sess-1",
        harness="claude-code",
        tool="Bash",
        signature="sig-python3",
        error_head="error",
        failed_args_head="python3 -m pytest tests/",
        fix_args_head="uv run python -m pytest tests/",
        resolved=True,
        repeat_count=5,
        project_path=str(cwd),
        cwd=cwd,
    )
    cli._cmd_rule_scan_arcs([])
    return candidates.list_pending(cwd=cwd)[0].id


class TestScanArcsCli:
    def test_scan_arcs_reports_count(self, tmp_path: pathlib.Path, capsys) -> None:
        arcs.upsert_arc(
            session_source="sess-1",
            harness="claude-code",
            tool="Bash",
            signature="sig-rg-rn",
            error_head="error",
            failed_args_head="rg -rn pattern",
            fix_args_head="rg -n pattern",
            resolved=True,
            repeat_count=5,
            project_path=str(tmp_path),
            cwd=tmp_path,
        )
        rc = cli._cmd_rule_scan_arcs([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 candidate(s) (1 new)" in out

    def test_scan_arcs_no_arcs_is_zero(self, tmp_path: pathlib.Path, capsys) -> None:
        rc = cli._cmd_rule_scan_arcs([])
        assert rc == 0
        assert "0 candidate(s) (0 new)" in capsys.readouterr().out

    def test_rescan_reports_zero_new(self, tmp_path: pathlib.Path, capsys) -> None:
        arcs.upsert_arc(
            session_source="sess-1",
            harness="claude-code",
            tool="Bash",
            signature="sig-rg-rn",
            error_head="error",
            failed_args_head="rg -rn pattern",
            fix_args_head="rg -n pattern",
            resolved=True,
            repeat_count=5,
            project_path=str(tmp_path),
            cwd=tmp_path,
        )
        cli._cmd_rule_scan_arcs([])
        capsys.readouterr()
        rc = cli._cmd_rule_scan_arcs([])
        assert rc == 0
        assert "1 candidate(s) (0 new)" in capsys.readouterr().out


class TestRulePromoteListing:
    def test_lists_pending_with_diff_and_rule(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        _seed_pattern_candidate(tmp_path)
        rc = cli._cmd_rule_promote([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 pending rule candidate(s)" in out
        assert "- rg -rn pattern" in out
        assert "+ rg -n pattern" in out
        assert "rule: pattern" in out
        assert "x5 across 1 session(s)" in out
        assert "reason:" in out

    def test_empty_listing(self, tmp_path: pathlib.Path, capsys) -> None:
        rc = cli._cmd_rule_promote([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no pending rule candidates" in out
        assert "simba rule scan-arcs" in out

    def test_json_listing(self, tmp_path: pathlib.Path, capsys) -> None:
        _seed_pattern_candidate(tmp_path)
        capsys.readouterr()
        rc = cli._cmd_rule_promote(["--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload) == 1
        assert payload[0]["ruleKind"] == "pattern"
        assert payload[0]["evidenceCount"] == 5


class TestRulePromoteApprove:
    def test_approve_program_candidate_writes_deny_rule(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        cid = _seed_program_candidate(tmp_path)
        rc = cli._cmd_rule_promote([str(cid)])
        assert rc == 0
        out = capsys.readouterr().out
        assert f"approved candidate #{cid}" in out
        assert "python3" in out
        assert "uv run python" in out
        assert "mode=deny" in out

        project_id = simba.db.resolve_project_id(tmp_path)
        rules = store.load_rules(tmp_path, project_path=project_id)
        assert len(rules) == 1
        assert rules[0].program == "python3"
        assert rules[0].replacement == "uv run python"
        assert rules[0].mode == "deny"

        row = candidates.get(cid, cwd=tmp_path)
        assert row.status == "approved"
        assert row.decided_at

    def test_approve_pattern_candidate_writes_deny_rule(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        cid = _seed_pattern_candidate(tmp_path)
        rc = cli._cmd_rule_promote([str(cid)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mode=deny" in out
        assert "pattern-kind rules have no CLI graduation path" in out

        project_id = simba.db.resolve_project_id(tmp_path)
        rules = store.load_rules(tmp_path, project_path=project_id)
        assert len(rules) == 1
        assert rules[0].pattern
        assert rules[0].mode == "deny"

    def test_approve_unknown_id_errors(self, tmp_path: pathlib.Path, capsys) -> None:
        rc = cli._cmd_rule_promote(["999"])
        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_double_approve_errors(self, tmp_path: pathlib.Path, capsys) -> None:
        cid = _seed_program_candidate(tmp_path)
        assert cli._cmd_rule_promote([str(cid)]) == 0
        capsys.readouterr()
        rc = cli._cmd_rule_promote([str(cid)])
        assert rc == 1
        assert "already approved" in capsys.readouterr().err

    def test_approve_never_bulk(self, tmp_path: pathlib.Path) -> None:
        """There is deliberately no flag that approves more than one id at a
        time -- only a single positional id is ever accepted."""
        import inspect

        src = inspect.getsource(cli._cmd_rule_promote)
        assert "--all" not in src
        assert "--bulk" not in src


class TestRulePromoteReject:
    def test_reject_marks_rejected_no_rule_written(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        cid = _seed_pattern_candidate(tmp_path)
        rc = cli._cmd_rule_promote([str(cid), "--reject"])
        assert rc == 0
        assert f"rejected candidate #{cid}" in capsys.readouterr().out

        project_id = simba.db.resolve_project_id(tmp_path)
        assert store.load_rules(tmp_path, project_path=project_id) == []

        row = candidates.get(cid, cwd=tmp_path)
        assert row.status == "rejected"

    def test_rejected_candidate_never_resurfaces_on_rescan(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        cid = _seed_pattern_candidate(tmp_path)
        cli._cmd_rule_promote([str(cid), "--reject"])
        capsys.readouterr()

        rc = cli._cmd_rule_scan_arcs([])
        assert rc == 0
        assert "0 new" in capsys.readouterr().out
        assert candidates.list_pending(cwd=tmp_path) == []

    def test_double_reject_errors(self, tmp_path: pathlib.Path, capsys) -> None:
        cid = _seed_pattern_candidate(tmp_path)
        assert cli._cmd_rule_promote([str(cid), "--reject"]) == 0
        capsys.readouterr()
        rc = cli._cmd_rule_promote([str(cid), "--reject"])
        assert rc == 1
        assert "already rejected" in capsys.readouterr().err


class TestRuleRedirectModeFlag:
    def test_add_with_mode_rewrite(self, tmp_path: pathlib.Path, capsys) -> None:
        rc = cli._cmd_rule_redirect(
            ["add", "cargo", "soldr cargo", "--mode", "rewrite"]
        )
        assert rc == 0
        project_id = simba.db.resolve_project_id(tmp_path)
        rules = store.load_rules(tmp_path, project_path=project_id)
        assert rules[0].mode == "rewrite"

    def test_add_with_invalid_mode_errors(self, tmp_path: pathlib.Path, capsys) -> None:
        rc = cli._cmd_rule_redirect(["add", "cargo", "soldr cargo", "--mode", "bogus"])
        assert rc == 1
        assert "Error" in capsys.readouterr().err

    def test_add_without_mode_defaults_empty(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        rc = cli._cmd_rule_redirect(["add", "cargo", "soldr cargo"])
        assert rc == 0
        project_id = simba.db.resolve_project_id(tmp_path)
        rules = store.load_rules(tmp_path, project_path=project_id)
        assert rules[0].mode == ""

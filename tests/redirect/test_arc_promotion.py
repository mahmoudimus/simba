"""Tests for the arc -> redirect-rule promotion eligibility filter
(redirect/arc_promotion.py).

Deterministic, LLM-free: classifies a failed->fixed shell-command pair into
one of three mechanical shapes (flag-drop, token-replace, prefix-insert) or
rejects it, then aggregates resolved failure_arc rows into candidate
redirect rules once an evidence threshold is met and there's no
contradicting fix for the same failure signature.
"""

from __future__ import annotations

import pathlib
import types

import pytest

import simba.db
import simba.redirect.arc_promotion as arc_promotion
import simba.redirect.candidates as candidates
import simba.transcripts.arcs as arcs


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


# ── classify_transform: the three shapes ────────────────────────────────────


class TestClassifyFlagDrop:
    def test_combined_short_flag_replace(self) -> None:
        """rg -rn pattern -> rg -n pattern: equal token count, the single
        differing token is a combined short-flag bundle with 'r' dropped."""
        c = arc_promotion.classify_transform("rg -rn pattern", "rg -n pattern")
        assert c is not None
        assert c.shape == "flag-drop"
        assert c.rule_kind == "pattern"
        assert c.rule_pattern and c.rule_rewrite

    def test_combined_short_flag_pattern_actually_rewrites(self) -> None:
        """The derived pattern/rewrite must generalize to a fresh occurrence,
        not just the literal example command it was mined from."""
        import re

        c = arc_promotion.classify_transform("rg -rn pattern", "rg -n pattern")
        assert c is not None
        fresh = "rg -rn 'TODO' src/"
        fixed = re.sub(c.rule_pattern, c.rule_rewrite, fresh)
        assert fixed == "rg -n 'TODO' src/"

    def test_true_token_removal(self) -> None:
        """A whole flag token removed (not bundled): rg -r -n pat -> rg -n pat."""
        c = arc_promotion.classify_transform("rg -r -n pat", "rg -n pat")
        assert c is not None
        assert c.shape == "flag-drop"
        assert c.rule_kind == "pattern"

    def test_true_token_removal_pattern_actually_rewrites(self) -> None:
        import re

        c = arc_promotion.classify_transform("rg -r -n pat", "rg -n pat")
        assert c is not None
        fresh = "rg -r -n 'TODO' src/"
        fixed = re.sub(c.rule_pattern, c.rule_rewrite, fresh)
        assert "-r" not in fixed.split()


class TestClassifyTokenReplace:
    def test_vmmap_summarize_to_summary(self) -> None:
        c = arc_promotion.classify_transform("vmmap --summarize X", "vmmap -summary X")
        assert c is not None
        assert c.shape == "token-replace"
        assert c.rule_kind == "pattern"
        assert c.rule_pattern and c.rule_rewrite

    def test_token_replace_pattern_actually_rewrites(self) -> None:
        import re

        c = arc_promotion.classify_transform("vmmap --summarize X", "vmmap -summary X")
        assert c is not None
        fresh = "vmmap --summarize other-pid"
        fixed = re.sub(c.rule_pattern, c.rule_rewrite, fresh)
        assert fixed == "vmmap -summary other-pid"


class TestClassifyPrefixInsert:
    def test_python3_to_uv_run_python(self) -> None:
        c = arc_promotion.classify_transform(
            "python3 -m pytest tests/", "uv run python -m pytest tests/"
        )
        assert c is not None
        assert c.shape == "prefix-insert"
        assert c.rule_kind == "program"
        assert c.rule_program == "python3"
        assert c.rule_replacement == "uv run python"

    def test_pure_prepend_no_leading_token_change(self) -> None:
        c = arc_promotion.classify_transform("cargo build", "soldr cargo build")
        assert c is not None
        assert c.shape == "prefix-insert"
        assert c.rule_kind == "program"
        assert c.rule_program == "cargo"
        assert c.rule_replacement == "soldr cargo"


class TestClassifyRejects:
    def test_operand_change_rejected(self) -> None:
        """cd /a -> cd /b: the changed token is a value/path operand, not a
        flag and not the program position -- reject."""
        assert arc_promotion.classify_transform("cd /a", "cd /b") is None

    def test_multi_token_diff_rejected(self) -> None:
        c = arc_promotion.classify_transform(
            "rg -rn pattern src/", "rg --hidden -n other src/other"
        )
        assert c is None

    def test_no_op_rejected(self) -> None:
        assert arc_promotion.classify_transform("rg -n x", "rg -n x") is None

    def test_shlex_unparseable_failed_rejected(self) -> None:
        assert (
            arc_promotion.classify_transform("echo 'unterminated", "echo done") is None
        )

    def test_shlex_unparseable_fixed_rejected(self) -> None:
        assert (
            arc_promotion.classify_transform("echo done", "echo 'unterminated") is None
        )

    def test_grown_by_more_than_three_tokens_rejected(self) -> None:
        c = arc_promotion.classify_transform(
            "pytest tests/", "uv run --no-cache --python 3.13 pytest tests/"
        )
        assert c is None

    def test_shrunk_by_more_than_one_token_rejected(self) -> None:
        c = arc_promotion.classify_transform("rg -r -n -i pat", "rg pat")
        assert c is None


# ── build_candidates: eligibility, evidence, contradiction ─────────────────


def _arc(**overrides) -> types.SimpleNamespace:
    base = dict(
        session_source="sess-1",
        harness="claude-code",
        tool="Bash",
        signature="sig-rg-rn",
        error_head="error",
        failed_args_head="rg -rn pattern",
        fix_args_head="rg -n pattern",
        resolved=True,
        repeat_count=1,
        project_path="/repo",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class TestBuildCandidates:
    def test_below_evidence_threshold_rejected(self) -> None:
        rows = [_arc(repeat_count=1)]
        proposals = arc_promotion.build_candidates(rows, min_evidence=3)
        assert proposals == []

    def test_evidence_by_repeat_count_sum(self) -> None:
        rows = [_arc(repeat_count=3)]
        proposals = arc_promotion.build_candidates(rows, min_evidence=3)
        assert len(proposals) == 1
        assert proposals[0].evidence_count == 3
        assert proposals[0].session_count == 1

    def test_evidence_by_distinct_sessions(self) -> None:
        rows = [
            _arc(session_source="sess-1", repeat_count=1),
            _arc(session_source="sess-2", repeat_count=1),
        ]
        proposals = arc_promotion.build_candidates(rows, min_evidence=100)
        assert len(proposals) == 1
        assert proposals[0].session_count == 2

    def test_unresolved_arc_excluded(self) -> None:
        rows = [_arc(resolved=False, repeat_count=10)]
        assert arc_promotion.build_candidates(rows, min_evidence=1) == []

    def test_non_shell_tool_excluded(self) -> None:
        rows = [_arc(tool="Read", repeat_count=10)]
        assert arc_promotion.build_candidates(rows, min_evidence=1) == []

    def test_codex_exec_tool_included(self) -> None:
        rows = [_arc(tool="exec", repeat_count=3)]
        proposals = arc_promotion.build_candidates(rows, min_evidence=3)
        assert len(proposals) == 1

    def test_contradicted_signature_rejected(self) -> None:
        """Same signature, two different fix shapes -> context-dependent,
        reject the whole group even though evidence is otherwise plenty."""
        rows = [
            _arc(
                session_source="sess-1",
                failed_args_head="rg -rn pattern",
                fix_args_head="rg -n pattern",
                repeat_count=5,
            ),
            _arc(
                session_source="sess-2",
                failed_args_head="rg -rn pattern",
                fix_args_head="rg -l pattern",
                repeat_count=5,
            ),
        ]
        assert arc_promotion.build_candidates(rows, min_evidence=1) == []

    def test_unclassifiable_arc_ignored_not_fatal(self) -> None:
        """One arc in the group is shlex-unparseable; the other is a clean,
        consistent fix with enough evidence on its own -- the group still
        produces a candidate from the classifiable arcs."""
        rows = [
            _arc(
                session_source="sess-1",
                failed_args_head="echo 'unterminated",
                fix_args_head="echo done",
                repeat_count=1,
            ),
            _arc(
                session_source="sess-2",
                failed_args_head="rg -rn pattern",
                fix_args_head="rg -n pattern",
                repeat_count=5,
            ),
        ]
        proposals = arc_promotion.build_candidates(rows, min_evidence=3)
        assert len(proposals) == 1
        assert proposals[0].evidence_count == 5

    def test_cross_project_signature_yields_empty_project_path(self) -> None:
        rows = [
            _arc(session_source="sess-1", project_path="/repo-a", repeat_count=2),
            _arc(session_source="sess-2", project_path="/repo-b", repeat_count=2),
        ]
        proposals = arc_promotion.build_candidates(rows, min_evidence=1)
        assert len(proposals) == 1
        assert proposals[0].project_path == ""

    def test_same_project_signature_keeps_project_path(self) -> None:
        rows = [_arc(project_path="/repo-a", repeat_count=5)]
        proposals = arc_promotion.build_candidates(rows, min_evidence=1)
        assert proposals[0].project_path == "/repo-a"

    def test_reason_cites_evidence(self) -> None:
        rows = [_arc(repeat_count=7)]
        proposals = arc_promotion.build_candidates(rows, min_evidence=3)
        reason = proposals[0].reason
        assert "7" in reason
        assert "1" in reason  # session count


# ── scan(): DB-backed orchestration + idempotent re-scan ───────────────────


class TestScan:
    def test_scan_writes_pending_candidates(self, tmp_path: pathlib.Path) -> None:
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
            project_path="/repo",
            cwd=tmp_path,
        )
        summary = arc_promotion.scan(tmp_path, min_evidence=3)
        assert summary.total == 1
        assert summary.new == 1
        pending = candidates.list_pending(cwd=tmp_path)
        assert len(pending) == 1
        assert pending[0].signature == "sig-rg-rn"
        assert pending[0].status == "pending"

    def test_rescan_is_idempotent_no_duplicate(self, tmp_path: pathlib.Path) -> None:
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
            project_path="/repo",
            cwd=tmp_path,
        )
        arc_promotion.scan(tmp_path, min_evidence=3)
        summary2 = arc_promotion.scan(tmp_path, min_evidence=3)
        assert summary2.total == 1
        assert summary2.new == 0
        assert len(candidates.list_pending(cwd=tmp_path)) == 1

    def test_rescan_never_resurrects_rejected(self, tmp_path: pathlib.Path) -> None:
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
            project_path="/repo",
            cwd=tmp_path,
        )
        arc_promotion.scan(tmp_path, min_evidence=3)
        pending = candidates.list_pending(cwd=tmp_path)
        candidates.reject(pending[0].id, cwd=tmp_path)

        summary2 = arc_promotion.scan(tmp_path, min_evidence=3)
        assert summary2.new == 0
        assert candidates.list_pending(cwd=tmp_path) == []

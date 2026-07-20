"""Compact relay leg B: SessionStart(source="compact") re-injects the most
recently distilled failure->fix arcs (transcripts/arcs.py's failure_arc
sidecar) -- the summarizer that runs across a compaction boundary drops this
kind of specific, hard-won context, so this re-surfaces it directly to the
model.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.hooks.config as hooks_config
import simba.hooks.session_start as session_start
import simba.transcripts.arcs as arcs


@pytest.fixture
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the shared sqlite sidecar under ``tmp_path`` -- NOT autouse: the
    DB-error test below deliberately leaves ``simba.db.get_db_path`` real so
    a bogus, non-writable cwd produces a genuine connect() failure (mirrors
    ``tests/hooks/test_lifecycle_nudges.py``'s
    ``test_rule_candidates_inbox_line_fail_soft_on_bad_db``)."""
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _arc(**overrides) -> dict:
    base = dict(
        session_source="sess-1",
        harness="claude-code",
        tool="Bash",
        signature="sig-abc",
        error_head="boom: :LINE: in /PATH/foo.py",
        failed_args_head="pytest -x",
        fix_args_head="pytest -k test_foo",
        resolved=True,
        repeat_count=1,
        project_path="/repo",
    )
    base.update(overrides)
    return base


class TestConfigDefault:
    def test_compact_relay_arcs_k_default(self) -> None:
        assert hooks_config.HooksConfig().compact_relay_arcs_k == 5


class TestListRecentResolved:
    pytestmark = pytest.mark.usefixtures("_tmp_db")

    def test_returns_only_resolved_newest_first(self, tmp_path) -> None:
        arcs.upsert_arc(
            **_arc(signature="sig-1", resolved=True, now="2026-07-01T00:00:00Z")
        )
        arcs.upsert_arc(
            **_arc(signature="sig-2", resolved=False, now="2026-07-02T00:00:00Z")
        )
        arcs.upsert_arc(
            **_arc(signature="sig-3", resolved=True, now="2026-07-03T00:00:00Z")
        )

        rows = arcs.list_recent_resolved("/repo", 5)
        assert [r.signature for r in rows] == ["sig-3", "sig-1"]

    def test_respects_limit(self, tmp_path) -> None:
        for i in range(10):
            arcs.upsert_arc(
                **_arc(signature=f"sig-{i}", now=f"2026-07-{i + 1:02d}T00:00:00Z")
            )

        rows = arcs.list_recent_resolved("/repo", 3)
        assert len(rows) == 3
        assert [r.signature for r in rows] == ["sig-9", "sig-8", "sig-7"]

    def test_scoped_to_project_path(self, tmp_path) -> None:
        arcs.upsert_arc(**_arc(signature="sig-a", project_path="/repoA"))
        arcs.upsert_arc(**_arc(signature="sig-b", project_path="/repoB"))

        rows = arcs.list_recent_resolved("/repoA", 5)
        assert [r.signature for r in rows] == ["sig-a"]

    def test_empty_when_no_arcs(self, tmp_path) -> None:
        assert arcs.list_recent_resolved("/repo", 5) == []


class TestCompactRelayArcBlock:
    def test_seeds_more_than_k_mixed_resolved_returns_k_newest_resolved(
        self, tmp_path, _tmp_db
    ) -> None:
        # 7 resolved + 2 unresolved arcs, newest last -> only the 5 (default k)
        # newest RESOLVED arcs should surface, newest first.
        for i in range(7):
            arcs.upsert_arc(
                **_arc(
                    signature=f"resolved-{i}",
                    resolved=True,
                    fix_args_head=f"fix-{i}",
                    now=f"2026-07-01T00:00:{i:02d}Z",
                )
            )
        arcs.upsert_arc(
            **_arc(
                signature="unresolved-a",
                resolved=False,
                fix_args_head=None,
                now="2026-07-01T00:00:08Z",
            )
        )
        arcs.upsert_arc(
            **_arc(
                signature="unresolved-b",
                resolved=False,
                fix_args_head=None,
                now="2026-07-01T00:00:09Z",
            )
        )

        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        block = session_start._compact_relay_arc_block("/repo", cfg)

        assert "Recent failure->fix arcs" in block
        assert "unresolved-a" not in block
        assert "unresolved-b" not in block
        for i in (6, 5, 4, 3, 2):
            assert f"resolved-{i}" in block
        for i in (1, 0):
            assert f"resolved-{i}" not in block

        # newest-first ordering
        assert block.index("resolved-6") < block.index("resolved-2")

    def test_zero_k_disables_block(self, tmp_path, _tmp_db) -> None:
        arcs.upsert_arc(**_arc())
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=0)
        assert session_start._compact_relay_arc_block("/repo", cfg) == ""

    def test_no_cwd_yields_no_block(self, tmp_path, _tmp_db) -> None:
        arcs.upsert_arc(**_arc())
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        assert session_start._compact_relay_arc_block("", cfg) == ""

    def test_no_arcs_yields_no_block(self, tmp_path, _tmp_db) -> None:
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        assert session_start._compact_relay_arc_block("/repo", cfg) == ""

    def test_each_line_capped_around_200_chars(self, tmp_path, _tmp_db) -> None:
        arcs.upsert_arc(
            **_arc(
                signature="sig-long",
                error_head="x" * 500,
                fix_args_head="y" * 500,
                resolved=True,
            )
        )
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        block = session_start._compact_relay_arc_block("/repo", cfg)
        lines = block.split("\n")
        arc_lines = [line for line in lines if line.startswith("- ")]
        assert arc_lines
        for line in arc_lines:
            assert len(line) <= 210

    def test_whole_block_hard_capped_around_1500_chars(self, tmp_path, _tmp_db) -> None:
        for i in range(20):
            arcs.upsert_arc(
                **_arc(
                    signature=f"sig-{i}",
                    error_head="e" * 190,
                    fix_args_head="f" * 190,
                    resolved=True,
                    now=f"2026-07-01T00:{i:02d}:00Z",
                )
            )
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=20)
        block = session_start._compact_relay_arc_block("/repo", cfg)
        assert len(block) <= 1600  # ~1500 + a little slack for the truncation marker

    def test_db_error_fails_open_no_block_no_exception(self, tmp_path) -> None:
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        bogus_cwd = str(pathlib.Path("/nonexistent/not-a-real-path-xyz"))
        assert session_start._compact_relay_arc_block(bogus_cwd, cfg) == ""


class TestSessionStartCompactBranchIntegration:
    pytestmark = pytest.mark.usefixtures("_tmp_db")

    def test_compact_source_injects_arc_block(self, tmp_path, monkeypatch) -> None:
        import unittest.mock

        arcs.upsert_arc(**_arc(project_path=str(tmp_path)))

        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = session_start.run(
                {"cwd": str(tmp_path), "source": "compact", "session_id": "s1"}
            )

        assert "Recent failure->fix arcs" in result.additional_context

    def test_non_compact_source_does_not_inject_arc_block(
        self, tmp_path, monkeypatch
    ) -> None:
        import unittest.mock

        arcs.upsert_arc(**_arc(project_path=str(tmp_path)))

        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = session_start.run({"cwd": str(tmp_path), "session_id": "s1"})

        assert "Recent failure->fix arcs" not in result.additional_context

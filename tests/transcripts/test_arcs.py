"""Tests for the failure-arc sidecar table (transcripts/arcs.py).

Cross-session mining store for failure->fix arcs surfaced by the transcript
distiller (transcripts/distill.py). Mirrors episodes/watermark.py's peewee
sidecar pattern: `simba.db.BaseModel` + `register_model` + `simba.db.connect`.
Upserts are keyed on (session_source, signature) so re-distilling the same
session never duplicates rows.
"""

from __future__ import annotations

import pathlib

import pytest

import simba.db
import simba.transcripts.arcs as arcs


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _arc(**overrides) -> dict:
    base = dict(
        session_source="sess-1",
        harness="codex",
        tool="exec",
        signature="sig-abc",
        error_head="boom: :LINE: in /PATH/foo.py",
        failed_args_head="pytest -x",
        fix_args_head=None,
        resolved=False,
        repeat_count=1,
        project_path="/repo",
    )
    base.update(overrides)
    return base


def test_upsert_then_list_roundtrip() -> None:
    arcs.upsert_arc(**_arc())
    rows = arcs.list_for_session("sess-1")
    assert len(rows) == 1
    row = rows[0]
    assert row.tool == "exec"
    assert row.signature == "sig-abc"
    assert row.resolved is False
    assert row.repeat_count == 1
    assert row.fix_args_head is None


def test_upsert_is_idempotent_keyed_on_session_and_signature() -> None:
    """Re-distilling the same session must not duplicate rows -- a second
    upsert for the same (session_source, signature) replaces the row."""
    arcs.upsert_arc(**_arc(repeat_count=1))
    arcs.upsert_arc(**_arc(repeat_count=50, resolved=True, fix_args_head="pytest"))

    rows = arcs.list_for_session("sess-1")
    assert len(rows) == 1
    assert rows[0].repeat_count == 50
    assert rows[0].resolved is True
    assert rows[0].fix_args_head == "pytest"


def test_distinct_signatures_produce_distinct_rows() -> None:
    arcs.upsert_arc(**_arc(signature="sig-a"))
    arcs.upsert_arc(**_arc(signature="sig-b"))
    rows = arcs.list_for_session("sess-1")
    assert {r.signature for r in rows} == {"sig-a", "sig-b"}


def test_arcs_are_scoped_per_session_source() -> None:
    arcs.upsert_arc(**_arc(session_source="sess-1"))
    arcs.upsert_arc(**_arc(session_source="sess-2"))
    assert len(arcs.list_for_session("sess-1")) == 1
    assert len(arcs.list_for_session("sess-2")) == 1
    assert len(arcs.list_for_session("sess-3")) == 0


def test_resolved_arc_persists_fix_fields() -> None:
    arcs.upsert_arc(
        **_arc(
            resolved=True,
            fix_args_head="pytest -k test_foo",
            repeat_count=3,
        )
    )
    row = arcs.list_for_session("sess-1")[0]
    assert row.resolved is True
    assert row.fix_args_head == "pytest -k test_foo"
    assert row.repeat_count == 3

"""Failure-arc sidecar table -- cross-session mining store.

The transcript distiller (``transcripts/distill.py``) treats a tool call
that fails, followed later by the same tool succeeding, as the single
highest-priority learning signal in a transcript (a "failure -> fix arc").
Arcs are written into ``transcript.md`` for the per-session learning-extraction
flow, AND upserted here so they can be mined across sessions (e.g. "which
failure signatures recur across the whole project, resolved or not").

Mirrors ``episodes/watermark.py``'s sidecar pattern: a peewee ``BaseModel`` +
``register_model`` + ``simba.db.connect`` -- own module, own table, sharing
the project's ``.simba/simba.db``. Upserts are keyed on
``(session_source, signature)`` (delete-then-create, same idiom as
``redirect/store.py``'s ``add``) so re-distilling the same session is
idempotent: never duplicates a row, just refreshes it (repeat_count/resolved/
fix fields) to the latest single-pass computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db
import simba.workflow._time as _time

if TYPE_CHECKING:
    import pathlib


class FailureArc(simba.db.BaseModel):
    session_source = pw.CharField()
    harness = pw.CharField()  # "claude-code" | "codex"
    tool = pw.CharField()
    signature = pw.CharField()  # normalized error signature (see distill.py)
    error_head = pw.TextField(default="")
    failed_args_head = pw.TextField(default="")
    fix_args_head = pw.TextField(null=True)
    resolved = pw.BooleanField(default=False)
    repeat_count = pw.IntegerField(default=1)
    project_path = pw.CharField(default="")
    created_at = pw.CharField(default="")

    class Meta:
        table_name = "failure_arc"
        indexes = ((("session_source", "signature"), True),)  # UNIQUE


simba.db.register_model(FailureArc)


def upsert_arc(
    session_source: str,
    harness: str,
    tool: str,
    signature: str,
    error_head: str,
    failed_args_head: str,
    *,
    fix_args_head: str | None = None,
    resolved: bool = False,
    repeat_count: int = 1,
    project_path: str = "",
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> None:
    """Insert or replace the arc for ``(session_source, signature)``.

    Delete-then-create (mirrors ``redirect/store.py``'s ``add``) rather than
    an in-place UPDATE: a re-distilled session recomputes every field from
    scratch in one pass, so "replace" is the correct semantics -- there is
    no partial/incremental state to preserve across runs.
    """
    created_at = _time.resolve(now)
    with simba.db.connect(cwd):
        FailureArc.delete().where(
            (FailureArc.session_source == session_source)
            & (FailureArc.signature == signature)
        ).execute()
        FailureArc.create(
            session_source=session_source,
            harness=harness,
            tool=tool,
            signature=signature,
            error_head=error_head,
            failed_args_head=failed_args_head,
            fix_args_head=fix_args_head,
            resolved=resolved,
            repeat_count=repeat_count,
            project_path=project_path,
            created_at=created_at,
        )


def list_for_session(
    session_source: str, *, cwd: pathlib.Path | None = None
) -> list[FailureArc]:
    """Return all arcs recorded for ``session_source`` (any order)."""
    with simba.db.connect(cwd):
        return list(
            FailureArc.select().where(FailureArc.session_source == session_source)
        )


def list_all(*, cwd: pathlib.Path | None = None) -> list[FailureArc]:
    """Return every arc recorded (any session, any order).

    Used by the redirect rule-candidate scan (``redirect/arc_promotion.py``),
    which mines cross-session failure->fix patterns -- unlike
    ``list_for_session``, this deliberately spans every session so recurring
    signatures across the whole project's history are visible in one pass.
    """
    with simba.db.connect(cwd):
        return list(FailureArc.select())

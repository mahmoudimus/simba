"""Rule-candidate sidecar table: reviewable redirect-rule proposals mined
from failure->fix arcs by ``redirect/arc_promotion.py``.

Mirrors ``transcripts/arcs.py``'s peewee sidecar pattern: a
``simba.db.BaseModel`` + ``register_model`` + ``simba.db.connect``, own
table, sharing the project's ``.simba/simba.db``. A candidate is a proposal
only -- nothing here ever writes to the redirect DB store
(``redirect/store.py``) or activates a rule; that happens exactly once, on
explicit human approval, in ``approve()`` below (called by
``simba rule promote <id>``).

Upserts are keyed on ``signature`` (unique): a re-scan refreshes a still-
PENDING candidate's evidence/fields in place, but a candidate that has
already been decided (approved or rejected) is left untouched -- re-scanning
never resurrects a rejected candidate, never duplicates a pending one, and
never silently mutates an already-approved one out from under the rule it
wrote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db
import simba.workflow._time as _time

if TYPE_CHECKING:
    import pathlib

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"


class RuleCandidate(simba.db.BaseModel):
    signature = pw.CharField(unique=True)
    tool = pw.CharField()
    failed_example = pw.TextField(default="")
    fixed_example = pw.TextField(default="")
    rule_kind = pw.CharField()  # "program" | "pattern"
    rule_program = pw.CharField(default="")
    rule_replacement = pw.CharField(default="")
    rule_pattern = pw.TextField(default="")
    rule_rewrite = pw.TextField(default="")
    reason = pw.TextField(default="")
    evidence_count = pw.IntegerField(default=0)
    session_count = pw.IntegerField(default=0)
    project_path = pw.CharField(default="")  # "" == cross-project
    status = pw.CharField(default=STATUS_PENDING)
    created_at = pw.CharField(default="")
    decided_at = pw.CharField(null=True)

    class Meta:
        table_name = "rule_candidate"


simba.db.register_model(RuleCandidate)


def upsert_candidate(
    *,
    signature: str,
    tool: str,
    failed_example: str,
    fixed_example: str,
    rule_kind: str,
    reason: str,
    evidence_count: int,
    session_count: int,
    rule_program: str = "",
    rule_replacement: str = "",
    rule_pattern: str = "",
    rule_rewrite: str = "",
    project_path: str = "",
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> str:
    """Insert a new pending candidate for ``signature``, or refresh an
    existing PENDING one's fields/evidence in place.

    Returns ``"new"``, ``"updated"``, or ``"skipped"`` (already approved or
    rejected -- left untouched). Never raises on a decided candidate; the
    caller (``arc_promotion.scan``) just tallies the outcome.
    """
    created_at = _time.resolve(now)
    with simba.db.connect(cwd):
        existing = RuleCandidate.get_or_none(RuleCandidate.signature == signature)
        if existing is None:
            RuleCandidate.create(
                signature=signature,
                tool=tool,
                failed_example=failed_example,
                fixed_example=fixed_example,
                rule_kind=rule_kind,
                rule_program=rule_program,
                rule_replacement=rule_replacement,
                rule_pattern=rule_pattern,
                rule_rewrite=rule_rewrite,
                reason=reason,
                evidence_count=evidence_count,
                session_count=session_count,
                project_path=project_path,
                status=STATUS_PENDING,
                created_at=created_at,
            )
            return "new"
        if existing.status != STATUS_PENDING:
            return "skipped"
        existing.tool = tool
        existing.failed_example = failed_example
        existing.fixed_example = fixed_example
        existing.rule_kind = rule_kind
        existing.rule_program = rule_program
        existing.rule_replacement = rule_replacement
        existing.rule_pattern = rule_pattern
        existing.rule_rewrite = rule_rewrite
        existing.reason = reason
        existing.evidence_count = evidence_count
        existing.session_count = session_count
        existing.project_path = project_path
        existing.save()
        return "updated"


def list_pending(*, cwd: pathlib.Path | None = None) -> list[RuleCandidate]:
    """Pending candidates, oldest first (stable ids for the review CLI)."""
    with simba.db.connect(cwd):
        return list(
            RuleCandidate.select()
            .where(RuleCandidate.status == STATUS_PENDING)
            .order_by(RuleCandidate.id)
        )


def count_pending(*, cwd: pathlib.Path | None = None) -> int:
    with simba.db.connect(cwd):
        return (
            RuleCandidate.select().where(RuleCandidate.status == STATUS_PENDING).count()
        )


def get(candidate_id: int, *, cwd: pathlib.Path | None = None) -> RuleCandidate | None:
    with simba.db.connect(cwd):
        return RuleCandidate.get_or_none(RuleCandidate.id == candidate_id)


def approve(
    candidate_id: int, *, project_path: str, cwd: pathlib.Path | None = None
) -> RuleCandidate:
    """Approve a pending candidate: write its derived rule to the redirect
    DB store (``redirect/store.py``) in DENY mode, then mark approved.

    DENY, never rewrite: a wrong auto-rewrite silently corrupts a command,
    while a wrong deny just blocks it (self-correcting -- the human sees the
    denial and can reject the rule). Graduating a specific rule to
    ``mode="rewrite"`` is a deliberate later action via
    ``simba rule redirect`` (program rules) or a direct edit for pattern
    rules -- never automatic here.

    Raises ``KeyError`` if the id doesn't exist, ``ValueError`` if it isn't
    pending (already decided -- approval is a one-way, one-time action).
    """
    import simba.redirect.store as redirect_store

    with simba.db.connect(cwd):
        row = RuleCandidate.get_or_none(RuleCandidate.id == candidate_id)
        if row is None:
            raise KeyError(f"no rule candidate #{candidate_id}")
        if row.status != STATUS_PENDING:
            raise ValueError(f"candidate #{candidate_id} already {row.status}")

        if row.rule_kind == "program":
            redirect_store.add(
                row.rule_program,
                row.rule_replacement,
                reason=row.reason,
                mode="deny",
                project_path=project_path,
                cwd=cwd,
            )
        else:
            redirect_store.add_pattern(
                row.rule_pattern,
                row.rule_rewrite,
                reason=row.reason,
                mode="deny",
                project_path=project_path,
                cwd=cwd,
            )

        row.status = STATUS_APPROVED
        row.decided_at = _time.resolve(None)
        row.save()
    return row


def reject(candidate_id: int, *, cwd: pathlib.Path | None = None) -> RuleCandidate:
    """Reject a pending candidate. Never writes to the redirect store.

    Raises ``KeyError`` if the id doesn't exist, ``ValueError`` if it isn't
    pending (already decided).
    """
    with simba.db.connect(cwd):
        row = RuleCandidate.get_or_none(RuleCandidate.id == candidate_id)
        if row is None:
            raise KeyError(f"no rule candidate #{candidate_id}")
        if row.status != STATUS_PENDING:
            raise ValueError(f"candidate #{candidate_id} already {row.status}")
        row.status = STATUS_REJECTED
        row.decided_at = _time.resolve(None)
        row.save()
    return row

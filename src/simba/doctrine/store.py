"""Doctrine-triggers store (spec 28) — a CLI-managed sqlite table.

Each row is a high-value doctrine/gate that carries trigger phrases + their
precomputed embeddings (so the hot-path intent match is a pure cosine, no
re-embed), a risk-tier flag (mandate preflight), and the applicable
TOOL_RULEs/redirects. Project-scoped, like the redirect store.

The embeddings are stored as JSON in a column (append-only doesn't apply — this
is a rebuildable CLI-managed index, same class as ``redirect_rules``).
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db

if TYPE_CHECKING:
    import pathlib


@dataclasses.dataclass
class Doctrine:
    """One doctrine entry: the guidance + its intent triggers + metadata."""

    id: str
    doctrine: str
    triggers: list[str]
    trigger_embeddings: list[list[float]]
    risk_tier: bool = False
    applicable_rules: list[str] = dataclasses.field(default_factory=list)
    project_path: str = ""


class DoctrineRow(simba.db.BaseModel):
    doctrine_id = pw.CharField(null=True)
    doctrine = pw.TextField(null=True)
    triggers = pw.TextField(null=True)  # JSON list[str]
    trigger_embeddings = pw.TextField(null=True)  # JSON list[list[float]]
    risk_tier = pw.BooleanField(default=False)
    applicable_rules = pw.TextField(null=True)  # JSON list[str]
    project_path = pw.CharField(null=True)

    class Meta:
        table_name = "doctrine_triggers"
        primary_key = False  # rowid table


simba.db.register_model(DoctrineRow)


def _loads(raw: str | None, default: object) -> object:
    try:
        return json.loads(raw) if raw else default
    except (json.JSONDecodeError, TypeError):
        return default


def _row_to_doctrine(r: DoctrineRow) -> Doctrine:
    return Doctrine(
        id=r.doctrine_id or "",
        doctrine=r.doctrine or "",
        triggers=list(_loads(r.triggers, [])),
        trigger_embeddings=[list(e) for e in _loads(r.trigger_embeddings, [])],
        risk_tier=bool(r.risk_tier),
        applicable_rules=list(_loads(r.applicable_rules, [])),
        project_path=r.project_path or "",
    )


def add(
    *,
    doctrine: str,
    triggers: list[str],
    trigger_embeddings: list[list[float]],
    risk_tier: bool = False,
    applicable_rules: list[str] | None = None,
    project_path: str,
    cwd: pathlib.Path | None = None,
) -> str:
    """Insert a doctrine entry; returns its generated id."""
    doctrine_id = f"doc_{uuid.uuid4().hex[:8]}"
    with simba.db.connect(cwd):
        DoctrineRow.create(
            doctrine_id=doctrine_id,
            doctrine=doctrine,
            triggers=json.dumps(list(triggers)),
            trigger_embeddings=json.dumps([list(e) for e in trigger_embeddings]),
            risk_tier=bool(risk_tier),
            applicable_rules=json.dumps(list(applicable_rules or [])),
            project_path=project_path,
        )
    return doctrine_id


def list_doctrines(
    *, project_path: str, cwd: pathlib.Path | None = None
) -> list[Doctrine]:
    """Return the doctrine entries scoped to ``project_path``."""
    with simba.db.connect(cwd):
        rows = list(
            DoctrineRow.select().where(DoctrineRow.project_path == project_path)
        )
    return [_row_to_doctrine(r) for r in rows]


def remove(
    doctrine_id: str, *, project_path: str, cwd: pathlib.Path | None = None
) -> int:
    """Delete a doctrine entry by id (project-scoped); returns the count deleted."""
    with simba.db.connect(cwd):
        return (
            DoctrineRow.delete()
            .where(
                (DoctrineRow.doctrine_id == doctrine_id)
                & (DoctrineRow.project_path == project_path)
            )
            .execute()
        )

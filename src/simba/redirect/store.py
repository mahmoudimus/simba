"""Redirect rule storage: a repo TOML file + a CLI-managed DB table, merged.

Two sources so rules can be either version-controlled with the project
(`.simba/redirects.toml`) or managed via `simba rule redirect …` (the
``redirect_rules`` table in simba.db). ``load_rules`` merges both, project-scoped.
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db
from simba.redirect.rules import RedirectRule

if TYPE_CHECKING:
    import pathlib

TOML_RELPATH = ".simba/redirects.toml"


class RedirectRow(simba.db.BaseModel):
    program = pw.CharField(null=True)
    replacement = pw.CharField(null=True)
    reason = pw.CharField(null=True)
    project_path = pw.CharField(null=True)

    class Meta:
        table_name = "redirect_rules"
        primary_key = False  # rowid table
        indexes = ((("program", "project_path"), True),)  # UNIQUE


simba.db.register_model(RedirectRow)


def load_toml(path: pathlib.Path) -> list[RedirectRule]:
    """Parse a redirects TOML file ([[redirect]] tables). Missing file -> []."""
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []
    rules = []
    for entry in data.get("redirect", []):
        reason = (entry.get("reason") or "").strip()
        pat = (entry.get("pattern") or "").strip()
        if pat:  # pattern (regex) rule — flag-level fixes
            rules.append(
                RedirectRule(
                    pattern=pat,
                    rewrite=(entry.get("rewrite") or "").strip(),
                    reason=reason,
                    source="toml",
                )
            )
            continue
        prog = (entry.get("program") or "").strip()
        repl = (entry.get("replacement") or "").strip()
        if prog and repl:
            rules.append(
                RedirectRule(
                    program=prog,
                    replacement=repl,
                    reason=reason,
                    source="toml",
                )
            )
    return rules


def add(
    program: str,
    replacement: str,
    *,
    reason: str = "",
    project_path: str,
    cwd: pathlib.Path | None = None,
) -> None:
    """Add or replace a redirect rule in the DB store."""
    with simba.db.connect(cwd):
        RedirectRow.delete().where(
            (RedirectRow.program == program)
            & (RedirectRow.project_path == project_path)
        ).execute()
        RedirectRow.create(
            program=program,
            replacement=replacement,
            reason=reason,
            project_path=project_path,
        )


def remove(program: str, *, project_path: str, cwd: pathlib.Path | None = None) -> int:
    """Remove a rule by program; returns the number deleted."""
    with simba.db.connect(cwd):
        return (
            RedirectRow.delete()
            .where(
                (RedirectRow.program == program)
                & (RedirectRow.project_path == project_path)
            )
            .execute()
        )


def list_rules(
    *, project_path: str, cwd: pathlib.Path | None = None
) -> list[RedirectRule]:
    """Return the DB-store rules for a project."""
    with simba.db.connect(cwd):
        rows = list(
            RedirectRow.select().where(RedirectRow.project_path == project_path)
        )
    return [
        RedirectRule(
            program=r.program,
            replacement=r.replacement,
            reason=r.reason or "",
            source="store",
        )
        for r in rows
    ]


def load_rules(cwd: pathlib.Path, *, project_path: str) -> list[RedirectRule]:
    """Merge TOML-file rules and DB-store rules for this project."""
    toml_rules = load_toml(cwd / TOML_RELPATH)
    store_rules = list_rules(project_path=project_path, cwd=cwd)
    return [*toml_rules, *store_rules]

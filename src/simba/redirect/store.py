"""Redirect rule storage: a repo TOML file + a CLI-managed DB table, merged.

Two sources so rules can be either version-controlled with the project
(`.simba/redirects.toml`) or managed via `simba rule redirect …` (the
``redirect_rules`` table in simba.db). ``load_rules`` merges both, project-scoped.
"""

from __future__ import annotations

import contextlib
import sqlite3
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
    # Pattern-kind rule fields (flag-level fixes -- see rules.RedirectRule's
    # pattern/rewrite) plus a per-rule mode override, added for arc-derived
    # candidates (redirect/arc_promotion.py, redirect/candidates.py). NULL
    # for plain program rules. See ``_migrate_redirect_rules_columns`` below
    # for how a pre-existing table gets these columns added.
    pattern = pw.TextField(null=True)
    rewrite = pw.TextField(null=True)
    mode = pw.CharField(null=True)

    class Meta:
        table_name = "redirect_rules"
        primary_key = False  # rowid table
        indexes = ((("program", "project_path"), True),)  # UNIQUE


simba.db.register_model(RedirectRow)


def _migrate_redirect_rules_columns(conn: sqlite3.Connection) -> None:
    """Additive migration: add pattern/rewrite/mode to a pre-existing
    ``redirect_rules`` table.

    New databases get these columns straight from the ``RedirectRow`` model
    when ``create_tables`` runs (right after schema initializers, see
    ``simba.db.connect``); this backfills databases created before those
    fields existed. Idempotent (checks ``PRAGMA table_info`` first) and
    guarded on the table already existing -- a brand-new DB hasn't created it
    yet when schema initializers run, so there is nothing to migrate.
    Mirrors ``neuron/schema.py``'s ``_migrate_dormant_flag``.
    """
    has_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='redirect_rules'"
    ).fetchone()
    if not has_table:
        return
    cols = {row[1] for row in conn.execute("PRAGMA table_info(redirect_rules)")}
    for name in ("pattern", "rewrite", "mode"):
        if name not in cols:
            with contextlib.suppress(sqlite3.OperationalError):
                conn.execute(f"ALTER TABLE redirect_rules ADD COLUMN {name} TEXT")


simba.db.register_schema(_migrate_redirect_rules_columns)


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
    mode: str = "",
    project_path: str,
    cwd: pathlib.Path | None = None,
) -> None:
    """Add or replace a program redirect rule in the DB store.

    Delete-then-create keyed on ``(program, project_path)`` -- a project has
    at most one active redirect per program, so re-adding the same program
    is how a rule is edited in place (e.g. graduating an arc-approved
    deny-mode candidate to ``mode="rewrite"`` via
    ``simba rule redirect add <program> <replacement> --mode rewrite``).
    """
    with simba.db.connect(cwd):
        RedirectRow.delete().where(
            (RedirectRow.program == program)
            & (RedirectRow.project_path == project_path)
        ).execute()
        RedirectRow.create(
            program=program,
            replacement=replacement,
            reason=reason,
            mode=mode,
            project_path=project_path,
        )


def add_pattern(
    pattern: str,
    rewrite: str,
    *,
    reason: str = "",
    mode: str = "",
    project_path: str,
    cwd: pathlib.Path | None = None,
) -> None:
    """Add a pattern-kind redirect rule to the DB store.

    Unlike ``add()``, this is a plain insert (no delete-then-replace): a
    regex/rewrite pair has no natural single-per-project key the way a
    program name does, so there's nothing to identify-and-replace by. The
    ``rule_candidate`` approval flow (redirect/candidates.py) only ever
    calls this once per candidate (a candidate can't be re-approved once
    decided), so duplicate rows from repeated calls are not a concern here.
    """
    with simba.db.connect(cwd):
        RedirectRow.create(
            program=None,
            replacement=None,
            pattern=pattern,
            rewrite=rewrite,
            reason=reason,
            mode=mode,
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
            program=r.program or "",
            replacement=r.replacement or "",
            pattern=r.pattern or "",
            rewrite=r.rewrite or "",
            mode=r.mode or "",
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

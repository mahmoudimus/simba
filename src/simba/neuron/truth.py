"""Truth database — record and query proven facts via MCP tools."""

from __future__ import annotations

import sqlite3

import simba.db


def _init_truth_schema(conn: sqlite3.Connection) -> None:
    """Create the ``proven_facts`` table, scoped by ``project_path``.

    Facts are scoped per-project to prevent cross-project leakage.  An older
    unscoped table (no ``project_path`` column) is sidelined once to
    ``proven_facts_legacy`` — its rows are preserved for later re-analysis but
    no longer injected — and a fresh scoped table is created in its place.
    """
    info = conn.execute("PRAGMA table_info(proven_facts)").fetchall()
    cols = [row[1] for row in info]
    if cols and "project_path" not in cols:
        has_legacy = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='proven_facts_legacy'"
        ).fetchone()
        if has_legacy:
            conn.execute("DROP TABLE proven_facts")
        else:
            conn.execute("ALTER TABLE proven_facts RENAME TO proven_facts_legacy")

    conn.execute(
        """CREATE TABLE IF NOT EXISTS proven_facts
           (subject TEXT, predicate TEXT, object TEXT, proof TEXT,
           project_path TEXT NOT NULL,
           UNIQUE(subject, predicate, object, project_path))"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_proven_facts_project "
        "ON proven_facts(project_path)"
    )


simba.db.register_schema(_init_truth_schema)


def truth_add(
    subject: str,
    predicate: str,
    object: str,
    proof: str,
    project_path: str | None = None,
) -> str:
    """Record a proven fact into the Truth DB, scoped to a project.

    Use this ONLY when a verifier (Z3/Datalog) has proven a hypothesis.
    ``project_path`` defaults to the current repo's stable project id.
    """
    if project_path is None:
        project_path = simba.db.resolve_project_id()
    with simba.db.get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO proven_facts VALUES (?, ?, ?, ?, ?)",
                (subject, predicate, object, proof, project_path),
            )
            conn.commit()
            return f"Fact recorded: {subject} {predicate} {object}"
        except sqlite3.IntegrityError:
            return f"Fact already exists: {subject} {predicate} {object}"
        except Exception as exc:
            return f"Database Error: {exc}"


def truth_query(
    subject: str | None = None,
    predicate: str | None = None,
    project_path: str | None = None,
) -> str:
    """Query the Truth DB for existing proven facts.

    Use this BEFORE assuming capabilities or behavior about the codebase.
    Pass ``project_path`` to scope results to one project; omit it to search
    across all projects.
    """
    with simba.db.get_db() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM proven_facts WHERE 1=1"
        params: list[str] = []

        if subject:
            query += " AND subject=?"
            params.append(subject)
        if predicate:
            query += " AND predicate=?"
            params.append(predicate)
        if project_path:
            query += " AND project_path=?"
            params.append(project_path)

        rows = cursor.execute(query, params).fetchall()

        if not rows:
            return "No facts found matching criteria."

        return "\n".join(f"FACT: {r[0]} {r[1]} {r[2]} (Proof: {r[3]})" for r in rows)

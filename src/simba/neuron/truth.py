"""Truth database — record and query proven facts via MCP tools."""

from __future__ import annotations

import sqlite3

import simba.db


def _init_truth_schema(conn: sqlite3.Connection) -> None:
    """Create the proven_facts table if it does not exist."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS proven_facts
           (subject TEXT, predicate TEXT, object TEXT, proof TEXT,
           UNIQUE(subject, predicate, object))"""
    )


simba.db.register_schema(_init_truth_schema)


def truth_add(subject: str, predicate: str, object: str, proof: str) -> str:
    """Record a proven fact into the Truth DB.

    Use this ONLY when a verifier (Z3/Datalog) has proven a hypothesis.
    """
    with simba.db.get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO proven_facts VALUES (?, ?, ?, ?)",
                (subject, predicate, object, proof),
            )
            conn.commit()
            return f"Fact recorded: {subject} {predicate} {object}"
        except sqlite3.IntegrityError:
            return f"Fact already exists: {subject} {predicate} {object}"
        except Exception as exc:
            return f"Database Error: {exc}"


def truth_query(subject: str | None = None, predicate: str | None = None) -> str:
    """Query the Truth DB for existing proven facts.

    Use this BEFORE assuming capabilities or behavior about the codebase.
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

        rows = cursor.execute(query, params).fetchall()

        if not rows:
            return "No facts found matching criteria."

        return "\n".join(f"FACT: {r[0]} {r[1]} {r[2]} (Proof: {r[3]})" for r in rows)

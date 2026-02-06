"""Truth database — record and query proven facts via MCP tools."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

import simba.neuron.config

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection]:
    """Yield a connection to the truth database, creating the schema if needed."""
    db_path = simba.neuron.config.CONFIG.resolved_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS facts
               (subject TEXT, predicate TEXT, object TEXT, proof TEXT,
               UNIQUE(subject, predicate, object))"""
        )
        yield conn
    finally:
        conn.close()


def truth_add(subject: str, predicate: str, object: str, proof: str) -> str:
    """Record a proven fact into the Truth DB.

    Use this ONLY when a verifier (Z3/Datalog) has proven a hypothesis.
    """
    with get_db_connection() as conn:
        try:
            conn.execute(
                "INSERT INTO facts VALUES (?, ?, ?, ?)",
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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM facts WHERE 1=1"
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

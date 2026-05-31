"""Temporal knowledge-graph store backed by ``kg_edges`` + FTS5/bm25.

Edges are bitemporal-lite: each row carries ``valid_from`` (set on insert) and
``valid_to`` (``NULL`` while the fact holds, stamped when invalidated).  This
lets ``kg_query`` return the *currently* valid graph by default, or the graph
"as of" an arbitrary timestamp.  An external-content FTS5 table mirrors the
``subject``/``predicate``/``object`` columns (trigram tokenizer) so substring
recall and bm25 ranking come for free.  Edges are scoped per-project via
``project_path`` to prevent cross-project leakage.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time

import simba.db

_SCHEMA_BASE_SQL = """\
CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY,
    subject TEXT, predicate TEXT, object TEXT,
    subject_type TEXT, object_type TEXT,
    proof TEXT, transcript_id TEXT, char_start INTEGER,
    valid_from TEXT, valid_to TEXT,
    project_path TEXT NOT NULL, created_at TEXT,
    UNIQUE(subject, predicate, object, project_path, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_kg_edges_subject
    ON kg_edges(subject, project_path);
CREATE INDEX IF NOT EXISTS idx_kg_edges_project
    ON kg_edges(project_path);
"""

_SCHEMA_FTS_SQL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS kg_edges_fts USING fts5(
    subject, predicate, object,
    content='kg_edges', content_rowid='id', tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS kg_edges_ai AFTER INSERT ON kg_edges BEGIN
    INSERT INTO kg_edges_fts(rowid, subject, predicate, object)
    VALUES (NEW.id, NEW.subject, NEW.predicate, NEW.object);
END;

CREATE TRIGGER IF NOT EXISTS kg_edges_ad AFTER DELETE ON kg_edges BEGIN
    INSERT INTO kg_edges_fts(kg_edges_fts, rowid, subject, predicate, object)
    VALUES ('delete', OLD.id, OLD.subject, OLD.predicate, OLD.object);
END;

CREATE TRIGGER IF NOT EXISTS kg_edges_au AFTER UPDATE ON kg_edges BEGIN
    INSERT INTO kg_edges_fts(kg_edges_fts, rowid, subject, predicate, object)
    VALUES ('delete', OLD.id, OLD.subject, OLD.predicate, OLD.object);
    INSERT INTO kg_edges_fts(rowid, subject, predicate, object)
    VALUES (NEW.id, NEW.subject, NEW.predicate, NEW.object);
END;
"""


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create ``kg_edges`` plus its FTS5 mirror and sync triggers.

    Core table/indexes are always created; the FTS5 virtual table and its
    triggers are installed only when the SQLite build supports FTS5 (otherwise
    ``kg_query`` falls back to non-FTS filtering).
    """
    conn.executescript(_SCHEMA_BASE_SQL)
    with contextlib.suppress(sqlite3.OperationalError):
        conn.executescript(_SCHEMA_FTS_SQL)


simba.db.register_schema(_init_schema)


def _now() -> str:
    """Return the current UTC time as an ISO-8601 ``Z`` timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    """Project a ``kg_edges`` row into a stable public dict."""
    return {
        "id": row["id"],
        "subject": row["subject"],
        "predicate": row["predicate"],
        "object": row["object"],
        "subject_type": row["subject_type"],
        "object_type": row["object_type"],
        "proof": row["proof"],
        "transcript_id": row["transcript_id"],
        "char_start": row["char_start"],
        "valid_from": row["valid_from"],
        "valid_to": row["valid_to"],
    }


def kg_add(
    subject: str,
    predicate: str,
    object: str,
    proof: str,
    *,
    subject_type: str = "concept",
    object_type: str = "concept",
    transcript_id: str | None = None,
    char_start: int | None = None,
    project_path: str | None = None,
) -> str:
    """Insert an *open* edge (``valid_from`` = now, ``valid_to`` = NULL).

    Returns ``"added"`` on success, or ``"exists"`` when an edge with the same
    ``(subject, predicate, object, project_path, valid_from)`` already exists
    (UNIQUE collision).  ``project_path`` defaults to the current repo's
    stable project id.
    """
    if project_path is None:
        project_path = simba.db.resolve_project_id()
    now = _now()
    with simba.db.get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO kg_edges (subject, predicate, object, "
                "subject_type, object_type, proof, transcript_id, char_start, "
                "valid_from, valid_to, project_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    subject,
                    predicate,
                    object,
                    subject_type,
                    object_type,
                    proof,
                    transcript_id,
                    char_start,
                    now,
                    project_path,
                    now,
                ),
            )
            conn.commit()
            return "added"
        except sqlite3.IntegrityError:
            return "exists"


def kg_invalidate(
    subject: str,
    predicate: str,
    object: str,
    *,
    project_path: str | None = None,
) -> int:
    """Close every matching *open* edge by stamping ``valid_to`` = now.

    Returns the number of edges closed.  ``project_path`` defaults to the
    current repo's stable project id.
    """
    if project_path is None:
        project_path = simba.db.resolve_project_id()
    with simba.db.get_db() as conn:
        cursor = conn.execute(
            "UPDATE kg_edges SET valid_to=? "
            "WHERE subject=? AND predicate=? AND object=? "
            "AND project_path=? AND valid_to IS NULL",
            (_now(), subject, predicate, object, project_path),
        )
        conn.commit()
        return cursor.rowcount


def kg_query(
    query: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    *,
    project_path: str | None = None,
    as_of: str | None = None,
    include_expired: bool = False,
    limit: int = 10,
) -> list[dict[str, object]]:
    """Query the knowledge graph, with FTS/bm25 ranking and temporal filters.

    When *query* is set, results are matched against the trigram FTS index and
    ordered by ``bm25``; otherwise rows are filtered by *subject*/*predicate*.
    *project_path* scopes results when given.  Temporal semantics: unless
    *include_expired*, only currently-valid edges (``valid_to IS NULL``) are
    returned; when *as_of* is given, an edge is kept iff
    ``valid_from <= as_of AND (valid_to IS NULL OR as_of < valid_to)``.

    Returns a list of row dicts.  A malformed FTS ``MATCH`` expression is
    swallowed and yields ``[]``.
    """
    clauses: list[str] = []
    params: list[object] = []

    if query:
        sql = (
            "SELECT e.* FROM kg_edges e "
            "JOIN kg_edges_fts f ON f.rowid = e.id "
            "WHERE kg_edges_fts MATCH ?"
        )
        params.append(query)
    else:
        sql = "SELECT e.* FROM kg_edges e WHERE 1=1"
        if subject:
            clauses.append("e.subject = ?")
            params.append(subject)
        if predicate:
            clauses.append("e.predicate = ?")
            params.append(predicate)

    if project_path:
        clauses.append("e.project_path = ?")
        params.append(project_path)

    if as_of is not None:
        clauses.append("e.valid_from <= ?")
        params.append(as_of)
        clauses.append("(e.valid_to IS NULL OR ? < e.valid_to)")
        params.append(as_of)
    elif not include_expired:
        clauses.append("e.valid_to IS NULL")

    for clause in clauses:
        sql += f" AND {clause}"

    if query:
        sql += " ORDER BY bm25(kg_edges_fts)"
    sql += " LIMIT ?"
    params.append(limit)

    with simba.db.get_db() as conn:
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
    return [_row_to_dict(row) for row in rows]

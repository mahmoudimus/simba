"""SQLite FTS5 keyword mirror for hybrid memory recall (L3).

A *derived* ``bm25`` index over memory ``content``/``context``, co-located with
the LanceDB at ``<data_dir>/memory_fts.db``.  The daemon owns the LanceDB table,
so it owns this mirror: it is written on ``/store``, ``/delete``, ``/patch`` and
reconciled against LanceDB on startup.  ``SYSTEM`` memories are never indexed
(recall excludes them).

Append-only applies to the LanceDB *source* of truth; this index is rebuildable
and may be deleted from (same as the KG's external-content FTS).
"""

from __future__ import annotations

import re
import sqlite3
import typing

FTS_FILENAME = "memory_fts.db"

_DEFAULT_TOKENIZE = "trigram"
# Tokenizer name is interpolated into DDL (can't be a bound parameter), so it is
# constrained to a known-safe allowlist.
_ALLOWED_TOKENIZE = frozenset({"trigram", "porter", "unicode61", "ascii"})
# Trigram MATCH needs >= 3-char terms; shorter terms are dropped (the vector arm
# covers those).
_MIN_TOKEN_LEN = 3
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+")

_COLUMNS = (
    "memory_id",
    "project_path",
    "type",
    "confidence",
    "created_at",
    "content",
    "context",
)


def schema_sql(tokenize: str = _DEFAULT_TOKENIZE) -> str:
    """Return the ``CREATE VIRTUAL TABLE`` DDL for the mirror."""
    tok = tokenize if tokenize in _ALLOWED_TOKENIZE else _DEFAULT_TOKENIZE
    return (
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5("
        "memory_id UNINDEXED, project_path UNINDEXED, type UNINDEXED, "
        "confidence UNINDEXED, created_at UNINDEXED, content, context, "
        f"tokenize='{tok}')"
    )


def connect(path: typing.Any) -> sqlite3.Connection:
    """Open a connection to the mirror with a ``Row`` factory."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init(path: typing.Any, tokenize: str = _DEFAULT_TOKENIZE) -> None:
    """Create the mirror table if it does not exist."""
    conn = connect(path)
    try:
        conn.execute(schema_sql(tokenize))
        conn.commit()
    finally:
        conn.close()


def _insert(conn: sqlite3.Connection, memory: dict[str, typing.Any]) -> bool:
    """Insert one memory row (no commit). Returns False if skipped."""
    if memory.get("type") == "SYSTEM":
        return False
    mid = memory.get("id") or memory.get("memory_id")
    if not mid:
        return False
    conn.execute(
        "INSERT INTO memory_fts "
        "(memory_id, project_path, type, confidence, created_at, "
        "content, context) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            mid,
            memory.get("projectPath", "") or "",
            memory.get("type", "") or "",
            float(memory.get("confidence", 0.0) or 0.0),
            memory.get("createdAt", "") or "",
            memory.get("content", "") or "",
            memory.get("context", "") or "",
        ),
    )
    return True


def upsert(conn: sqlite3.Connection, memory: dict[str, typing.Any]) -> None:
    """Idempotently index a memory (delete-then-insert by ``memory_id``).

    ``SYSTEM`` memories and rows without an id are skipped.
    """
    mid = memory.get("id") or memory.get("memory_id")
    if mid:
        conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (mid,))
    _insert(conn, memory)
    conn.commit()


def delete(conn: sqlite3.Connection, memory_id: str) -> None:
    """Remove a memory from the mirror."""
    conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
    conn.commit()


def set_project(conn: sqlite3.Connection, memory_id: str, project_path: str) -> None:
    """Update the scoping ``project_path`` for a memory (keeps /patch in sync)."""
    conn.execute(
        "UPDATE memory_fts SET project_path = ? WHERE memory_id = ?",
        (project_path or "", memory_id),
    )
    conn.commit()


def count(conn: sqlite3.Connection) -> int:
    """Return the number of indexed rows."""
    return conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]


def rebuild(
    conn: sqlite3.Connection, memories: typing.Iterable[dict[str, typing.Any]]
) -> int:
    """Replace the whole mirror from ``memories`` (skips SYSTEM). Returns count."""
    conn.execute("DELETE FROM memory_fts")
    n = 0
    for m in memories:
        if _insert(conn, m):
            n += 1
    conn.commit()
    return n


def _build_match(query: str, min_token_len: int = _MIN_TOKEN_LEN) -> str:
    """Build a safe FTS5 ``MATCH`` expression: OR of quoted literal terms.

    Quoting each term neutralizes FTS5 operators, so arbitrary user text can't
    produce a malformed expression from term content.  Returns ``""`` when no
    usable terms remain (caller treats that as "no keyword hits").
    """
    seen: set[str] = set()
    terms: list[str] = []
    for m in _TOKEN_RE.finditer(query or ""):
        tok = m.group()
        if len(tok) < min_token_len:
            continue
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        terms.append('"' + tok.replace('"', '""') + '"')
    return " OR ".join(terms)


def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    project_path: str | None = None,
    types: list[str] | None = None,
    limit: int = 20,
) -> list[dict[str, typing.Any]]:
    """Return up to ``limit`` bm25-ranked memories matching ``query``.

    Scoped to ``project_path`` (exact match) and ``types`` when given.  Never
    raises: a malformed ``MATCH`` or missing table yields ``[]``.
    """
    match = _build_match(query)
    if not match:
        return []

    sql = (
        "SELECT memory_id, project_path, type, confidence, created_at, "
        "content, context FROM memory_fts WHERE memory_fts MATCH ?"
    )
    params: list[typing.Any] = [match]
    if project_path:
        sql += " AND project_path = ?"
        params.append(project_path)
    if types:
        sql += f" AND type IN ({','.join('?' * len(types))})"
        params.extend(types)
    sql += " ORDER BY bm25(memory_fts) LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for r in rows:
        raw_conf = r["confidence"]
        try:
            conf = float(raw_conf) if raw_conf not in (None, "") else 0.0
        except (TypeError, ValueError):
            conf = 0.0
        results.append(
            {
                "memory_id": r["memory_id"],
                "type": r["type"],
                "content": r["content"],
                "context": r["context"],
                "confidence": conf,
                "createdAt": r["created_at"],
                "projectPath": r["project_path"],
            }
        )
    return results

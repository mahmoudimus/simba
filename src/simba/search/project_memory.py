"""SQLite-backed per-project memory for session history, knowledge, and facts.

Ported from claude-turbo-search/memory/memory-db.sh + schema.sql.
"""

from __future__ import annotations

import contextlib
import sqlite3
import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

_SCHEMA_BASE_SQL = """\
-- Session summaries - what was worked on
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary TEXT NOT NULL,
    files_touched TEXT,
    tools_used TEXT,
    topics TEXT
);

-- Code area knowledge - accumulated understanding of codebase areas
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    area TEXT UNIQUE NOT NULL,
    summary TEXT NOT NULL,
    patterns TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quick facts - key decisions, conventions, important notes
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_area ON knowledge(area);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
"""

_SCHEMA_FTS_SQL = """\
-- Full-text search virtual table for fast local search
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    source_type,
    source_id,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync

-- Session FTS triggers
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO memory_fts(content, source_type, source_id)
    VALUES (NEW.summary || ' ' || COALESCE(NEW.topics, ''), 'session', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    DELETE FROM memory_fts WHERE source_type = 'session' AND source_id = OLD.id;
    INSERT INTO memory_fts(content, source_type, source_id)
    VALUES (NEW.summary || ' ' || COALESCE(NEW.topics, ''), 'session', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    DELETE FROM memory_fts WHERE source_type = 'session' AND source_id = OLD.id;
END;

-- Knowledge FTS triggers
CREATE TRIGGER IF NOT EXISTS knowledge_ai AFTER INSERT ON knowledge BEGIN
    INSERT INTO memory_fts(content, source_type, source_id)
    VALUES (NEW.area || ' ' || NEW.summary || ' ' || COALESCE(NEW.patterns, ''),
            'knowledge', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_au AFTER UPDATE ON knowledge BEGIN
    DELETE FROM memory_fts WHERE source_type = 'knowledge' AND source_id = OLD.id;
    INSERT INTO memory_fts(content, source_type, source_id)
    VALUES (NEW.area || ' ' || NEW.summary || ' ' || COALESCE(NEW.patterns, ''),
            'knowledge', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS knowledge_ad AFTER DELETE ON knowledge BEGIN
    DELETE FROM memory_fts WHERE source_type = 'knowledge' AND source_id = OLD.id;
END;

-- Facts FTS triggers
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO memory_fts(content, source_type, source_id)
    VALUES (NEW.fact || ' ' || COALESCE(NEW.category, ''), 'fact', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    DELETE FROM memory_fts WHERE source_type = 'fact' AND source_id = OLD.id;
    INSERT INTO memory_fts(content, source_type, source_id)
    VALUES (NEW.fact || ' ' || COALESCE(NEW.category, ''), 'fact', NEW.id);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    DELETE FROM memory_fts WHERE source_type = 'fact' AND source_id = OLD.id;
END;
"""

# Combined schema for reference / documentation purposes.
_SCHEMA_SQL = _SCHEMA_BASE_SQL + _SCHEMA_FTS_SQL


def init_db(db_path: pathlib.Path) -> sqlite3.Connection:
    """Create parent dirs, initialize DB with schema, return connection.

    FTS5 tables and triggers are installed when supported by the SQLite build;
    core tables and indexes are always created.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_BASE_SQL)
    with contextlib.suppress(sqlite3.OperationalError):
        # FTS5 module not available in this SQLite build; full-text search
        # will gracefully return empty results.
        conn.executescript(_SCHEMA_FTS_SQL)
    return conn


def add_session(
    conn: sqlite3.Connection,
    summary: str,
    files_touched: str,
    tools_used: str,
    topics: str,
) -> int:
    """Insert a session summary and return its rowid."""
    cursor = conn.execute(
        "INSERT INTO sessions (summary, files_touched, tools_used, topics) "
        "VALUES (?, ?, ?, ?)",
        (summary, files_touched, tools_used, topics),
    )
    conn.commit()
    return typing.cast("int", cursor.lastrowid)


def add_knowledge(
    conn: sqlite3.Connection,
    area: str,
    summary: str,
    patterns: str,
) -> int:
    """Upsert knowledge for a code area and return its rowid."""
    cursor = conn.execute(
        "INSERT INTO knowledge (area, summary, patterns) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(area) DO UPDATE SET "
        "summary = excluded.summary, "
        "patterns = excluded.patterns, "
        "updated_at = CURRENT_TIMESTAMP",
        (area, summary, patterns),
    )
    conn.commit()
    return typing.cast("int", cursor.lastrowid)


def add_fact(
    conn: sqlite3.Connection,
    fact: str,
    category: str = "general",
) -> int:
    """Insert a fact and return its rowid."""
    cursor = conn.execute(
        "INSERT INTO facts (fact, category) VALUES (?, ?)",
        (fact, category),
    )
    conn.commit()
    return typing.cast("int", cursor.lastrowid)


def _escape_fts_query(query: str) -> str:
    """Escape special FTS5 characters so the query is treated as plain terms."""
    # FTS5 special characters that need quoting: * " ( ) : ^
    # Wrap each token in double quotes to treat as a literal phrase/term.
    tokens = query.split()
    escaped = []
    for token in tokens:
        # Strip characters that would break even inside quotes
        cleaned = token.replace('"', "")
        if cleaned:
            escaped.append(f'"{cleaned}"')
    return " ".join(escaped)


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
) -> list[dict[str, typing.Any]]:
    """Full-text search with snippet extraction.

    Returns list of {source_type, source_id, match}.
    Returns an empty list when FTS5 is unavailable.
    """
    safe_query = _escape_fts_query(query)
    if not safe_query:
        return []

    try:
        rows = conn.execute(
            "SELECT source_type, source_id, "
            "snippet(memory_fts, 0, '**', '**', '...', 32) AS match "
            "FROM memory_fts "
            "WHERE memory_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (safe_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # FTS5 virtual table may not exist if SQLite lacks FTS5 support
        return []

    return [
        {
            "source_type": row["source_type"],
            "source_id": row["source_id"],
            "match": row["match"],
        }
        for row in rows
    ]


def get_context(
    conn: sqlite3.Connection,
    query: str,
    token_budget: int = 500,
) -> str:
    """Build a markdown context string combining facts, knowledge, sessions, and FTS.

    The result is truncated to approximately *token_budget* tokens (estimated at
    4 characters per token).
    """
    char_limit = token_budget * 4
    parts: list[str] = []

    # 1. Facts (highest value, lowest cost)
    fact_rows = conn.execute(
        "SELECT fact FROM facts ORDER BY created_at DESC LIMIT 5"
    ).fetchall()
    if fact_rows:
        lines = ["## Project Facts"]
        for row in fact_rows:
            lines.append(f"- {row['fact']}")
        parts.append("\n".join(lines))

    # 2. Relevant knowledge areas (LIKE match on query)
    if query:
        like_pattern = f"%{query}%"
        knowledge_rows = conn.execute(
            "SELECT area, summary FROM knowledge "
            "WHERE area LIKE ? OR summary LIKE ? "
            "LIMIT 3",
            (like_pattern, like_pattern),
        ).fetchall()
        if knowledge_rows:
            lines = ["## Relevant Code Areas"]
            for row in knowledge_rows:
                lines.append(f"- **{row['area']}**: {row['summary']}")
            parts.append("\n".join(lines))

    # 3. Recent sessions
    session_rows = conn.execute(
        "SELECT summary FROM sessions ORDER BY created_at DESC LIMIT 3"
    ).fetchall()
    if session_rows:
        lines = ["## Recent Work"]
        for row in session_rows:
            lines.append(f"- {row['summary']}")
        parts.append("\n".join(lines))

    # 4. FTS results for query-specific context
    if query:
        fts_results = search_fts(conn, query, limit=5)
        if fts_results:
            lines = ["## Related Context"]
            for result in fts_results:
                lines.append(f"- {result['match']}")
            parts.append("\n".join(lines))

    output = "\n\n".join(parts)
    return output[:char_limit]


def get_recent_sessions(
    conn: sqlite3.Connection,
    limit: int = 5,
) -> list[dict[str, typing.Any]]:
    """Return recent sessions as a list of dicts."""
    rows = conn.execute(
        "SELECT id, created_at, summary, topics "
        "FROM sessions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "summary": row["summary"],
            "topics": row["topics"],
        }
        for row in rows
    ]


def get_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Return counts for sessions, knowledge, and facts."""
    sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
    knowledge = conn.execute("SELECT COUNT(*) AS c FROM knowledge").fetchone()["c"]
    facts = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
    return {
        "sessions": sessions,
        "knowledge": knowledge,
        "facts": facts,
    }


def find_repo_root(cwd: pathlib.Path) -> pathlib.Path | None:
    """Walk up from *cwd* looking for a ``.git`` directory.

    Returns the repo root path, or ``None`` if not found.
    """
    current = cwd.resolve()
    while True:
        if (current / ".git").is_dir():
            return current
        parent = current.parent
        if parent == current:
            # Reached filesystem root without finding .git
            return None
        current = parent


def get_db_path(cwd: pathlib.Path) -> pathlib.Path:
    """Determine the memory database path for a working directory.

    Uses the repository root if one is found, otherwise falls back to *cwd*.
    """
    root = find_repo_root(cwd)
    base = root if root is not None else cwd
    return base / ".claude-memory" / "memory.db"


def get_connection(cwd: pathlib.Path) -> sqlite3.Connection | None:
    """Open a connection to the memory database if it already exists.

    Returns ``None`` when the database file does not exist.
    """
    db_path = get_db_path(cwd)
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

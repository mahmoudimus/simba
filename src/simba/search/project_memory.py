"""SQLite-backed per-project memory for session history, knowledge, and facts.

Ported from claude-turbo-search/memory/memory-db.sh + schema.sql.
"""

from __future__ import annotations

import contextlib
import sqlite3
import typing

import simba._vendor.peewee as pw
import simba.db
from simba._vendor.playhouse.sqlite_ext import FTS5Model, RowIDField, SearchField

if typing.TYPE_CHECKING:
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


def _init_schema(conn: sqlite3.Connection) -> None:
    """Initialize project memory tables and FTS indexes.

    FTS5 tables and triggers are installed when supported by the SQLite build;
    core tables and indexes are always created.
    """
    conn.executescript(_SCHEMA_BASE_SQL)
    with contextlib.suppress(sqlite3.OperationalError):
        # FTS5 module not available in this SQLite build; full-text search
        # will gracefully return empty results.
        conn.executescript(_SCHEMA_FTS_SQL)


simba.db.register_schema(_init_schema)


class Session(simba.db.BaseModel):
    summary = pw.TextField()
    files_touched = pw.TextField(null=True)
    tools_used = pw.TextField(null=True)
    topics = pw.TextField(null=True)
    created_at = pw.TextField(null=True)  # DB default CURRENT_TIMESTAMP

    class Meta:
        table_name = "sessions"


class Knowledge(simba.db.BaseModel):
    area = pw.TextField(unique=True)
    summary = pw.TextField()
    patterns = pw.TextField(null=True)
    updated_at = pw.TextField(null=True)  # DB default CURRENT_TIMESTAMP

    class Meta:
        table_name = "knowledge"


class Fact(simba.db.BaseModel):
    fact = pw.TextField()
    category = pw.TextField(null=True)  # DB default 'general'
    created_at = pw.TextField(null=True)  # DB default CURRENT_TIMESTAMP

    class Meta:
        table_name = "facts"


class ProjectMemoryFTS(FTS5Model):
    # Query-only view of the porter-tokenized FTS mirror (creation + sync stay
    # in the raw DDL/triggers above). Used for snippet() + rank in search_fts.
    rowid = RowIDField()
    content = SearchField()
    source_type = SearchField()
    source_id = SearchField()

    class Meta:
        database = simba.db.database
        table_name = "memory_fts"


def add_session(
    summary: str,
    files_touched: str,
    tools_used: str,
    topics: str,
    *,
    cwd: pathlib.Path | None = None,
) -> int:
    """Insert a session summary and return its rowid."""
    with simba.db.connect(cwd):
        return Session.create(
            summary=summary,
            files_touched=files_touched,
            tools_used=tools_used,
            topics=topics,
        ).id


def add_knowledge(
    area: str,
    summary: str,
    patterns: str,
    *,
    cwd: pathlib.Path | None = None,
) -> int:
    """Upsert knowledge for a code area and return its rowid."""
    with simba.db.connect(cwd):
        return (
            Knowledge.insert(area=area, summary=summary, patterns=patterns)
            .on_conflict(
                conflict_target=[Knowledge.area],
                update={
                    Knowledge.summary: summary,
                    Knowledge.patterns: patterns,
                    Knowledge.updated_at: pw.SQL("CURRENT_TIMESTAMP"),
                },
            )
            .execute()
        )


def add_fact(
    fact: str,
    category: str = "general",
    *,
    cwd: pathlib.Path | None = None,
) -> int:
    """Insert a fact and return its rowid."""
    with simba.db.connect(cwd):
        return Fact.create(fact=fact, category=category).id


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
    query: str,
    limit: int = 10,
    *,
    cwd: pathlib.Path | None = None,
) -> list[dict[str, typing.Any]]:
    """Full-text search with snippet extraction.

    Returns list of {source_type, source_id, match}.
    Returns an empty list when FTS5 is unavailable.
    """
    safe_query = _escape_fts_query(query)
    if not safe_query:
        return []

    snippet = pw.fn.snippet(ProjectMemoryFTS._meta.entity, 0, "**", "**", "...", 32)
    with simba.db.connect(cwd):
        q = (
            ProjectMemoryFTS.select(
                ProjectMemoryFTS.source_type,
                ProjectMemoryFTS.source_id,
                snippet.alias("match"),
            )
            .where(ProjectMemoryFTS.match(safe_query))
            .order_by(ProjectMemoryFTS.bm25())
            .limit(limit)
        )
        try:
            return list(q.dicts())
        except Exception:
            # FTS5 virtual table may not exist if SQLite lacks FTS5 support
            return []


def get_context(
    query: str,
    token_budget: int = 500,
    *,
    cwd: pathlib.Path | None = None,
) -> str:
    """Build a markdown context string combining facts, knowledge, sessions, and FTS.

    The result is truncated to approximately *token_budget* tokens (estimated at
    4 characters per token).
    """
    char_limit = token_budget * 4
    parts: list[str] = []

    with simba.db.connect(cwd):
        # 1. Facts (highest value, lowest cost)
        facts = list(Fact.select(Fact.fact).order_by(Fact.created_at.desc()).limit(5))
        if facts:
            lines = ["## Project Facts"]
            lines += [f"- {f.fact}" for f in facts]
            parts.append("\n".join(lines))

        # 2. Relevant knowledge areas (LIKE match on query)
        if query:
            knowledge = list(
                Knowledge.select(Knowledge.area, Knowledge.summary)
                .where(
                    Knowledge.area.contains(query) | Knowledge.summary.contains(query)
                )
                .limit(3)
            )
            if knowledge:
                lines = ["## Relevant Code Areas"]
                lines += [f"- **{k.area}**: {k.summary}" for k in knowledge]
                parts.append("\n".join(lines))

        # 3. Recent sessions
        sessions = list(
            Session.select(Session.summary).order_by(Session.created_at.desc()).limit(3)
        )
        if sessions:
            lines = ["## Recent Work"]
            lines += [f"- {s.summary}" for s in sessions]
            parts.append("\n".join(lines))

        # 4. FTS results for query-specific context
        if query:
            fts_results = search_fts(query, limit=5, cwd=cwd)
            if fts_results:
                lines = ["## Related Context"]
                lines += [f"- {r['match']}" for r in fts_results]
                parts.append("\n".join(lines))

    output = "\n\n".join(parts)
    return output[:char_limit]


def get_recent_sessions(
    limit: int = 5,
    *,
    cwd: pathlib.Path | None = None,
) -> list[dict[str, typing.Any]]:
    """Return recent sessions as a list of dicts."""
    with simba.db.connect(cwd):
        rows = (
            Session.select(
                Session.id, Session.created_at, Session.summary, Session.topics
            )
            .order_by(Session.created_at.desc())
            .limit(limit)
        )
        return list(rows.dicts())


def get_stats(cwd: pathlib.Path | None = None) -> dict[str, int]:
    """Return counts for sessions, knowledge, and facts."""
    with simba.db.connect(cwd):
        return {
            "sessions": Session.select().count(),
            "knowledge": Knowledge.select().count(),
            "facts": Fact.select().count(),
        }

"""Phase 7 schema: ``kg_derived_edges`` + ``kg_rules`` + ``dormant`` column.

Two new tables (materialized derived edges, induced Horn rules) plus an additive
``dormant`` flag on ``kg_edges`` for AGM-retracted facts. All changes are
idempotent migrations registered with ``simba.db.register_schema`` and never
touch existing rows (append-only contract).
"""

from __future__ import annotations

import contextlib
import sqlite3

import simba.db
import simba.kg.store  # ensures kg_edges schema is registered first

_ = simba.kg.store

_DERIVED_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS kg_derived_edges (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    subject_type TEXT,
    object_type TEXT,
    proof TEXT NOT NULL,
    source_edge_ids TEXT NOT NULL,
    rule_id INTEGER,
    confidence REAL DEFAULT 0.8,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    occurred_at TEXT,
    project_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(subject, predicate, object, project_path, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_kg_derived_project
    ON kg_derived_edges(project_path);

CREATE TABLE IF NOT EXISTS kg_rules (
    id INTEGER PRIMARY KEY,
    rule_text TEXT NOT NULL,
    head_predicate TEXT NOT NULL,
    confidence REAL DEFAULT 0.7,
    activation_count INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    last_fired_at TEXT,
    UNIQUE(rule_text)
);
"""


def _migrate_dormant_flag(conn: sqlite3.Connection) -> None:
    """Add the ``dormant`` column to ``kg_edges`` (marks AGM-retracted edges).

    Idempotent: a no-op once the column exists. Skips silently if ``kg_edges``
    has not been created yet (the kg store initializer creates it first).
    """
    has_kg = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_edges'"
    ).fetchone()
    if not has_kg:
        return
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kg_edges)")}
    if "dormant" not in cols:
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("ALTER TABLE kg_edges ADD COLUMN dormant INTEGER DEFAULT 0")


def _init_neuron_schema(conn: sqlite3.Connection) -> None:
    """Create the Phase 7 tables and run additive migrations."""
    conn.executescript(_DERIVED_SCHEMA_SQL)
    _migrate_dormant_flag(conn)


simba.db.register_schema(_init_neuron_schema)

"""Phase 7 schema: ``kg_derived_edges`` + ``kg_rules`` + resolution tables.

Tables registered here (all idempotent, append-only, never touch existing rows):

  - ``kg_derived_edges`` — materialised derived edges (DERIVE phase)
  - ``kg_rules`` — induced Horn rules (INDUCE phase)
  - ``kg_audit_resolutions`` — loser-preserving audit rows from the typed
    contradiction-resolution operators (N3 defence: the superseded fact stays
    recoverable from the audit trail)
  - ``neuron_judge_log`` — keyed, append-only judge-verdict log so an
    AwaitConfirm / PerRule verdict replays deterministically after a crash
    (N1 defence)

Plus an additive ``dormant`` flag on ``kg_edges`` for AGM-retracted facts. All
changes are idempotent migrations registered with ``simba.db.register_schema``.
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

# Append-only audit trail emitted by every typed resolution (N3 defence). The
# loser's object, edge id, and merged provenance are preserved so the superseded
# fact is always recoverable. UNIQUE(loser_edge_id, system_time) keeps repeated
# resolutions of the same loser as distinct append rows (never a clobber).
_AUDIT_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS kg_audit_resolutions (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    winner_object TEXT NOT NULL,
    loser_object TEXT NOT NULL,
    winner_edge_id INTEGER NOT NULL,
    loser_edge_id INTEGER NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    occurred_at TEXT,
    system_time TEXT NOT NULL,
    provenance_merge TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    confidence_winner REAL,
    confidence_loser REAL,
    judge_verdict TEXT,
    project_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(loser_edge_id, system_time)
);

CREATE INDEX IF NOT EXISTS idx_kg_audit_loser_edge
    ON kg_audit_resolutions(loser_edge_id);
CREATE INDEX IF NOT EXISTS idx_kg_audit_project
    ON kg_audit_resolutions(project_path);
"""

# Keyed, append-only judge-verdict log (N1 defence). A verdict committed here
# under (r_key, theta) BEFORE the operator commit replays deterministically
# after a crash. ``vote`` is a non-negative selection index; the binary path
# constrains it to {0, 1} in Python. ``winner_edge_id`` is the elected winner
# for AwaitConfirm / PerRule replay.
_JUDGE_LOG_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS neuron_judge_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    r_key TEXT NOT NULL,
    theta TEXT NOT NULL,
    vote INTEGER NOT NULL CHECK (vote >= 0),
    winner_edge_id INTEGER,
    system_time TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_neuron_judge_log_r_key_theta
    ON neuron_judge_log(r_key, theta);
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
    conn.executescript(_AUDIT_SCHEMA_SQL)
    conn.executescript(_JUDGE_LOG_SCHEMA_SQL)
    _migrate_dormant_flag(conn)


simba.db.register_schema(_init_neuron_schema)

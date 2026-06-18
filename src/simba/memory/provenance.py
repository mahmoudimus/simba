"""Append-only temporal/provenance metadata for general memories."""

from __future__ import annotations

import time
import typing

import simba._vendor.peewee as pw
import simba.db

TRUST_SOURCES = {
    "user_stated",
    "agreed_upon",
    "agent_suggested",
    "llm_extracted",
    "hook_auto_learned",
    "external",
}

_SOURCE_WEIGHTS = {
    "agreed_upon": 1.30,
    "user_stated": 1.20,
    "agent_suggested": 0.85,
    "external": 0.75,
    "llm_extracted": 0.70,
    "hook_auto_learned": 0.60,
}

_ORIGIN_WEIGHTS = {
    "user": 1.15,
    "cli": 1.10,
    "store": 1.00,
    "external": 0.85,
    "codex_extract": 0.80,
    "transcript_extract": 0.75,
    "hook": 0.65,
}

_TYPE_WEIGHTS = {
    "DECISION": 1.08,
    "PREFERENCE": 1.08,
    "GOTCHA": 1.04,
    "WORKING_SOLUTION": 1.02,
    "PATTERN": 1.00,
    "FAILURE": 1.00,
}


def _ensure_column(conn, table: str, name: str, spec: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")


def _init_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_provenance ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "memory_id VARCHAR(64) NOT NULL, "
        "occurred_at VARCHAR(64) NOT NULL DEFAULT '', "
        "observed_at VARCHAR(64) NOT NULL DEFAULT '', "
        "source_file TEXT NOT NULL DEFAULT '', "
        "source_span TEXT NOT NULL DEFAULT '', "
        "source_url TEXT NOT NULL DEFAULT '', "
        "extraction_agent TEXT NOT NULL DEFAULT '', "
        "extraction_version TEXT NOT NULL DEFAULT '', "
        "source_session TEXT NOT NULL DEFAULT '', "
        "trust_source VARCHAR(32) NOT NULL DEFAULT 'agent_suggested', "
        "capture_origin VARCHAR(64) NOT NULL DEFAULT 'store', "
        "trust_score REAL NOT NULL DEFAULT 0.0, "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "created_at_iso VARCHAR(32) NOT NULL DEFAULT '')"
    )
    _ensure_column(
        conn,
        "memory_provenance",
        "trust_source",
        "VARCHAR(32) NOT NULL DEFAULT 'agent_suggested'",
    )
    _ensure_column(
        conn,
        "memory_provenance",
        "capture_origin",
        "VARCHAR(64) NOT NULL DEFAULT 'store'",
    )
    _ensure_column(
        conn,
        "memory_provenance",
        "trust_score",
        "REAL NOT NULL DEFAULT 0.0",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_provenance_memory "
        "ON memory_provenance(memory_id, created_at)"
    )


simba.db.register_schema(_init_schema)


class MemoryProvenance(simba.db.BaseModel):
    id = pw.AutoField()
    memory_id = pw.CharField(max_length=64)
    occurred_at = pw.CharField(max_length=64, default="")
    observed_at = pw.CharField(max_length=64, default="")
    source_file = pw.TextField(default="")
    source_span = pw.TextField(default="")
    source_url = pw.TextField(default="")
    extraction_agent = pw.TextField(default="")
    extraction_version = pw.TextField(default="")
    source_session = pw.TextField(default="")
    trust_source = pw.CharField(max_length=32, default="agent_suggested")
    capture_origin = pw.CharField(max_length=64, default="store")
    trust_score = pw.FloatField(default=0.0)
    created_at = pw.FloatField(default=0.0)
    created_at_iso = pw.CharField(max_length=32, default="")

    class Meta:
        table_name = "memory_provenance"


simba.db.register_model(MemoryProvenance)


def normalize_trust_source(value: str) -> str:
    source = (value or "").strip().lower().replace("-", "_")
    return source if source in TRUST_SOURCES else "agent_suggested"


def normalize_capture_origin(value: str) -> str:
    origin = (value or "").strip().lower().replace("-", "_")
    return origin or "store"


def compute_trust_score(
    *,
    trust_source: str,
    capture_origin: str,
    confidence: float,
    memory_type: str,
    usage: typing.Any = None,
) -> float:
    """Compute a compact trust score for supersession decisions.

    This is a policy score, not a model score: source/origin dominate, confidence
    is honored, and explicit user feedback gives a bounded reinforcement bump.
    """
    source = normalize_trust_source(trust_source)
    origin = normalize_capture_origin(capture_origin)
    base = max(0.0, min(1.0, float(confidence or 0.0)))
    score = (
        base
        * _SOURCE_WEIGHTS.get(source, 0.85)
        * _ORIGIN_WEIGHTS.get(origin, 1.0)
        * _TYPE_WEIGHTS.get(str(memory_type or "").upper(), 1.0)
    )
    if usage is not None:
        use_count = int(getattr(usage, "use_count", 0) or 0)
        save_count = int(getattr(usage, "save_count", 0) or 0)
        match_count = int(getattr(usage, "match_count", 0) or 0)
        score += min(0.20, 0.03 * use_count + 0.02 * save_count + 0.01 * match_count)
    return round(score, 4)


def append_event(
    *,
    memory_id: str,
    occurred_at: str = "",
    observed_at: str = "",
    source_file: str = "",
    source_span: str = "",
    source_url: str = "",
    extraction_agent: str = "",
    extraction_version: str = "",
    source_session: str = "",
    trust_source: str = "agent_suggested",
    capture_origin: str = "store",
    trust_score: float = 0.0,
    now: float,
) -> MemoryProvenance:
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    normalized_source = normalize_trust_source(trust_source)
    normalized_origin = normalize_capture_origin(capture_origin)
    return MemoryProvenance.create(
        memory_id=memory_id,
        occurred_at=occurred_at,
        observed_at=observed_at or created_at_iso,
        source_file=source_file,
        source_span=source_span,
        source_url=source_url,
        extraction_agent=extraction_agent,
        extraction_version=extraction_version,
        source_session=source_session,
        trust_source=normalized_source,
        capture_origin=normalized_origin,
        trust_score=float(trust_score or 0.0),
        created_at=now,
        created_at_iso=created_at_iso,
    )


def latest_for(memory_ids: list[str]) -> dict[str, MemoryProvenance]:
    if not memory_ids:
        return {}
    rows = (
        MemoryProvenance.select()
        .where(MemoryProvenance.memory_id.in_(memory_ids))
        .order_by(MemoryProvenance.created_at.asc(), MemoryProvenance.id.asc())
    )
    out: dict[str, MemoryProvenance] = {}
    for row in rows:
        out[row.memory_id] = row
    return out

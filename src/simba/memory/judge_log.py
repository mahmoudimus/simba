"""Append-only judge decision log for replayable memory adjudication.

Rows record the inputs and outcome for a write-time judge decision before the
caller commits the derived memory side effect. The first row for a
``decision_key`` wins; later attempts read it back and reuse the same decision
instead of asking the judge again.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import simba._vendor.peewee as pw
import simba.db


def _ensure_column(conn, table: str, name: str, spec: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")


def _init_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_judge_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "decision_key VARCHAR(128) NOT NULL UNIQUE, "
        "project_path TEXT NOT NULL DEFAULT '', "
        "strategy VARCHAR(128) NOT NULL DEFAULT '', "
        "input_memory_ids TEXT NOT NULL DEFAULT '[]', "
        "winner_id VARCHAR(64) NOT NULL DEFAULT '', "
        "loser_ids TEXT NOT NULL DEFAULT '[]', "
        "judge_kind VARCHAR(64) NOT NULL DEFAULT '', "
        "judge_model TEXT NOT NULL DEFAULT '', "
        "prompt_hash VARCHAR(64) NOT NULL DEFAULT '', "
        "config_hash VARCHAR(64) NOT NULL DEFAULT '', "
        "decision TEXT NOT NULL DEFAULT '{}', "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "created_at_iso VARCHAR(32) NOT NULL DEFAULT '')"
    )
    for name, spec in (
        ("project_path", "TEXT NOT NULL DEFAULT ''"),
        ("strategy", "VARCHAR(128) NOT NULL DEFAULT ''"),
        ("input_memory_ids", "TEXT NOT NULL DEFAULT '[]'"),
        ("winner_id", "VARCHAR(64) NOT NULL DEFAULT ''"),
        ("loser_ids", "TEXT NOT NULL DEFAULT '[]'"),
        ("judge_kind", "VARCHAR(64) NOT NULL DEFAULT ''"),
        ("judge_model", "TEXT NOT NULL DEFAULT ''"),
        ("prompt_hash", "VARCHAR(64) NOT NULL DEFAULT ''"),
        ("config_hash", "VARCHAR(64) NOT NULL DEFAULT ''"),
        ("decision", "TEXT NOT NULL DEFAULT '{}'"),
        ("created_at", "REAL NOT NULL DEFAULT 0.0"),
        ("created_at_iso", "VARCHAR(32) NOT NULL DEFAULT ''"),
    ):
        _ensure_column(conn, "memory_judge_log", name, spec)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_judge_log_key "
        "ON memory_judge_log(decision_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_judge_log_project "
        "ON memory_judge_log(project_path, created_at)"
    )


simba.db.register_schema(_init_schema)


class MemoryJudgeLog(simba.db.BaseModel):
    id = pw.AutoField()
    decision_key = pw.CharField(max_length=128, unique=True)
    project_path = pw.TextField(default="")
    strategy = pw.CharField(max_length=128, default="")
    input_memory_ids = pw.TextField(default="[]")
    winner_id = pw.CharField(max_length=64, default="")
    loser_ids = pw.TextField(default="[]")
    judge_kind = pw.CharField(max_length=64, default="")
    judge_model = pw.TextField(default="")
    prompt_hash = pw.CharField(max_length=64, default="")
    config_hash = pw.CharField(max_length=64, default="")
    decision = pw.TextField(default="{}")
    created_at = pw.FloatField(default=0.0)
    created_at_iso = pw.CharField(max_length=32, default="")

    class Meta:
        table_name = "memory_judge_log"


simba.db.register_model(MemoryJudgeLog)


def stable_hash(value: Any) -> str:
    """Return a stable sha256 hex digest for strings or JSON-ish values."""
    if isinstance(value, str):
        raw = value
    else:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_conflict_key(
    *,
    project_path: str,
    new_id: str,
    neighbor_id: str,
    prompt_hash: str,
    config_hash: str,
) -> str:
    """Stable key for one write-time conflict pair adjudication."""
    lo, hi = (new_id, neighbor_id) if new_id <= neighbor_id else (neighbor_id, new_id)
    return stable_hash(
        {
            "kind": "memory_write_conflict_v1",
            "project_path": project_path,
            "lo": lo,
            "hi": hi,
            "prompt_hash": prompt_hash,
            "config_hash": config_hash,
        }
    )


def get(decision_key: str) -> MemoryJudgeLog | None:
    return MemoryJudgeLog.get_or_none(MemoryJudgeLog.decision_key == decision_key)


def decision_payload(row: MemoryJudgeLog) -> dict[str, Any]:
    try:
        payload = json.loads(row.decision or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def loser_ids(row: MemoryJudgeLog) -> list[str]:
    try:
        payload = json.loads(row.loser_ids or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def record(
    *,
    decision_key: str,
    project_path: str,
    strategy: str,
    input_memory_ids: list[str],
    winner_id: str,
    loser_ids: list[str],
    judge_kind: str,
    judge_model: str,
    prompt_hash: str,
    config_hash: str,
    decision: dict[str, Any],
    now: float,
) -> MemoryJudgeLog:
    """Append one decision, or return the existing first decision for the key."""
    existing = get(decision_key)
    if existing is not None:
        return existing
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    try:
        return MemoryJudgeLog.create(
            decision_key=decision_key,
            project_path=project_path,
            strategy=strategy,
            input_memory_ids=json.dumps(input_memory_ids),
            winner_id=winner_id,
            loser_ids=json.dumps(loser_ids),
            judge_kind=judge_kind,
            judge_model=judge_model,
            prompt_hash=prompt_hash,
            config_hash=config_hash,
            decision=json.dumps(decision, sort_keys=True),
            created_at=now,
            created_at_iso=created_at_iso,
        )
    except pw.IntegrityError:
        row = get(decision_key)
        if row is None:
            raise
        return row


def recent(*, project_path: str = "", limit: int = 20) -> list[MemoryJudgeLog]:
    query = MemoryJudgeLog.select()
    if project_path:
        query = query.where(MemoryJudgeLog.project_path == project_path)
    return list(query.order_by(MemoryJudgeLog.created_at.desc()).limit(limit))

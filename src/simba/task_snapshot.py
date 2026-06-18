"""Append-only active task snapshots.

Snapshots are compact current-work records, separate from semantic memory. A
``save`` appends an active row; ``clear`` appends a cleared row. The newest row
for a project/session is authoritative, so old snapshots remain auditable while
the latest active snapshot wins.
"""

from __future__ import annotations

import json
import time
from typing import Any

import simba._vendor.peewee as pw
import simba.db

STATUS_ACTIVE = "active"
STATUS_CLEARED = "cleared"


def _ensure_column(conn, table: str, name: str, spec: str) -> None:
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")


def _init_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS task_snapshots ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "project_path TEXT NOT NULL DEFAULT '', "
        "session_id TEXT NOT NULL DEFAULT '', "
        "task TEXT NOT NULL DEFAULT '', "
        "summary TEXT NOT NULL DEFAULT '', "
        "branch TEXT NOT NULL DEFAULT '', "
        "worktree TEXT NOT NULL DEFAULT '', "
        "files TEXT NOT NULL DEFAULT '[]', "
        "blockers TEXT NOT NULL DEFAULT '[]', "
        "next_step TEXT NOT NULL DEFAULT '', "
        "status VARCHAR(32) NOT NULL DEFAULT 'active', "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "created_at_iso VARCHAR(32) NOT NULL DEFAULT '')"
    )
    for name, spec in (
        ("project_path", "TEXT NOT NULL DEFAULT ''"),
        ("session_id", "TEXT NOT NULL DEFAULT ''"),
        ("task", "TEXT NOT NULL DEFAULT ''"),
        ("summary", "TEXT NOT NULL DEFAULT ''"),
        ("branch", "TEXT NOT NULL DEFAULT ''"),
        ("worktree", "TEXT NOT NULL DEFAULT ''"),
        ("files", "TEXT NOT NULL DEFAULT '[]'"),
        ("blockers", "TEXT NOT NULL DEFAULT '[]'"),
        ("next_step", "TEXT NOT NULL DEFAULT ''"),
        ("status", "VARCHAR(32) NOT NULL DEFAULT 'active'"),
        ("created_at", "REAL NOT NULL DEFAULT 0.0"),
        ("created_at_iso", "VARCHAR(32) NOT NULL DEFAULT ''"),
    ):
        _ensure_column(conn, "task_snapshots", name, spec)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_snapshots_project "
        "ON task_snapshots(project_path, session_id, created_at)"
    )


simba.db.register_schema(_init_schema)


class TaskSnapshot(simba.db.BaseModel):
    id = pw.AutoField()
    project_path = pw.TextField(default="")
    session_id = pw.TextField(default="")
    task = pw.TextField(default="")
    summary = pw.TextField(default="")
    branch = pw.TextField(default="")
    worktree = pw.TextField(default="")
    files = pw.TextField(default="[]")
    blockers = pw.TextField(default="[]")
    next_step = pw.TextField(default="")
    status = pw.CharField(max_length=32, default=STATUS_ACTIVE)
    created_at = pw.FloatField(default=0.0)
    created_at_iso = pw.CharField(max_length=32, default="")

    class Meta:
        table_name = "task_snapshots"


simba.db.register_model(TaskSnapshot)


def _json_list(values: list[str]) -> str:
    return json.dumps([str(v) for v in values if str(v).strip()])


def _parse_list(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(v) for v in data] if isinstance(data, list) else []


def _iso(now: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


def save(
    *,
    project_path: str,
    session_id: str = "",
    task: str,
    summary: str = "",
    branch: str = "",
    worktree: str = "",
    files: list[str] | None = None,
    blockers: list[str] | None = None,
    next_step: str = "",
    now: float,
) -> TaskSnapshot:
    return TaskSnapshot.create(
        project_path=project_path,
        session_id=session_id,
        task=task,
        summary=summary,
        branch=branch,
        worktree=worktree,
        files=_json_list(files or []),
        blockers=_json_list(blockers or []),
        next_step=next_step,
        status=STATUS_ACTIVE,
        created_at=now,
        created_at_iso=_iso(now),
    )


def clear(
    *,
    project_path: str,
    session_id: str = "",
    reason: str = "",
    now: float,
) -> TaskSnapshot:
    return TaskSnapshot.create(
        project_path=project_path,
        session_id=session_id,
        task=reason,
        status=STATUS_CLEARED,
        created_at=now,
        created_at_iso=_iso(now),
    )


def latest(*, project_path: str, session_id: str = "") -> TaskSnapshot | None:
    query = TaskSnapshot.select().where(TaskSnapshot.project_path == project_path)
    if session_id:
        query = query.where(TaskSnapshot.session_id == session_id)
    row = query.order_by(TaskSnapshot.created_at.desc(), TaskSnapshot.id.desc()).first()
    if row is None or row.status != STATUS_ACTIVE:
        return None
    return row


def to_dict(row: TaskSnapshot) -> dict[str, Any]:
    return {
        "id": int(row.id),
        "project_path": row.project_path,
        "session_id": row.session_id,
        "task": row.task,
        "summary": row.summary,
        "branch": row.branch,
        "worktree": row.worktree,
        "files": _parse_list(row.files),
        "blockers": _parse_list(row.blockers),
        "next_step": row.next_step,
        "status": row.status,
        "created_at": row.created_at,
        "created_at_iso": row.created_at_iso,
    }


def render(row: TaskSnapshot) -> str:
    data = to_dict(row)
    lines = ["<active-task-snapshot>"]
    for key in ("task", "summary", "branch", "worktree", "next_step"):
        val = str(data.get(key) or "").strip()
        if val:
            lines.append(f"{key}: {val}")
    if data["files"]:
        lines.append("files: " + ", ".join(data["files"]))
    if data["blockers"]:
        lines.append("blockers: " + ", ".join(data["blockers"]))
    lines.append(f"updated_at: {data['created_at_iso']}")
    lines.append("</active-task-snapshot>")
    return "\n".join(lines)

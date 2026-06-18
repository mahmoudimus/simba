"""Append-only anticipated-query metadata for memories."""

from __future__ import annotations

import time

import simba._vendor.peewee as pw
import simba.db


def _init_schema(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_anticipated_queries ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "memory_id VARCHAR(64) NOT NULL, "
        "query TEXT NOT NULL, "
        "source VARCHAR(64) NOT NULL DEFAULT 'store', "
        "created_at REAL NOT NULL DEFAULT 0.0, "
        "created_at_iso VARCHAR(32) NOT NULL DEFAULT '')"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_anticipated_memory "
        "ON memory_anticipated_queries(memory_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_anticipated_query "
        "ON memory_anticipated_queries(query)"
    )


simba.db.register_schema(_init_schema)


class AnticipatedQuery(simba.db.BaseModel):
    id = pw.AutoField()
    memory_id = pw.CharField(max_length=64)
    query = pw.TextField()
    source = pw.CharField(max_length=64, default="store")
    created_at = pw.FloatField(default=0.0)
    created_at_iso = pw.CharField(max_length=32, default="")

    class Meta:
        table_name = "memory_anticipated_queries"


simba.db.register_model(AnticipatedQuery)


def normalize_queries(queries: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in queries:
        query = " ".join(str(raw or "").strip().split())
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(query[:300])
        if len(out) >= max(0, limit):
            break
    return out


def append_queries(
    *,
    memory_id: str,
    queries: list[str],
    source: str = "store",
    now: float,
    limit: int = 5,
) -> list[AnticipatedQuery]:
    rows: list[AnticipatedQuery] = []
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    for query in normalize_queries(queries, limit=limit):
        rows.append(
            AnticipatedQuery.create(
                memory_id=memory_id,
                query=query,
                source=source or "store",
                created_at=now,
                created_at_iso=created_at_iso,
            )
        )
    return rows


def list_for(memory_id: str) -> list[AnticipatedQuery]:
    return list(
        AnticipatedQuery.select()
        .where(AnticipatedQuery.memory_id == memory_id)
        .order_by(AnticipatedQuery.created_at.asc(), AnticipatedQuery.id.asc())
    )

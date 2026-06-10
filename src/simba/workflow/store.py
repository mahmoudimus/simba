"""SQLite models for the durable workflow engine (mirrors ``rlm/jobs.py``).

Control-plane only: durable task rows, projection checkpoints, and asset
freshness cursors. All mutable status/cursor state lives here in
``.simba/simba.db`` — never in LanceDB.

The ``(queue, dedup_key)`` UNIQUE index makes enqueue idempotent. SQLite
treats ``NULL`` as distinct in a UNIQUE index, so rows *without* a dedup key
always insert (unlimited), while rows *with* a key collide — exactly the
"unique where dedup_key not null" semantics the spec calls for, without a
partial index.
"""

from __future__ import annotations

import simba._vendor.peewee as pw
import simba.db


class WfTask(simba.db.BaseModel):
    queue = pw.CharField()
    dedup_key = pw.CharField(null=True)
    status = pw.CharField()  # pending | running | done | failed | dead
    payload = pw.TextField()  # JSON
    attempts = pw.IntegerField(default=0)
    max_attempts = pw.IntegerField()
    available_at = pw.CharField()
    created_at = pw.CharField()
    started_at = pw.CharField(null=True)
    finished_at = pw.CharField(null=True)
    error = pw.TextField(null=True)
    result = pw.TextField(null=True)

    class Meta:
        table_name = "wf_tasks"
        indexes = ((("queue", "dedup_key"), True),)  # UNIQUE (NULL = distinct)


class WfCheckpoint(simba.db.BaseModel):
    name = pw.CharField(unique=True)
    position = pw.IntegerField(default=0)
    updated_at = pw.CharField()

    class Meta:
        table_name = "wf_checkpoints"


class WfAsset(simba.db.BaseModel):
    name = pw.CharField(unique=True)
    last_materialized_at = pw.CharField(null=True)
    last_source_position = pw.IntegerField(default=0)

    class Meta:
        table_name = "wf_assets"


simba.db.register_model(WfTask, WfCheckpoint, WfAsset)

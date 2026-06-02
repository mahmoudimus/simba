"""rlm_jobs — coordination table for the autonomous RLM engine.

Control-plane only (dedup + status), never transcript data. The UNIQUE
(transcript_id, project_path) constraint makes claim() idempotent so a
transcript is digested at most once.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db

if TYPE_CHECKING:
    import pathlib


class RlmJob(simba.db.BaseModel):
    transcript_id = pw.CharField(null=True)
    project_path = pw.CharField(null=True)
    status = pw.CharField(null=True)
    engine = pw.CharField(null=True)
    started_at = pw.CharField(null=True)
    finished_at = pw.CharField(null=True)
    n_stored = pw.IntegerField(default=0)

    class Meta:
        table_name = "rlm_jobs"
        primary_key = False  # rowid table, matching the original schema
        indexes = ((("transcript_id", "project_path"), True),)  # UNIQUE


simba.db.register_model(RlmJob)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def claim(
    transcript_id: str,
    project_path: str,
    engine: str,
    *,
    cwd: pathlib.Path | None = None,
) -> bool:
    """Insert a 'running' job. Return True if claimed, False if one exists."""
    with simba.db.connect(cwd):
        try:
            RlmJob.create(
                transcript_id=transcript_id,
                project_path=project_path,
                status="running",
                engine=engine,
                started_at=_now(),
            )
            return True
        except pw.IntegrityError:
            return False


def complete(
    transcript_id: str,
    project_path: str,
    n_stored: int,
    *,
    cwd: pathlib.Path | None = None,
) -> None:
    with simba.db.connect(cwd):
        RlmJob.update(status="done", finished_at=_now(), n_stored=n_stored).where(
            (RlmJob.transcript_id == transcript_id)
            & (RlmJob.project_path == project_path)
        ).execute()

"""episode_jobs — coordination table for episodic consolidation.

Control-plane only (dedup + status), never memory content. The UNIQUE
(session_source, project_path) constraint makes claim() idempotent so a session
is consolidated at most once (until its EPISODE exists, which is the durable
dedup). Modeled on :mod:`simba.rlm.jobs`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db

if TYPE_CHECKING:
    import pathlib


class EpisodeJob(simba.db.BaseModel):
    session_source = pw.CharField(null=True)
    project_path = pw.CharField(null=True)
    status = pw.CharField(null=True)
    started_at = pw.CharField(null=True)
    finished_at = pw.CharField(null=True)

    class Meta:
        table_name = "episode_jobs"
        primary_key = False  # rowid table, matching the original schema
        indexes = ((("session_source", "project_path"), True),)  # UNIQUE


simba.db.register_model(EpisodeJob)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def claim(
    session_source: str,
    project_path: str,
    *,
    cwd: pathlib.Path | None = None,
    stale_after_seconds: int | None = None,
) -> bool:
    """Insert a 'running' job. Return True if claimed, False if one exists.

    A still-'running' job older than ``stale_after_seconds`` is treated as a
    dead dispatch (the detached agent never closed it) and reclaimed, so a
    session is never permanently locked out of re-consolidation. ``done`` jobs
    are never reclaimed — durable dedup there is the stored EPISODE itself.
    """
    with simba.db.connect(cwd):
        if stale_after_seconds is not None:
            cutoff = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - stale_after_seconds),
            )
            EpisodeJob.delete().where(
                (EpisodeJob.session_source == session_source)
                & (EpisodeJob.project_path == project_path)
                & (EpisodeJob.status == "running")
                & (EpisodeJob.started_at < cutoff)
            ).execute()
        try:
            EpisodeJob.create(
                session_source=session_source,
                project_path=project_path,
                status="running",
                started_at=_now(),
            )
            return True
        except pw.IntegrityError:
            return False


def complete(
    session_source: str,
    project_path: str,
    *,
    cwd: pathlib.Path | None = None,
) -> None:
    with simba.db.connect(cwd):
        EpisodeJob.update(status="done", finished_at=_now()).where(
            (EpisodeJob.session_source == session_source)
            & (EpisodeJob.project_path == project_path)
        ).execute()

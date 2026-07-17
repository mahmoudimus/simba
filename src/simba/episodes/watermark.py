"""Incremental discovery watermark (2026-07-17 RSS-storm fix).

Per-project high-water mark of the max ``createdAt`` seen by the last
COMPLETED episode-discovery sweep (``simba.episodes.consolidate.
consolidate_eligible``). NOT config -- mutable sweep state, so it lives in
the peewee sidecar DB (``.simba/simba.db``) next to the ``episode_jobs``
lease table (``simba.episodes.jobs`` / ``simba.workflow.lease``), following
the same ``simba.db.BaseModel`` + ``register_model`` + ``simba.db.connect``
pattern.

Discovery re-runs with ``since=<watermark>`` (see routes.py's ``/list?
since=``); any session that gained a new memory since the watermark
reappears in the scan and is rechecked in full via ``consolidate.
_fetch_session`` -- so the watermark only ever narrows the *rescan* window,
never the correctness of what gets processed (see consolidate.py's module
docstring for the full argument).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import simba._vendor.peewee as pw
import simba.db
import simba.workflow._time as _time

if TYPE_CHECKING:
    import pathlib

# Sentinel key for consolidate_eligible(all_projects=True) sweeps -- distinct
# from "" (a real, if unusual, untagged-projectPath value in the corpus) so
# a global sweep's watermark never collides with -- or is silently read by --
# a per-project one.
ALL_PROJECTS_KEY = "\x00__all_projects__\x00"


class EpisodeDiscoveryWatermark(simba.db.BaseModel):
    project_path = pw.CharField(unique=True)
    high_water_mark = pw.CharField()
    updated_at = pw.CharField()

    class Meta:
        table_name = "episode_discovery_watermark"


simba.db.register_model(EpisodeDiscoveryWatermark)


def _key(project_path: str, *, all_projects: bool) -> str:
    return ALL_PROJECTS_KEY if all_projects else project_path


def get(
    project_path: str,
    *,
    all_projects: bool = False,
    cwd: pathlib.Path | None = None,
) -> str | None:
    """Return the stored watermark for ``project_path`` (or the
    all-projects sentinel when ``all_projects``), or ``None`` if no sweep
    has completed yet."""
    key = _key(project_path, all_projects=all_projects)
    with simba.db.connect(cwd):
        row = EpisodeDiscoveryWatermark.get_or_none(
            EpisodeDiscoveryWatermark.project_path == key
        )
        return row.high_water_mark if row else None


def advance(
    project_path: str,
    high_water_mark: str,
    *,
    all_projects: bool = False,
    now: str | None = None,
    cwd: pathlib.Path | None = None,
) -> None:
    """Set the watermark for ``project_path`` (or the all-projects
    sentinel). Called only after a sweep with no dispatch error -- see
    ``consolidate.consolidate_eligible``'s advance rule."""
    key = _key(project_path, all_projects=all_projects)
    now = _time.resolve(now)
    with simba.db.connect(cwd):
        existing = EpisodeDiscoveryWatermark.get_or_none(
            EpisodeDiscoveryWatermark.project_path == key
        )
        if existing is None:
            EpisodeDiscoveryWatermark.create(
                project_path=key, high_water_mark=high_water_mark, updated_at=now
            )
        else:
            EpisodeDiscoveryWatermark.update(
                high_water_mark=high_water_mark, updated_at=now
            ).where(EpisodeDiscoveryWatermark.project_path == key).execute()

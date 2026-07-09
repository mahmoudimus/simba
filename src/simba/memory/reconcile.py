"""Cross-store drift audit for the memory subsystem (rlm-claude borrow).

Three stores can drift out of sync with each other:

* **LanceDB** (``memories.lance``) -- the append-only source of truth for
  content and vectors.
* **The FTS5 keyword mirror** (``memory_fts.db``) -- derived and rebuildable,
  and deliberately excludes ``SYSTEM`` memories (see :mod:`simba.memory.fts`).
* **The usage sidecar** (``memory_usage`` in the shared ``simba.db``) -- mutable
  ranking signals keyed by the LanceDB id (see :mod:`simba.memory.usage`).

:func:`reconcile` computes the three drift directions -- Lance ids missing
from FTS, FTS ids with no matching Lance row ("ghost" rows), and usage rows
with no matching Lance row ("orphaned" rows) -- and, only when ``apply=True``,
repairs the one safe direction: re-upserting Lance rows missing from FTS.
Ghost FTS rows and orphaned usage rows are always report-only; this module
never deletes anything and never touches Lance vectors.
"""

from __future__ import annotations

import dataclasses
import pathlib
import typing

import simba.db
import simba.memory.config
import simba.memory.fts
import simba.memory.usage

_EXAMPLE_LIMIT_DEFAULT = 10


@dataclasses.dataclass
class ReconcileReport:
    """Full 3-way drift snapshot. ``*_ids`` lists are sorted and complete."""

    lance_total: int
    lance_non_system: int
    fts_total: int
    usage_total: int
    missing_fts_ids: list[str]
    ghost_fts_ids: list[str]
    orphan_usage_ids: list[str]
    repaired_ids: list[str] = dataclasses.field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when no drift was found in any direction."""
        return not (self.missing_fts_ids or self.ghost_fts_ids or self.orphan_usage_ids)


def resolve_data_dir(cwd: pathlib.Path) -> pathlib.Path:
    """Resolve the directory holding ``memories.lance`` / ``memory_fts.db``.

    Mirrors the daemon's own resolution (``simba.memory.server._resolve_db_path``):
    an explicit ``memory.db_path`` config override wins, else ``<cwd>/.simba/memory``.
    Deliberately NOT repo-root-aware (same as the daemon) -- whatever directory
    is passed in is what "wins" unless ``memory.db_path`` is configured.
    """
    config = simba.memory.config.load_config()
    if config.db_path:
        return pathlib.Path(config.db_path)
    return cwd / ".simba" / "memory"


async def _lance_rows_by_id(table: typing.Any) -> dict[str, dict[str, typing.Any]]:
    """Read every row keyed by id, refreshing the handle across process boundaries."""
    if hasattr(table, "checkout_latest"):
        await table.checkout_latest()
    rows = await table.query().to_list()
    return {row["id"]: dict(row) for row in rows if row.get("id")}


def _fts_ids(fts_path: pathlib.Path) -> set[str]:
    with simba.memory.fts.connect(fts_path):
        return {
            row.memory_id
            for row in simba.memory.fts.MemoryFTS.select(
                simba.memory.fts.MemoryFTS.memory_id
            )
        }


def _usage_ids(cwd: pathlib.Path) -> set[str]:
    with simba.db.connect(cwd):
        return {
            row.memory_id
            for row in simba.memory.usage.MemoryUsage.select(
                simba.memory.usage.MemoryUsage.memory_id
            )
        }


async def reconcile(
    table: typing.Any,
    fts_path: pathlib.Path,
    cwd: pathlib.Path,
    *,
    apply: bool = False,
) -> ReconcileReport:
    """Audit 3-way drift; when ``apply``, repair ONLY the safe direction.

    The safe direction is re-upserting Lance rows missing from the FTS mirror
    (using :func:`simba.memory.fts.upsert`, one row per missing id). This NEVER
    deletes Lance rows, NEVER deletes usage rows, and NEVER touches vectors.
    Ghost FTS rows and orphaned usage rows are always report-only -- deleting
    them is a future, separate decision.
    """
    lance_by_id = await _lance_rows_by_id(table)
    lance_ids = set(lance_by_id)
    # FTS never indexes SYSTEM memories (simba.memory.fts._insert), so only
    # non-SYSTEM Lance ids are ever "expected" to appear in the mirror.
    non_system_ids = {
        mid for mid, row in lance_by_id.items() if row.get("type") != "SYSTEM"
    }
    fts_ids = _fts_ids(fts_path)
    usage_ids = _usage_ids(cwd)

    missing_fts_ids = sorted(non_system_ids - fts_ids)
    ghost_fts_ids = sorted(fts_ids - lance_ids)
    orphan_usage_ids = sorted(usage_ids - lance_ids)

    repaired_ids: list[str] = []
    if apply and missing_fts_ids:
        with simba.memory.fts.connect(fts_path):
            for mid in missing_fts_ids:
                simba.memory.fts.upsert(lance_by_id[mid])
        repaired_ids = list(missing_fts_ids)

    return ReconcileReport(
        lance_total=len(lance_ids),
        lance_non_system=len(non_system_ids),
        fts_total=len(fts_ids),
        usage_total=len(usage_ids),
        missing_fts_ids=missing_fts_ids,
        ghost_fts_ids=ghost_fts_ids,
        orphan_usage_ids=orphan_usage_ids,
        repaired_ids=repaired_ids,
    )


def _format_bucket(name: str, desc: str, ids: list[str], limit: int) -> str:
    if not ids:
        return f"{name}: 0 ({desc})"
    shown = ", ".join(ids[:limit])
    extra = len(ids) - limit
    more = f", +{extra} more" if extra > 0 else ""
    return f"{name}: {len(ids)} ({desc}) -> {shown}{more}"


def format_report(
    report: ReconcileReport, *, example_limit: int = _EXAMPLE_LIMIT_DEFAULT
) -> str:
    """Compact human-readable rendering: counts + up to ``example_limit`` ids."""
    lines = [
        f"lance: {report.lance_total} row(s) ({report.lance_non_system} non-SYSTEM)",
        f"fts:   {report.fts_total} row(s)",
        f"usage: {report.usage_total} row(s)",
        "",
        _format_bucket(
            "missing-fts",
            "Lance rows not yet indexed in FTS",
            report.missing_fts_ids,
            example_limit,
        ),
        _format_bucket(
            "ghost-fts",
            "FTS rows with no matching Lance id; report-only",
            report.ghost_fts_ids,
            example_limit,
        ),
        _format_bucket(
            "orphan-usage",
            "usage rows with no matching Lance id; report-only",
            report.orphan_usage_ids,
            example_limit,
        ),
    ]
    if report.repaired_ids:
        lines.append("")
        lines.append(f"repaired: upserted {len(report.repaired_ids)} row(s) into FTS")
    return "\n".join(lines)

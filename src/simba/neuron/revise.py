"""Sub-phase REVISE: AGM contraction via entrenchment order (Task B.5).

Given the UNSAT-core edge ids from VERIFY, drop the weaker conflicting fact in
each pair by stamping it dormant (``valid_to`` set + ``dormant=1``). The row is
never deleted — this is the append-only contract. Entrenchment order: higher
``(occurred_at, valid_from, confidence)`` wins (more entrenched = retained).

Fail-open: any error returns an empty result with ``errors`` incremented.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import typing

import simba.config
import simba.db

if typing.TYPE_CHECKING:
    from simba.neuron.config import NeuronConfig

logger = logging.getLogger("simba.neuron.revise")


@dataclasses.dataclass
class ReviseResult:
    dormant_edge_ids: list[int] = dataclasses.field(default_factory=list)
    retained_edge_ids: list[int] = dataclasses.field(default_factory=list)
    skipped: int = 0
    errors: int = 0


def _neuron_cfg() -> NeuronConfig:
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    return simba.config.load("neuron")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def entrenchment_score(edge: dict) -> tuple[str, str, float]:
    """Return ``(occurred_at, valid_from, confidence)`` for ordering.

    Higher tuple = more entrenched = kept.
    """
    return (
        edge.get("occurred_at") or "",
        edge.get("valid_from") or "",
        float(edge.get("confidence") or 0.0),
    )


def _fetch_edges(unsat_edge_ids: list[int], project_path: str) -> list[dict]:
    import simba.neuron.schema  # ensures dormant column exists
    from simba.kg.store import KgEdge

    _ = simba.neuron.schema
    if not unsat_edge_ids:
        return []
    out: list[dict] = []
    with simba.db.connect():
        q = KgEdge.select().where(
            (KgEdge.id.in_(unsat_edge_ids)) & (KgEdge.project_path == project_path)
        )
        for r in q:
            out.append(
                {
                    "id": r.id,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "occurred_at": r.occurred_at,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                    "confidence": getattr(r, "confidence", 0.8) or 0.8,
                }
            )
    return out


def _stamp_dormant(edge_ids: list[int]) -> None:
    if not edge_ids:
        return
    now = _now()
    with simba.db.connect() as db:
        for eid in edge_ids:
            db.execute_sql(
                "UPDATE kg_edges SET valid_to=?, dormant=1 WHERE id=?",
                (now, eid),
            )


def revise_unsat_core(
    unsat_edge_ids: list[int],
    *,
    project_path: str,
    cfg: NeuronConfig | None = None,
) -> ReviseResult:
    """Mark the weaker edge in each conflicting pair as dormant. Never raises."""
    cfg = cfg or _neuron_cfg()
    if not cfg.revise_enabled:
        return ReviseResult()

    try:
        edges = _fetch_edges(unsat_edge_ids, project_path)
    except Exception:
        logger.debug("revise: fetching edges failed", exc_info=True)
        return ReviseResult(errors=1)

    # Group by (subject, object) to find conflicting pairs.
    groups: dict[tuple, list[dict]] = {}
    for e in edges:
        groups.setdefault((e["subject"], e["object"]), []).append(e)

    dormant: list[int] = []
    retained: list[int] = []
    skipped = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=entrenchment_score)
        # Pair lowest-entrenched against the rest; the weakest goes dormant
        # only if strictly weaker than the strongest in the group.
        weakest = members_sorted[0]
        strongest = members_sorted[-1]
        if entrenchment_score(weakest) == entrenchment_score(strongest):
            skipped += 1
            continue
        dormant.append(weakest["id"])
        retained.extend(m["id"] for m in members if m["id"] != weakest["id"])

    try:
        _stamp_dormant(dormant)
    except Exception:
        logger.debug("revise: stamping dormant failed", exc_info=True)
        return ReviseResult(
            dormant_edge_ids=[],
            retained_edge_ids=retained,
            skipped=skipped,
            errors=1,
        )

    return ReviseResult(
        dormant_edge_ids=dormant,
        retained_edge_ids=retained,
        skipped=skipped,
    )

"""Sub-phase REVISE: AGM contraction via entrenchment order (Task B.5).

Given the UNSAT-core edge ids from VERIFY, drop the weaker conflicting fact in
each pair by stamping it dormant (``valid_to`` set + ``dormant=1``). The row is
never deleted — this is the append-only contract. Entrenchment order: higher
``(occurred_at, valid_from, confidence)`` wins (more entrenched = retained).

Phase 7 (default-OFF): when ``neuron.resolution_ops_enabled`` is True, the
weaker edge is chosen by a typed resolution operator (LWW / Evidence / ...) and
an append-only audit row is written so the superseded fact stays recoverable
(N3 defence). When False, the legacy entrenchment-only path runs unchanged.

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
    """Mark the weaker edge in each conflicting pair as dormant. Never raises.

    Dispatches to the typed-operator path (default-OFF) when
    ``cfg.resolution_ops_enabled`` is True, otherwise the legacy
    entrenchment-only path.
    """
    cfg = cfg or _neuron_cfg()
    if not cfg.revise_enabled:
        return ReviseResult()

    if getattr(cfg, "resolution_ops_enabled", False):
        return _revise_with_operators(
            unsat_edge_ids, project_path=project_path, cfg=cfg
        )
    return _revise_legacy(unsat_edge_ids, project_path=project_path, cfg=cfg)


def _revise_legacy(
    unsat_edge_ids: list[int],
    *,
    project_path: str,
    cfg: NeuronConfig,
) -> ReviseResult:
    """Legacy entrenchment-only dormancy (Phase 7 ops disabled)."""
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


def _to_fact(edge: dict) -> dict:
    """Project a fetched edge dict into the resolve_ops fact shape."""
    return {
        "edge_id": edge["id"],
        "subject": edge["subject"],
        "predicate": edge["predicate"],
        "object": edge["object"],
        "confidence": edge.get("confidence") or 0.8,
        "valid_from": edge.get("valid_from") or "",
        # Open-ended belief-time upper bound so same-subject conflicts overlap.
        "valid_to": edge.get("valid_to") or "9999-12-31",
        "occurred_at": edge.get("occurred_at"),
        "provenance": edge.get("proof") or "",
    }


def _simba_conflict(e_a: dict, e_b: dict) -> bool:
    """True iff the two edges conflict under simba's verifier model.

    Reuses the Z3 verifier's exclusion relation: a shared ``(subject, object)``
    endpoint with either an antonym predicate pair or the same predicate, and
    overlapping belief-time. This is simba's contradiction topology (distinct
    from toki's same-pred / different-object model), so the operator path
    resolves the SAME witnesses the verifier flagged as the UNSAT core.
    """
    from simba.neuron.z3_verify import _antonym_lookup, _overlaps

    if e_a.get("subject") != e_b.get("subject"):
        return False
    if e_a.get("object") != e_b.get("object"):
        return False
    pred_a, pred_b = e_a.get("predicate"), e_b.get("predicate")
    antonym = _antonym_lookup().get(pred_a) == pred_b
    same_pred = pred_a == pred_b
    return (antonym or same_pred) and _overlaps(e_a, e_b)


def _revise_with_operators(
    unsat_edge_ids: list[int],
    *,
    project_path: str,
    cfg: NeuronConfig,
) -> ReviseResult:
    """Phase 7 path: resolve each conflict via a typed operator, write an
    append-only audit row (N3 recoverability), then stamp the loser dormant.

    Conflicts are detected with simba's verifier relation (antonym / same-pred
    on shared endpoints); the winner/loser within a pair is chosen by the typed
    operator's selection key (LWW = most-recent, Evidence = highest-confidence).
    """
    import simba.neuron.resolve_ops as resolve_ops

    try:
        edges = _fetch_edges(unsat_edge_ids, project_path)
    except Exception:
        logger.debug("revise: fetching edges failed", exc_info=True)
        return ReviseResult(errors=1)

    strategy = getattr(cfg, "resolution_default_operator", "lww")
    dormant: list[int] = []
    retained: list[int] = []
    skipped = 0
    errors = 0
    resolved_pairs: set[tuple[int, int]] = set()

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e_a, e_b = edges[i], edges[j]
            if not _simba_conflict(e_a, e_b):
                continue
            pair_key = tuple(sorted((e_a["id"], e_b["id"])))
            if pair_key in resolved_pairs:
                continue
            resolved_pairs.add(pair_key)
            inc, chal = _to_fact(e_a), _to_fact(e_b)
            try:
                winner, audit = resolve_ops.resolve_pair_unchecked(
                    inc, chal, strategy_id=strategy
                )
            except ValueError:
                logger.debug("revise: resolution failed", exc_info=True)
                errors += 1
                continue
            loser_id = audit.loser_edge_id
            try:
                resolve_ops.record_audit(audit, project_path=project_path)
                _stamp_dormant([loser_id])
            except Exception:
                logger.debug("revise: audit/dormant failed", exc_info=True)
                errors += 1
                continue
            dormant.append(loser_id)
            retained.append(winner["edge_id"])

    return ReviseResult(
        dormant_edge_ids=list(dict.fromkeys(dormant)),
        retained_edge_ids=[r for r in dict.fromkeys(retained) if r not in dormant],
        skipped=skipped,
        errors=errors,
    )

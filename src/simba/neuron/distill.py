"""Sub-phase DISTILL: write verified derived edges to ``kg_derived_edges``.

Persists ``DerivedEdge`` candidates (from DERIVE, surviving VERIFY+REVISE) with
their proof chain (``proof="derived:<rule_id|adhoc>"`` + the source edge ids).
Dedup is on the logical key ``(subject, predicate, object, project_path)`` for
currently-valid rows, with the table UNIQUE constraint as a backstop. Never
overwrites — append-only. Fail-open: any error increments ``errors``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import typing

if typing.TYPE_CHECKING:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import DerivedEdge

logger = logging.getLogger("simba.neuron.distill")


@dataclasses.dataclass
class DistillResult:
    added: int = 0
    duplicates: int = 0
    errors: int = 0


def _neuron_cfg() -> NeuronConfig:
    import simba.config
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    return simba.config.load("neuron")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def distill_edges(
    candidates: list[DerivedEdge],
    *,
    project_path: str,
    cfg: NeuronConfig | None = None,
) -> DistillResult:
    """Write verified DerivedEdge candidates to kg_derived_edges. Never raises."""
    cfg = cfg or _neuron_cfg()
    if not cfg.distill_enabled:
        return DistillResult()

    import simba.db
    import simba.neuron.schema  # ensures tables exist

    _ = simba.neuron.schema

    added = 0
    duplicates = 0
    errors = 0
    now = _now()
    try:
        with simba.db.connect() as db:
            for cand in candidates:
                rule_tag = cand.rule_id if cand.rule_id is not None else "adhoc"
                proof = f"derived:{rule_tag}"
                source_ids = json.dumps(cand.source_edge_ids)
                # Logical-key dedup: skip if a currently-valid derived edge with
                # the same (subject, predicate, object, project_path) exists.
                existing = db.execute_sql(
                    "SELECT 1 FROM kg_derived_edges WHERE subject=? AND "
                    "predicate=? AND object=? AND project_path=? AND "
                    "valid_to IS NULL LIMIT 1",
                    (cand.subject, cand.predicate, cand.object, project_path),
                ).fetchone()
                if existing:
                    duplicates += 1
                    continue
                cur = db.execute_sql(
                    "INSERT OR IGNORE INTO kg_derived_edges "
                    "(subject, predicate, object, proof, source_edge_ids, "
                    "rule_id, confidence, valid_from, valid_to, occurred_at, "
                    "project_path, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cand.subject,
                        cand.predicate,
                        cand.object,
                        proof,
                        source_ids,
                        cand.rule_id,
                        cand.confidence,
                        now,
                        None,
                        cand.occurred_at,
                        project_path,
                        now,
                    ),
                )
                if cur.rowcount and cur.rowcount > 0:
                    added += 1
                else:
                    duplicates += 1
    except Exception:
        logger.debug("distill: insert failed", exc_info=True)
        return DistillResult(added=added, duplicates=duplicates, errors=errors + 1)

    return DistillResult(added=added, duplicates=duplicates, errors=errors)

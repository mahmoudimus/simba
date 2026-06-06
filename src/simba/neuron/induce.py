"""Sub-phase INDUCE: promote recurring derivation patterns to ``kg_rules``.

Scans ``kg_derived_edges`` for rules that fired ``>= induce_min_activations``
times at ``avg_confidence >= induce_min_confidence`` and promotes them to
``kg_rules`` via ``INSERT OR IGNORE`` (a re-promotion is ``already_known``, never
a duplicate). The LLM is not involved here — promotion is a frequency/confidence
gate over already-materialized provenance. Fail-open: errors increment ``errors``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import typing

if typing.TYPE_CHECKING:
    from simba.neuron.config import NeuronConfig

logger = logging.getLogger("simba.neuron.induce")


@dataclasses.dataclass
class InduceResult:
    promoted: int = 0
    already_known: int = 0
    below_threshold: int = 0
    errors: int = 0


def _neuron_cfg() -> NeuronConfig:
    import simba.config
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    return simba.config.load("neuron")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def induce_rules(
    *,
    project_path: str,
    cfg: NeuronConfig | None = None,
) -> InduceResult:
    """Promote recurring derivation patterns to kg_rules. Never raises."""
    cfg = cfg or _neuron_cfg()
    if not cfg.induce_enabled:
        return InduceResult()

    import simba.db
    import simba.neuron.schema  # ensures tables exist

    _ = simba.neuron.schema

    promoted = 0
    already_known = 0
    below_threshold = 0
    now = _now()
    try:
        with simba.db.connect() as db:
            rows = db.execute_sql(
                "SELECT rule_id, predicate, COUNT(*) AS n, AVG(confidence) AS c "
                "FROM kg_derived_edges "
                "WHERE rule_id IS NOT NULL AND project_path=? "
                "GROUP BY rule_id, predicate",
                (project_path,),
            ).fetchall()
            for rule_id, predicate, n, avg_conf in rows:
                if (
                    n < cfg.induce_min_activations
                    or (avg_conf or 0.0) < cfg.induce_min_confidence
                ):
                    below_threshold += 1
                    continue
                rule_text = f"rule:{rule_id}:{predicate}"
                cur = db.execute_sql(
                    "INSERT OR IGNORE INTO kg_rules "
                    "(rule_text, head_predicate, confidence, activation_count, "
                    "created_at, last_fired_at) VALUES (?,?,?,?,?,?)",
                    (rule_text, predicate, avg_conf, n, now, now),
                )
                if cur.rowcount and cur.rowcount > 0:
                    promoted += 1
                else:
                    already_known += 1
    except Exception:
        logger.debug("induce: promotion failed", exc_info=True)
        return InduceResult(
            promoted=promoted,
            already_known=already_known,
            below_threshold=below_threshold,
            errors=1,
        )

    return InduceResult(
        promoted=promoted,
        already_known=already_known,
        below_threshold=below_threshold,
    )

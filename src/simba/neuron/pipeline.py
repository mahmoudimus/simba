"""Neuro-symbolic deductive distillation pipeline (Task B.8).

Wires the five sub-phases into one pass:

    DERIVE → VERIFY → (REVISE if UNSAT) → DISTILL → INDUCE

The LLM lives only in extraction upstream; this pass is a pure
consistency/closure engine. Every derived edge keeps its rule provenance and a
retracted edge is stamped dormant (never deleted). Fail-open: never raises.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

from simba.neuron.derive import run_derive
from simba.neuron.distill import distill_edges
from simba.neuron.induce import induce_rules
from simba.neuron.revise import revise_unsat_core
from simba.neuron.z3_verify import run_verify

if typing.TYPE_CHECKING:
    from simba.neuron.config import NeuronConfig

logger = logging.getLogger("simba.neuron.pipeline")


@dataclasses.dataclass
class DistillationResult:
    status: str  # "ok" | "disabled" | "error"
    candidates: int = 0
    satisfiable: bool = True
    unsat_core_size: int = 0
    dormant: int = 0
    distilled: int = 0
    promoted_rules: int = 0
    errors: int = 0


def _neuron_cfg() -> NeuronConfig:
    import simba.config
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    return simba.config.load("neuron")


def distillation_pass(
    *,
    project_path: str,
    cfg: NeuronConfig | None = None,
) -> DistillationResult:
    """Run the full 5-step deductive distillation pipeline. Never raises."""
    cfg = cfg or _neuron_cfg()
    if not cfg.enabled:
        return DistillationResult(status="disabled")

    try:
        derive_res = run_derive(project_path, cfg=cfg)
        verify_res = run_verify(project_path, cfg=cfg)

        dormant = 0
        if not verify_res.satisfiable:
            revise_res = revise_unsat_core(
                verify_res.unsat_edge_ids, project_path=project_path, cfg=cfg
            )
            dormant = len(revise_res.dormant_edge_ids)

        distill_res = distill_edges(
            derive_res.candidates, project_path=project_path, cfg=cfg
        )
        induce_res = induce_rules(project_path=project_path, cfg=cfg)
    except Exception:
        logger.debug("distillation pass failed", exc_info=True)
        return DistillationResult(status="error", errors=1)

    return DistillationResult(
        status="ok",
        candidates=len(derive_res.candidates),
        satisfiable=verify_res.satisfiable,
        unsat_core_size=len(verify_res.unsat_edge_ids),
        dormant=dormant,
        distilled=distill_res.added,
        promoted_rules=induce_res.promoted,
        errors=(
            derive_res.errors
            + verify_res.errors
            + distill_res.errors
            + induce_res.errors
        ),
    )

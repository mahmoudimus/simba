"""Sub-phase VERIFY: Z3 constraint encoding + UNSAT core (Task B.4).

Encodes a set of edges as Z3 Boolean assertions under antonym + bitemporal
mutual-exclusion constraints, checks satisfiability, and on UNSAT extracts the
minimal conflicting subset (the offending ``kg_edges.id`` values).

Caveat: the solver guarantees consistency *of the formalization*, not that the
NL→logic translation is faithful. It is a closure/consistency engine, not a
truth oracle. Fail-open: any error returns ``satisfiable=True`` with no core.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

if typing.TYPE_CHECKING:
    from simba.neuron.config import NeuronConfig

logger = logging.getLogger("simba.neuron.z3_verify")

# Antonym predicate pairs: two edges sharing (subject, object) but with
# mutually-exclusive predicates cannot both hold.
_ANTONYMS: list[tuple[str, str]] = [
    ("uses", "does_not_use"),
    ("prefers", "avoids"),
    ("fixes", "breaks"),
]


@dataclasses.dataclass
class VerifyResult:
    satisfiable: bool
    unsat_edge_ids: list[int] = dataclasses.field(default_factory=list)
    checked_edges: int = 0
    errors: int = 0
    raw_output: str = ""


def _neuron_cfg() -> NeuronConfig:
    import simba.config
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    return simba.config.load("neuron")


def _antonym_lookup() -> dict[str, str]:
    table: dict[str, str] = {}
    for a, b in _ANTONYMS:
        table[a] = b
        table[b] = a
    return table


def _overlaps(e_a: dict, e_b: dict) -> bool:
    """True when the two edges' belief-time windows overlap (open = +inf)."""
    a_from = e_a.get("valid_from") or ""
    a_to = e_a.get("valid_to")
    b_from = e_b.get("valid_from") or ""
    b_to = e_b.get("valid_to")
    # [a_from, a_to) overlaps [b_from, b_to)  (None upper bound = open)
    if a_to is not None and a_to <= b_from:
        return False
    return not (b_to is not None and b_to <= a_from)


def build_z3_script(edges: list[dict]) -> str:
    """Return a self-contained Z3 script that checks edge-set consistency.

    Declares one Bool per edge (named ``e<id>``), asserts each edge holds
    (tracked, so the UNSAT core names the offending ids), adds antonym and
    overlapping-belief-time mutual-exclusion constraints, then prints ``SAT`` or
    ``UNSAT:<id1>,<id2>,...``.
    """
    antonyms = _antonym_lookup()
    exclusions: list[tuple[int, int]] = []
    for i, e_a in enumerate(edges):
        for e_b in edges[i + 1 :]:
            same_endpoints = e_a.get("subject") == e_b.get("subject") and e_a.get(
                "object"
            ) == e_b.get("object")
            if not same_endpoints:
                continue
            pred_a = e_a.get("predicate")
            pred_b = e_b.get("predicate")
            antonym = antonyms.get(pred_a) == pred_b
            same_pred = pred_a == pred_b
            if (antonym or same_pred) and _overlaps(e_a, e_b):
                exclusions.append((e_a["id"], e_b["id"]))

    ids = [int(e["id"]) for e in edges]
    lines = [
        "s = Solver()",
        "s.set(unsat_core=True)",
    ]
    for eid in ids:
        lines.append(f'e{eid} = Bool("e{eid}")')
    for eid in ids:
        lines.append(f's.assert_and_track(e{eid}, "e{eid}")')
    for a, b in exclusions:
        lines.append(f"s.add(Not(And(e{a}, e{b})))")
    lines.append("r = s.check()")
    lines.append("if r == unsat:")
    lines.append("    core = s.unsat_core()")
    lines.append('    print("UNSAT:" + ",".join(str(c) for c in core))')
    lines.append("else:")
    lines.append('    print("SAT")')
    return "\n".join(lines) + "\n"


def _fetch_edges(project_path: str, sample_size: int) -> list[dict]:
    import simba.db
    import simba.neuron.schema  # ensures dormant column exists
    from simba.kg.store import KgEdge

    _ = simba.neuron.schema
    out: list[dict] = []
    with simba.db.connect():
        q = (
            KgEdge.select()
            .where((KgEdge.project_path == project_path) & (KgEdge.valid_to.is_null()))
            .limit(sample_size)
        )
        for r in q:
            if getattr(r, "dormant", 0):
                continue
            out.append(
                {
                    "id": r.id,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                }
            )
    return out


def _parse_output(output: str) -> tuple[bool, list[int]]:
    for line in output.splitlines():
        line = line.strip()
        if line == "SAT":
            return True, []
        if line.startswith("UNSAT:"):
            ids: list[int] = []
            for tok in line[len("UNSAT:") :].split(","):
                tok = tok.strip()
                if tok.startswith("e") and tok[1:].isdigit():
                    ids.append(int(tok[1:]))
            return False, ids
    # No verdict parsed → treat as satisfiable (fail-open).
    return True, []


def run_verify(
    project_path: str,
    *,
    cfg: NeuronConfig | None = None,
    extra_edges: list[dict] | None = None,
) -> VerifyResult:
    """Run Z3 consistency check. Returns VerifyResult. Never raises."""
    cfg = cfg or _neuron_cfg()
    if not cfg.verify_enabled:
        return VerifyResult(satisfiable=True, unsat_edge_ids=[])

    try:
        edges = _fetch_edges(project_path, cfg.contradiction_sample_size)
    except Exception:
        logger.debug("verify: fetching edges failed", exc_info=True)
        return VerifyResult(satisfiable=True, unsat_edge_ids=[], errors=1)

    if extra_edges:
        edges = [*edges, *extra_edges]

    if not edges:
        return VerifyResult(satisfiable=True, unsat_edge_ids=[], checked_edges=0)

    try:
        import simba.neuron.verify as verify

        script = build_z3_script(edges)
        output = verify.verify_z3(script)
    except Exception:
        logger.debug("verify: z3 run failed", exc_info=True)
        return VerifyResult(
            satisfiable=True,
            unsat_edge_ids=[],
            checked_edges=len(edges),
            errors=1,
        )

    satisfiable, unsat_ids = _parse_output(output)
    return VerifyResult(
        satisfiable=satisfiable,
        unsat_edge_ids=unsat_ids,
        checked_edges=len(edges),
        raw_output=output,
    )

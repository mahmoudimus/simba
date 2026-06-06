"""Sub-phase DERIVE: Datalog materialization over ``kg_edges`` (Task B.3).

Exports the currently-valid edges for a project as Soufflé ``.facts``, runs the
seed Horn rules, and collects candidate derived edges with provenance (the
source ``kg_edges.id`` values that fired each rule). Fail-open: any error returns
an empty ``DeriveResult`` with ``errors`` incremented; never raises.

The LLM lives only in the extraction pipeline — Soufflé here is a pure closure
engine, so every derived edge is tagged with its rule provenance downstream.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import pathlib
import shutil
import subprocess
import tempfile
import typing

if typing.TYPE_CHECKING:
    from simba.neuron.config import NeuronConfig

logger = logging.getLogger("simba.neuron.derive")


@dataclasses.dataclass
class DerivedEdge:
    subject: str
    predicate: str
    object: str
    source_edge_ids: list[int]
    rule_id: int | None
    confidence: float = 0.8
    occurred_at: str | None = None


@dataclasses.dataclass
class DeriveResult:
    candidates: list[DerivedEdge] = dataclasses.field(default_factory=list)
    rules_applied: int = 0
    edges_fed: int = 0
    errors: int = 0
    souffle_output: str = ""


_SEED_RULES: str = """\
// Seed Horn rules for Phase 7 DERIVE pass
.decl edge(sub:symbol, pred:symbol, obj:symbol, id:number)
.input edge
.decl derived(sub:symbol, pred:symbol, obj:symbol, via1:number, via2:number)

derived(A, "transitively_uses", C, ID1, ID2) :-
    edge(A, "uses", B, ID1),
    edge(B, "uses", C, ID2),
    A != C.

derived(A, "co_occurs_with", B, ID1, ID2) :-
    edge(A, "causes", X, ID1),
    edge(B, "causes", X, ID2),
    A != B.

.output derived(IO=stdout)
"""


def _neuron_cfg() -> NeuronConfig:
    import simba.config
    import simba.neuron.config  # registers section

    _ = simba.neuron.config
    return simba.config.load("neuron")


def _fetch_edges(project_path: str, max_edges: int) -> list[tuple]:
    """Return up to ``max_edges`` currently-valid, non-dormant edges.

    Each tuple is ``(id, subject, predicate, object)``.
    """
    import simba.db
    import simba.neuron.schema  # ensures dormant column exists
    from simba.kg.store import KgEdge

    _ = simba.neuron.schema
    rows: list[tuple] = []
    with simba.db.connect():
        q = (
            KgEdge.select(KgEdge.id, KgEdge.subject, KgEdge.predicate, KgEdge.object)
            .where((KgEdge.project_path == project_path) & (KgEdge.valid_to.is_null()))
            .limit(max_edges)
        )
        for r in q:
            dormant = getattr(r, "dormant", 0)
            if dormant:
                continue
            rows.append((r.id, r.subject, r.predicate, r.object))
    return rows


def _parse_derived(stdout: str) -> list[DerivedEdge]:
    candidates: list[DerivedEdge] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        sub, pred, obj, via1, via2 = parts[0], parts[1], parts[2], parts[3], parts[4]
        source_ids: list[int] = []
        for v in (via1, via2):
            with contextlib.suppress(ValueError):
                source_ids.append(int(v))
        # Skip Soufflé's column-header row (via columns are non-numeric there).
        if not source_ids:
            continue
        candidates.append(
            DerivedEdge(
                subject=sub,
                predicate=pred,
                object=obj,
                source_edge_ids=source_ids,
                rule_id=None,
            )
        )
    return candidates


def run_derive(
    project_path: str,
    *,
    cfg: NeuronConfig | None = None,
    extra_rules: str = "",
) -> DeriveResult:
    """Materialize derived edges via Soufflé. Returns DeriveResult. Never raises."""
    cfg = cfg or _neuron_cfg()
    if not cfg.derive_enabled or not cfg.souffle_cmd:
        return DeriveResult()

    facts_dir: str | None = None
    dl_file: str | None = None
    try:
        edges = _fetch_edges(project_path, cfg.derive_max_edges)
    except Exception:
        logger.debug("derive: fetching edges failed", exc_info=True)
        return DeriveResult(errors=1)

    try:
        facts_dir = tempfile.mkdtemp(prefix="simba_derive_")
        facts_path = pathlib.Path(facts_dir) / "edge.facts"
        facts_path.write_text(
            "".join(f"{s}\t{p}\t{o}\t{eid}\n" for (eid, s, p, o) in edges)
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dl", delete=False) as f:
            f.write(_SEED_RULES + extra_rules)
            dl_file = f.name

        result = subprocess.run(
            [cfg.souffle_cmd, "-F", facts_dir, "-D", "-", dl_file],
            capture_output=True,
            text=True,
            timeout=60,
        )
        candidates = _parse_derived(result.stdout)
        return DeriveResult(
            candidates=candidates,
            rules_applied=2,
            edges_fed=len(edges),
            souffle_output=result.stdout,
        )
    except Exception:
        logger.debug("derive: souffle run failed", exc_info=True)
        return DeriveResult(errors=1, edges_fed=len(edges))
    finally:
        if dl_file:
            with contextlib.suppress(OSError):
                pathlib.Path(dl_file).unlink()
        if facts_dir:
            shutil.rmtree(facts_dir, ignore_errors=True)

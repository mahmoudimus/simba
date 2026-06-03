"""Bundled-dataset lookup + human-readable report formatting."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simba.eval.runner import EvalReport


def default_dataset_path() -> pathlib.Path:
    """Path to the bundled seed dataset."""
    return pathlib.Path(__file__).parent / "datasets" / "seed.json"


def format_report(rep: EvalReport, *, top_n_worst: int = 0) -> str:
    """Render an :class:`EvalReport` as a readable text block."""
    plural = "" if rep.n_cases == 1 else "s"
    lines = [f"eval: {rep.dataset_name}  ({rep.n_cases} case{plural})", ""]

    # Aggregate metrics, grouped by family, in a stable order.
    order: list[str] = []
    for fam in ("recall", "precision", "hit", "ndcg"):
        order += [f"{fam}@{k}" for k in rep.ks]
    order.append("mrr")
    width = max((len(n) for n in order), default=3)
    for name in order:
        if name in rep.aggregate:
            lines.append(f"  {name.ljust(width)}  {rep.aggregate[name]:.3f}")

    if top_n_worst > 0:
        ranked = sorted(rep.per_case, key=lambda c: c.metrics.get("mrr", 0.0))
        worst = [c for c in ranked if c.metrics.get("mrr", 0.0) < 1.0][:top_n_worst]
        if worst:
            lines += ["", f"worst {len(worst)} case(s) by MRR:"]
            for c in worst:
                lines.append(
                    f"  [{c.case_id}] mrr={c.metrics.get('mrr', 0.0):.2f}  "
                    f'"{c.query[:60]}"  -> {c.ranked[:3]}'
                )

    return "\n".join(lines)

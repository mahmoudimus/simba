"""Score bake-off runs: per-case metrics, paired bootstrap vs baseline, report.

Pure functions over the run JSONs written by ``arms`` — re-scoring never needs a
model. Ratios are computed with ``fractions.Fraction`` so aggregates are exact
until the final float conversion.
"""

from __future__ import annotations

import json
import math
import typing
from fractions import Fraction

if typing.TYPE_CHECKING:
    import pathlib

KS = (1, 3, 5, 10, 20, 50)
BOOT_COLS = ("R@3", "R@10", "All@10", "MRR", "nDCG@10")
HEADLINE = {"dense": ("R@10", "All@10", "MRR", "nDCG@10"), "pipeline": ("R@3", "MRR")}


def case_metrics(gold: list[str], ranked: list[str]) -> dict[str, Fraction | float]:
    """Recall@k, complete-evidence@k (all gold in top k), MRR, nDCG@10."""
    g = set(gold)
    if not g:
        raise ValueError("case has no gold evidence")
    m: dict[str, Fraction | float] = {}
    for k in KS:
        hit = len(g & set(ranked[:k]))
        m[f"R@{k}"] = Fraction(hit, len(g))
        m[f"All@{k}"] = Fraction(int(hit == len(g)))
    m["MRR"] = next(
        (Fraction(1, i + 1) for i, r in enumerate(ranked) if r in g), Fraction(0)
    )
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(ranked[:10]) if r in g)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))
    m["nDCG@10"] = dcg / idcg
    return m


def mean(values: list) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def paired_bootstrap(
    deltas: list[float], resamples: int, seed: int = 7
) -> tuple[float, float, float]:
    """(mean, lo, hi) 95% percentile CI of the mean of paired per-case deltas.

    Same seed and n ⇒ same resample indices, so every model is resampled on the
    identical case draws.
    """
    import numpy as np

    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    d = np.asarray(deltas, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    boots = np.sort(d[idx].mean(axis=1))
    lo = float(boots[int(0.025 * resamples)])
    hi = float(boots[max(0, math.ceil(0.975 * resamples) - 1)])
    return float(d.mean()), lo, hi


def load_runs(run_dir: pathlib.Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))}


def score_group(
    runs: dict[str, dict], baseline: str, resamples: int, seed: int = 7
) -> dict[str, typing.Any]:
    """Score runs of one (arm, benchmark) on their common case set."""
    if baseline not in runs:
        raise KeyError(f"baseline {baseline!r} has no run here; run it first")
    common = set.intersection(*({c["id"] for c in r["cases"]} for r in runs.values()))
    order = [c["id"] for c in runs[baseline]["cases"] if c["id"] in common]
    dropped = {k: len(r["cases"]) - len(order) for k, r in runs.items()}
    per: dict[str, list[dict]] = {}
    intents: list[str] = []
    for key, r in runs.items():
        by_id = {c["id"]: c for c in r["cases"]}
        per[key] = [case_metrics(by_id[i]["gold"], by_id[i]["ranked"]) for i in order]
        if key == baseline:
            intents = [by_id[i]["intent"] for i in order]
    cols = [f"R@{k}" for k in KS] + ["All@10", "All@50", "MRR", "nDCG@10"]
    summary = {k: {c: mean([m[c] for m in per[k]]) for c in cols} for k in per}
    deltas: dict[str, dict[str, tuple[float, float, float]]] = {}
    for key in per:
        if key == baseline:
            continue
        deltas[key] = {}
        for c in BOOT_COLS:
            d = [
                float(a[c] - b[c]) for a, b in zip(per[key], per[baseline], strict=True)
            ]
            deltas[key][c] = paired_bootstrap(d, resamples, seed)
    by_intent: dict[str, dict[str, float]] = {}
    for it in sorted(set(intents)):
        idx = [i for i, x in enumerate(intents) if x == it]
        by_intent[it] = {"n": len(idx)} | {
            k: mean([per[k][i]["R@10"] for i in idx]) for k in per
        }
    return {
        "n": len(order),
        "dropped": dropped,
        "summary": summary,
        "deltas": deltas,
        "by_intent": by_intent,
        "truncated": {k: r.get("truncated", 0) for k, r in runs.items()},
        "fingerprints": {k: r.get("vector_fingerprint", "") for k, r in runs.items()},
        "runtime": sorted({r.get("runtime", "") for r in runs.values()}),
        # >1 ⇒ runs were made on different benchmark files/samples; re-run them
        "bench_variants": len(
            {json.dumps(r.get("bench", {}).get("case_ids_sha")) for r in runs.values()}
        ),
    }


def _fmt_delta(t: tuple[float, float, float]) -> str:
    m, lo, hi = t
    star = "*" if lo > 0 or hi < 0 else ""
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}"


def render(groups: dict[str, dict], baseline: str) -> str:
    lines = [
        "# Embedder bake-off",
        "",
        f"Baseline: `{baseline}`. Deltas are paired-bootstrap means with 95% CIs; "
        "`*` = CI excludes 0.",
        "",
    ]
    for name, g in groups.items():
        arm = name.split("/")[0]
        head = HEADLINE.get(arm, HEADLINE["dense"])
        lines += [f"## {name} (n={g['n']} cases)", ""]
        if g.get("bench_variants", 1) > 1 or any(g["dropped"].values()):
            lines += [
                "> **Warning:** runs differ in benchmark sample; scored on the "
                f"{g['n']} common cases (dropped: {g['dropped']}). Re-run with "
                "`--force` for a clean comparison.",
                "",
            ]
        lines.append("| model | " + " | ".join(head) + " | truncated |")
        lines.append("|---" * (len(head) + 2) + "|")
        for k, s in sorted(g["summary"].items(), key=lambda kv: -kv[1][head[0]]):
            cells = [f"{s[c]:.4f}" for c in head]
            lines.append(f"| {k} | " + " | ".join(cells) + f" | {g['truncated'][k]} |")
        lines += ["", f"Δ vs `{baseline}`:", ""]
        lines.append("| model | " + " | ".join(f"Δ{c}" for c in head) + " |")
        lines.append("|---" * (len(head) + 1) + "|")
        for k, d in g["deltas"].items():
            lines.append(
                f"| {k} | " + " | ".join(_fmt_delta(d[c]) for c in head) + " |"
            )
        if arm == "dense":
            ks = list(g["summary"])
            lines += ["", "R@10 by intent:", ""]
            lines.append("| intent | n | " + " | ".join(ks) + " |")
            lines.append("|---" * (len(ks) + 2) + "|")
            for it, row in g["by_intent"].items():
                lines.append(
                    f"| {it} | {row['n']} | "
                    + " | ".join(f"{row[k]:.4f}" for k in ks)
                    + " |"
                )
        lines += ["", f"Runtime(s): {', '.join(g['runtime'])}", ""]
    return "\n".join(lines)


def score_all(
    runs_root: pathlib.Path, baseline: str, resamples: int, seed: int = 7
) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for arm_dir in sorted(p for p in runs_root.glob("*") if p.is_dir()):
        for bench_dir in sorted(p for p in arm_dir.glob("*") if p.is_dir()):
            runs = load_runs(bench_dir)
            if baseline in runs and len(runs) > 1:
                groups[f"{arm_dir.name}/{bench_dir.name}"] = score_group(
                    runs, baseline, resamples, seed
                )
    return groups

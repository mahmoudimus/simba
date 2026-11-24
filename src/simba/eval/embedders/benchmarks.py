"""Benchmark loading for the bake-off: LoCoMo (full) and LongMemEval-S (stratified)."""

from __future__ import annotations

import hashlib
import pathlib
import random
import typing

if typing.TYPE_CHECKING:
    from simba.eval.dataset import Dataset

    from .config import EmbedderBakeoffConfig

BENCHMARKS = ("locomo", "lme")
LME_REPO = "xiaowu0162/longmemeval-cleaned"
LME_FILE = "longmemeval_s_cleaned.json"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)


def stratified_sample(datasets: list[Dataset], n: int, seed: int) -> list[Dataset]:
    """Fixed-seed subsample proportional to each question type's share.

    One Dataset per LongMemEval question; ``cases[0].intent`` is its type. Every
    type keeps at least one question. Deterministic for a given (input, n, seed).
    """
    if n <= 0 or n >= len(datasets):
        return list(datasets)
    groups: dict[str, list[Dataset]] = {}
    for d in datasets:
        groups.setdefault(d.cases[0].intent, []).append(d)
    rng = random.Random(seed)
    picked: list[Dataset] = []
    for intent in sorted(groups):
        grp = sorted(groups[intent], key=lambda d: d.name)
        k = max(1, round(n * len(grp) / len(datasets)))
        picked += rng.sample(grp, min(k, len(grp)))
    return picked


def ensure_locomo(cfg: EmbedderBakeoffConfig, root: pathlib.Path) -> pathlib.Path:
    path = cfg.locomo_file(root)
    if path.exists():
        return path
    import httpx

    path.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(LOCOMO_URL, follow_redirects=True, timeout=120)
    r.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    return path


def ensure_lme(cfg: EmbedderBakeoffConfig, root: pathlib.Path) -> pathlib.Path:
    path = cfg.lme_path(root)
    if path.exists():
        return path
    import huggingface_hub

    got = huggingface_hub.hf_hub_download(
        LME_REPO, LME_FILE, repo_type="dataset", local_dir=path.parent
    )
    return pathlib.Path(got)


def load(
    name: str, cfg: EmbedderBakeoffConfig, root: pathlib.Path
) -> tuple[list[Dataset], dict[str, typing.Any]]:
    """Return (datasets, provenance) for a benchmark name."""
    if name == "locomo":
        from simba.eval.benchmarks.locomo import load_locomo

        path = ensure_locomo(cfg, root)
        ds = load_locomo(path)
        sample = {}
    elif name == "lme":
        from simba.eval.benchmarks.longmemeval import load_longmemeval

        path = ensure_lme(cfg, root)
        full = load_longmemeval(path)
        ds = stratified_sample(full, cfg.lme_questions, cfg.sample_seed)
        sample = {"of": len(full), "n": cfg.lme_questions, "seed": cfg.sample_seed}
    else:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    prov = {
        "benchmark": name,
        "path": str(path),
        "sha256_16": digest,
        "sample": sample,
        "n_cases": sum(len(d.cases) for d in ds),
        "n_docs": sum(len(d.corpus) for d in ds),
        "case_ids_sha": hashlib.sha256(
            "\n".join(c.id for d in ds for c in d.cases).encode()
        ).hexdigest()[:16],
    }
    return ds, prov

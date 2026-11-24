"""The two measurement arms.

dense    — rank every corpus unit by cosine alone (no floors, caps, BM25 or
           reranker): isolates the representation.
pipeline — the real Simba recall path with the candidate embedder, in the
           cold-query production shape: effective deployed config with the
           reranker OFF, because the daemon serves the un-reranked fused order on
           a rerank-cache miss (memory/hybrid.py, async rerank branch). LoCoMo only.

Each run writes one JSON with full provenance and per-case rankings, so scoring is
a pure function of files on disk.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import subprocess
import tempfile
import time
import typing

import numpy as np

from . import benchmarks, engine

if typing.TYPE_CHECKING:
    import pathlib

    from .candidates import EmbedderSpec
    from .config import EmbedderBakeoffConfig


def _git_sha(root: pathlib.Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def result_path(out_dir: pathlib.Path, arm: str, bench: str, key: str) -> pathlib.Path:
    return out_dir / "runs" / arm / bench / f"{key}.json"


def fingerprint(spec, cfg, runtime: str, artifact: str) -> str:
    return spec.vector_fingerprint(runtime, artifact, cfg.token_headroom)


def pipeline_memory_config() -> typing.Any:
    """Effective deployed memory config in the cold-query shape (reranker off)."""
    import simba.memory.config as mc

    mcfg = mc.load_config()
    mcfg.llm_rerank_enabled = False  # daemon serves fused order on rerank-cache miss
    return mcfg


def memory_snapshot(mcfg: typing.Any) -> dict[str, typing.Any]:
    return json.loads(json.dumps(dataclasses.asdict(mcfg), default=str))


def identity(
    arm: str,
    spec: EmbedderSpec,
    cfg: EmbedderBakeoffConfig,
    runtime: str,
    artifact: str,
    bench_prov: dict[str, typing.Any],
    memory: dict[str, typing.Any] | None = None,
) -> dict[str, typing.Any]:
    """Everything a saved result depends on. A run is reusable iff it is equal.

    Covers the vectors (fingerprint), the full spec (incl. the post-hoc ``dims``
    slice), the exact benchmark file + sample + case set, the ranking depth, and —
    for the pipeline arm — the whole effective memory config.
    """
    ident = {
        "arm": arm,
        "spec": dataclasses.asdict(spec),
        "vector_fingerprint": fingerprint(spec, cfg, runtime, artifact),
        "bench": {k: bench_prov[k] for k in ("sha256_16", "sample", "case_ids_sha")},
        "top_k": cfg.top_k,
    }
    if arm == "pipeline":
        ident["memory_config"] = memory
    return json.loads(json.dumps(ident))


def _provenance(
    ident, cfg, root, runtime, artifact, bench_prov
) -> dict[str, typing.Any]:
    return {
        "identity": ident,
        "vector_fingerprint": ident["vector_fingerprint"],
        "spec": ident["spec"],
        "runtime": runtime,
        "artifact_sha": artifact,
        "llama_server": cfg.llama_server,
        "token_headroom": cfg.token_headroom,
        "simba_git": _git_sha(root),
        "when": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "bench": bench_prov,
    }


def _server(spec, cfg, gguf, port, out_dir):
    return engine.llama_server(
        cfg.llama_server,
        gguf,
        pooling=spec.pooling,
        ctx=spec.ctx,
        port=port,
        slots=cfg.slots,
        log_path=out_dir / "logs" / f"server_{spec.key}.log",
        gpu_layers=cfg.gpu_layers,
        start_timeout_s=cfg.server_start_timeout_s,
    )


def _embedder(spec, cfg, url, out_dir, runtime, artifact) -> engine.HttpEmbedder:
    return engine.HttpEmbedder(
        spec,
        url,
        out_dir / "cache",
        runtime,
        artifact_sha=artifact,
        batch=cfg.batch,
        headroom=cfg.token_headroom,
        timeout_s=cfg.request_timeout_s,
    )


def _write(out_dir, arm, bench, spec, res) -> pathlib.Path:
    path = result_path(out_dir, arm, bench, spec.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return path


def run_dense(
    spec: EmbedderSpec,
    bench: str,
    cfg: EmbedderBakeoffConfig,
    root: pathlib.Path,
    gguf: pathlib.Path,
    *,
    runtime: str,
    artifact: str,
    progress: typing.Callable[[str], None] = print,
) -> pathlib.Path:
    out_dir = cfg.resolve(cfg.out_dir, root)
    datasets, bprov = benchmarks.load(bench, cfg, root)
    t0 = time.monotonic()
    cases: list[dict[str, typing.Any]] = []
    with _server(spec, cfg, gguf, cfg.port, out_dir) as url:
        emb = _embedder(spec, cfg, url, out_dir, runtime, artifact)
        for i, d in enumerate(datasets, 1):
            ids = [m.id for m in d.corpus]
            docs = emb.embed([m.content for m in d.corpus], "d")
            qs = emb.embed([c.query for c in d.cases], "q")
            for c, s in zip(d.cases, qs @ docs.T, strict=True):
                top = np.argsort(-s, kind="stable")[: cfg.top_k]
                cases.append(
                    {
                        "id": c.id,
                        "intent": c.intent,
                        "gold": c.relevant_ids,
                        "ranked": [ids[j] for j in top],
                    }
                )
            if i % 10 == 0 or i == len(datasets):
                progress(f"  {spec.key} {bench} {i}/{len(datasets)} "
                         f"{time.monotonic() - t0:.0f}s")  # fmt: skip
    ident = identity("dense", spec, cfg, runtime, artifact, bprov)
    res = _provenance(ident, cfg, root, runtime, artifact, bprov)
    res.update(seconds=time.monotonic() - t0, truncated=emb.truncated, cases=cases)
    return _write(out_dir, "dense", bench, spec, res)


def run_pipeline(
    spec: EmbedderSpec,
    cfg: EmbedderBakeoffConfig,
    root: pathlib.Path,
    gguf: pathlib.Path,
    *,
    runtime: str,
    artifact: str,
    progress: typing.Callable[[str], None] = print,
) -> pathlib.Path:
    import simba.eval.recall_adapter as ra
    import simba.eval.runner as rn

    out_dir = cfg.resolve(cfg.out_dir, root)
    datasets, bprov = benchmarks.load("locomo", cfg, root)
    mcfg = pipeline_memory_config()
    t0 = time.monotonic()
    cases: list[dict[str, typing.Any]] = []
    with _server(spec, cfg, gguf, cfg.port + 1, out_dir) as url:
        emb = _embedder(spec, cfg, url, out_dir, runtime, artifact)

        def ed(t: str) -> list[float]:
            return emb.embed([t], "d")[0].tolist()

        def eq(t: str) -> list[float]:
            return emb.embed([t], "q")[0].tolist()

        for d in datasets:
            intent = {c.id: c.intent for c in d.cases}
            gold = {c.id: c.relevant_ids for c in d.cases}
            with tempfile.TemporaryDirectory(prefix="simba-embedders-") as td:
                r = ra.build_retriever(
                    d, mcfg, embed_doc=ed, embed_query=eq, data_dir=td
                )
                rep = rn.run_eval(d, r, ks=(1, 3, 5, 10))
            cases += [
                {
                    "id": c.case_id,
                    "intent": intent[c.case_id],
                    "gold": gold[c.case_id],
                    "ranked": c.ranked,
                    "latency_ms": c.latency_ms,
                }
                for c in rep.per_case
            ]
        progress(f"  {spec.key} pipeline {time.monotonic() - t0:.0f}s")
    snap = memory_snapshot(mcfg)
    ident = identity("pipeline", spec, cfg, runtime, artifact, bprov, snap)
    res = _provenance(ident, cfg, root, runtime, artifact, bprov)
    res.update(seconds=time.monotonic() - t0, truncated=emb.truncated, cases=cases)
    return _write(out_dir, "pipeline", "locomo", spec, res)

"""Correctness and cost checks that gate a candidate before its numbers are trusted.

fetch     — download a candidate GGUF into ``embedders.models_dir``.
refcheck  — GGUF (llama-server) vs the authors' Sentence-Transformers model on
            identical prefixed inputs. A conversion/pooling bug would otherwise
            show up as "this model is worse". Needs optional deps, e.g.
            ``uv run --with sentence-transformers --with torch --with torchvision
            --with pillow simba eval embedders refcheck <key>``.
inprocess — load the GGUF the way the daemon does (in-process llama-cpp-python,
            ``EmbeddingService``'s forced MEAN pooling), report whether it loads,
            the output dim, whether its vectors match the reference-pooled server
            vectors, and latency / throughput / peak RSS.
"""

from __future__ import annotations

import pathlib
import resource
import time
import typing

import numpy as np

from . import engine

if typing.TYPE_CHECKING:
    from .candidates import EmbedderSpec
    from .config import EmbedderBakeoffConfig

# Fixed probe texts: short dialogue turns + questions, the shape Simba stores.
PROBE_DOCS = [
    "Caroline: I went to a LGBTQ support group yesterday and it was so powerful.",
    "Melanie: That's awesome! I just finished a painting of a sunset by the lake.",
    "Caroline: I'm researching adoption agencies; I want to give a kid a home.",
    "Melanie: We took the kids camping at the beach last weekend.",
    "Caroline: My friends threw a surprise party for my transition anniversary.",
    "Melanie: I signed up for a pottery class, it calms me down.",
    "Disable pointwise reranking for count queries; it demotes required members.",
    "The daemon freezes config at startup; restart after `simba config set`.",
    "LanceDB reembed keeps the old vector when a row fails to embed.",
    "Use uv tool install --force --reinstall to refresh the global simba tool.",
    "Prefer hybrid recall: vector plus BM25 fused with reciprocal rank fusion.",
    "Hooks must use bounded tail readers; transcripts can be multiple gigabytes.",
]
PROBE_QUERIES = [
    "When did Caroline go to the support group?",
    "What does Melanie do to relax?",
    "Why is reranking disabled for counting questions?",
    "How do I refresh the installed simba tool?",
]


def fetch(spec: EmbedderSpec, cfg: EmbedderBakeoffConfig) -> pathlib.Path:
    path = spec.local_path(cfg.models_path())
    if path.exists():
        return path
    import huggingface_hub

    got = huggingface_hub.hf_hub_download(spec.repo, spec.file, local_dir=path.parent)
    return pathlib.Path(got)


def _probe_inputs(spec: EmbedderSpec) -> list[str]:
    return [spec.doc_prefix + d for d in PROBE_DOCS] + [
        spec.query_prefix + q for q in PROBE_QUERIES
    ]


def _server_vectors(spec, cfg, gguf, port, log_dir) -> np.ndarray:
    import httpx

    with engine.llama_server(
        cfg.llama_server, gguf, pooling=spec.pooling, ctx=min(spec.ctx, 2048),
        port=port, slots=1, log_path=log_dir / f"check_{spec.key}.log",
    ) as url:  # fmt: skip
        inputs = [t + spec.eos for t in _probe_inputs(spec)]
        r = httpx.post(f"{url}/v1/embeddings", json={"input": inputs}, timeout=300)
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda d: d["index"])
    v = np.array([d["embedding"] for d in rows], dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Per-text cosine + top-5 overlap of query→doc rankings."""
    nd = len(PROBE_DOCS)
    if a.shape != b.shape:
        return {"cos_min": float("nan"), "cos_mean": float("nan"), "top5": float("nan")}
    cos = (a * b).sum(axis=1)
    overlap = []
    for qi in range(nd, len(a)):
        ra = set(np.argsort(-(a[qi] @ a[:nd].T))[:5])
        rb = set(np.argsort(-(b[qi] @ b[:nd].T))[:5])
        overlap.append(len(ra & rb) / 5)
    return {
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "top5": float(np.mean(overlap)),
    }


def refcheck(
    spec: EmbedderSpec,
    cfg: EmbedderBakeoffConfig,
    gguf: pathlib.Path,
    log_dir: pathlib.Path,
) -> dict[str, typing.Any]:
    if not spec.ref_repo:
        raise ValueError(f"{spec.key} has no ref_repo")
    try:
        from sentence_transformers import SentenceTransformer, models
    except ImportError as exc:
        msg = (
            "refcheck needs sentence-transformers + torch: uv run --with "
            "sentence-transformers --with torch --with torchvision --with pillow "
            "simba eval embedders refcheck ..."
        )
        raise RuntimeError(msg) from exc
    g = _server_vectors(spec, cfg, gguf, cfg.port + 2, log_dir)
    if spec.pooling == "mean" and spec.key.startswith("bge"):
        # mirror Simba's MEAN override on BGE's reference transformer
        t = models.Transformer(spec.ref_repo)
        m = SentenceTransformer(
            modules=[t, models.Pooling(t.get_word_embedding_dimension(), "mean")],
            device="cpu",
        )
    else:
        try:
            m = SentenceTransformer(spec.ref_repo, device="cpu")
        except Exception:
            m = SentenceTransformer(spec.ref_repo, device="cpu", trust_remote_code=True)
    r = m.encode(
        _probe_inputs(spec), prompt="", normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float64)
    out = {"key": spec.key, "ref_repo": spec.ref_repo, "dim_gguf": g.shape[1],
           "dim_ref": r.shape[1]} | _agreement(g, r)  # fmt: skip
    out["pass"] = bool(out["cos_min"] >= 0.99 and out["top5"] >= 0.9)
    return out


def inprocess(
    spec: EmbedderSpec,
    cfg: EmbedderBakeoffConfig,
    gguf: pathlib.Path,
    log_dir: pathlib.Path,
) -> dict[str, typing.Any]:
    """Load like the daemon (forced MEAN, see memory/embeddings.py) and measure."""
    import ctypes

    import llama_cpp

    out: dict[str, typing.Any] = {
        "key": spec.key,
        "llama_cpp_python": llama_cpp.__version__,
    }
    cb = llama_cpp.llama_log_callback(lambda *_: None)
    llama_cpp.llama_log_set(cb, ctypes.c_void_p(0))
    t0 = time.perf_counter()
    try:
        m = llama_cpp.Llama(
            model_path=str(gguf), embedding=True, n_ctx=min(spec.ctx, 2048),
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN, n_gpu_layers=-1,
            verbose=False,
        )  # fmt: skip
    except Exception as exc:
        return out | {"loads": False, "error": str(exc)[:200]}
    out["loads"] = True
    out["load_s"] = round(time.perf_counter() - t0, 2)
    texts = _probe_inputs(spec)
    v = np.array(
        [m.create_embedding(t + spec.eos)["data"][0]["embedding"] for t in texts],
        dtype=np.float64,
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out["dim"] = v.shape[1]
    s = _server_vectors(spec, cfg, gguf, cfg.port + 2, log_dir)
    out["dim_server"] = s.shape[1]
    out |= {f"vs_server_{k}": x for k, x in _agreement(v, s).items()}
    out["matches_server"] = bool(
        v.shape == s.shape and out["vs_server_cos_min"] >= 0.999
    )
    qs = [spec.query_prefix + q for q in PROBE_QUERIES] * 25
    for q in qs[:5]:
        m.create_embedding(q)
    lat = []
    for q in qs:
        t = time.perf_counter()
        m.create_embedding(q)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    docs = [spec.doc_prefix + d for d in PROBE_DOCS] * 20
    t = time.perf_counter()
    for d in docs:
        m.create_embedding(d)
    out |= {
        "query_p50_ms": round(lat[len(lat) // 2], 2),
        "query_p95_ms": round(lat[int(0.95 * len(lat))], 2),
        "docs_per_s": round(len(docs) / (time.perf_counter() - t), 1),
        # macOS reports ru_maxrss in bytes, Linux in KiB
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (2**20 if __import__("sys").platform == "darwin" else 2**10),
            1,
        ),
    }
    return out

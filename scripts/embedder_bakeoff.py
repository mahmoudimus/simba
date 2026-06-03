"""Embedder bake-off: measure recall on seed+temporal across embedders.

Not shipped — a dev measurement script. Compares nomic-Q4 (baseline) vs nomic-Q8
vs Qwen3-Embedding-0.6B over the bundled eval datasets, all local GGUF in-process.
"""

from __future__ import annotations

import pathlib
import tempfile

import simba.eval.report as report
import simba.eval.run as run
import simba.memory.config as mc

CANDIDATES = {
    "nomic-Q4 (baseline)": dict(
        model_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
        model_file="nomic-embed-text-v1.5.Q4_K_M.gguf",
        embedding_dims=768,
        embed_doc_prefix="search_document: ",
        embed_query_prefix="search_query: ",
    ),
    "nomic-Q8_0": dict(
        model_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
        model_file="nomic-embed-text-v1.5.Q8_0.gguf",
        embedding_dims=768,
        embed_doc_prefix="search_document: ",
        embed_query_prefix="search_query: ",
    ),
    "Qwen3-Embedding-0.6B-Q8_0": dict(
        model_repo="Qwen/Qwen3-Embedding-0.6B-GGUF",
        model_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
        embedding_dims=1024,
        embed_doc_prefix="",
        embed_query_prefix=(
            "Instruct: Given a query, retrieve relevant memories\nQuery: "
        ),
    ),
}

DATASETS = {
    "seed": report.default_dataset_path(),
    "temporal": report.resolve_dataset("temporal"),
}


def main() -> None:
    base = pathlib.Path("src/simba/eval/datasets")
    _ = base
    for label, overrides in CANDIDATES.items():
        cfg = mc.MemoryConfig(**overrides)
        # rerank off + scoring off: isolate raw embedder retrieval quality
        cfg.llm_rerank_enabled = False
        cfg.scoring_enabled = False
        try:
            ed, eq = run.sync_embedders(cfg)
        except Exception as exc:
            print(f"{label}: LOAD FAILED: {exc}")
            continue
        line = [f"{label:>28}"]
        for ds_name, ds_path in DATASETS.items():
            with tempfile.TemporaryDirectory() as td:
                rep = run.run_dataset(
                    ds_path, ks=(1, 3, 5), data_dir=td,
                    embed_doc=ed, embed_query=eq, cfg=cfg,
                )
            a = rep.aggregate
            line.append(
                f"{ds_name}: r@1={a['recall@1']:.3f} r@3={a['recall@3']:.3f} "
                f"mrr={a['mrr']:.3f}"
            )
        print("  |  ".join(line))


if __name__ == "__main__":
    main()

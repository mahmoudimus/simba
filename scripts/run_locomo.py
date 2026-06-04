"""Run simba recall@k on the LoCoMo benchmark (deterministic, no LLM judge).

Usage: uv run python scripts/run_locomo.py [path-to-locomo10.json] [n-conversations]
Measures whether simba's recall surfaces the gold evidence dia_ids per question
category. Not shipped — a dev measurement script.
"""

from __future__ import annotations

import sys

import simba.eval.benchmarks.locomo as locomo
import simba.eval.benchmarks.run as bench
import simba.eval.run as run
import simba.memory.config as mc


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/locomo10.json"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    datasets = locomo.load_locomo(path)[:n]
    # Wide retrieval so recall@10 is meaningful; rerank off (no LLM); scoring off
    # to isolate raw retrieval quality.
    cfg = mc.MemoryConfig(
        max_results=20,
        max_results_broad=20,
        fts_candidate_pool=40,
        fts_candidate_pool_broad=60,
        llm_rerank_enabled=False,
        scoring_enabled=False,
        expansion_enabled=False,
    )
    embed_doc, embed_query = run.sync_embedders(cfg)
    report = bench.run_recall(
        datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=cfg
    )

    print(
        f"\nLoCoMo recall ({report['n_conversations']} conversations, "
        f"{report['n_cases']} questions)"
    )
    o = report["overall"]
    print(
        f"  OVERALL  recall@1={o['recall@1']:.3f} recall@3={o['recall@3']:.3f} "
        f"recall@5={o['recall@5']:.3f} recall@10={o['recall@10']:.3f} "
        f"mrr={o['mrr']:.3f}"
    )
    for cat, m in report["by_category"].items():
        print(
            f"  {cat:<18} n={m['n']:<4} r@1={m['recall@1']:.3f} "
            f"r@5={m['recall@5']:.3f} r@10={m['recall@10']:.3f} mrr={m['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()

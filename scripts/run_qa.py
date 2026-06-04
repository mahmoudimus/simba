"""Run the LLM-judge QA layer on a sample of LoCoMo / LongMemEval questions.

Usage: uv run python scripts/run_qa.py [locomo|longmemeval] [path] [n] [k]
  bench : "locomo" (default) or "longmemeval"
  path  : dataset json (defaults: /tmp/locomo10.json, /tmp/lme_oracle.json)
  n     : "50" first-N answerable | "perN" N-per-category (representative) |
          "all" everything (default 50)
  k     : retrieved context size (default 10)

Retrieve top-k context -> generate an answer -> grade vs gold, all with the
configured llm client (deepseek-v4-flash). Reports answer accuracy overall +
by category. A dev measurement script, not shipped.
"""

from __future__ import annotations

import sys

import simba.eval.benchmarks.judge as judge
import simba.eval.benchmarks.locomo as locomo
import simba.eval.benchmarks.longmemeval as lme
import simba.eval.run as run
import simba.llm.client as llm_client
import simba.memory.config as mc


def main() -> None:
    bench = sys.argv[1] if len(sys.argv) > 1 else "locomo"
    default_path = "/tmp/locomo10.json" if bench == "locomo" else "/tmp/lme_oracle.json"
    path = sys.argv[2] if len(sys.argv) > 2 else default_path
    # n: "first N" sample; or "perN" for N-per-category stratified (representative)
    n_arg = sys.argv[3] if len(sys.argv) > 3 else "50"
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    if bench == "locomo":
        datasets = locomo.load_locomo(path)
    else:
        datasets = lme.load_longmemeval(path)
    if n_arg.startswith("per"):
        datasets = judge.sample_cases(datasets, per_category=int(n_arg[3:]))
    elif n_arg != "all":
        datasets = judge.sample_cases(datasets, n=int(n_arg))

    cfg = mc.MemoryConfig(
        max_results=max(20, k),
        max_results_broad=max(20, k),
        fts_candidate_pool=40,
        fts_candidate_pool_broad=60,
        llm_rerank_enabled=False,
        scoring_enabled=False,
        expansion_enabled=False,
    )
    embed_doc, embed_query = run.sync_embedders(cfg)
    llm = llm_client.get_client()
    print(f"llm: provider={llm._cfg.provider} model={llm._cfg.model} "
          f"thinking={llm._cfg.thinking!r} available={llm.available()}")

    report = judge.run_qa(
        datasets, embed_doc=embed_doc, embed_query=embed_query, cfg=cfg, llm=llm, k=k
    )

    print(f"\n{bench} QA accuracy (graded={report['n_graded']}, "
          f"skipped={report['n_skipped']}, k={k})")
    print(f"  OVERALL  accuracy={report['overall']['accuracy']:.3f}")
    for cat, m in report["by_category"].items():
        print(f"  {cat:<26} n={m['n']:<4} accuracy={m['accuracy']:.3f}")


if __name__ == "__main__":
    main()

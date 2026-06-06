"""Run the LLM-judge QA layer on a sample of LoCoMo / LongMemEval questions.

Usage: uv run python scripts/run_qa.py [locomo|longmemeval] [path] [n] [k] [flags]
  bench : "locomo" (default) or "longmemeval"
  path  : dataset json (defaults: /tmp/locomo10.json, /tmp/lme_oracle.json)
  n     : "50" first-N answerable | "perN" N-per-category (representative) |
          "all" everything (default 50)
  k     : retrieved context size (default 10)

  --split dev|test|all : case split (default "all")
  --out PATH           : write the JSON report to PATH
  --baseline           : append the report to .simba/eval/baselines/<bench>_qa.jsonl
  --cache PATH         : judge-verdict cache db (default .simba/eval/judge_cache.db)

Retrieve top-k context -> generate an answer (llm) -> grade vs gold (judge).
The answerer comes from the ``llm`` config, the grader from the ``judge`` config,
so the model never grades its own answer. Reports answer accuracy overall + by
category. A dev measurement script, not shipped.
"""

from __future__ import annotations

import json
import pathlib
import sys

import simba.eval.benchmarks.baseline_store as baseline_store
import simba.eval.benchmarks.judge as judge
import simba.eval.benchmarks.judge_cache as judge_cache
import simba.eval.benchmarks.locomo as locomo
import simba.eval.benchmarks.longmemeval as lme
import simba.eval.run as run
import simba.llm.client as llm_client
import simba.llm.judge_config as jcfg
import simba.memory.config as mc


def _flag_value(argv: list[str], name: str, default: str = "") -> str:
    """Return the value following ``--name`` in argv, or default."""
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def main() -> None:
    argv = sys.argv
    # Collect positional args, skipping flags and their values.
    flags_with_values = ("--split", "--out", "--cache")
    skip_next = False
    positional: list[str] = []
    for a in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a in flags_with_values:
            skip_next = True
            continue
        if a.startswith("--"):
            continue
        positional.append(a)

    bench = positional[0] if len(positional) > 0 else "locomo"
    default_path = "/tmp/locomo10.json" if bench == "locomo" else "/tmp/lme_oracle.json"
    path = positional[1] if len(positional) > 1 else default_path
    n_arg = positional[2] if len(positional) > 2 else "50"
    k = int(positional[3]) if len(positional) > 3 else 10

    split = _flag_value(argv, "--split", "all")
    out_path = _flag_value(argv, "--out", "")
    cache_path = _flag_value(argv, "--cache", ".simba/eval/judge_cache.db")
    want_baseline = "--baseline" in argv

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
    judge_client = jcfg.get_judge_client()
    print(
        f"answerer: provider={llm._cfg.provider} model={llm._cfg.model} "
        f"thinking={llm._cfg.thinking!r} available={llm.available()}"
    )
    print(
        f"judge: provider={judge_client._cfg.provider} "
        f"model={judge_client._cfg.model} available={judge_client.available()}"
    )

    cache = judge_cache.JudgeCache(cache_path)
    report = judge.run_qa(
        datasets,
        embed_doc=embed_doc,
        embed_query=embed_query,
        cfg=cfg,
        llm=llm,
        judge=judge_client,
        k=k,
        cache=cache,
        judge_model=judge_client._cfg.model,
    )

    print(
        f"\n{bench} QA accuracy (graded={report['n_graded']}, "
        f"skipped={report['n_skipped']}, k={k}, split={split})"
    )
    print(f"  OVERALL  accuracy={report['overall']['accuracy']:.3f}")
    for cat, m in report["by_category"].items():
        print(f"  {cat:<26} n={m['n']:<4} accuracy={m['accuracy']:.3f}")
    if "latency" in report:
        lat = report["latency"]
        print(f"  p50={lat['p50_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms")

    if out_path:
        pathlib.Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"wrote report -> {out_path}")
    if want_baseline:
        written = baseline_store.append_baseline(
            f"{bench}_qa",
            report,
            metadata={
                "answerer": llm._cfg.model,
                "judge": judge_client._cfg.model,
                "k": k,
                "n_arg": n_arg,
                "split": split,
            },
        )
        print(f"appended baseline -> {written}")


if __name__ == "__main__":
    main()

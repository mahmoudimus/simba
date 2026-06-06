"""Run simba recall@k (and optional QA) on LongMemEval.

Usage: uv run python scripts/run_longmemeval.py [path-to-longmemeval_*.json] [n] [flags]
Turn-level recall of the gold (has_answer) evidence turns per question type.

  --full        : treat the file as the full longmemeval_s haystack (distractors)
  --abstention  : keep resolvable _abs questions and score abstention in QA
  --qa          : also run the QA layer (answer + judge) after recall
  --baseline    : append results to .simba/eval/baselines/longmemeval_s_*.jsonl

NOTE: against the *oracle* haystack (only evidence sessions) recall is an upper
bound — the real test is the full longmemeval_s haystack. Not shipped — a dev
measurement script.
"""

from __future__ import annotations

import sys

import simba.eval.benchmarks.baseline_store as baseline_store
import simba.eval.benchmarks.judge as judge
import simba.eval.benchmarks.longmemeval as lme
import simba.eval.benchmarks.run as bench
import simba.eval.run as run
import simba.llm.client as llm_client
import simba.llm.judge_config as jcfg
import simba.memory.config as mc


def main() -> None:
    argv = sys.argv
    positional = [a for a in argv[1:] if not a.startswith("--")]
    path = positional[0] if len(positional) > 0 else "/tmp/lme_oracle.json"

    include_abstention = "--abstention" in argv
    want_qa = "--qa" in argv
    want_baseline = "--baseline" in argv

    datasets = lme.load_longmemeval(path, include_abstention=include_abstention)
    if len(positional) > 1:
        datasets = datasets[: int(positional[1])]

    # Wide retrieval so recall@10 is meaningful; rerank/scoring/expansion off to
    # isolate raw retrieval quality.
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

    print(f"\nLongMemEval recall ({report['n_cases']} questions)")
    o = report["overall"]
    print(
        f"  OVERALL  recall@1={o['recall@1']:.3f} recall@3={o['recall@3']:.3f} "
        f"recall@5={o['recall@5']:.3f} recall@10={o['recall@10']:.3f} "
        f"mrr={o['mrr']:.3f}"
    )
    for cat, m in report["by_category"].items():
        print(
            f"  {cat:<26} n={m['n']:<4} r@1={m['recall@1']:.3f} "
            f"r@5={m['recall@5']:.3f} r@10={m['recall@10']:.3f} mrr={m['mrr']:.3f}"
        )
    if "latency" in report:
        lat = report["latency"]
        print(f"  p50={lat['p50_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms")

    if want_baseline:
        written = baseline_store.append_baseline(
            "longmemeval_s_recall",
            report,
            metadata={"abstention": include_abstention, "full": "--full" in argv},
        )
        print(f"appended baseline -> {written}")

    if want_qa:
        llm = llm_client.get_client()
        judge_client = jcfg.get_judge_client()
        print(
            f"answerer: model={llm._cfg.model} judge: model={judge_client._cfg.model}"
        )
        qa_report = judge.run_qa(
            datasets,
            embed_doc=embed_doc,
            embed_query=embed_query,
            cfg=cfg,
            llm=llm,
            judge=judge_client,
            include_abstention=include_abstention,
            judge_model=judge_client._cfg.model,
        )
        print(
            f"\nLongMemEval QA accuracy (graded={qa_report['n_graded']}, "
            f"skipped={qa_report['n_skipped']})"
        )
        print(f"  OVERALL  accuracy={qa_report['overall']['accuracy']:.3f}")
        ab = qa_report["abstention"]
        print(f"  ABSTENTION n={ab['n']:<4} accuracy={ab['accuracy']:.3f}")
        if "latency" in qa_report:
            lat = qa_report["latency"]
            print(f"  p50={lat['p50_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms")
        if want_baseline:
            written = baseline_store.append_baseline(
                "longmemeval_s_qa",
                qa_report,
                metadata={
                    "answerer": llm._cfg.model,
                    "judge": judge_client._cfg.model,
                    "abstention": include_abstention,
                },
            )
            print(f"appended baseline -> {written}")


if __name__ == "__main__":
    main()

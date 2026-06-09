"""Configuration for ``simba eval bench`` / ``simba eval leaderboard``.

The ``bench`` section holds the dataset paths, cache paths, the results log
location, and the MemoryConfig overrides applied during a bench run (wide
retrieval, reranker/scoring/expansion off — to isolate raw retrieval quality).
"""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("bench")
@dataclasses.dataclass
class BenchConfig:
    locomo_path: str = ""
    longmemeval_path: str = ""
    # HotpotQA distractor dev set (genuine bridge-entity multi-hop). Default to the
    # gitignored benchmarks dir; fetch via scripts/fetch_benchmarks.sh (Wayback,
    # since the CMU host is offline). Scored with bridge_recall@k (all hops in k).
    hotpotqa_path: str = ".simba/benchmarks/hotpot_dev_distractor_v1.json"
    # HaluMem (memory-hallucination eval; docs/plans/10). >1M tokens/user, so
    # default to a small subsample (0 = all users).
    halumem_path: str = ".simba/benchmarks/HaluMem-Medium.jsonl"
    halumem_user_limit: int = 10
    # SubtleMemory (relational / contradiction eval). One dir per persona
    # (persona_0..9), each ~100 cases / ~2.5k turns. Default to the cloned data
    # dir; subsample to a small persona count for cheap runs (0 = all personas).
    subtlememory_path: str = (
        "/Users/mahmoud/src/ai/memory/SubtleMemory/data/subtlememory"
    )
    subtlememory_persona_limit: int = 1
    embedding_cache_path: str = ".simba/eval/embedding_cache.db"
    judge_cache_path: str = ".simba/eval/judge_cache.db"
    default_k: int = 10
    default_n: int = 50
    results_path: str = ".simba/eval/results.jsonl"
    leaderboard_path: str = "BENCHMARKS.md"
    max_results_eval: int = 20
    max_results_broad_eval: int = 20
    fts_candidate_pool_eval: int = 40
    fts_candidate_pool_broad_eval: int = 60
    llm_rerank_enabled_eval: bool = False
    scoring_enabled_eval: bool = False
    expansion_enabled_eval: bool = False

    def eval_memory_config_overrides(self) -> dict[str, object]:
        """Return the MemoryConfig field overrides for bench runs."""
        return {
            "max_results": self.max_results_eval,
            "max_results_broad": self.max_results_broad_eval,
            "fts_candidate_pool": self.fts_candidate_pool_eval,
            "fts_candidate_pool_broad": self.fts_candidate_pool_broad_eval,
            "llm_rerank_enabled": self.llm_rerank_enabled_eval,
            "scoring_enabled": self.scoring_enabled_eval,
            "expansion_enabled": self.expansion_enabled_eval,
        }

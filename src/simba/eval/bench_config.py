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

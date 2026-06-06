"""Configuration for the memory daemon.

Ported from claude-memory/config.json.
"""

from __future__ import annotations

import dataclasses
import typing

import simba.config


@simba.config.configurable("memory")
@dataclasses.dataclass
class MemoryConfig:
    port: int = 8741
    db_path: str = ""
    embedding_model: str = "nomic-embed-text"
    embedding_dims: int = 768
    model_repo: str = "nomic-ai/nomic-embed-text-v1.5-GGUF"
    model_file: str = "nomic-embed-text-v1.5.Q4_K_M.gguf"
    model_path: str = ""
    n_gpu_layers: int = -1
    embed_url: str = ""
    # Embedding backend: "gguf" (in-process llama-cpp, default), "http" (when
    # embed_url is set), or "llm-cli" (shell `llm embed`; note: only as local as
    # the chosen llm model — cloud models cross the no-external-service line).
    embed_provider: str = "gguf"
    # Asymmetric task prefixes (model-specific). nomic uses "search_document: " /
    # "search_query: "; Qwen3-Embedding uses "" for docs and an "Instruct: …\n
    # Query: " instruction for queries. Prepended to the text before embedding.
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "
    min_similarity: float = 0.35
    max_results: int = 3
    duplicate_threshold: float = 0.92
    # Supersession (Phase 3): on store, replace an older same-type memory whose
    # similarity is in [supersede_threshold, duplicate_threshold). On by default
    # (experimental); set false to keep every near-duplicate.
    supersede_enabled: bool = True
    supersede_threshold: float = 0.85
    max_content_length: int = 200
    auto_start: bool = True
    diagnostics_after: int = 50
    # Max latency samples kept per endpoint for p50/p95 in /metrics.
    diagnostics_reservoir_size: int = 1000
    # Memory hygiene (Phase 7 ops): expire stale TOOL_RULE memories older than
    # this many days (0 = disabled). Solves the stale false-warning injection.
    tool_rule_max_age_days: int = 30
    # Let the background sync scheduler run the hygiene pass.
    hygiene_scheduler_enabled: bool = True
    sync_interval: int = 0
    shutdown_timeout: int = 10
    # Hybrid recall (L3): a BM25 keyword arm fused with the vector arm via RRF.
    hybrid_enabled: bool = True
    rrf_k: int = 60
    fts_candidate_pool: int = 20
    fts_tokenize: str = "trigram"
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    # Read-path query intelligence (Phase 0).
    intent_aware: bool = True  # auto-pick the cosine floor from query intent
    min_similarity_broad: float = 0.28  # recall floor for aggregation/exploration
    fts_max_terms: int = 12  # cap on high-signal terms fed to the keyword arm
    # Broad-query widening (Phase 0.1): aggregation queries pull a wider net.
    max_results_broad: int = 8  # results returned for broad queries
    fts_candidate_pool_broad: int = 40  # RRF candidate pool for broad queries
    # Multi-arm HyDE (Phase 0.2): a 2nd vector arm over the focused-term string.
    expansion_enabled: bool = True  # on by default (costs one extra embed per recall)
    # HyDE mode (C3): how the 2nd vector arm's text is derived.
    #   "keyword" (default) = the focus-term string (current behavior).
    #   "llm" = a short hypothetical answer generated via simba.llm.client, embedded
    #           as the 2nd arm. Falls back to the keyword string (or "") on any
    #           failure. In the daemon the LLM call is OFF the hot path: the first
    #           recall serves the keyword fallback and warms a per-process cache, so
    #           recurring queries get the HyDE text free.
    hyde_mode: str = "keyword"
    hyde_cache_size: int = 256  # LRU capacity for the HyDE cache (daemon only)
    # Composite re-scoring: blend RRF relevance with recency + importance after
    # fusion. On by default (experimental). The default weights are the measured
    # blend — relevance-dominant, with recency + importance as tie-breakers
    # (never the sole signal, which would ignore the query) — so it benefits
    # time-sensitive recall and is a no-op when a corpus has uniform
    # dates/confidence. See datasets/temporal.json.
    scoring_enabled: bool = True
    score_weight_relevance: float = 1.0
    score_weight_recency: float = 0.5
    score_weight_importance: float = 0.3  # uses the stored confidence as importance
    recency_halflife_days: float = 90.0
    # LLM reranker: an LLM relevance pass over the candidate pool before truncating
    # to max_results (the cross-encoder's role). On by default (experimental).
    # In the daemon it is NON-BLOCKING — recall serves the fast order and reranks
    # off the hot path, caching by (query, candidate-set), so novel queries pay no
    # latency and recurring ones get the rerank for free. Needs an llm provider;
    # always fail-open (any error leaves the RRF + composite ordering intact).
    llm_rerank_enabled: bool = True
    llm_rerank_candidates: int = 20  # cap of candidates sent to the reranker
    # "async" (default): non-blocking — serve fast order, rerank off the hot path
    # via the cache. "sync": block on the rerank every recall (test/measure mode).
    llm_rerank_mode: str = "async"
    # Async rerank: when a cache is wired (the daemon), recall never blocks on the
    # LLM — it serves the fast order and reranks off the hot path, caching the
    # result by (query, candidate-set) for the next recurrence. Cache capacity:
    rerank_cache_size: int = 256
    # Decay / forgetting + feedback-aware ranking (Phase 6). Mutable per-memory
    # ranking signals live in the sqlite ``memory_usage`` table; these tunables
    # drive how strength decays over time, how access reinforces it, and how
    # outcome feedback nudges it. The scheduler runs a periodic decay pass.
    decay_enabled: bool = True  # master switch for all decay/dormancy updates
    # Time (days) at which an unaccessed memory's decay factor reaches 0.5.
    decay_half_life_days: float = 30.0
    # How much each access "lifts" strength (logistic scale). Smaller saturates
    # faster: with scale=0.5, 1 access → ~0.86, 2 → ~0.98, 3 → ~1.00.
    reinforcement_scale: float = 0.5
    # Weight applied to feedback_score when computing final strength:
    # final = base * (1 + feedback_weight * feedback_score), feedback ∈ [-1, 1].
    feedback_weight: float = 0.2
    # Memories whose strength falls below this after a decay pass become dormant.
    strength_dormancy_threshold: float = 0.1
    # Max non-dormant memories per (type, project_path). 0 = unlimited. When > 0,
    # the weakest memories beyond this cap are set dormant.
    decay_capacity_per_type: int = 0
    # Weight of the strength term in composite_rescore. Missing usage rows score
    # 1.0 (no penalty for never-recalled memories).
    score_weight_strength: float = 0.4
    # When True, dormant memories are excluded from recall results.
    dormant_filter_enabled: bool = True
    # Default delta applied per good/bad feedback signal. Overridable per-call.
    feedback_default_weight: float = 0.3


def load_config(**overrides: typing.Any) -> MemoryConfig:
    """Load config from TOML files, then apply CLI/keyword overrides."""
    base = simba.config.load("memory")
    valid_keys = {f.name for f in dataclasses.fields(MemoryConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid_keys}
    if not filtered:
        return base
    # Merge overrides on top of TOML-loaded base
    base_dict = dataclasses.asdict(base)
    base_dict.update(filtered)
    return MemoryConfig(**base_dict)

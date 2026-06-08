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
    # Default embedder: bge-large-en-v1.5 (1024-d). Bake-off (2026-06-06) on a
    # discriminating eval showed it clearly beats nomic-embed-text (768-d) on
    # both LoCoMo (r@5 0.595->0.614) and LongMemEval (r@5 0.780->0.814), lifting
    # both weak axes (multi-hop, open-domain) with no single-hop regression. See
    # docs/plans/07-recall-excellence.md. Switching embedders changes the vector
    # dimension, so an existing store must be migrated with `simba memory reembed`
    # (a guard raises a clear error on dim mismatch).
    embedding_model: str = "bge-large-en-v1.5"
    embedding_dims: int = 1024
    model_repo: str = "CompendiumLabs/bge-large-en-v1.5-gguf"
    model_file: str = "bge-large-en-v1.5-q8_0.gguf"
    model_path: str = ""
    n_gpu_layers: int = -1
    embed_url: str = ""
    # Embedding backend: "gguf" (in-process llama-cpp, default), "http" (when
    # embed_url is set), or "llm-cli" (shell `llm embed`; note: only as local as
    # the chosen llm model — cloud models cross the no-external-service line).
    embed_provider: str = "gguf"
    # Asymmetric task prefixes (model-specific). bge-large uses "" for docs and a
    # "Represent this sentence for searching relevant passages: " instruction for
    # queries; nomic used "search_document: " / "search_query: ". Prepended to
    # the text before embedding.
    embed_doc_prefix: str = ""
    embed_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )
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
    # RRF rank constant; lower = sharper top-rank weighting. Swept 2026-06-06:
    # k=20 beat k=60 on LoCoMo recall (r@5 0.573->0.595, multi-hop + single-hop
    # both up) and was neutral on LongMemEval — recall is ranking-limited, not
    # candidate-pool-limited (widening the pools regressed). See
    # docs/plans/07-recall-excellence.md.
    rrf_k: int = 20
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
    # Entity-bridge multi-hop (spec 09): after RRF, fold memories that share a
    # *named entity* with the top seeds into the candidate set (BFS depth N over
    # the shared-entity graph), as a third RRF arm before composite rescore +
    # reranker. Default-OFF — the one multi-hop mechanism with a positive external
    # result (YourMemory +12pp HotpotQA), distinct from kg co-occurrence/PPR
    # (sparse, high-precision links). No-op unless a caller supplies the index.
    entity_bridge_enabled: bool = False
    entity_bridge_hops: int = 1  # BFS depth from the seeds over shared entities
    entity_bridge_seeds: int = 3  # top fused hits used as traversal seeds
    entity_bridge_max: int = 10  # cap on bridged ids folded in
    entity_bridge_weight: float = 1.0  # RRF-arm weight of the bridge contribution
    entity_bridge_min_shared: int = 1  # min seed-entities a bridge must share
    entity_bridge_max_df: int = 0  # drop entities in > N memories (0 = off); precision


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

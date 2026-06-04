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

    # Query decomposition (C4): split a multi-hop query into single-fact
    # sub-queries (via the llm client), retrieve each, and RRF-fuse the ranked
    # lists — so each evidence piece gets its own targeted retrieval. "none"
    # (default) keeps single-query recall; "llm" decomposes. Needs an llm
    # provider; fail-open to the original query.
    decompose_mode: str = "none"  # "none" | "llm"
    decompose_max_sub: int = 4  # cap on sub-queries (besides the original)


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

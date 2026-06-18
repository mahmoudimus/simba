"""Configuration for the memory daemon.

Ported from claude-memory/config.json.
"""

from __future__ import annotations

import dataclasses
import typing

import simba.config

if typing.TYPE_CHECKING:
    import pathlib


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
    # Low-confidence rejection (abstention) gate. When enabled, if the TOP recall
    # candidate's similarity is below `recall_reject_threshold`, the whole recall is
    # suppressed (return nothing) instead of surfacing weak/spurious context.
    # Distinct from `min_similarity` (a per-result floor): this judges the BEST
    # candidate and abstains. Off by default. Borrowed from MemX (arXiv 2603.16171).
    recall_reject_enabled: bool = False
    recall_reject_threshold: float = 0.0
    # Score-adaptive truncation (SmartSearch, arXiv 2603.15599). When > 0, recall
    # returns the longest score-ranked prefix that fits this many estimated tokens
    # (~chars/recall_chars_per_token) instead of a fixed `max_results` count — so
    # co-required evidence isn't cut at a hard count cap (the completeness gate).
    # 0 = off (use max_results). The top hit is always included.
    recall_token_budget: int = 0
    recall_chars_per_token: int = 4
    # Store-time anticipated query metadata. This first slice records likely
    # future query phrasings append-only; recall expansion remains separately
    # measured before it can affect ranking.
    anticipated_query_max_per_memory: int = 5
    # Entropy-gated exact-term boost. Trigram FTS collides high-information tokens
    # (50815 -> 508/081/815 overlaps other codes), so an exact error code / symbol /
    # path can rank #14 behind trigram-collision noise. When on, rare/identifier query
    # tokens (general-English rarity via wordfreq + identifier shape) boost memories
    # that contain them verbatim. No-op on prose (no such token).
    # DEFAULT ON (2026-06-14): measured on real data — acme "INTERR 50815" #15 -> #1
    # (deterministic), and LME-S A/B (n=60) showed NO recall@k regression (recall@5/@10
    # flat, recall@1 +0.017, fired harmlessly on 8/60 prose Qs). Harm-free + targeted
    # win -> graduates per the SoTA-lever policy. Set false to disable.
    recall_exact_boost_enabled: bool = True
    # A query token with Zipf frequency >= this is common English and is skipped.
    recall_exact_zipf_common: float = 3.0
    # Dimensional tagging (DimMem, arXiv 2605.15759). When on, each stored memory gets
    # a parseable time/keyword blob appended to its `context` (deterministic extract;
    # never embedded), so aggregation can filter/count by field later instead of
    # re-individuating raw text at answer time. Off by default (see dimensions.py).
    dimensions_enabled: bool = False
    max_results: int = 3
    duplicate_threshold: float = 0.92
    # Supersession (Phase 3): on store, replace an older same-type memory whose
    # similarity is in [supersede_threshold, duplicate_threshold). On by default
    # (experimental); set false to keep every near-duplicate.
    supersede_enabled: bool = True
    supersede_threshold: float = 0.85
    # Trust gate for supersession: weak/automatic evidence may store as a new
    # memory, but it cannot actively supersede stronger user/agreed knowledge
    # without explicit confirmation.
    supersede_trust_gate_enabled: bool = True
    supersede_trust_margin: float = 0.05
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
    # Continuous extraction (the "Continuous" gap). When on, the Stop hook reads only
    # the NEW transcript window each turn (incremental cursor, O(new)) and enqueues it
    # for the scored extract->score->keep/drop worker. DEFAULT-OFF: this ships the
    # cursor + enqueue rails only; the worker + Importance rubric are gated behind a
    # gold-set Evaluator (see simba.eval.extraction_eval). Off -> prior behavior
    # (extraction stays at PreCompact).
    continuous_extraction_enabled: bool = False
    continuous_extraction_max_bytes: int = 2_000_000  # per-turn window cap (bytes)
    # Hierarchical (ancestor-prefix) project recall (spec 26). Today recall is a
    # STRICT exact-match on projectPath: a memory scoped to /repo/api never recalls
    # from the /repo root, and a project filter excludes global memories entirely.
    # When on, recall at cwd C returns memories scoped to C plus every ancestor of C
    # up to the git root plus global — root facts inherit DOWN to packages, package
    # facts stay put (don't leak to siblings). The CLIENT computes the chain (it owns
    # the filesystem; the daemon stays path-agnostic, string-membership only) and
    # sends it as ``project_scopes``. Both retrieval arms (vector + FTS/BM25) honor
    # the same set. DEFAULT-OFF: unmeasured — widening the candidate pool can dilute
    # precision; graduate to ON only after a measured no-regression on recall@k.
    hierarchical_recall: bool = False
    # Treat global (empty-path) memories as the root of the tree: include them in a
    # project-scoped recall (fixes the "global excluded under a project filter"
    # quirk). Separate lever so it can be measured independently. No effect unless
    # ``hierarchical_recall`` is also on.
    hierarchical_recall_include_global: bool = True
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
    # Intent-aware candidate DEPTH for count queries. Counting an open class is
    # recall-BREADTH-bound, not pointwise-rerank-bound: it needs every member, and
    # a pointwise reranker cannot recover what a narrow pool never retrieved
    # (measured on LongMemEval: count pool_complete@20 = 0.50 — half the gold never
    # enters a width-20 pool). So for count intent, widen the first-stage pool +
    # context and skip the (pointwise) reranker. Off -> count uses normal sizing.
    count_depth_enabled: bool = True
    count_candidate_pool_n: int = 80  # wide RRF candidate pool (pool_complete@80=1.0)
    # Returned context size. Measured (LME count, avg 3.52 gold): complete@20=0.40 is
    # too small — gold lives at ranks 20-40; complete@40=0.96 (vs @80=1.0 at 2x cost).
    # Answer accuracy tracks it: 0.40(k=8) -> 0.48(k=20) -> 0.56(k=40).
    count_context_k: int = 40  # results returned (context size) for count queries
    count_disable_rerank: bool = True  # skip the pointwise reranker for count
    # Intent-aware candidate DEPTH for multi-session / aggregation queries — the
    # count-depth lever generalized past instance-counting. Measured on
    # LongMemEval-S: multi-session/aggregation questions are recall-BREADTH-bound
    # exactly like counting (multi-session evidence sets complete@80 = 0.90 vs
    # complete@20 = 0.33). Widening the answer context k=20 -> k=80 lifted the
    # multi-session category 0.557 -> 0.686 (+0.13) and overall 0.7495 -> 0.7702 —
    # the lever in the 0.823 stack config (k_by_type multi-session:80).
    # DEFAULT ON (2026-06-14): policy — a lever measured to reach SoTA/par graduates
    # to default-on so the shipped product runs at its measured ceiling. Cost: fires
    # a wider, costlier retrieval on multi-session/aggregation queries (gated by
    # is_aggregation). Set false to revert to the conservative narrow net.
    aggregation_depth_enabled: bool = True
    aggregation_candidate_pool_n: int = 80  # wide RRF candidate pool (complete@80=0.90)
    # Returned context size. k=80 (not count's 40): the gate showed multi-session
    # 3+-span evidence needs the wider window — complete@80 = 0.90 is the
    # cost-balanced knee for these questions.
    aggregation_context_k: int = 80
    # Same-session expansion (SubtleMemory driver). When on, if normal recall
    # touches a transcript/session via ``sessionSource``, pull a bounded number of
    # same-session rows into the candidate pool before scoring/reranking. This is
    # non-oracle: sessions must already be retrieved. Default-OFF pending
    # held-out eval because it can add noisy transcript turns.
    session_expansion_enabled: bool = False
    session_expansion_top_sessions: int = 2
    session_expansion_max_per_session: int = 12
    session_expansion_weight: float = 2.0
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
    # Reranker backend selection (spec 22). The reranker is a relevance re-scoring
    # pass over the fused candidate pool before truncation to max_results (the
    # cross-encoder's role). Backends (measured LoCoMo n=60 recall@5 / latency):
    #   "cross-encoder" (default) bge-reranker-v2-m3 GGUF via llama-cpp RANK
    #       pooling — 1 relevance score per (query, doc) pair. 0.688 @ ~240ms.
    #   "local-llm"     zerank-2 Qwen3-4B GGUF via llama-cpp; score = the logit of
    #       the model's "Yes"/true token at the final position. 0.718 @ ~1.3s.
    #   "llm"           the cloud LLM client (existing llm_rerank.rerank). 0.721
    #       @ ~23s — the latency trap this spec retires from the default.
    #   "none"          skip reranking entirely.
    # Both local backends run GGUF via the EXISTING llama-cpp stack (no torch),
    # mirroring EmbeddingService loading; the model auto-downloads from the repo/
    # file fields below. Default "cross-encoder" per the measured win — 95% of the
    # LLM's recall@5 at ~96x lower latency. Always fail-open.
    reranker_mode: str = "cross-encoder"
    # cross-encoder backend (bge-reranker-v2-m3 GGUF, XLM-RoBERTa based).
    reranker_model_repo: str = "gpustack/bge-reranker-v2-m3-GGUF"
    reranker_model_file: str = "bge-reranker-v2-m3-Q4_K_M.gguf"
    # local-llm backend (zerank-2 Qwen3-4B GGUF). true_token_id=9454 ("Yes") is
    # the relevance logit per zeroentropy/zerank-2's 1_LogitScore module.
    reranker_local_llm_repo: str = "godkingleto/zerank-2-Q4_K_M-GGUF"
    reranker_local_llm_file: str = "zerank-2-q4_k_m.gguf"
    reranker_local_llm_true_token: int = 9454
    reranker_n_ctx: int = 4096  # context window for the local-llm backend
    # Intent-gated reranking (spec 22, LME-gate correction). The reranker is a
    # POINTWISE relevance pass, which HELPS latest/compositional-multihop but HURTS
    # multi-evidence temporal (it promotes the most-relevant turn and demotes a
    # co-required one, breaking the evidence set: LME complete@5 0.65->0.20). When
    # on, skip reranking for query shapes it measurably harms (multi-endpoint
    # temporal) — making reranking a router decision, not a global flag. Off ->
    # reranking fires whenever the mode/client gate allows (prior behavior).
    rerank_intent_gating: bool = True
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
    # Persistent query-embedding cache (daemon). Identical recall queries hit a
    # sha1(model|prefix|text) sqlite lookup instead of re-running the GGUF embed
    # under the process-global llama lock — a pure speedup (same vector), default
    # ON. Big win when a query repeats (e.g. the conflict detector firing the same
    # pairwise check N times) and across the frequent daemon restarts. Path empty
    # -> ``<db dir>/embed_cache.db``.
    embed_cache_enabled: bool = True
    embed_cache_path: str = ""
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
    # Optional outcome-counter contribution to feedback before strength is
    # computed. 0.0 is a no-op; when >0, use_count vs noise_count nudges
    # effective feedback, enabling measured outcome-driven half-life.
    outcome_quality_weight: float = 0.0
    # Memories whose strength falls below this after a decay pass become dormant.
    strength_dormancy_threshold: float = 0.1
    # Arousal-modulated decay (Phase 6.5): a multiplier applied to the time-decay
    # factor (in [0, 1]) as ``d ** arousal_decay_multiplier`` before reinforcement.
    #   == 1.0 → exact no-op (default; behavior unchanged, zero runtime effect)
    #   <  1.0 → slower decay (more arousal / importance), memory retains longer
    #   >  1.0 → faster decay (less arousal), memory is forgotten sooner
    # Default-OFF: the multiplier is 1.0 and ``arousal_decay_enabled`` is False
    # until a measured win justifies turning it on. The enabled flag is advisory
    # documentation — the multiplier of 1.0 is already a behavioral no-op.
    arousal_decay_enabled: bool = False
    arousal_decay_multiplier: float = 1.0  # sensible range [0.1, 3.0]; 1.0 = no-op
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
    entity_bridge_ner: str = "regex"  # entity extractor: "regex" | "spacy" (real NER)
    # Retrieval-time GraphRAG (Track B): after RRF, fold PPR-ranked KG neighbors
    # (seeded by the query's entities) into the candidate set as a third arm,
    # before composite rescore + the reranker. Default-OFF — the measured ceiling
    # is marginal (LoCoMo multi-hop +0.013, LME multi-session +0.046 upper bounds;
    # docs/plans/06-multihop.md), so it stays gated pending a proven delta. The KG
    # itself is supplied by the caller (the eval throwaway KG; a kg_edges adjacency
    # on the live path), so this is a no-op until one is wired in.
    kg_ppr_enabled: bool = False
    kg_ppr_top: int = 10  # how many PPR-ranked memories to fold
    kg_ppr_weight: float = 1.0  # RRF-arm weight of the PPR contribution
    kg_ppr_damping: float = 0.85
    # Answer-time conflict surfacing (src/simba/memory/conflict.py): after recall,
    # detection asks whether any two retrieved memories CONFLICT for the query;
    # if so, a directive that NAMES the specific conflict is appended to the
    # injected context so the answerer surfaces it (states what must be confirmed)
    # instead of silently picking a side. Default-ON since 0.7.0: the pairwise
    # detector + directive took the SubtleMemory contradictory both-sides slice
    # 0.111 -> 0.944 with a net-positive harm check (docs/plans/14). The lever is
    # gated on a DETECTED, NAMED conflict (a generic always-on directive measured
    # harmful) and fail-open (any error leaves context intact). Zero LLM cost
    # below the minimum candidate count; disable via
    # `simba config set memory.conflict_surfacing_enabled false`.
    conflict_surfacing_enabled: bool = True
    conflict_surfacing_min_memories: int = 2  # min candidates before detection runs
    # Detection strategy. "single" (default) = one LLM call over all top-k
    # memories at once ("do any of these conflict?"). "pairwise" = check candidate
    # pairs in isolation, returning the first flagged pair. Isolating the pair
    # lifts detection recall on subtle/buried conflicts (the all-at-once prompt
    # buries the conflicting pair among k distractors); pairwise costs up to
    # ``conflict_detect_max_pairs`` LLM calls (short-circuits on the first hit).
    conflict_detect_strategy: str = "pairwise"  # "single" | "pairwise"
    conflict_detect_max_pairs: int = 45  # cap on pairs checked in "pairwise" mode
    # Pairwise checks run in waves of this width (bounded threads), so a recall
    # with k memories pays ~ceil(pairs/width) LLM latencies, not one per pair.
    # Result is deterministic (lowest-index flagged pair, same as sequential).
    conflict_detect_parallel: int = 8
    # Query-intent gate (0.7.1 regression fix). v0.7.0 shipped surfacing default-ON
    # and WON on genuine contradictions (SubtleMemory both-sides 0.111 -> 0.944) but
    # REGRESSED knowledge-update QA: on LME-S knowledge-update OFF=0.958 vs the
    # directive=0.25, because a "what is X now?" query retrieves BOTH the old and
    # the new value of a fact, the pairwise detector flags that as a conflict, and
    # the directive tells the answerer not to pick a side — exactly wrong when the
    # correct answer is most-recent-wins. When True (default), conflict surfacing
    # SKIPS its directive (and pays zero detection cost) for current-value /
    # knowledge-update-shaped queries (intent.is_knowledge_update); recency /
    # most-recent-wins handles them. All other queries stay on the strict pairwise
    # path unchanged, so the genuine-contradiction win is untouched. Gating by
    # query INTENT — not by detecting date-disjointness on the memories: the ARM3
    # date-disjoint carve-out FAILED its SubtleMemory gate (0.722 < 0.9) because
    # genuine preference conflicts are also date-disjoint.
    conflict_skip_on_current_value: bool = True
    # Write-time conflict engine (B2, src/simba/memory/conflict_store.py): move
    # detection OFF the answer-time path. When enabled, on store a new memory is
    # compared against its nearest neighbors (one focused pairwise LLM call each,
    # capped at conflict_write_max_neighbors); any real contradiction is persisted
    # to the append-only memory_conflicts table. At recall the precomputed conflict
    # among the recalled set is READ (no detection latency) and surfaced via the
    # same directive. Default-OFF — the engine is exposed for measurement; the live
    # daemon store-route hook is a deferred follow-up (B2b).
    conflict_detect_on_write: bool = False
    conflict_write_max_neighbors: int = 5  # neighbors checked per write
    # Query-aware recall re-check (B2b): the write-time pass can be a GENEROUS,
    # high-recall PRE-FILTER (store candidate conflicts query-independently, off
    # the hot path); this flag recovers precision at recall. When True, recall
    # runs ONE query-aware confirm over the stored candidate(s) among the recalled
    # set (gives the LLM the question + candidate descriptions, asks which — if any
    # — is a real conflict that matters for THIS question). A confirmed candidate
    # is surfaced; a query-irrelevant candidate is dropped (the precision win).
    # Default-OFF — when False, recall reads the first stored candidate with no LLM
    # call (current behavior). Always fail-open (any error falls back).
    conflict_recall_recheck: bool = False


def resolve_max_content_length(root: pathlib.Path | None = None) -> int:
    """Resolve the configured memory content cap (``memory.max_content_length``).

    Single source of truth for both enforcement (store truncation/validation) and
    the "keep content under N chars" guidance the daemon emits to extraction /
    digest / episode / reflection agents. Fail-open to the dataclass default so
    prompt building and the CLI never crash on a missing/broken config.
    """
    default = int(MemoryConfig.max_content_length)
    try:
        cfg = simba.config.load("memory", root)
        max_len = int(getattr(cfg, "max_content_length", default))
        return max_len if max_len > 0 else default
    except Exception:
        return default


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

# simba roadmap

Living doc for what's next on the memory/recall system. Driven by the honest
SOTA assessment + the 4-repo borrow investigation (2026-06-03).

## Where we are

Shipped this session (all on `main`): episodic consolidation (L2, #16), the recall
**eval harness** (#17), composite **recency+importance scoring** (#18), KG **entity
resolution** (#19), **multi-hop KG** traversal (#20), the **LLM reranker + extraction +
local providers** with experimental defaults on (#21), the **non-blocking rerank
cache** (#22), the **swappable embedder** (#23), and **eval hardening** — real-corpus
builder + dev/test split (#24).

Honest standing: the *architecture* is at/near SOTA on most axes; the **evidence**
is thin — the eval datasets saturate (MRR→1.0), so the now-default features and the
phases below are not yet proven on real data.

## Next → Phase 3 (do this first next session)

**LLM extraction as the primary KG feed.** Today `sync/extractor` runs regex first
and only falls back to LLM extraction on misses, so the bitemporal / entity-res /
multi-hop machinery is under-fed by default ("schema ahead of data"). Add
`sync.extract_strategy` (`regex | llm | llm+regex`, default `llm+regex` when an
`llm.provider` is available): run `llm_extract.extract_triples` on every new memory,
union with regex, dedup, canonicalize via `kg/entities` on write. Bounded by the
watermark + a per-cycle cap; background pipeline; fail-open to regex. Measure KG
density (edges/entities) before/after.

## The three options discussed (2026-06-03)

1. **Refine the eval builder + real baseline** *(prerequisite for trusting phases 3–6)* —
   the real-corpus builder saturates at small N because deepseek's generated questions
   paraphrase the source memory closely (recall@1=1.0 on 3 cases). Fix: a harder
   query-gen prompt (ask about the topic without restating the memory's wording) +
   run `simba eval build --n 200` to establish a discriminating real-corpus dev/test
   baseline. ~30 min. Without this, a saturating eval can't prove the phases below.
2. **Phase 3 — LLM extraction as default** (see ↑ next).
3. **Phase 6 — decay/forgetting + feedback-aware ranking** (see below; the biggest).

## Remaining phases (from the approved plan)

- **Phase 4 — true LLM HyDE.** `memory.hyde_mode = keyword | llm`. When `llm`, generate
  a hypothetical answer via `simba.llm.client` and embed *that* as the 2nd vector arm
  (replacing the focus-term string). Reuse the non-blocking cache so it's off the hot
  path. Measure on the harness.
- **Phase 5 — reflection.** Scheduler-driven LLM pass (like consolidation) that
  synthesizes cross-session higher-level insights into a new `REFLECTION` memory type
  (Generative-Agents style; importance-gated). Distinct from episodic consolidation
  (one session → EPISODE).
- **Phase 6 — decay / forgetting + feedback-aware ranking** *(the cross-repo convergent
  idea; biggest impact).* Persisted per-memory **strength** = decay(age, half-life) ×
  retrieval-reinforcement(access_count); a reversible **dormant** tier (excluded from
  recall, never deleted → append-only-safe) + carrying-capacity prune of the weakest;
  an **outcome feedback** signal (good/bad on recalled memories, tag-not-delete) that
  modulates strength. Fold strength into `memory/scoring.composite_rescore`. Needs
  mutable usage columns, a recall-time bump, a scheduler decay pass, `@configurable`
  weights/thresholds, and a usage/feedback eval fixture.

## Discipline (applies to every phase)

- Each phase = its own PR, **measured on the (refined) real-corpus dev/test split** —
  not the authored synthetic sets. See `eval-do-not-chase-1.0`: never tune config to
  saturate the benchmark; report deltas on held-out data; keep the dataset ahead of
  the system.
- Experimental features stay `@configurable` and fail-open.
- The reranker fills the cross-encoder role via the LLM (no learned model); the
  embedder swap is config-only (bake-off showed nomic-Q4 ≈ Qwen3-0.6B on the current
  eval, so the default stays nomic-Q4 until the eval discriminates).

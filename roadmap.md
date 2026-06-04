# simba roadmap

Living doc for what's next on the memory/recall system. Driven by the honest
SOTA assessment + the 4-repo borrow investigation (2026-06-03).

## Where we are

Shipped this session (all on `main`): episodic consolidation (L2, #16), the recall
**eval harness** (#17), composite **recency+importance scoring** (#18), KG **entity
resolution** (#19), **multi-hop KG** traversal (#20), the **LLM reranker + extraction +
local providers** with experimental defaults on (#21), the **non-blocking rerank
cache** (#22), the **swappable embedder** (#23), **eval hardening** — real-corpus
builder + dev/test split (#24), **LLM extraction as the primary KG feed** (Phase 3,
#25), and the **tool-call redirect** layer (#26/#27). In flight on `feat/benchmarks`:
the **LoCoMo / LongMemEval recall@k harness** (`simba.eval.benchmarks`).

Honest standing: the *architecture* is at/near SOTA on most axes; the **evidence**
is firming up — the authored eval datasets saturate (MRR→1.0), but the new external
benchmark harness measures recall@k against labelled evidence on real conversations
(first LoCoMo numbers below). The phases below remain unproven until measured there.

## First external numbers (2026-06-03)

Harness: `src/simba/eval/benchmarks/` (locomo, longmemeval, run, judge); scripts
`run_locomo.py` / `run_longmemeval.py` / `run_qa.py`. Hybrid recall only
(reranker/scoring/expansion off) unless noted. Turns carry their session date so
relative time is groundable (a loader fix that lifted QA 0.082 → 0.280; see below).

**LoCoMo recall@k of gold `dia_id` evidence** (no LLM judge), 10 convs / 1977 Q:
`OVERALL r@1=0.334 r@5=0.573 r@10=0.682 mrr=0.490`; single-hop r@5=0.684,
single-hop-factual 0.663, adversarial 0.554, **multi-hop 0.305, open-domain 0.270**.

**LongMemEval recall@k** (turn-level, **oracle haystack = upper bound**), 470 Q:
`OVERALL r@1=0.368 r@5=0.779 r@10=0.893`; weakest arm **multi-session r@5=0.624**.

**LoCoMo QA accuracy** (deepseek-v4-flash answer + judge), stratified 50/category
(202 Q): `OVERALL 0.391` balanced — single-hop-factual 0.60, single-hop 0.54,
multi-hop 0.24, open-domain 0.16. Distribution-weighted (single-hop-factual is
43% of real Q) ≈ **0.50**. The session-date fix was decisive: 0.082 → 0.280 on
the conv-1 sample before stratification.

The consistent headline across all three: **multi-hop / cross-session is the weak
axis** — exactly what the KG (#4/#19/#20) should win and doesn't yet feed recall.
Caveats: numbers are recall@k / deepseek-judged (not GPT-4-judged like Mem0/Zep);
LongMemEval is oracle (not full `longmemeval_s`).

## Next → Phase 4 (Phase 3 shipped in #25)

True LLM HyDE — see **Remaining phases** below. Prerequisite still standing: refine
the eval builder so the real-corpus split discriminates (option 1 below).

## The three options discussed (2026-06-03)

1. **Refine the eval builder + real baseline** *(prerequisite for trusting phases 3–6)* —
   the real-corpus builder saturates at small N because deepseek's generated questions
   paraphrase the source memory closely (recall@1=1.0 on 3 cases). Fix: a harder
   query-gen prompt (ask about the topic without restating the memory's wording) +
   run `simba eval build --n 200` to establish a discriminating real-corpus dev/test
   baseline. ~30 min. Without this, a saturating eval can't prove the phases below.
2. **Phase 3 — LLM extraction as default** — ✅ shipped in #25.
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

## Phase 7 — neuro-symbolic deductive distillation ("learning")

The distinctive bet, grounded in a prior-art survey (2026-06-03): **a persistent,
bitemporal, prover-maintained memory where the LLM is confined to extraction/proposal
and the solver guarantees consistency + deductive closure.** The survey's finding:
every ingredient is open-source and locally reimplementable (Z3 + Souffle/Datalog +
the bitemporal KG we already have), but **no packaged system wires the full pipeline
together** — contradiction-resolution systems (KARMA, TruthfulRAG, Zep) resolve
conflicts with *more LLM calls / heuristics, never a solver + UNSAT core + formal
revision*; LLM+solver systems (Logic-LM, SatLM, LLM+ASP) solve *one-shot puzzles,
never maintain a persistent KB*; memory/RAG systems (HippoRAG, GraphRAG, Zep)
*traverse, never prove*, and do temporal reasoning by prompt rather than by SMT over
validity intervals. simba already has the bitemporal KG (#4) + Z3/Datalog via the
neuron MCP — the gap is the loop between them.

The deductive learning loop (each step its own sub-phase, all `@configurable`, all
fail-open):

1. **Derive** — run Datalog (Souffle) over `kg_edges` with learned/seeded Horn rules
   to materialise the deductive closure (new candidate edges, each with provenance:
   the supporting source edges). Most directly reusable: AnyBURL → Datalog rules with
   confidences, then exact closure in Souffle.
2. **Verify** — encode the live edge set (+ bitemporal validity intervals) as Z3
   constraints; on UNSAT, extract the **minimal UNSAT core** to isolate the exact
   contradicting facts. This is the single cleanest reusable primitive from the survey.
3. **Revise** — AGM-style contraction over an **entrenchment order keyed on
   (occurred_at/ingestion_time, extraction_confidence)**: drop the weaker conflicting
   fact (tag dormant, never delete → append-only-safe). Don't make the solver
   adjudicate two equally-trusted facts.
4. **Distill** — write verified derived edges back as **proof-carrying facts** (store
   the derivation / supporting edge ids alongside), so recall can return *"provable
   from F under R, here's the chain"* rather than a bare hit.
5. **Induce** — periodically promote recurring derivation patterns into new rules
   (ILP / AnyBURL-style), gated by confidence — the actual "learning" step.

**Cautionary findings (must design around):** the solver only guarantees correctness
*with respect to the formalization* — not that the LLM's NL→logic translation is
faithful (documented fabricated-axiom / paraphrase-instability failure modes that the
solver cannot catch). So: keep the LLM in extraction/proposal only, prefer
high-confidence + recent facts in revision, and keep a source-verification path for
extraction — treat the prover as a consistency/closure engine, **not** a truth oracle.

Sequenced after the recall phases (3–6) since it builds on a well-fed KG. Measured by
KG density + a contradiction-injection fixture (does the UNSAT core find the planted
conflict?) + proof-carrying-recall coverage, not just recall@k.

## Discipline (applies to every phase)

- Each phase = its own PR, **measured on the (refined) real-corpus dev/test split** —
  not the authored synthetic sets. See `eval-do-not-chase-1.0`: never tune config to
  saturate the benchmark; report deltas on held-out data; keep the dataset ahead of
  the system.
- Experimental features stay `@configurable` and fail-open.
- The reranker fills the cross-encoder role via the LLM (no learned model); the
  embedder swap is config-only (bake-off showed nomic-Q4 ≈ Qwen3-0.6B on the current
  eval, so the default stays nomic-Q4 until the eval discriminates).

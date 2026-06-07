# 07 — Recall excellence program

Make simba's recall *exceptionally* strong — as a measured program, not a pile of
levers. The binding constraint today is the **measuring instrument**, not ideas:
we have more recall levers than we can currently tell apart (C1 was a negative,
the IRCoT probe was noise). So the order below is deliberate.

Baseline to beat (committed `BENCHMARKS.md`, hybrid recall, levers off):
LoCoMo recall@5 **0.573** (multi-hop **0.305**, open-domain **0.211**, single-hop
0.684); LongMemEval oracle recall@5 0.780 (multi-session 0.662). The weak axes —
multi-hop and open-domain — are the headroom; single-hop must never regress.

## Discipline (every pillar)
Tune on dev, report on test; never tune to saturate ([[eval-do-not-chase-1.0]]);
each lever = a measured delta on recall@k (deterministic, local, no judge) where
it's a retrieval lever; LLM-judged deltas need ≥30/category (small-n is noise —
[[eval-ablation-latency-trap]]). Levers plug into `plan_recall`/`hybrid_search`
so the bench measures what ships.

---

## Pillar 1 — Fix the instrument (unblocks everything)
Right now most wins are invisible.
- **P1.1 Discriminating internal eval.** The authored real-corpus builder
  saturates (recall@1=1.0) because the question-gen restates the memory. Harder
  query-gen prompt (ask about the topic without echoing wording) + `simba eval
  build --n 200`. (External LoCoMo/LongMemEval recall@k already discriminate — use
  them as the primary instrument meanwhile.)
- **P1.2 Cheap ablation path.** Cloud LLM is ~17s/call. Add a fast **local**
  eval LLM option (mlx-lm) for judge/answerer/rerank sweeps; prefer recall@k
  (no LLM) for retrieval levers.
- **P1.3 Full `longmemeval_s`** (distractor haystack) vs today's oracle upper
  bound. First concrete PR after this program lands.

## Pillar 2 — Attack the weak axes (see `06-multihop.md`)
- **Multi-hop (0.305) → Track B GraphRAG** (PPR + community over `kg_edges`,
  recall@k-measurable locally) + reranker (done) + IRCoT (deferred, scaled run).
- **Open-domain (0.211) → HyDE-LLM** (untested) + query understanding.

## Pillar 3 — Strengthen the foundation (highest ceiling, underexplored)
- **P3.1 Embedder bake-off, redone on a *discriminating* eval.** The prior
  "nomic-Q4 ≈ Qwen3" result was on the saturated eval — likely hiding the single
  biggest lever. Compare nomic-Q4 vs nomic-Q8 (quantization cost) vs a stronger
  local GGUF (e.g. Qwen3-Embedding-0.6B, bge) on LoCoMo/LongMemEval recall@k via
  the existing `embed_provider`/`reembed` infra. Local only (CORE rule).
- **P3.2 Representation/extraction quality + KG density.** What we store and how
  we embed it (200-char + context split, chunking); KG density gates Track B and
  Phase 7. Garbage-in caps everything above.

## Pillar 4 — The feedback flywheel (the real-world differentiator)
Phase 6 is shipped but default-off: strength = decay × reinforcement × feedback;
`simba memory feedback` tags outcomes; dormant tier. Recall that **gets better
with use.** Run its usage/feedback eval fixture; if it helps without regressing
recall@k, earn it default-on.

## Pillar 5 — Fusion tuning (cheap, local, don't skip)
RRF `k`, candidate-pool widths, arm weights, intent thresholds (0.28/0.35),
returned-count — all sweepable on recall@k locally/fast. Probably leaving easy
points on the table. **Executed first (below) to prove the loop.**

## The long game — *exceptional*, not just *strong*
Pillars 1–5 → very good RAG memory. The moat no one ships: **Phase 7
proof-carrying, contradiction-checked, bitemporal recall** ("provable from F under
R, here's the chain") with a solver guaranteeing consistency. Scaffolded,
unproven; the differentiator.

## Sequencing (after 0.4.0)
**P1 (instrument)** → **P2 (Track B) ‖ P3.1 (embedder bake-off)** (both recall@k-
measurable) → **P4 (flywheel proof)** → **P5 continuous** → **Phase 7**.

---

## First execution — Pillar 5 fusion sweep (recall@k, local, zero cloud)

Swept four configs on full LoCoMo recall@k (cache-warm, no cloud), 2026-06-06:

| config | r@5 | r@10 | mrr | multi-hop r@5 | multi-hop r@10 | single-hop r@5 |
|---|---|---|---|---|---|---|
| baseline (rrf_k=60) | 0.573 | 0.682 | 0.490 | 0.305 | 0.410 | 0.684 |
| wide FTS pool (120/160) | 0.576 | 0.661 | 0.485 | 0.297 | 0.384 | 0.687 |
| wide vector (mr 50) | 0.571 | 0.660 | 0.490 | 0.300 | 0.398 | 0.684 |
| **rrf_k=20** | **0.595** | **0.699** | **0.496** | **0.312** | **0.424** | **0.703** |

**Findings**
- **`rrf_k=20` is a clean win**: overall +0.022 r@5 / +0.017 r@10, multi-hop up
  (0.305→0.312, r@10 0.410→0.424), single-hop *improved* (0.684→0.703), open-domain
  flat. Cross-checked on LongMemEval oracle: **neutral** (r@5 0.780→0.780,
  multi-session 0.662→0.667, no regression) — helps where there's headroom,
  doesn't hurt near the ceiling. Shipped as the new `memory.rrf_k` default.
- **Widening candidate pools regressed** (both FTS and vector). The gold is
  already in the pool — simba is **ranking-limited, not pool-limited**. This
  validates the strategy: spend on *ranking* (RRF tuning, reranker, Track B
  graph evidence) rather than *recall breadth*. Don't widen the pools.

**Methodology note:** tuned on LoCoMo, confirmed neutral-or-better on LongMemEval
(cross-dataset generalization > single-split tuning). Not saturating (0.595 ≪ 1.0).

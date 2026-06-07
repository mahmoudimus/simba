# 06 — Multi-hop: close the weak axis (evidence-gated)

Multi-hop / cross-session is simba's weakest axis on every external benchmark
(LoCoMo multi-hop r@5 ≈ 0.30, QA ≈ 0.24). This spec is the **dedicated multi-hop
phase** — and unlike the first attempt, every track here is gated on a measured
delta vs the committed baseline. We already paid for one blind attempt (C1); we
don't repeat it.

## What we already know (don't relitigate)

| Lever | Result | Lesson |
|---|---|---|
| **C1** KG-into-recall (co-occurrence fold) | **negative** | Folding raw KG neighbors into candidates didn't move multi-hop recall. The KG over a near-complete corpus is non-discriminating; naive co-occurrence is not the mechanism. |
| **C4** query decomposition | **neutral** (n=281) | Splitting the query + RRF-fusing sub-results didn't help. |
| **C2** reranker | **win** (multi-hop r@5 ≈ 0.28→0.48) | Re-ranking *already-retrieved* evidence is the lever that works. Live + default. |
| **IRCoT** (answer-time) | measured in the lever-ablation pass | Reasoning-time interleaved retrieve+reason. **Eval-only today.** |

The crux finding ([[multihop-is-reasoning-not-retrieval]]): **multi-hop is a
reasoning problem, not a pure-retrieval one.** That reframes the work into two
tracks below. **Phase 7** (`05-reflection-neurosymbolic-ops.md`) is a third,
longer-horizon track (multi-hop *by proof*).

## Decision gate (fill from the lever ablation)

> **STATUS: pending the lever-ablation result** (baseline vs +reranker/+HyDE on
> recall; baseline vs +IRCoT on multi-hop QA). Record the numbers here, then:
> - **If IRCoT shows a real multi-hop-QA delta** → lead with **Track A**
>   (productionize reasoning-time multi-hop).
> - **If IRCoT is flat / marginal** → lead with **Track B** (retrieval-time
>   GraphRAG done right), since the reranker already covers the cheap reasoning
>   gain and the remaining headroom is in getting the *right evidence* in front
>   of the reranker.
>
> Ablation table (to be pasted):
>
> | config | multi-hop r@5 | multi-hop QA acc | open-domain r@5 | latency p50/p95 |
> |---|---|---|---|---|
> | baseline | _ | _ | _ | _ |
> | +reranker | _ | — | _ | _ |
> | +HyDE | _ | — | _ | _ |
> | +IRCoT | — | _ | — | _ |

---

## Track A — Reasoning-time multi-hop (productionize IRCoT)

**Current state.** `src/simba/eval/benchmarks/ircot.py` (`build_step_prompt`,
`build_final_prompt`, `ircot_answer`, `score_case_ircot`) exists **only in the
eval harness**. The live daemon *retrieves* (returns memories for the host to
reason over); it does not *answer*. So productionizing IRCoT is a genuine product
decision, not a flag flip.

### A0 — Product decision (do this first)

Pick one, write it down, build only that:

1. **simba answers (new surface).** A multi-hop *answer* mode: an endpoint
   `POST /answer` (+ a `simba answer` CLI / skill) that runs the iterative
   retrieve→reason→retrieve→answer loop and returns a synthesized answer with
   its supporting memory ids. New capability, larger surface, owns latency.
2. **simba stays retrieval-only (recommended default).** Keep IRCoT as an
   *eval-validated* technique; in production lean on the reranker + better
   retrieval, and let the host (Claude) do the reasoning over simba's evidence.
   Ship instead a lightweight **`/recall` "iterative" mode** that does ≤N
   retrieve+reason *expansion* steps to assemble a better evidence set (no final
   answer synthesis) — multi-hop *retrieval assist*, not answering.

> Recommendation: **option 2's iterative-recall mode** unless A0 evidence shows
> answering is wanted. It keeps simba a memory layer (its contract) while
> capturing the multi-hop gain at retrieval time.

### A1 — Extract IRCoT into a shared, non-eval module

- **New:** `src/simba/memory/ircot.py` — move the reusable loop (step-prompt,
  retrieve-expand, stop condition) out of `eval/benchmarks/ircot.py`. The eval
  module imports from here so the benchmark measures exactly what ships.
- Pure functions + an `llm_client` param; **fail-open** (any LLM failure → fall
  back to the single-pass retrieved set).

### A2 — Config (`@configurable`, `memory` section)

- `multihop_mode: str = "off"` (`off` | `iterative` | `answer`)
- `multihop_max_steps: int = 3`
- `multihop_k_per_step: int = 3`
- `multihop_k_final: int = 10`
- `multihop_intents: str = "multi-hop"` (comma list of intents to route; empty =
  apply a cheap heuristic, e.g. multi-entity / temporal queries)

### A3 — Wire into the live recall path

- `recall_plan.plan_recall` / `routes.recall_memories`: when `multihop_mode !=
  "off"` and the query looks multi-hop, run the iterative expansion against the
  shared `memory/ircot.py`, dedup into the candidate set **before**
  `composite_rescore` (so the reranker still ranks the assembled evidence).
- `answer` mode (only if A0 chose it): add the endpoint/CLI that also synthesizes
  the final answer.
- Off the hot path: gate behind the mode flag; default `off`; non-blocking where
  possible; latency counted in diagnostics.

### A4 — Tests / acceptance
- Unit: step routing, stop condition, fail-open (LLM down → single-pass set),
  dedup. Fakes per existing `test_ircot.py` style.
- Bench: `simba eval bench locomo --qa` with `eval.ircot_enabled` already
  measures the eval side; add a live-path test that the iterative mode changes
  the candidate set for a multi-hop query and is a no-op when `off`.
- **Acceptance:** measured multi-hop-QA (or multi-hop r@k for iterative mode)
  delta vs baseline on the test split + latency p50/p95; default stays `off`
  until the delta justifies it.

---

## Track B — Retrieval-time multi-hop done right (GraphRAG, not co-occurrence)

C1 failed because it was naive co-occurrence. The roadmap's real plan
(HippoRAG-style) was **deferred**: community detection and personalized PageRank
were never built. This is the retrieval-time swing C1 never actually took.

**Current state.** `src/simba/kg/store.py` has `kg_neighbors` /
`kg_query(expand_hops=...)` (BFS traversal). `hybrid.py` has `rrf_fuse`
(~line 170) → `composite_rescore`. No community detection, no PPR.

### B1 — Community detection (pure-Python, no deps)
- **New:** `src/simba/kg/community.py` — **label propagation** (~40 LOC, simpler
  than Louvain; upgrade later). Assigns each KG node a community id from the
  `kg_edges` adjacency. Deterministic (seed order by node id).

### B2 — Personalized PageRank over `kg_edges`
- **New:** `src/simba/kg/ppr.py` — PPR seeded by the entities of the top-N fused
  vector/RRF hits; power-iteration to a fixed tolerance/cap. Returns neighbor
  nodes ranked by PPR mass (not raw adjacency, which is what C1 used and why it
  was non-discriminating).

### B3 — Fold PPR-ranked evidence into recall
- `hybrid.py` after `rrf_fuse`: map PPR-top nodes → their source memory ids,
  dedup into candidates **before** `composite_rescore` (so the reranker ranks the
  graph-assembled set). Optionally weight by community agreement with the seeds.
- **Config (`memory`):** `kg_ppr_enabled: bool = False`, `kg_ppr_seeds: int = 5`,
  `kg_ppr_top: int = 10`, `kg_ppr_damping: float = 0.85`,
  `kg_community_enabled: bool = False`. Fail-open if the KG is empty/sparse.

### B4 — Make it measurable (the C1 blocker)
- Bench corpora have no KG. `recall_adapter.build_retriever` must optionally
  build a **throwaway KG** from the corpus (existing extractor; regex for speed,
  optional LLM on a subset) so the PPR/community lever can be measured. **If KG
  density over the bench corpus is too low to help, that is itself the finding**
  (→ feeds the extraction-density work) — and it is reported, not hidden.

### B5 — Tests / acceptance
- Unit: label propagation on a toy graph (stable communities); PPR mass ordering;
  fold dedup; fail-open on empty KG. `tmp_path` + real sqlite KG per existing
  `tests/kg/` style.
- **Acceptance:** measured multi-hop r@k delta vs baseline on the test split with
  the throwaway-KG path + latency; **explicitly compare to C1's co-occurrence**
  to confirm PPR/community is the difference. Default `off` until justified.

---

## Track C — Phase 7 deductive closure (cross-reference)

Multi-hop *by proof* — the derive→verify→revise→distill→induce loop over
`kg_edges` (see `05-reflection-neurosymbolic-ops.md`, Group B). Deductive closure
*is* multi-hop inference; proof-carrying recall can return "provable from F under
R, here's the chain." Longest-horizon, depends on KG density (shared dependency
with Track B's B4 finding). Validate with the contradiction-injection fixture,
not recall@k alone.

---

## Sequencing & discipline

1. **Read the ablation** → fill the Decision Gate → pick the lead track.
2. **Lead track first** as a measured-delta PR (ablation table + latency, test
   split, default-off until the delta earns default-on). Never tune to saturate
   ([[eval-do-not-chase-1.0]]).
3. Slot after **0.4.0** so each track ships against a stable, committed baseline.
4. Track B and Track C share the **KG-density** dependency — if B4 shows the
   throwaway KG is too sparse to help, invest in **extraction density** before
   either, and say so in `BENCHMARKS.md`.

**Open product question threaded through all of this:** does simba *answer*
(Track A option 1 / proof-carrying recall) or only *retrieve better* (Track A
option 2 / Track B)? Resolve A0 before building Track A.

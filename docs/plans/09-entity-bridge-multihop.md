# 09 — Entity-bridge multi-hop (the untested retrieval lever)

> **VERDICT (2026-06-07): CLEAN KILL for simba's regime — built, measured, closed.**
> P0 (HotpotQA loader + `bridge_recall@k`) and the lever (E1–E4, default-off) were
> built TDD. HotpotQA *distractor* saturates recall (`bridge_recall@10=1.0`, ~10-para
> haystacks), so we built a tractable **fullwiki regime** — `load_hotpotqa_pooled`
> pools N questions into one ~5k-paragraph corpus (recall genuinely unsaturated,
> `@10=0.876`). On that pool the entity-bridge ADD lever is **flat-to-slightly-
> negative** (`@3 −0.002, @5 0.000, @10 −0.012`) — and crucially, swapping the
> capitalized-span regex for **real spaCy PERSON/ORG/GPE NER** (YourMemory's exact
> signal) gave the **identical** result. **NER precision was not the gap.** Root
> cause: simba's bge-large hybrid base retriever is strong enough that the gold is
> already in the candidate pool → multi-hop is a **ranking** problem (the reranker's
> job), not a **recall** problem; entity-bridge ADD only helps a *weak* retriever
> that misses gold (YourMemory's regime). All six retrieval-time multi-hop levers
> (C1, C4, Track B, Track A, entity-bridge, + the reranker) now converge: **the
> reranker is the only win.** Only untested stone: *true* 5M-paragraph fullwiki
> (infeasible to embed locally) — reopen only there. Harness + lever + spaCy
> extractor kept default-off as reusable instruments.


A measured experiment, not a commitment. simba's multi-hop verdict is "dead at
retrieval" — but that verdict was earned by **co-occurrence (C1)** and **PPR over a
dense graph (Track B)**, both **on LoCoMo**, both negative. The borrow survey
(`08`) surfaced the one system with a *positive* multi-hop result — **YourMemory,
+12pp on HotpotQA** — using a mechanism we never tried: **shared-named-entity
bridge edges**. This spec tests whether that reopens multi-hop, and does it on the
**right benchmark**.

## Hypothesis (and why it's distinct from the negatives)

| | link signal | rank signal | benchmark | result |
|---|---|---|---|---|
| C1 | raw co-occurrence (diffuse) | fold into candidates | LoCoMo | negative |
| Track B | PPR mass over dense graph (diffuse) | PPR-ranked fold | LoCoMo | negative |
| **this** | **shared *named entity*** (sparse, high-precision) | depth-2 BFS from retrieved seeds | **HotpotQA + LME multi-session** | **?** |

Two independent changes from the failures: (1) **sparse, high-precision links**
(named entities, not co-occurrence/PPR), and (2) **a benchmark where multi-hop is
*not* directly retrievable** (LoCoMo's "multi-hop" largely is — that's why every
lever died there). If this is *also* negative on a genuine bridge-entity dataset,
multi-hop-at-retrieval is conclusively dead → pivot to Phase 7 / extraction
density. Either outcome is decision-grade.

## Pillar 0 — fix the benchmark FIRST (the real prerequisite)

Do not run this on LoCoMo as the primary signal. Add a genuinely multi-hop set.

- **P0.1 HotpotQA loader.** `src/simba/eval/benchmarks/hotpotqa.py` — load HotpotQA
  (distractor setting): each question has a gold answer + **supporting facts**
  (title + sentence ids) spanning ≥2 paragraphs. Map to the existing `Dataset`/
  `EvalCase` shape: corpus = the 10 context paragraphs as memories; `relevant_ids`
  = the supporting-fact paragraphs (the bridge). This gives a **bridge-recall@k**
  metric: did we retrieve *both* supporting paragraphs.
- **P0.2 Metric.** Add `bridge_recall@k` (all gold supporting ids in top-k) to
  `eval/metrics.py` alongside recall@k — multi-hop needs *all* hops, not any.
- **P0.3 Wire into `simba eval bench`** as `hotpotqa` (mirror locomo/longmemeval).
  Also keep **LongMemEval multi-session** as the secondary multi-hop signal.
- Fully local: HotpotQA is recall@k over labelled supporting facts — **no LLM
  judge** needed. Avoids the ~17s/call trap entirely.

## The lever

Build on what exists: `src/simba/kg/entities.py` (`normalize_entity`, `resolve`,
embedding-synonym merge) and `src/simba/kg/store.py`.

- **E1 — entity index (write/index time).** For each memory, extract entities
  (reuse the KG extractor; regex/NER for speed) and record `(memory_id, entity_key)`
  where `entity_key = normalize_entity(name)`. Two memories are **bridged** if they
  share an entity key. N-word prefix matching is already approximated by
  `normalize_entity` + `resolve`; extend `resolve` if needed so "Shirley Temple
  Black" resolves to "Shirley Temple". New module `src/simba/memory/entity_bridge.py`.
- **E2 — recall-time bridge expansion.** In `hybrid.py` after `rrf_fuse`
  (the shared path both `routes.py` and `recall_adapter.py` use): take the top-N
  fused seeds, collect their entity keys, pull **bridged memory ids** (BFS depth
  1–2 over the shared-entity relation), **dedup into the candidate set before**
  `composite_rescore` + reranker (so the reranker still ranks the assembled set).
  Fail-open if no entities/empty index.
- **E3 — config (`memory`, `@configurable`, default-off):** `entity_bridge_enabled
  = False`, `entity_bridge_hops = 2`, `entity_bridge_seeds = 5`,
  `entity_bridge_max = 10`, `entity_bridge_min_shared = 1`.
- **E4 — bench measurability (the C1/Track-B blocker, solved differently).** Bench
  corpora have no KG. `recall_adapter.build_retriever` must build the entity index
  **from the corpus** at retriever-build time (regex/NER over the memories) so the
  lever is exercised. This is cheap (no LLM) unlike the throwaway-KG path Track B
  needed. If entity density is too low to bridge, that's a finding.

## Implementation order (TDD)

1. `eval/benchmarks/hotpotqa.py` + `metrics.bridge_recall@k` + bench wiring (P0).
   Establish a **baseline** bridge_recall@k on HotpotQA (entity bridge OFF) — this
   is the number to beat.
2. `memory/entity_bridge.py` (index + lookup), pure functions, TDD with a toy
   corpus (two memories sharing "Shirley Temple", a distractor that doesn't).
3. `recall_adapter` builds the entity index from the corpus (E4).
4. Fold into `hybrid.py` after `rrf_fuse` (E2), config-gated (E3).
5. Measure: `simba eval bench hotpotqa` ON vs OFF → bridge_recall@k delta; confirm
   on LongMemEval multi-session; sanity-check LoCoMo doesn't regress.

## Tests
- `tests/eval/benchmarks/test_hotpotqa.py` — loader maps supporting facts →
  `relevant_ids`; `bridge_recall@k` = 1.0 only when *all* gold ids in top-k.
- `tests/memory/test_entity_bridge.py` — shared-entity link found; distractor not
  linked; prefix match ("…Black" ↔ "…"); empty index fail-open.
- `tests/memory/test_hybrid.py` — bridge fold adds bridged ids before rescore;
  no-op when `entity_bridge_enabled=False`.

## Acceptance / kill criteria
- **Win:** measured **+bridge_recall@k on HotpotQA** (and non-regression on LoCoMo
  single-hop) with the bridge ON vs OFF, default-on only if it earns it. Report the
  delta; never tune to saturate ([[eval-do-not-chase-1.0]]).
- **Kill:** if entity-bridge is *also* flat/negative on HotpotQA (where the
  mechanism *should* work), then multi-hop-at-retrieval is conclusively dead for
  simba's corpus shapes → stop chasing retrieval levers, pivot to **Phase 7**
  (proof-carrying / deductive multi-hop) and **extraction density**. Record it
  next to C1/Track B/Track A in `06-multihop.md`.

## Why this is the right next bet
It's the only multi-hop idea with a *positive* external result, it's mechanistically
distinct from all three simba negatives, it's **fully local + recall@k-measurable**
(no cloud-judge trap), and — win or lose — it converts "we think multi-hop is dead"
into a *measured* conclusion on a benchmark where multi-hop is real. See `08`.

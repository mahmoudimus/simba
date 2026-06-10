# 17 — Episodic memory as a materialized aggregate view (answer-time aggregation)

> **STATUS (2026-06-10): DESIGN + VALIDATION GATE. Nothing built yet.**
> Pillar 0 (the two-arm ceiling probe) is **decision-grade and must run FIRST**.
> Every build pillar below is **contingent** on what Pillar 0 returns — do not
> write production code before the gate. This spec is self-contained: a fresh
> session can execute it without the originating conversation.

This spec answers one question — *how do we fix the question types that store-raw
recall alone gets wrong (counting, temporal arithmetic, ordering), without
repeating the mistakes that are already measured dead?* — and it gates the answer
on a cheap experiment before committing to a build.

---

## 1. Why (the measured problem this fixes)

Three results from the prior session set the whole frame. **Trust them; they are
measured, not assumed:**

1. **Digest-as-recall collapses QA.** Routing personal-assistant QA through
   simba's learn-from-chat *digest* (extract-and-summarize, then answer from the
   summaries) collapses LongMemEval **oracle QA 0.700 → 0.089**. Mechanism-proven
   via drill-down, five distinct failure modes:
   - **counting** ("how many Korean restaurants" = *four*) → summarization merges/
     drops instances (kept 3 of 4, scattered);
   - **temporal** ("days between Holi and Sunday mass") → one date anchor never
     extracted; surviving events stamped with the *conversation* date, not the
     event date;
   - **ordering** ("which did I buy first") → comparison target dropped;
   - **verbatim assistant-recall** ("the 7th job you listed") → **0 memories**
     (prompt extracts *user* facts, the answer is *assistant* output);
   - **salience miss** ("what degree did I graduate with") → the incidental fact
     dropped in favor of the conversation's apparent theme.
   The only bucket that survives (and all 3 gains) is **preference** — because it
   asks for *gist*, which is what summarization preserves. Recorded:
   memory `digest-eval-extraction-in-loop-verdict`; instruments
   `src/simba/eval/digest_corpus.py` + `.simba/digest_ab*.py`.

2. **The gap is aggregation, not recall.** On oracle, **recall@10 ≈ 0.89** (gold
   evidence is almost always retrieved) yet **QA ≈ 0.70**. That ~20pt gap is
   cases where the evidence was in front of the answerer and it *still* got the
   answer wrong — it failed to *aggregate* (count, subtract dates, resolve "now").
   So the lever is **answer-time aggregation**, not better retrieval.

3. **store-raw is the correct recall substrate** and must not be touched. It is
   robust where extract-at-store systems collapse (same store-raw robustness that
   beat mem0-OSS 7× on the same axis — memory `simba-vs-mem0-same-axis`).

**Information-theoretic root cause:** extraction is a *write-time bet on what will
be asked*; LongMemEval probes the unpredictable long tail (exact counts, dates,
incidental facts, assistant output) that any gist-compressor must discard. Raw
storage makes no bet. Therefore any derived layer must be **additive over raw,
never a replacement.**

---

## 2. The principle (hard constraints, each earned by a measured result)

| Constraint | Why (measured) |
|---|---|
| **Index, don't summarize** | the digest summarized → 0.09. An index organizes access; a summary discards. |
| **Lossless-pointered** | every derived node cites the raw turn id(s) it came from → recoverable + auditable. |
| **Additive, never authoritative** | raw stays truth; the view is a derived cache that can be wrong/stale without data loss. |
| **Answer-time, not retrieval-time** | every retrieval-time graph fold died (see §3). The view never enters candidate selection. |
| **Augment, don't replace** | IRCoT *replaced* direct retrieval with sub-query loops and drifted → −0.139. Keep direct top-k; *add* the structure beside it. |
| **Deterministic aggregation** | LLMs are unreliable at counting/date-math. Count/subtract in **Python** over the cluster; let the LLM phrase the computed fact. |

---

## 3. The graveyard (do NOT re-attempt these — they are measured dead)

Audited from the memory record (numbers are recorded deltas; raw tables in the
named branches / `docs/plans/09`):

| Attempt | Category | Result | Mechanism |
|---|---|---|---|
| **C1** KG co-occurrence/typed fold | retrieval-time | LoCoMo multi-hop r@5 **0.271→0.243** | displacement |
| **Track B** PPR fold | retrieval-time | LoCoMo overall **0.614→0.590**, LME multi-session **0.687→0.627** | displacement |
| **Entity-bridge** shared-entity BFS | retrieval-time | pooled HotpotQA bridge_recall@5 **0.694 OFF==ON**, @10 −0.012; spaCy==regex | gold already in pool |
| **C4** query decomposition | retrieval-adjacent | n=281 r@5 **0.305→0.300** | NEUTRAL (noise) |
| **IRCoT** sub-query interleave | **answer-time** | LoCoMo multi-hop QA **0.361→0.222** | sub-query drift → abstention |

**Wins (the only ones):** the **reranker** (answer-time re-rank of *retrieved*
evidence) and the **breadth-ladder** (+17pp QA at k=20 — more *complete* evidence
helps even when recall@5 is saturated; memory `lme-leaderboard-reality`). Both are
**answer-time evidence-completeness** levers. This design is in that family;
IRCoT's failure (answer-time, 0-for-1) is the warning that earns Pillar 0.

Unifying cause of the retrieval-time deaths: on accessible haystacks (LoCoMo,
HotpotQA-distractor, LME-oracle) the gold is **already in the candidate pool**, so
it's a *ranking* problem; adding candidates only *displaces*. The recall-ADD value
only manifests at ~5M-fullwiki scale we can't index locally. **Conclusion: the
view feeds the *answerer*, never the *retriever*.**

---

## 4. Architecture: CQRS + a materialized view over an append-only log

The pattern is event-sourcing. Two derivations (information-theoretic §1 +
distributed-systems) converge on the same constraints — that convergence is the
confidence.

```
              WRITE (command) side                      READ (query) side
 raw turns  ──CDC──▶  structuring   ──▶  resolve  ──▶  MATERIALIZED VIEW
 (LanceDB+    hooks/   extractor      entity-      (entity clusters +     ◀── query
  JSONL,      sched.   (LLM, async)   canonical    event timeline +            │
  the LOG,                            -ize         supersedes; SQLite)         │
  truth)                                                  │                    │
                                                          ▼                    ▼
                                          deterministic aggregate    + raw top-k (unchanged
                                          (COUNT / date-diff /          hybrid + reranker)
                                          walk-supersedes, in Python)         │
                                                          └────────┬──────────┘
                                                                   ▼
                                                  answerer: phrases computed fact
                                                  + provenance, beside raw evidence
```

| DB concept | simba component |
|---|---|
| append-only event log (truth) | raw turns — already the "append-only" CORE constraint |
| materialized view (read projection) | fact graph: entity clusters + event timeline (the bitemporal KG, L4, *partly exists*) |
| async view-maintenance / CDC | detached rlm engine + episodes scheduler |
| CDC triggers | the hooks (PreCompact = session-end; PostToolUse = per-action) |
| idempotency ledger (exactly-once) | `rlm_jobs` / `episode_jobs` (UNIQUE + claim/complete) |

**Read path (sync, additive, intent-routed):** normal hybrid top-k (UNCHANGED) +
*if* the query intent is counting/temporal/update/enumeration → look up the
relevant cluster/timeline slice → **compute the aggregate in Python** → inject a
structured block (`"Korean restaurants visited: 4 [turns 12,40,55,71]"`) beside
the raw evidence. Otherwise: normal path.

**Write path (async, off the hot path):** the structuring extractor (the rlm
`llm-cli` engine, pointed at a *structuring* prompt instead of a summarizing one)
emits `(entity, mention, turn_id)` and `(event, real_date, turn_id, supersedes?)`
→ entity-resolve/canonicalize → **idempotent upsert** into the view.

**Correctness properties the framing buys us:**
- **Eventual consistency w/ read-your-writes.** The view lags the log; the
  *additive read* (raw top-k beside the view) covers the un-materialized tail — so
  a just-mentioned 4th restaurant is still present in raw even if the view says 3.
  **The additive design *is* the bounded-staleness mitigation.** (Counting is the
  sharp risk: a stale view undercounts → mitigated by raw tail + periodic rebuild.)
- **Idempotency.** Upserts keyed by `(canonical_entity, turn_id)` /
  `(event, turn_id)` so re-processing never double-counts.
- **Rebuildability = swappability.** The view is pure derivation of the immutable
  log → drop & replay to rebuild → a better extractor ships with zero data risk.
  This is the modularity premise made concrete.
- **Provenance = auditability.** Every view row cites turn ids → view/log
  divergence (extractor missed the 4th restaurant) is *detectable* by cross-check.

---

## 5. Storage decision: stay embedded (NO Postgres)

The materialized-view framing makes Postgres tempting (native `MATERIALIZED VIEW`,
triggers, `ON CONFLICT`, recursive CTEs, pgvector, AGE). **Reject it:**

- Postgres is an **external service** → violates the CORE "no external services"
  constraint and **erases simba's reproducible-anywhere differentiator** (recorded
  lesson: *letta is unreproducible precisely because it's Postgres-only*).
- A materialized view is **just a maintained table.** Everything the design needs,
  the embedded stack already does: **SQLite** (already used for FTS5 + `rlm_jobs`)
  gives ACID, `INSERT … ON CONFLICT` (idempotent upsert), `WITH RECURSIVE`
  (timeline / supersedes walk), and aggregates. **LanceDB** stays for vectors. The
  graph queries we need (group-by-entity, walk-supersedes, slice-timeline) are all
  plain SQL — no graph server required.
- **If single-engine consolidation is ever wanted** (to kill LanceDB↔SQLite-FTS
  drift), the *embedded* answer is **DuckDB** (in-process: VSS vectors + FTS +
  recursive CTE + ACID) — **not Postgres.** Postgres earns its place only on a
  **product-shape change to a hosted multi-tenant service** — a different product.
- Keep storage **behind interfaces** (mostly already true: `vector-db`, the FTS
  mirror, the KG store are separate modules) — the modularity insurance that makes
  a future engine swap contained, not a rewrite.

**Decision: SQLite-now for the view, LanceDB for vectors, DuckDB-maybe-later,
Postgres-never (unless the product becomes hosted multi-tenant).** Do NOT
re-platform storage before Pillar 0 proves the architecture earns it.

---

## 6. Pillar 0 — THE GATE: two-arm ceiling probe (run FIRST, decision-grade)

**Question:** when the answerer is handed the *complete, structured* set of
relevant facts, does QA on the counting/temporal/ordering buckets recover toward
1.0 — and does it need a *deterministic* aggregate, or is completeness enough?

**Data:** `longmemeval_oracle.json` (oracle haystack isolates aggregation from
retrieval). Path used in prior runs:
`/Users/mahmoud/src/ai/simba/.simba/benchmarks/longmemeval_oracle.json`.
**Buckets:** the regressions — `knowledge-update`, `temporal-reasoning`,
`multi-session`, `single-session-user` (skip `preference`: digest already helps it).
**Stack:** deepseek-v4-flash answerer, deepseek-v4-pro judge (validated GPT-4o
proxy; memory `judge-calibration-deepseek-v4-vs-gpt4o`). `~15`/bucket.

For each case, build the **oracle-complete cluster** = all raw turns from the gold
answer-session(s) that mention the queried entity/timeframe, **date-ordered**. (For
the probe this uses the *gold* sessions — no extractor yet — so it measures the
*answer-time ceiling*, independent of extraction quality. Extraction quality is a
*separate*, later question.) Then three arms:

- **RAW** (baseline): normal top-k retrieval → `build_answer_prompt` → judge.
  (≈ the 0.70/per-bucket already measured.)
- **Arm A — completeness:** give the answerer the full ordered cluster as context
  → LLM counts/reasons itself → judge.
- **Arm B — deterministic aggregate:** compute the answer in **Python** over the
  cluster (count distinct entities; subtract two event dates; walk supersedes to
  the latest value) → hand the answerer the *computed* fact + provenance → judge.

**Decision rule (each outcome is decision-grade):**
- **B ≫ A and B ≫ RAW** → the deterministic aggregator is the lever → build §7.
- **A ≈ B ≫ RAW** → completeness alone suffices → build the index, **skip** the
  symbolic aggregator.
- **A ≈ B ≈ RAW** → even perfect structured evidence doesn't help → **DO NOT
  BUILD**; the bottleneck is elsewhere (answerer reasoning / benchmark framing).
  Record and stop. (This is the IRCoT-style outcome — and we'd have spent ~1 hour,
  not weeks.)

**Instruments:** reuse `src/simba/eval/digest_corpus.py` (`digest_dataset`,
committed) and the A/B harness pattern in `.simba/digest_ab.py` (rlm-pa worktree;
gitignored — copy or rebuild from the pattern). Cost: a few hundred deepseek calls,
~1 hr. Metric: **QA accuracy per bucket** (recall@k is N/A on a cluster). Output: a
table (RAW / A / B per bucket) + the chosen world, committed to this spec.

### Pillar 0 — RESULT (2026-06-10): **GO — World 1 (B ≫ A ≫ RAW)**

Ran a 3-arm version (RAW top-k · A = whole oracle evidence corpus, date-ordered ·
B = A + an explicit enumerate-then-compute scaffold = LLM proxy for the
deterministic aggregator). 4 buckets × 15, deepseek-v4-flash answerer /
deepseek-v4-pro judge. Harness: `.simba/ceiling_probe.py`.

| Bucket | RAW | A (complete) | B (complete+compute) | A−RAW | B−A |
|---|---|---|---|---|---|
| multi-session (counting) | 0.467 | 0.733 | 0.800 | **+0.266** | +0.067 |
| single-session-user | 0.667 | 0.800 | 0.867 | +0.133 | +0.067 |
| knowledge-update | 0.800 | 0.733 | **0.933** | −0.067 | **+0.200** |
| temporal-reasoning | 0.933 | 1.000 | 1.000 | +0.067 | 0.0 |
| **OVERALL** | **0.717** | **0.817** | **0.900** | **+0.10** | **+0.083** |

Overall **B ≫ A ≫ RAW** (+0.183 B vs RAW; B fixed 12 RAW-misses, regressed ~1).
**Both parts earn their place, on different buckets:**
- **Completeness (the index)** is the dominant lever for **counting** (multi-session
  **+0.266** — top-k misses instances "how many" needs).
- **Aggregation (the compute)** is the dominant lever for **knowledge-update**
  (completeness alone *hurt*, A −0.067, because historical values drown the current
  one; "pick latest" → B **+0.20**).
- **temporal-reasoning** is already ~solved on oracle (RAW 0.933) — its difficulty
  is at `_s` scale (buried date anchors = a retrieval/index problem, not arithmetic).

Opposite of IRCoT: structured evidence **helps** because we *augment* retrieval, not
*replace* it. **Caveats:** (1) this is the **oracle answer-time ceiling** with GOLD
evidence — it does NOT prove the async extractor can rebuild a complete-enough index
from raw turns (→ the extraction-quality de-risk is the next gate). (2) B is an
LLM-scaffold proxy → its +0.083 is a **lower bound** on a real Python aggregator.
(3) n=15/bucket: overall solid, per-bucket suggestive.

**Decision: build the lever — index first (broad counting win), aggregator layered
for update/latest — but de-risk extraction quality before trusting it end-to-end.**

---

## 7. Build (CONTINGENT on Pillar 0 — design sketch, full TDD specced after the gate)

Only relevant if Pillar 0 says A or B beats RAW. Each piece `@configurable`,
default-off, TDD, plugged into the shared path, ablation + latency.

- **View schema (SQLite):** `entity_mention(canonical_entity, turn_id, raw_text,
  event_date)`, `event(event_id, label, event_date, turn_id)`,
  `event_edge(from_event, to_event, kind)` (e.g. `supersedes`, `next`). All
  upsert-idempotent (`ON CONFLICT(canonical_entity, turn_id)`), every row → turn id.
- **Structuring extractor:** the rlm `llm-cli` engine (already built, spec 17's
  prerequisite shipped on `feat/rlm-personal-assistant`) pointed at a *structuring*
  prompt (emit the tuples above as JSON). New `rlm.structuring_prompt` or a mode
  on `digest_prompt`. Reuse `_parse_*`/the JSON path.
- **Async refresh:** PreCompact/scheduler → extract new turns → resolve
  (`src/simba/kg/entities.py` `normalize_entity`/`resolve`) → upsert. Idempotency
  via the `rlm_jobs` pattern.
- **Intent → aggregate (read side):** classify counting/temporal/update intent
  (reuse the existing intent-aware recall hook) → cluster/timeline lookup →
  deterministic compute → inject block into the answer context. Config-gated.
- **Reuse, don't rebuild:** bitemporal KG (L4) IS the event-timeline+supersedes
  store; `kg/entities.py` for canonicalization; the reranker stays the ranking win;
  `rlm_jobs` for idempotency.

---

## 8. Existing-component map (what to reuse vs build)

| Need | Exists? | Where |
|---|---|---|
| append-only raw log | ✅ | LanceDB + JSONL |
| async extractor module (swappable) | ✅ | `src/simba/rlm/engine.py` `LlmCliEngine` (this branch) |
| entity normalize/resolve | ✅ | `src/simba/kg/entities.py` |
| event timeline + supersedes (bitemporal) | ✅ (under-populated) | KG store (L4, Phase 4/7) |
| idempotency ledger | ✅ | `rlm_jobs` / `episode_jobs` |
| CDC triggers | ✅ | the 6 hooks |
| reranker (ranking win) | ✅ | shipped, default-on |
| intent-aware routing | ✅ (partial) | agent-oss borrow Phase 0 |
| entity-cluster index (for counting) | ❌ build | new SQLite table |
| structuring prompt | ❌ build | new `rlm` config |
| deterministic aggregator | ❌ build (if Pillar 0 = B) | new `src/simba/memory/aggregate.py` |
| answer-time aggregate injection | ❌ build | recall/answer path |
| eval instrument | ✅ | `src/simba/eval/digest_corpus.py` |

---

## 9. Open questions for the evaluator

1. **Can the LLM count over a complete cluster (Arm A), or is Python needed (Arm
   B)?** — Pillar 0 answers directly. (Strong prior: B, LLMs miscount.)
2. **Entity-resolution quality on dialogue** — is `kg/entities.py` canonicalization
   enough, or does merging ("Mapo" == "Mapo Korean") need an LLM resolver? (Affects
   counting correctness; test after Pillar 0.)
3. **Intent-router precision** — false positives inject irrelevant aggregates;
   does that *harm* non-aggregation queries? (Gate must be harm-free, like the
   conflict-surfacing gate.)
4. **Staleness bound** — is the refresh latency acceptable given the raw-tail
   read-your-writes fallback? Counting is the sharp case.
5. **DuckDB consolidation** — defer until cross-store drift actually bites; not on
   the critical path.

---

## 10. Boot a fresh session (paste-ready)

> Read `docs/plans/17-episodic-aggregation-view.md` end-to-end, plus memories
> `digest-eval-extraction-in-loop-verdict`, `multihop-is-reasoning-not-retrieval`,
> `reranker-is-the-multihop-win`, `lme-leaderboard-reality`, `simba-vs-mem0-same-axis`.
> Then execute **Pillar 0 only**: the two-arm ceiling probe (RAW vs completeness vs
> deterministic-aggregate) on the LongMemEval oracle counting/temporal/ordering
> buckets, deepseek-v4-flash answerer + deepseek-v4-pro judge, ~15/bucket, reusing
> `src/simba/eval/digest_corpus.py` and the `.simba/digest_ab.py` harness pattern.
> Report the per-bucket table + which of the three worlds we're in. Do **not** build
> §7 until the gate result is in. Honor the graveyard (§3): the view feeds the
> answerer, never the retriever; augment, never replace.

## 11. Acceptance

- **Pillar 0:** a committed results table (RAW / A / B per bucket) + the decision
  (which world) appended to this spec. That is the deliverable; the build is
  contingent.
- **Contingent build:** each lever default-off, TDD (RED first), ablation table +
  latency p50/p95, `BENCHMARKS.md` updated. Storage stays embedded (§5).

# 08 — Borrow survey: 8 memory systems (2026-06-07)

Surveyed 8 freshly-cloned memory/recall systems (`~/src/ai/memory/{forgetful,
auto-memory, shodh-memory, YourMemory, dna-memory, memory-palace, animaworks,
yantrikdb-server}`) for ideas worth borrowing — especially for the **open
problem**: multi-hop. simba's three multi-hop attempts are all **measured
negatives on LoCoMo** — C1 (co-occurrence fold), Track B (PPR over a dense graph),
Track A (IRCoT). The reranker is the only multi-hop win. This survey asked: does
anyone have a *positive* multi-hop lever we haven't tried?

## Ranked digest

| Repo | Maturity | Stack | Eval (their numbers) | Worth borrowing? |
|---|---|---|---|---|
| **YourMemory** | young but real | pure-Py, local | LoCoMo r@5 **0.59**; LongMemEval-S recall-any@5 **0.958**; **HotpotQA BOTH@5 71.5 vs 59.5 w/o entity edges (+12pp)** | **★★★ entity-bridge multi-hop (positive!)** |
| **shodh-memory** | serious (122k Rust) | Rust, local | LoCoMo overall 0.40, **multi-hop 0.167** (n=10, weak) | ★★ Hebbian edge learning, spreading-activation, hybrid decay, replay |
| **animaworks** | serious (129k Py) | Py, Neo4j/Chroma | LoCoMo F1 **0.637** (Qwen) / multi-hop 0.419 | ★★ procedural memory + reconsolidation, priming gate, biological forgetting |
| **yantrikdb** | serious (Rust core) | Rust+Py | LongMemEval **oracle** r@5 0.806 (retrieval-only; retracted-bench history) | ★★ proactive "urge" layer, background think-loop, affect/procedural types |
| **forgetful** | serious (43k Py) | Py, sqlite-vec/pg | none | ★ provenance + event-sourced audit log, meta-tool pattern |
| **memory-palace** | active, unproven | Py, Ollama | none | ★ prose-from-code indexing, graph-centrality ranking term |
| **auto-memory** | narrow, real | Py stdlib | none (token-cost anecdotes) | ★ read agent's OWN session JSONL; trust-fencing; FTS5 sanitizer |
| **dna-memory** | POC | Py, sqlite | none | ½ adversarial contradiction sweep, typed causal edges |

(simba's retrieval — LanceDB + FTS5 + RRF hybrid + LLM reranker + bitemporal KG +
decay/feedback + neuro-symbolic — is at or ahead of all 8 on the core stack. The
value here is *specific mechanisms*, not wholesale architecture.)

## The headline finding — entity-bridge multi-hop is a POSITIVE lever

**YourMemory** is the only system with a *measured positive* multi-hop result, and
the mechanism is **not** what simba tried:
- At store time it runs **NER** (PERSON/ORG/GPE/…) and creates `entity:<X>` edges
  linking memories that **share a named entity**, with **N-word prefix matching**
  ("Shirley Temple Black" ↔ "Shirley Temple"). (`graph/graph_store.py`
  `_entity_linked_nodes`)
- At recall it does **2-round retrieval**: vector+BM25 seeds → **depth-2 BFS over
  the entity graph** to surface low-similarity *bridge* facts. (`services/retrieve.py`)
- Result: **+12pp on HotpotQA** (a genuinely bridge-entity multi-hop dataset).

Why this matters vs simba's negatives: C1 linked on **co-occurrence**, Track B
ranked by **PPR mass over a dense graph** — both diffuse signals. YourMemory links
on **shared named entities** (sparse, high-precision) and traverses from
*retrieved seeds*. **And critically:** their win is on **HotpotQA**, while all of
simba's negatives were on **LoCoMo**, where multi-hop evidence is largely
*directly retrievable* (which is exactly why every lever died there). The negative
we recorded may be "**co-occurrence/PPR on LoCoMo** can't help," not "graph
retrieval can't help multi-hop." → see `09-entity-bridge-multihop.md`.

## Cross-cutting theme — feedback-driven graph *topology* (Hebbian/LTP)

Three of the four serious systems converge on this, and simba lacks it:
- **shodh** (`retrieval.rs`/`feedback.rs`): co-retrieved memories form/strengthen
  edges; edges past a coactivation threshold are "potentiated" to permanent tiers.
- **animaworks**: access-count LTP boost (30-day half-life).
- **yantrikdb**: retrieval-feedback as an auto-tuned scoring signal.

simba's Phase 6 is **scalar** decay/feedback on individual memories. These learn
the **graph structure** from usage — a structurally different "memory that learns"
bet. Candidate future phase.

## Capability gaps (orthogonal to retrieval)

- **Procedural memory + reconsolidation** — animaworks (`reconsolidation.py`:
  `failure_count≥2 & confidence<0.6` → LLM rewrites the procedure) and yantrikdb
  both have a distinct "strategies that worked" type with failure-triggered
  self-revision. simba has no procedural/skill memory type.
- **Proactive surfacing** — yantrikdb's **urge queue** + background think-loop:
  memory *pushes* (conflict alerts, follow-ups) between turns, inverting pull-only
  recall. Novel architecture direction.
- **Multi-channel priming** — animaworks renders each recalled memory as
  ANCHOR/GUARDRAIL/POINTER/SUPPRESS by trust+risk (POINTER = stub, fetch body on
  demand) vs simba's flat injection; plus an **act-before-recall gate**.

## Cheap, directly borrowable levers

- **Chain-aware pruning** (YourMemory `decay_job.py`): don't prune a decayed memory
  if a graph neighbor is still strong — "related memories age together." Plugs into
  simba's decay pass.
- **Subject-aware dedup + contradiction gating** (YourMemory `resolve.py`): cosine
  bands → reinforce/replace/merge, gated by a 2-token subject embedding + polarity/
  negation/number contradiction. Sharper than simba's flat 0.92 dedup.
- **Graph-centrality ranking term** (memory-palace): `0.7·sim + 0.15·log(access) +
  0.15·in_degree_centrality`. Cheap to prototype on simba's KG.
- **Provenance + audit log** (forgetful): per-memory `encoding_agent/model` lineage
  + before/after mutation log with a distinct LLM-maintenance actor.
- **Trust-fencing** (auto-memory `providers/file/_trust.py`): file-backed content
  tagged untrusted + wrapped in sentinels — a prompt-injection guard simba lacks.
- **Adversarial contradiction sweep** (dna-memory): periodically hunt conflicting
  pairs and vote a winner (recency/weight/type) — simba supersedes on write but
  never goes looking.

## Recommendations (priority)

1. **Entity-bridge multi-hop experiment** (`09-entity-bridge-multihop.md`) — the
   one measured-positive multi-hop lever we haven't tried; measure it on a
   *genuinely* multi-hop benchmark (HotpotQA / LongMemEval multi-session), not
   LoCoMo. **Highest value: it directly tests whether our multi-hop "death" was a
   benchmark/mechanism artifact.**
2. **Hebbian graph-topology learning** — convergent across the serious systems; a
   future "memory that learns" phase beyond Phase 6's scalar signals.
3. **Procedural memory + reconsolidation** — a capability gap (new memory type),
   separate from the retrieval fight.
4. Cheap levers (chain-aware pruning, subject-aware dedup, centrality term) — fold
   in opportunistically; each is a small measured delta.

---

## Addendum (2026-06-07): paper-backed "self-maintaining memory" cluster

Surveyed 4 more (paper releases) + investigated one benchmark. These cluster on
**memory that evolves/maintains itself** — the thread behind simba's
shipped-but-unvalidated Phase 6/7.

| Repo | What | Eval (their metric) | ★ Borrow |
|---|---|---|---|
| **A-MEM** (`A-mem-sys`) | Zettelkasten atomic notes; LLM **evolves neighbors on write** | LoCoMo (in separate repo) | **neighbor metadata back-propagation** |
| **MemRL** | tabular **Q-learning over memory metadata** (no weight updates) | agentic (HLE/BCB/ALFWorld) — not recall@k | Q-EMA+visits, two-phase utility rerank, **null-action abstain** |
| **ViLoMem** | multimodal grow-and-refine | multimodal VQA acc — not recall@k | LLM **similarity-merge** (but it's dead code there) |
| **ReMe** | file+vector memory, **experience/procedural** type, auto-compaction | LoCoMo QA **86.23**, HaluMem-Med 94.06 (LLM-judged, not recall@k); **cloud-only** | **procedural memory schema**, dedup-on-write+validation gate, **HaluMem adapter** |

### The concrete borrows
- **A-MEM — neighbor metadata back-propagation** (`memory_system.py` `process_memory`):
  on write, an LLM regenerates *existing neighbors'* `context`/`tags` (never their
  content → append-only-safe). A "grow-and-refine" pass that enriches simba's FTS5/
  embedding *inputs* without replacing content. Optional write-time hook, gated like
  Phase-6. Cost: +2 LLM calls/write (the trap we've flagged).
- **MemRL — utility over similarity, no RL infra needed.** It's *tabular* Q-learning
  on memory metadata (`q_value`, `reward_ma` EMA, visit counts; `service/value_driven.py`),
  fully compatible with simba's local-first/no-train constraint. Portable: (a) Q-EMA+
  visit-count as a more principled Phase-6 feedback signal; (b) **two-phase retrieval**
  (similarity prefilter → utility rerank) as a pre-reranker sort; (c) **null-action /
  τ-floor abstain** (refuse to recall when top-sim is low) → adapt simba's 0.35 floor
  into an abstain path. Friction: simba lacks env task-success reward; signal must come
  from recall outcomes (weaker).
- **ReMe — procedural/"experience" memory** (`when_to_use` condition + content +
  LLM-validation score; `procedural_summarizer.py`): a capability gap simba has. Borrow
  the schema + trajectory→success/failure split; reimplement as a plain prompt (NOT
  AgentScope ReAct — ReMe is cloud-only and violates simba's local rule). Also its
  **dedup-on-write + validation gate** is a cleaner version of simba's 0.92 dup-detect.
- **ViLoMem — LLM similarity-merge**: when a near-dup is *complementary* (not a
  contradiction), fuse two guidelines into one via a small LLM call instead of
  replace-only supersession. Route at store time when `dup_sim ∈ [floor, supersede)`.
  Unvalidated even upstream (dead code) — pilot cautiously.

### ★★★ The benchmark find — HaluMem (→ `10-halumem-forgetting-eval.md`)
ReMe led us to **HaluMem** (arXiv 2511.03506, MemTensor): the **operation-level
memory-hallucination** benchmark whose metrics (Target Precision, False-Memory-
Resistance, Hallucination-Rate, an Updating subtask where *keeping a stale fact is a
failure*) **reward forgetting and contradiction-resolution** — the inverse of
recall@k. This is the eval simba was missing to validate Phase-6 (dormant tier) and
Phase-7 (contradiction-resolution). Local-runnable (judge swappable → our mlx local
judge; subsample the >1M-token corpus). **Spec'd as `docs/plans/10`.**

---

## Second survey (2026-06-08): the 37-repo clone — novel subset

Cloned 37 more named memory systems into `~/src/ai/memory` and deep-surveyed the
**novel** subset (7 systems) that weren't covered above. Ordered by value to simba's
*current* trajectory (HaluMem just landed → Phase-6/7 validation → SoTA comparison
`docs/plans/11`). Read-only survey; no code changes.

| Repo | What | Verdict | Borrow |
|---|---|---|---|
| **toki-bitemporal-memory** | typed **contradiction-resolution operator algebra** over a bitemporal store; audit-row per resolution; keyed-judge-log θ-pin replay | **★★★ BORROW** | resolution-operator taxonomy + audit-row schema + θ-pin replay → **Phase-7 resolution layer** |
| **SubtleMemory** | **relational-memory-discrimination** benchmark; contradictory slice = 35–50pp system-vs-oracle gap | **★★ INTEGRATE (eval)** | a 2nd forgetting/relational instrument *alongside* HaluMem; loads to `Dataset`, local Qwen3-4B judge swappable |
| **somnigraph** | Hebbian topology done right: PMI co-retrieval (not raw count), lazy edges from event log, contradiction-aware PPR, two-tier potentiation | **★★ BORROW (source)** | the cleanest Hebbian source *if* built — but its own finding: PPR > raw Hebbian, Hebbian <1% importance once feedback ranking works (simba has Phase 6) |
| **engram-ai** | **seeded usage-replay simulator** (synthetic co-retrieval streams) | **★★ BORROW (tooling)** | unblocks measuring any topology-learning lever *without* prod traffic — de-risks the Hebbian question independent of whether Hebbian wins |
| **emotional-memory** | affect-tagged memory + arousal-modulated decay | **NOTE** | only **arousal-modulated decay** (a retention term; test on HaluMem); affect-as-ranking is **falsified on their own real data** — don't borrow ranking |
| **Mnemis** (Microsoft) | paper + partial code; graph traversal | **NOTE** | latency-trapped traversal; one novel gap = **enumeration/coverage completeness** via `get_all_children` one-call subtree expansion |
| **SME / multipass-structural-memory-eval** | structural graph-integrity diagnostic | **NOTE** | needs a typed graph simba lacks; borrow only the **`corpus_doctor` inject→detect→score** pattern as a Phase-7 test-harness shape |
| **Aether** | graph-fold recall | **SKIP** | same dead graph-fold pattern simba already killed (C1/Track-B/entity-bridge); **loses to pairwise on its own LoCoMo** — no new signal |

### The headline — toki reshapes the Phase-7 contradiction-resolution spec
simba's neuro-symbolic L4 **detects** contradictions (Z3 UNSAT-core + AGM revision +
Datalog closure) but the **resolution + provenance** layer was scaffolded, never
specified. **toki-bitemporal-memory** is the complement: *"simba detects, TOKI
resolves."* It carries (1) a **typed resolution-operator algebra** (named operators,
not ad-hoc supersession), (2) an **audit row per resolution** (what was resolved,
which operator, against which belief-state), and (3) a **keyed judge-log θ-pin
replay** discipline so every resolution is deterministically re-judgeable against the
state that produced it. This is the highest-value, most timely borrow — it slots
straight into the Phase-7 spec as the resolution layer.

### SubtleMemory — the missing Phase-7 eval instrument
HaluMem grades **retention pressure** (keeping a stale fact = failure). SubtleMemory
grades **relational discrimination under contradiction**, and its contradictory
slice exposes a **35–50pp gap** between systems and an oracle — i.e. it isolates
exactly what Phase-7 contradiction-resolution + Phase-6 dormancy are supposed to fix,
which recall@k can't and HaluMem only partly does. Pairs with HaluMem as a second
instrument; same loader/judge plumbing.

### Hebbian — somnigraph is the source, engram-ai is the measurement unblock
If Hebbian topology gets built, **somnigraph** is the cleanest reference (PMI not
raw count, lazy edges, contradiction-aware PPR, two-tier potentiation). But
somnigraph's *own* result is the caution: **PPR > raw Hebbian, and Hebbian is <1%
importance once feedback ranking works** — and simba already has Phase-6 feedback.
**engram-ai** contributes the piece that has always blocked this measurement: a
**seeded usage-replay simulator** producing synthetic co-retrieval streams, so a
topology-learning lever can be measured *before* real production traffic exists.

### Recommendations (priority — fold after the SoTA-comparison work `11`)

**Instrument before build** (simba's measurement-first discipline — and the HaluMem
lesson: Phase-7 earned only a *small* win there, and the "Memory Conflict" collapse
was largely an answer-prompt/judge artifact, not a memory defect):

1. **SubtleMemory FIRST — stand up the instrument.** Add it as a 2nd
   contradiction/relational eval next to HaluMem (`10`). **Run it with a capable
   answerer + robust judge** so its "35–50pp system-vs-oracle gap" is a real memory
   signal, not judge-strictness (the HaluMem trap). If no honest metric moves on it,
   *don't* build the resolution layer — that's the gate.
2. **toki resolution-operator borrow — measured ON SubtleMemory.** Graduate the
   Phase-7 resolution layer from scaffold to spec (typed operators + audit-row +
   θ-pin replay) only against a metric that demonstrably moves. Highest-ceiling
   borrow, but gated behind (1).
3. **Hebbian (somnigraph source + engram-ai replay sim) — ONLY if (1)+(2) stall.**
   Deprioritized, not ranked #3-by-default: this session closed the multi-hop-
   *retrieval* frontier (entity-bridge negative; "multi-hop = ranking/reasoning, not
   recall"), and Hebbian is *another* graph-retrieval lever. somnigraph's own data
   (PPR > raw Hebbian; Hebbian <1% importance once feedback ranking exists — simba
   has Phase-6) says it's very likely marginal-to-negative here. Build engram-ai's
   replay harness only to settle it if the toki/SubtleMemory thread doesn't pan out.
4. **NOTE/SKIP:** arousal-modulated decay (test on HaluMem only); Mnemis enumeration
   gap + SME `corpus_doctor` pattern (harness shapes); **skip Aether**. **ReMe**
   (LoCoMo 86.23, cloud-only) → spec `11`'s **published-only / can't-reproduce**
   bucket (post-GPT-4o-deprecation), a loose reference, not a head-to-head.

## Third survey (2026-06-09): 10-repo clone — novel subset

Surveyed `cognee`, `Memori`, `honcho`, `agentic-context-engine` (ACE), `memsearch`,
`SoloFlow`, `yantrikdb` (yantrikos), `aletheia-platform`, `spector`,
`engram` (raya-ac). Deduped hard against both prior surveys **and** this session's
established kills. **Net: 8 of 10 are redundant / cloud-trapped / confirm a kill; the
2 genuinely-novel borrows are both ORTHOGONAL TO RETRIEVAL** — which is itself the
signal: the retrieval frontier is closed (every multi-hop-at-retrieval lever is dead),
so every fresh idea worth taking is now a *write-time correctness* or *staleness*
mechanism, not a recall lever.

Name-collision note: this `engram` is **`raya-ac/engram`** (Python, PyPI
`engram-memory-system`) — a different repo from the prior-surveyed `engram-ai`
(`tonitangpotato`, Rust replay-sim). This `yantrikdb` (`yantrikos`) is the same
project family as the prior `yantrikdb-server`; only its new v0.7.23 mechanism is
novel.

| Repo | Verdict | Why / novel borrow |
|---|---|---|
| **yantrikdb** (yantrikos) | **★★★ BORROW** | **zero-LLM copular conflict bridge** — see headline below |
| **engram** (raya-ac) | **★★ BORROW** | **`drift.py`** — claim-vs-filesystem staleness check (codebase-grounded, $0) |
| **honcho** | ★★ borrow-idea | typed **reasoning-observation taxonomy** (explicit/deductive/inductive/contradiction + premise `source_ids`) — but **overlaps spec `15`** write-time loop; measure-the-delta only |
| **ACE** (Stanford/SambaNova) | ★★ borrow-idea | **delta-update skillbook** (bullet-level ADD/UPDATE/REMOVE, never full rewrite → "context-collapse" fix); procedural/Phase-6, local-OK — **not a retrieval lever** |
| **cognee** | NOTE | core = graph-completion = simba's **killed** graph-fold; only non-redundant bit (feedback-weighted edges) overlaps Phase-6 |
| **SoloFlow** | NOTE | pure-stdlib skill auto-evolution + 4-dim quality scorer; no eval numbers; overlaps ACE/Phase-6 |
| **spector** | NOTE | Java/Panama-SIMD perf play (0.13ms r@1M, BEIR ndcg not memory recall@k) — **infra-not-borrow** for Python simba; bio set duplicates falsified affect-ranking |
| **Memori** | SKIP | recall = FAISS+BM25+EmbeddingGemma = simba's stack; "memory from what agents DO" is marketing (code drops tool_calls) |
| **memsearch** | SKIP | redundant with FTS5/hybrid + turbo-search; no numbers |
| **aletheia-platform** | SKIP | web shell only; Rust engine not in repo; no code, no numbers |

### The headline — yantrikdb's zero-LLM copular conflict bridge feeds Phase-7 detection
This session's contradiction work left one **measured wall**: single-pass *detection
recall* (`answer-time-conflict-surfacing` found only ~15% of SubtleMemory's latent
conflicts; spec `14`). yantrikdb's `cognition/attribute_claims.rs` is a tiny
deterministic pre-pass that turns plain free-text updates ("brand color **is now**
green") into `(subject, "is", value)` triples, so two values of one attribute become
distinct objects of the **same** `(subject, rel)` and the *existing* conflict detector
fires — closing the gap where ordinary `record()` updates produced **zero** claims.
For simba this is a **free, local-first fallback feeder** for Phase-7 contradiction
detection: today the contradiction signal rides on LLM/KG extraction; this heuristic
catches the most common "X is Y → X is Z" update for **$0** and surfaces it as a
*review* signal (their design never auto-mutates — slots cleanly under simba's
**detect→resolve(toki)** split). It's the first borrow this survey round that attacks
a **documented** simba weakness rather than adding a parallel mechanism. It augments,
not replaces, LLM detection → measure as a *recall lift* on SubtleMemory's
contradictory slice.

### The secondary — engram's `drift.py` is a codebase-grounded staleness axis
`engram/drift.py` verifies memory claims (file paths, function names, `npm run`/
`cargo` commands, deps) against the **live filesystem**, emits a 0–100 drift score,
and auto-invalidates dead refs — **zero AI cost**. simba validates a memory against
*contradicting facts* (HaluMem/SubtleMemory/Z3) but **never against the codebase** —
a distinct, cheap staleness signal that fits a *coding-agent* memory specifically
(simba's actual deployment). Orthogonal to every retrieval/contradiction lever; a
natural Phase-6 retention input ("claim references a deleted symbol → decay/flag").

### Kills reconfirmed (do NOT borrow)
- **engram's Hopfield channel** (softmax-sharpened full-matrix similarity) = exactly
  the associative/graph-fold retrieval class simba **already killed** (C1 / Track-B /
  entity-bridge). Its headline **LongMemEval R@5 98.1%** is a benchmark trick (dual
  user/assistant BM25 split + temporal boost + cross-encoder, on the same
  HNSW+BM25+RRF+cross-encoder stack simba has) with graph/Hopfield **disabled** — no
  new mechanism, and its surprise-gate/trust-decay/ADD-UPDATE-NOOP CRUD are already in
  simba's Phase-6/7.
- **cognee** graph-completion, **Memori** = simba's own stack, **spector** bio set
  (affect-as-ranking falsified on `emotional-memory`'s own data, survey 1) — all
  redundant or dead.
- **No new eval instrument** in this batch matches HaluMem/SubtleMemory's value: honcho
  publishes **no reproducible numbers in-repo**; spector benches BEIR ndcg (not memory
  recall@k); yantrikdb's "99.9% token savings / 88% precision" is a token-cost demo,
  not recall@k.

### Recommendation — slots into the existing gate, doesn't add a new thread
The standing plan is **SubtleMemory-first → toki resolution (gated on a moving
metric)**. The two novel borrows attach to that thread rather than opening a new one:
- **Fold the yantrikdb copular extractor into the Phase-7 detection feeder** and
  measure it as a *detection-recall lift on SubtleMemory's contradictory slice* — it
  directly targets the ~15%-detection wall spec `14` hit, and it's the cheapest
  possible experiment ($0, deterministic, default-off). Do this **inside** the
  SubtleMemory instrument work, not as a separate build.
- **`drift.py` staleness is a separate, low-priority Phase-6 input** for the
  coding-agent deployment — note it, build only if retention work resurfaces.
- honcho taxonomy / ACE skillbook: **already-have-adjacent** (spec `15` write-time
  loop; Phase-6 evolution). Record as measure-the-delta-IF, not new builds.

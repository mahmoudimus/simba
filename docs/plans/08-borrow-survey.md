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

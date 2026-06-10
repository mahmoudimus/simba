# Implementation plans

> **Continuing this work?** Start with [`HANDOFF.md`](HANDOFF.md) — full session
> handoff (state, decisions, what's left, where to pick up) + a paste-ready prompt
> to boot a fresh context.

Detailed, implementer-ready specs for the remaining simba roadmap. Each spec is
written to be executed top-to-bottom by an implementer (human or a smaller model)
without further design work: it names exact files, signatures, config fields,
TDD test cases (RED first), acceptance criteria, and verification commands.

These trace back to the eval-program plan and the
[`roadmap.md`](../../roadmap.md) gap analysis. They are independent — pick any
one and implement it on its own branch off `main`.

## Discipline (applies to every spec)

- **TDD**: every spec lists its tests RED-first. Watch them fail, then implement.
- **All config via `@configurable`**: no hidden constants; every tunable is a
  field on a section dataclass, gettable/settable through `simba config get/set`.
- **Pure Python under `src/simba/`**, ruff-clean (88 cols, `pathlib` not
  `os.path`, `TYPE_CHECKING` for annotation-only imports).
- **Append-only storage**: mutable ranking/usage state lives in SQLite, never in
  LanceDB columns (which are write-once vectors).
- **Levers plug into the shared path** (`plan_recall` / `hybrid_search`) so the
  benchmark measures exactly what ships.
- **Never tune to saturate the benchmark** — report deltas on the held-out test
  split with an ablation table + latency p50/p95.

## The specs

| # | Spec | Scope | Roadmap |
|---|------|-------|---------|
| [01](01-eval-bench-infra.md) | Eval-program infrastructure | `simba eval bench` CLI, results store, `BENCHMARKS.md` leaderboard, CI smoke fixture | Workstream A (A4·A5·A6) |
| [02](02-judge-baselines.md) | Local judge + honest baselines | separate `judge` config section (different local model than the answerer), full LoCoMo / `longmemeval_s` baselines, abstention scoring, per-query latency | Workstream B (B1–B4) |
| [03](03-hyde-ircot.md) | HyDE + IRCoT | true LLM HyDE 2nd vector arm (cached, fail-open) and answer-time IRCoT for multi-hop QA | Lever C3 + answer-time multi-hop |
| [04](04-decay-forgetting.md) | Decay / forgetting + feedback-aware ranking | usage store, strength model, recall-time reinforcement, scheduler decay pass, dormant tier, outcome feedback (`simba memory feedback`) | Phase 6 |
| [05](05-reflection-neurosymbolic-ops.md) | Reflection + neuro-symbolic + ops | `REFLECTION` memory type + reflect pass (Phase 5); derive→verify→revise→distill→induce loop over the KG with Z3/Datalog (Phase 7); latency metrics, TOOL_RULE TTL, lighter install extras, release glob fix | Phases 5 & 7 + ops |
| [06](06-multihop.md) | Multi-hop: close the weak axis | evidence-gated; Track A productionize reasoning-time IRCoT, Track B retrieval-time GraphRAG (PPR + community, *not* C1's co-occurrence), Track C Phase 7 deductive closure | the multi-hop frontier |
| [07](07-recall-excellence.md) | Recall excellence program | 5 pillars — fix the eval instrument, attack weak axes, embedder/extraction foundation, feedback flywheel (Phase 6), fusion tuning; + proof-carrying recall as the moat | make recall exceptional |
| [08](08-borrow-survey.md) | Borrow survey (8 memory systems) | digest of forgetful/auto-memory/shodh/YourMemory/dna/memory-palace/animaworks/yantrikdb — entity-bridge multi-hop (positive!), Hebbian graph-topology learning, procedural memory, cheap levers | mine competitors for ideas |
| [09](09-entity-bridge-multihop.md) | Entity-bridge multi-hop experiment | the one *positive* multi-hop lever (shared-named-entity edges, YourMemory +12pp HotpotQA); distinct from C1/Track-B; measured on HotpotQA/LME-multi-session, not LoCoMo | reopen multi-hop — win or kill |
| [10](10-halumem-forgetting-eval.md) | HaluMem forgetting/hallucination eval | the inverse-pressure benchmark (Target Precision / FMR / Hallucination-Rate / Updating) where forgetting + contradiction-resolution PAY OFF — validates Phase-6 dormant tier + Phase-7; local judge, subsampled | measure what recall@k can't |
| [17](17-episodic-aggregation-view.md) | Episodic memory as a materialized aggregate view | answer-time aggregation (counting/temporal/ordering) as a lossless-pointered materialized view over the raw-turn log; CQRS/event-sourcing framing, embedded SQLite (no Postgres), deterministic Python aggregation; **gated on a two-arm ceiling probe** | fix what store-raw recall alone can't aggregate |

## Suggested order

The eval program (01 → 02) comes first: it makes iteration cheap and locks in
honest baselines, so every later lever is a *measured delta* rather than a guess.
03 (HyDE/IRCoT) and 04 (decay) are independent levers measurable against those
baselines. 05 is the largest and most exploratory (Phase 7 neuro-symbolic) —
land it once the measurement program exists to keep it honest. 06 (multi-hop) is
**evidence-gated**: its lead track is chosen from the lever-ablation result, and
it slots after 0.4.0 so it ships against a stable baseline.

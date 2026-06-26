# Changelog

## [Unreleased]

### Added

- **Raw session indexing and active task continuity.** Added a rebuildable
  SQLite/FTS session-message sidecar (`simba sessions index/search`) for exact
  transcript recovery without hook-context bloat, plus append-only active task
  snapshots (`simba task snapshot save|show|clear`) that UserPromptSubmit can
  inject as one compact lane.

- **Append-only memory audit sidecars.** General memories now record
  provenance/trust metadata, trust-gated supersession lineage, replayable
  write-time judge decisions, persisted conflicts, and explicit quality counters
  (`match`, `inject`, `use`, `noise`, `save`) without mutating historical rows.

- **SubtleMemory readback and driver loop.** `simba eval bench subtlememory`
  can compare normal recall against exact-session readback, write a per-case
  failure ledger, and run a default-off same-session expansion sweep with a
  promotion gate.

- **Codex extraction analysis traces.** `simba codex-extract --run --trace`
  writes JSONL analysis-run artifacts with candidate evidence/source spans,
  keep decisions, store outcomes, and negative lessons for failed candidates.

- **Codex trace curator reports.** `simba codex-curate` turns extraction trace
  JSONL into append-only markdown or JSON review reports under
  `.simba/curator_runs`, surfacing candidate memories, evidence, store
  outcomes, negative lessons, and playbook candidates. `simba codex-curate
  review <report>` appends reviewer labels to `.review.jsonl` and prints exact
  `simba memory store ...` commands for accepted candidates without executing
  them.

- **Default-off retrieval triage.** `hooks.recall_triage_enabled` adds a
  conservative no-LLM UserPromptSubmit classifier that skips recall/RAG only for
  narrow self-contained prompts; `simba eval triage` provides a re-runnable
  zero-false-negative gate.

- **Store-time anticipated queries.** Memories can carry append-only anticipated
  future query phrasings via `/store` `anticipatedQueries` or
  `simba memory store --anticipated-query`, capped by
  `memory.anticipated_query_max_per_memory`. A default-off
  `memory.anticipated_query_recall_enabled` lane can now fold matching
  anticipated-query FTS rows into hybrid recall for measured A/B runs.

- **Outcome-quality decay lever.** `memory.outcome_quality_weight` is a
  default-off strength/decay contribution from explicit `use_count` vs
  `noise_count` quality counters.

- **Executable ambiguity eval.** `simba eval ambiguity` runs small structured-data
  cases that preserve multiple interpretations as answer spaces instead of
  collapsing vague language to one count. The default Python oracle is fully
  offline; optional Souffle/Clingo backends and opt-in LLM codegen
  (`--generate python|souffle|clingo`) are off the daemon path and meant for
  research/eval only.

- **Ontology/lexicon eval scaffolding.** Added local lexicon bootstrap helpers
  for WordNet/FrameNet resources and a micro-schema ontology router that can
  cache/append ratification candidates for Fail18-style ambiguity probes.

- **Recursive fact formalizer boundary.** Candidate-unit formalizer payloads now
  ask providers for neutral recursive facts (`action`, `event`, `object_type`,
  `relation`, `value`, etc.) instead of answer-support labels or fail18-shaped
  predicates; legacy polarity outputs remain parseable for historical artifacts.
  A follow-up residual normalizer can now run over `other` facts only, converting
  them to generic predicates when possible while keeping answer labels and
  candidate-unit status out of the provider payload. A deterministic recursive
  fact compiler now consumes the normalized facts directly and emits
  candidate-unit rows for the five active fail18 candidate-unit cases, reaching
  5/5 recomputed gold matches on that slice without using the old provider
  candidate-unit outputs. The compiler now has adversarial/paraphrase tests for
  action-obligation synonyms, attended-event wording, and negative charity-sum
  controls. A full fail18 recursive-v2 formalizer payload artifact now covers all
  18 fail18 rows / 122 evidence sessions with gold, failure-mode, and raw
  session labels hidden. Candidate-unit labels now saturate incomplete
  relational noun phrases such as `new pair` only from explicit `sortal` /
  bridging facts, so answer units can display `new pair of boots` without
  treating the replacement pair as identical to the exchanged pair. The formalizer
  contract now keeps `coreference` for true identity and represents contrastive
  replacement mentions with `distinct` plus ordinary `relation` facts. Recursive
  fact compilation now hard-fails evidence-local fact sets that overload the
  same bare symbol as both entity and type, while leaving future ontology/type
  lattice namespaces separate. The provider contract also treats unknown optional
  argument values as omitted/null rather than concrete empty strings. The
  compiler now uses a local WordNet JSONL type-subsumption ratifier for target
  matching, so evidence-local facts such as `sortal(new_pair_1, boots)` can
  satisfy a `clothing` query without emitting ad hoc ontology edges; the ratifier
  is noun-only and definition-gated to avoid homonym shortcuts such as
  `book -> clothing`. Formalizer outputs may now use `facts: []` for irrelevant
  evidence sessions, and the recursive compiler quarantines evidence sessions
  with sorted-symbol or consistency violations instead of invalidating the whole
  case. The compiler reports inclusion-mutation sensitivity so aggregate answer
  scores expose balanced include/exclude compensation gaps. Fail18 numeric gold
  parsing now preserves decimal human answers such as `0.5 hours` instead of
  truncating them through integer `gold_count` fields. The typed-fact envelope
  intent planner now uses a closed operation set for `sum_value`, `sum_duration`,
  `lookup_scalar`, `date_answer`, and `stated_total` in addition to money, day,
  instance-count, and entity-selection shapes, so obvious fail18 shape errors
  are routed before membership aggregation rather than handled as free-form
  arithmetic. The envelope now classifies scalar value facts into answer,
  threshold, balance, subtotal, historical, and distractor roles so `sum_value`
  ignores unrelated prices/clicks/rates and `lookup_scalar` prefers requirement
  thresholds over current balances. On fail18, the value-role rerun improved
  gold-in-envelope coverage from 9/18 to 11/18; `60036106` now collapses exactly
  to `[12000,12000]`, while `9ee3ecd6` moves from the wrong 300-point scalar to
  a still-wide `[50,200]` threshold envelope. Lookup-scalar aggregation now also
  applies a strict threshold-consensus selector: if one threshold value has
  independent support from multiple resolved roots and every alternative has
  weaker support, the lookup collapses to that value; otherwise it remains an
  interval. This narrows `9ee3ecd6` to `[100,100]` and raises fail18 endpoint
  exactness from 7/18 to 8/18 without changing 11/18 gold-in-envelope coverage.
  Pickup/return clothing questions now count actionable obligations rather than
  only canonical wardrobe identities, so an exchanged pair of boots can
  contribute separate return/pickup obligations while ordinary instance-count
  questions still count one resolved entity once. This moves `0a995998` from
  `[1,2]` to `[2,3]`, raising fail18 gold-in-envelope coverage to 12/18 and
  endpoint exactness to 9/18.

- **LanceDB storage compaction CLI.** `simba memory compact` now reports live
  LanceDB bytes versus recursive on-disk bytes, retained versions, and fragments
  without mutating by default. `simba memory compact --run --older-than 24h`
  calls LanceDB optimize to compact fragments and prune old derived-store
  versions while preserving memory rows.

## [0.11.0] — 2026-06-17

Every new lever in this release defaults **OFF** (backward-compatible): with the
defaults, hook output stays byte-identical to 0.10.0.

### Added

- **`memory.max_content_length` is now the single source of truth for the memory
  content cap** (#79). The configured cap (default 200) drives both enforcement
  *and* the "keep content under N chars" guidance in every extraction / digest /
  episode / reflection prompt, plus the content-producing truncations (`simba rule
  add`, auto-learned `TOOL_RULE`, `SYSTEM` sync rows) — previously these hardcoded
  `200` in ~13 places, so raising the cap silently had no effect. Resolved via
  `simba.memory.config.resolve_max_content_length()`. Set it once
  (`simba config set memory.max_content_length <N>`, or per-project) and all
  guidance + truncation follow. `content` is what recall surfaces (not the
  unbounded `context`), so a larger cap means more tokens per memory and, under
  the recall token budget, can surface fewer memories per turn.

- **Conditional guardian re-injection** (spec 25; `hooks.guardian_signal_gated`,
  default off). The CLAUDE.md `SIMBA:core` block is re-injected every prompt
  (~2k tokens) regardless of need. When enabled, the block is re-injected only on
  the first prompt of a session, after compaction, or when the prior response
  dropped the `[✓ rules]` signal — reclaiming the per-turn token tax. Fail-open
  (inject when uncertain). A per-session flag written by `Stop`, read by
  `UserPromptSubmit`, reset by `PreCompact`/`SessionStart`. Plus documented
  gate-graduation discipline (a rule that becomes a deterministic gate leaves the
  CORE block).

- **Hierarchical (ancestor-prefix) project recall** (spec 26;
  `memory.hierarchical_recall` + `hierarchical_recall_include_global`, default
  off). A child cwd (`/repo/api`) inherits memories scoped to its ancestors
  (`/repo`) + global, bounded at the git root — the client computes the scope
  chain, the daemon does string-membership across both the vector and BM25 arms;
  `projectPath` is normalized to a resolved absolute path on store. Dissolves the
  dedup dual-home problem (place a fact once at the right level). **Measured**
  (A/B on the production stack): a *situational* lever — near-free at low
  ancestor:child noise ratios, a real recall@k cost at high ratios with sparse
  gold (LoCoMo 9:1 ≈ −0.7pp, longmemeval_s 14.8:1 −4 to −9pp) — so it stays
  default-off and is enabled per-project where the ancestor scope is small.

- **Intent-primed doctrine + mandated preflight** (spec 28;
  `hooks.intent_priming_enabled` / `preflight_mandate_enabled` /
  `preflight_mandate_risk_only`, default off). `UserPromptSubmit` classifies the
  prompt against project doctrine *triggers* (cosine over precomputed embeddings,
  no LLM) and injects the matched doctrine + applicable gates; on a risk-tier
  match it arms a mandate, and `PreToolUse` then blocks a mutating tool that runs
  without a `simba preflight <task>` this turn. New `simba preflight` CLI +
  `/preflight` endpoint.

- **Reasoning-layer verification** (spec 27; `hooks.engagement_marker_enabled` /
  `reasoning_verify_enabled`, default off). A *simba-emitted* `🦁☑` ledger of what
  simba surfaced this turn is injected at `UserPromptSubmit` (every turn, tools or
  not), the agent echoes it, and `Stop` verifies the echo — observability, not
  self-attestation. `reasoning_verify` lets `Stop`/`SubagentStop` (the latter newly
  wired) doctrine-check the finalized response and **block-to-reconsider** on a
  violation — the catch for tools-free / mid-reasoning output. On **pi** only, the
  `context` event re-injects doctrine before every LLM call and `message_end`
  annotates a doctrine-violating finalized message (Tier 2; pi `context` injection
  pending runtime verification).

### Changed

- **CI hardening.** A per-test `pytest-timeout` (`--timeout=120 --timeout-method=
  thread`) plus a job-level `timeout-minutes: 20` turn a hung test into a fast
  fail instead of running to the 6-hour Actions ceiling. A global autouse fixture
  forbids real GGUF model loads in the unit suite (reranker fail-opens;
  `gguf`-marked tests exempt), so CI never reaches Hugging Face.

## [0.10.0] — 2026-06-16

### Added

- **pi pitfall/doctrine tool gate** (spec 24 v2.1). pi's tool gate can now block
  on the pitfall/doctrine gate — "you're about to take the workaround you told me
  not to" — not just on redirects and `TOOL_RULE` matches. The pitfall gate keys
  on the agent's *reasoning*, which pi's `tool_call` event doesn't carry, so the
  bridge pulls the last assistant message from the live session
  (`ctx.sessionManager.getBranch()`) and passes it to the daemon as `thinking`.
  When a pending mutating tool (`hooks.pitfall_gate_tools`, default Edit/Write/Bash)
  violates a stored doctrine, the daemon escalates the directive to
  `escalated_block` so pi enforces it as a hard block; Claude/Codex still receive
  it as `additionalContext` and stay byte-identical (their payloads carry no
  `thinking` field, so `pre_tool` reads the transcript exactly as before). Gated
  by the existing `hooks.pitfall_gate_enabled` (default off) — no new config.

## [0.9.0] — 2026-06-16

### Added

- **pi tool-gating** (#77). The pi bridge now wires pi's `tool_call` event to
  simba's canonical `pre_tool` path, giving pi the same command-level
  enforcement Claude Code and Codex get from `PreToolUse` — reusing the same
  redirect and `TOOL_RULE` rules, with no logic duplicated in TypeScript. Before
  a tool runs, simba can: **block** the call on a redirect *deny* or a strong
  `TOOL_RULE` match (escalated to a hard block via `escalated_block`, since pi's
  `tool_call` has no context-injection channel); **silently rewrite** it on a
  redirect *rewrite* (e.g. the universal `rg -rn` → `rg -n` fix) by mutating
  `event.input` in place — no model round-trip; or **allow** it (context-only
  injections like recall hints and weak `TOOL_RULE` warnings are dropped, having
  nowhere to render). The `PreToolUse` path was canonicalized first and remains
  byte-identical for Claude/Codex (characterization-tested). The daemon stays
  authoritative — gating is governed by the existing `hooks.redirect_enabled` /
  `hooks.rule_check_enabled` / `permission_deny_similarity` config, so the
  extension always wires the gate and Python decides. The pitfall/doctrine gate
  (which keys on the agent's reasoning) is deferred.

## [0.8.2] — 2026-06-16

### Added

- **Persistent query-embedding cache for the daemon** (#76). After the 0.8.1
  concurrency fix serialized all llama.cpp calls, a *repeated* recall query
  re-embedded from scratch each time and the embeds queued behind the lock —
  tail latency climbed to ~8s under a burst (e.g. the conflict detector firing
  the same pairwise check N times). The existing content-hash `embedding_cache`
  is now wired into the daemon's `embed_query`: identical queries hit a ~ms
  sqlite lookup instead of a fresh GGUF embed, and the cache is **persistent**
  so it survives daemon restarts. `memory.embed_cache_enabled` (default on),
  `memory.embed_cache_path` (empty → `<db dir>/embed_cache.db`). Correctness-
  neutral — the key includes `model_id`, so a model swap can't serve a stale
  vector.

## [0.8.1] — 2026-06-16

### Fixed

- **Daemon crash under concurrent recalls** (#75). The embedder and the
  bge-reranker cross-encoder are two `llama_cpp.Llama` instances sharing the
  process-global ggml backend, which is not safe under concurrent context use:
  a simultaneous embed and rerank score corrupted shared buffers
  (`GGML_ASSERT … "tensor write out of bounds"` → `SIGSEGV`). The embedder's
  asyncio queue and the reranker's lock each serialized only themselves, not
  one another. A new `simba.memory._llama.LLAMA_LOCK` guards **every** native
  llama.cpp call — embedder embed/load and reranker load/score — so no two
  threads enter llama.cpp at once (all call sites run in worker threads, so the
  lock never blocks the event loop). Most likely to bite when multiple agents
  share one daemon.

## [0.8.0] — 2026-06-15

### Added

- **pi coding-agent runtime support** (#74). Adds **pi**
  (`@earendil-works/pi-coding-agent`) as a third runtime alongside Claude Code and
  Codex. A harness-agnostic hook core (`simba.harness.dispatch` → `CanonicalResult`)
  is exposed two ways — inline via the CLI (`simba hook` / new `simba hook-canonical`)
  and over HTTP via a new daemon `POST /hook/{event}` endpoint — and a thin bundled
  `simba.ts` extension bridges pi's lifecycle events (`session_start`,
  `before_agent_start`, `agent_end`, `session_before_compact`) to it. Install with
  `simba pi-install`; path driven by `pi.agent_home`. Claude/Codex hook output stays
  byte-identical (characterization-pinned); `hooks.dispatch_via_daemon` (default on)
  routes hooks through the warm daemon with inline fallback. MVP = the memory loop
  (recall-on-prompt, capture-on-stop, daemon health, transcript export); tool gating
  is deferred to a later release.
- **No-magic recall visibility**: the pi bridge prints `[simba: N memories injected]`
  (plus session-ready / captured / exported notes) to stderr — visible even in
  `pi -p`. `CanonicalResult.memory_count` carries the recalled count through the
  `/hook` endpoint and `simba hook-canonical`.

## [0.7.1] — 2026-06-13

### Fixed

- **Conflict surfacing no longer regresses knowledge-update questions** (#68). v0.7.0
  shipped answer-time conflict surfacing default-on; a "what is my current X?" query
  retrieves both the old and new value, which the pairwise detector flagged as a
  conflict and told the answerer not to pick a side — wrong, since recency resolves it.
  New `intent.is_knowledge_update` + `memory.conflict_skip_on_current_value` (default
  True) makes conflict_note skip its directive on current-value queries; genuine
  contradictions keep the strict surfacing path. Measured on LongMemEval-S
  knowledge-update: directive 0.25 -> 0.958.

### Added

- **Intent-gated retrieval breadth for multi-session/aggregation queries** (#69).
  Generalizes the count-depth lever: `intent.is_aggregation` + `memory.
  aggregation_depth_enabled` (default off), `aggregation_candidate_pool_n=80`,
  `aggregation_context_k=80`. Measured: multi-session evidence sets reach complete@80
  0.90 vs complete@20 0.33; enabling lifts the multi-session QA category +0.13.
- **Official LongMemEval per-type judge** (#67): `bench.judge_style` (default
  "official") routes grading by question type to the verbatim anscheck templates.
- **question_date threaded into the bench reader** (#66): the official "Current Date"
  anchor; +0.111 overall on the LME-S sweep, temporal-reasoning 0.417 -> 0.833.

## [0.7.0] — 2026-06-12

### Changed

- **Answer-time conflict surfacing is now default-ON with the pairwise detector**
  (`memory.conflict_surfacing_enabled=true`, `memory.conflict_detect_strategy=
  "pairwise"`). When a recall's memories contain a real contradiction for the
  query, the injected context gains a directive that NAMES the conflict and
  instructs the consumer to surface it instead of silently picking a side.
  Measured (SubtleMemory, paired A/B over identical contexts): contradictory
  accuracy 0.083 → 0.528 overall, **0.111 → 0.944** when both sides are
  retrieved, zero cases broken; harm check on non-contradiction slices is
  net-positive (fires 0.20, 1/8 fired cases harmed). Disable with
  `simba config set memory.conflict_surfacing_enabled false`. Spec:
  docs/plans/14.

### Added

- **`memory.conflict_detect_parallel` (default 8):** pairwise checks run in
  bounded waves, so a recall with k memories costs ~ceil(pairs/8) LLM latencies
  instead of one per pair. Deterministic: the lowest-index flagged pair wins,
  identical results to sequential.

## [0.6.1] — 2026-06-08

### Added

- **Built-in redirect rule for the grep→rg `-r` trap** — every project now gets it
  with no config. In ripgrep `-r` is `--replace` (it eats the next token), so a
  grep-style bundle like `rg -rn` / `-rl` / `-rln` / `-nr` silently became
  `--replace` and corrupted output. One general pattern (a single-dash clump of
  common grep flags `{r,l,n,i,c,w,o,h,v}` containing an `r`) drops the `r`
  (`rg -rn` → `rg -n`) and auto-rewrites — a real `--replace` (`rg -r new`,
  `rg -rnew old`, `rg --replace`) is left untouched. Disable via
  `hooks.redirect_enabled=false`.
- **Per-rule redirect `mode`** (`RedirectRule.mode`, "deny"|"rewrite"; empty ⇒
  global) so a safe, unambiguous fix can auto-apply even when the project default
  is `deny` (the built-in rg rule uses it).

### Fixed

- **Eval faithfulness:** `build_answer_prompt` now mirrors `format_memories` (date
  labels + newest flag, no recency-resolution instruction). An A/B (deepseek)
  showed the instruction is a no-op for a capable consumer — the date+newest
  annotation the daemon already ships is the lever — so the benchmark stops
  handing the answerer a hint the product never injects.

## [0.6.0] — 2026-06-08

### Added

- **Engine-agnostic persistent LLM server.** The local OpenAI-compatible server
  launcher is no longer MLX-only. New providers/presets join `mlx-server`:
  - `llama-server` — llama.cpp's `llama-server` (cross-platform, CUDA), with the
    same one-command auto-spawn convenience `mlx-server` had on Apple Silicon.
  - `openai-http` — talk to any running OpenAI `/v1` server (Ollama / llama.cpp /
    vLLM, local or on a remote GPU box); never auto-spawns — you run it.
  - `llm.serve_cmd` / `judge.serve_cmd` — a launch-command template
    (`{model}/{host}/{port}`) to drive any other server (e.g. vLLM); empty uses
    the provider's preset.
  These share one transport (`_complete_http`) and load the model once, so the
  reranker and LLM-judged eval stop paying the `llama-cli`/`mlx-lm` per-call reload.
  `ensure_for_config` auto-spawns only for **local** endpoints; a remote base_url
  is check-only (run the server on that host). See `docs/eval-remote-gpu.md`.
- **HaluMem forgetting eval** (`simba eval halumem`) — operation-level
  memory-hallucination benchmark (correct / hallucination / omission + boundary
  abstention). See `docs/plans/10`.
- **Recency-annotated answer context** in the eval — `build_answer_prompt` mirrors
  what the daemon injects (`format_memories`): date-labels each memory and flags
  the most recent. Closes a large temporal-accuracy gap the eval was hiding.
- **External recall@k benchmark harness** — `simba eval bench locomo|longmemeval|
  hotpotqa [--qa]` over real datasets, `simba eval leaderboard` renders a committed
  `BENCHMARKS.md`, and every run is appended to `.simba/eval/results.jsonl` with a
  config snapshot. Includes a HotpotQA pooled (fullwiki) loader.
- **Multi-hop retrieval instruments — both default-OFF, measured.** Two ways to fold
  a third graph arm into recall before composite-rescore + reranker, gated and no-op
  unless wired:
  - **Entity-bridge** (`memory.entity_bridge_enabled`, spec 09) — fold memories that
    share a *named entity* with the top seeds. The one multi-hop mechanism with a
    positive external signal (YourMemory +12pp HotpotQA); ships off pending a proven
    in-repo delta.
  - **Track B retrieval-time GraphRAG** (`memory.kg_ppr_enabled`, spec 06) — fold
    PPR-ranked KG neighbors seeded by the query's entities. A **measured negative**
    (marginal/regressive on LoCoMo/LME multi-session), kept as a default-off
    instrument with the `kg_ceiling` / `kg_corpus` apparatus that proves it.
- **CI ↔ local parity** — `scripts/checks.sh {lint|test|all}` is the single source of
  truth for ruff + pytest, called by **both** `.github/workflows/ci.yml` and the new
  `.githooks` (`pre-commit` = lint, `pre-push` = full). Enable: `git config
  core.hooksPath .githooks`.

### Changed

- **BREAKING: `simba/llm/mlx_server.py` → `simba/llm/local_server.py`** and its API
  is now engine-agnostic (`ensure_server(serve_cmd, *, base_url=...)`,
  `build_serve_cmd(template, model, host, port)`, `SERVE_PRESETS`). No back-compat
  shim. Update any direct imports.
- Result snapshots now record the answerer + judge model, so QA numbers are
  attributable. The leaderboard renders a HaluMem block.

### Fixed

- `LlmClient.available()` now rejects unknown providers (e.g. the vision runtime
  `mlx-vlm`) instead of silently returning `""` and skipping work.

## [0.5.1] — 2026-06-07

### Fixed

- **Embedding-dim migration guard now actually fires on the daemon.** After the
  0.5.0 bge-large default (1024-d), querying an un-migrated 768-d store raised a
  raw LanceDB `RuntimeError` and silently degraded recall to keyword-only
  (`-> N memories, top: 0.00`) instead of the promised actionable message. The
  guard read `table.schema` synchronously, but the daemon's `AsyncTable.schema`
  is a coroutine — so the stored dim was never determined and the check no-opped.
  It now awaits the async schema (sync tables still work), so a dim mismatch
  surfaces a clear "run `simba memory reembed`" error. Fix for an existing store:
  `simba memory reembed`.

### Changed

- **PreToolUse skips the tool-rule recall for projects with no learned rules.**
  The rule pre-check embedded every Bash command / file path and ran a
  `TOOL_RULE` vector search before each tool call — a guaranteed miss (and log
  noise) for the common case of a project with zero rules. It now consults a
  TTL-cached per-project `TOOL_RULE` count (`hooks.rule_count_ttl`, default 300s;
  0 disables) and skips the embed+recall when the count is zero. Fail-open: if the
  count can't be determined, the check still runs. `GET /list` gained a
  `projectPath` filter to source the count.

## [0.5.0] — 2026-06-07

### Changed — BREAKING

- **Default embedder is now `bge-large-en-v1.5` (1024-d), was nomic-embed-text
  (768-d).** A bake-off on a discriminating eval showed a clear, cross-dataset
  recall win — LoCoMo r@5 0.595→0.614, LongMemEval r@5 0.780→0.814, lifting both
  weak axes (multi-hop, open-domain) with no single-hop regression
  (docs/plans/07-recall-excellence.md). The vector dimension changes, so an
  **existing store must be migrated**: run `simba memory reembed`. Recall now
  guards against the mismatch — a 768-d store queried with the 1024-d model logs a
  clear "run `simba memory reembed`" error instead of silently returning nothing.
  The model is larger (~340 MB vs 81 MB) and slower to embed on store; pin the old
  embedder via `simba config set memory.embedding_model/...` if needed.

## [0.4.0] — 2026-06-06

A large feature release: the eval program is now a disciplined, measured pipeline,
and Phases 4–7 of the roadmap land behind config flags (default-off, fail-open).

### Added

- **Eval program (`simba eval bench` / `simba eval leaderboard`)**: config-driven
  benchmark CLI over LoCoMo / LongMemEval (recall@k + optional LLM-judge QA), an
  append-only results store (`.simba/eval/results.jsonl` with git SHA + config
  snapshot per run), a committed `BENCHMARKS.md` leaderboard with a methodology
  caveats footer, and a CI smoke fixture so the harness can't rot (#41, #46).
- **Configurable local judge + honest baselines**: a separate `judge` config
  section so the answerer never grades itself (B1), abstention scoring for
  LongMemEval `_abs` questions, and per-query latency p50/p95 in every report
  (#43).
- **True LLM HyDE** (`memory.hyde_mode = "llm"`): a hypothetical-answer second
  vector arm, cached + fail-open; and **answer-time IRCoT** for multi-hop QA in
  the eval harness (`eval.ircot_enabled`) (#44).
- **Decay / forgetting + feedback-aware ranking (Phase 6)**: a SQLite usage store,
  a deterministic strength model (decay × reinforcement × feedback), recall-time
  reinforcement, a scheduler decay pass, a reversible dormant tier, and
  `simba memory feedback <id> good|bad` (#42).
- **Reflection (Phase 5)**: a `REFLECTION` memory type + a scheduler reflect pass
  that synthesises cross-session insights (#45).
- **Neuro-symbolic deductive distillation (Phase 7)**: a derive→verify→revise→
  distill→induce loop over `kg_edges` (Datalog/Souffle closure, Z3 UNSAT-core
  contradiction detection, AGM-style revision, proof-carrying derived edges),
  scheduler-wired, gated + fail-open (#45).
- **First committed benchmark baseline**: LoCoMo recall@5 0.573 / QA acc 0.427;
  LongMemEval oracle recall@5 0.780 (see `BENCHMARKS.md`; numbers are
  DeepSeek-judged, oracle = upper bound) (#46).
- **`docs/plans/`**: implementer-ready specs for the roadmap, including the
  evidence-gated multi-hop plan (`06-multihop.md`, lead = retrieval-time GraphRAG)
  (#40, #47).

### Changed

- **Ops hardening**: latency p50/p95 in `DiagnosticsTracker` + a `/metrics`
  endpoint; a `TOOL_RULE` TTL hygiene pass; lighter install extras
  (`embed`/`full` optional-dependency split, lazy imports); the release workflow
  globs `*.whl`/`*.tar.gz` explicitly (#45).
- **`simba eval bench` threads an LLM client + the `eval` config** through recall
  and QA so the reranker / LLM-HyDE / IRCoT levers can be measured through the CLI
  (#47).
- **Recall ranking: `memory.rrf_k` 60 → 20.** A fusion sweep showed sharper RRF is
  a measured win on LoCoMo recall@k (r@5 0.573→0.595, both weak axes up) and
  neutral on LongMemEval; widening candidate pools regressed, so recall is
  ranking-limited, not pool-limited. First result of the recall-excellence program
  (`docs/plans/07-recall-excellence.md`) (#49).

## [0.3.0] — 2026-06-06

### Fixed

- **SessionStart cross-project extraction**: the `<learning-extraction-required>`
  reminder read the global `~/.claude/transcripts/latest.json`, so a session in
  one project could be told to extract *another* project's transcript — filing
  that project's learnings under the wrong `project_path`. Now resolved
  per-project via `simba.transcripts.find_pending(cwd)` (project + pending-status
  scoped); no reminder when nothing is pending for the current project. Completes
  the cross-project-attribution fix begun for `/memories-learn` in 0.2.0.

## [0.2.1] — 2026-06-06

### Fixed

- **Blank PyPI project page**: `pyproject.toml` was missing `[project].readme`, so
  no long-description metadata was uploaded. Declared `readme = "README.md"` — the
  README now renders on PyPI.

## [0.2.0] — 2026-06-05

### Added

- **Hybrid recall (L3)**: RRF fusion of the LanceDB vector arm + a SQLite FTS5
  bm25 keyword arm, with an intent-aware similarity floor, broad-query widening,
  and a multi-arm HyDE expansion arm.
- **Temporal knowledge graph (L4)**: bitemporal `kg_edges` (belief-time
  `valid_from`/`valid_to` + event-time `occurred_at`), entity resolution
  (normalize + embedding-synonym merge), and multi-hop traversal
  (`kg_neighbors` / `kg_query(expand_hops)`).
- **LLM layer (`simba.llm`)**: CLI-backed client (`claude-cli`, `llm-cli` cloud;
  `llama-cli`, `mlx-lm` 100%-local), fail-open. Powers the **LLM reranker**
  (cross-encoder role over the candidate pool; non-blocking read-through cache on
  the daemon), **LLM fact extraction** as the primary KG feed
  (`sync.extract_strategy`), and composite recency+importance scoring.
- **Swappable embedder**: `embed_provider` (gguf | llm-cli) + configurable task
  prefixes + `simba memory reembed`.
- **Episodic consolidation** (L2): session-summary EPISODE memories.
- **Eval program**: in-process recall harness (recall@k / MRR / nDCG, dev/test
  split, real-corpus builder) **plus an external benchmark suite** —
  `simba.eval.benchmarks` over LoCoMo / LongMemEval (recall@k of labelled
  evidence + an LLM-judge QA layer), a persistent **embedding cache** + **judge
  cache** so reruns are cheap, and `scripts/fetch_benchmarks.sh`.
- **Tool-call redirect**: PreToolUse steering of bare commands to better tooling
  (deny + opt-in silent rewrite), shlex-tokenized; rules from
  `.simba/redirects.toml` + a project-scoped DB store. Supports **program rules**
  (swap a leading command) and **regex pattern rules** (flag-level fixes, e.g.
  `rg -rln` → `rg -l`).
- **`simba config` CLI**: TOML-backed configuration. `@configurable` dataclasses
  in a global registry; git-style scoping (`~/.config/simba/config.toml` vs
  `.simba/config.toml`); `list`/`get`/`set [--global]`/`reset`/`show`/`edit`.
- **`simba markers` CLI**: discover/audit/update `<!-- BEGIN SIMBA:name -->`
  markers across `.md` files.
- **`simba transcript` CLI**: project-scoped resolution of pending transcripts
  for learning extraction (`pending` / `mark-extracted`).

### Changed

- **Context-low warning** now measures transcript bytes **since the last
  compaction** (the live-context proxy) instead of cumulative file size, and is
  recalibrated for large (~1M-token) windows (`hooks.context_low_bytes` default
  8 MB, configurable). Fixes false alarms after compaction.
- **Skill packaging**: bundled skills relocated under the package
  (`src/simba/skills/`, repo-root `skills/` symlinked); the installer now
  **updates** changed skills (was create-only) and matches `SKILL.md`.
- **Orchestration extracted from neuron** into a `simba.orchestration` package;
  neuron keeps only the formal-verification tools (truth DB, Z3, Datalog).
- **PyPI distribution name** is `simba-ai` (`simba` was taken); the import
  package and the `simba` CLI are unchanged. `pip install simba-ai`.

### Fixed

- **`/memories-learn` cross-wiring**: it resolved transcripts via a single global
  `latest.json` symlink, so running it in one project extracted another project's
  transcript. Now resolved by the current project + pending status, and the
  transcript is marked `extracted` so it isn't re-extracted.
- Config-CLI section registry, redirect, and `simba`-shadow import bugs.

## [0.1.0] — 2025-02-06

### Added

- **Context-low early warning**: PreToolUse hook monitors transcript file size
  and injects a one-time `<context-low-warning>` when context approaches the
  auto-compact threshold (default 4 MB). Gives Claude time to summarize work
  state before compaction.

- **Sync pipeline**: New `simba sync` CLI with subcommands (`run`, `index`,
  `extract`, `status`, `schedule`) and `SyncScheduler` for periodic background
  sync. `POST /sync` endpoint on the daemon triggers one-off cycles.
  SessionStart hook fires a sync on every new session.

- **Diagnostics tracker**: Periodic reporting of endpoint hits, recall/store
  stats, and automatic LanceDB compaction after every N requests.

- **Access tracking**: Recalled memories now have `lastAccessedAt` and
  `accessCount` updated via fire-and-forget background tasks.

- **`simba server --sync-interval`**: Start the daemon with automatic periodic
  sync.

### Fixed

- **Vector search returning 0 results**: Added `table.checkout_latest()` before
  every vector search to refresh stale LanceDB table handles. Fragment buildup
  (one per `table.add()`) caused searches to silently return empty results.

- **Silent error swallowing in vector_db.py**: Replaced bare
  `except Exception: return []` with `logger.warning(..., exc_info=True)` so
  search failures are visible in daemon logs.

- **`table.update()` parameter name**: Fixed `values=` (wrong) to `updates=`
  (correct LanceDB API) in access tracking.

- **PreCompact TypeError on `/compact`**: Fixed crash when transcript entries
  contain nested `tool_result` content (list instead of string). Added
  `isinstance(val, str)` guard.

- **Content length limit**: Bumped `max_content_length` from 200 to 1000 to
  prevent 400 errors during learning extraction.

- **LanceDB table compaction**: Added `compact_table()` function and periodic
  compaction during diagnostics to prevent fragment buildup.

### Changed

- **Test infrastructure overhaul**: Removed all dangerous mock classes
  (`MockTable`, `MockVectorSearch`, `MockQuery`, `TrackingMockTable`) that
  masked production bugs. Tests now use real LanceDB with `tmp_path` fixtures
  and `httpx.AsyncClient` for async route testing.

- **PreToolUse refactored**: `main()` now collects context parts into a list
  (context-low warning, memory recall, truth DB) and joins them, replacing the
  earlier early-return pattern.

### Test Coverage

- 516 tests, all passing
- Lint clean (`ruff check` + `ruff format`)

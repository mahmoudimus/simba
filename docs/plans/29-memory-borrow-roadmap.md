# Memory Borrow Roadmap

This plan captures memory-system mechanisms Simba should borrow. It is a
sanitized repo artifact: it records implementation direction and gates, not the
private research notes or personal-memory file that prompted the review.

## Target End State

Borrow mechanisms, not architectures. Keep Simba's local-first constraints:

- no hosted embedding services
- no mutable-row-first memory design
- no custom vector store unless a measured bottleneck proves LanceDB is the
  limiting factor
- every configurable lever lives in `simba config`
- every default-on graduation has a real, attributed, re-runnable A/B

## Borrow First

### 0. Transcript Message Index Sidecar

Status: implemented.

Simba now has a rebuildable SQLite/FTS sidecar for raw transcript messages,
separate from LanceDB semantic memories. `simba sessions index --latest` or
`simba sessions index --path <file>` records `session_id`, `project_path`,
transcript path, message index/span, role, text, tool refs, file refs, and
parent session id when present. `simba sessions search <query>` recovers exact
raw-message evidence without injecting raw transcript text into hook context.
The default search limit is configurable as `sessions.search_limit`.

### 1. Daemon Readiness And Parity

Expose a richer `/health` and make `simba codex-status` surface it:

- ready/degraded state
- LanceDB path, table, count, and size
- FTS mirror path/count/tokenizer
- embedding model, provider, and dimension
- reranker mode
- sync scheduler state
- latest request id and latest daemon error

This is an ops gate, not a recall-quality lever. It protects daemon-vs-inline
parity work and makes hook/runtime failures diagnosable without reading logs.

### 2. Benchmark Provenance

Every append-only benchmark result should record enough attribution for a future
lever-graduation review:

- dataset name/path/hash
- source commit
- config snapshot hash
- answerer and judge model identity
- excluded, abstained, and contaminated counts
- judge replay agreement when measured
- significance metadata when computed

The existing detailed `config` record remains the source of truth; provenance is
the compact audit header.

### 3. Append-Only Supersession Audit

Status: implemented.

Simba keeps superseded LanceDB rows, appends a SQLite audit sidecar that links
old id -> new id with project path, memory type, similarity, reason, provenance,
and created-at timestamp, and demotes recalled old rows with a `supersededBy`
pointer. `simba memory supersession <memory_id>` inspects the forward chain.
Trust-gated supersession is also implemented: provenance records trust
source/origin, store computes a trust score from source, origin, confidence,
memory type, and usage counters, and lower-trust replacements of stronger
knowledge become `pending_confirmation` instead of active supersession. Pending
rows are shown as recall diagnostics only; `simba memory supersession
confirm|reject <audit_id>` appends the decision and either activates the
successor or keeps the older memory active.

### 3.5 Persistent Conflict/Judge Log

Status: implemented.

Simba now has an append-only `memory_judge_log` sidecar for replayable
adjudication decisions. The default-off write-time conflict path
(`memory.conflict_detect_on_write`) compares a newly stored memory against its
nearest neighbors, checks the judge log before asking the LLM, appends the
decision with input ids, strategy, winner/loser ids, judge/model identity,
prompt/config hashes, and timestamp, then records confirmed conflicts in
`memory_conflicts`. A retry with the same decision key reuses the logged verdict
instead of re-adjudicating.

### 3.6 Active Task Snapshot

Status: implemented.

Simba now has an append-only `task_snapshots` sidecar for compact current-work
state. `simba task snapshot save|show|clear` records task, summary,
branch/worktree, files, blockers, and next step; `clear` appends a cleared row
instead of mutating history. UserPromptSubmit injects the latest active
same-project snapshot as one bounded `<active-task-snapshot>` lane when
`hooks.task_snapshot_injection_enabled` is on, so a new session can recover the
last task and next step without broad semantic recall.

### 3.7 Trace -> Playbook -> Curator Loop

Status: implemented first slice.

`simba codex-extract --run --trace` writes an append-only JSONL analysis run
under `.simba/analysis_runs` by default, or under `--trace-dir` /
`codex.extraction_trace_dir` when provided. Each run records transcript load,
candidate memory evidence, source spans, heuristic reason/score, the current
conservative `keep` decision, store result or store error, and final status.
`codex.extraction_trace_enabled` can enable the same trace for configured
automatic extraction, but it remains off by default until benchmark evidence
shows the added artifacts improve review or later curation.

Failed candidate persistence now also appends `negative_lesson` trace events
for store exceptions or unaccepted store statuses, so noisy/rolled-back
candidates remain inspectable without becoming memories.

## Borrow Next

### 4. Context-Injection Budget Lanes

Status: implemented.

The guardian/rules context must not grow back into a large per-turn tax. Build a
single allocator with named lanes:

- protected guardian/rules lane
- capped active recall lane
- small doctrine/redirect lane
- bounded RAG/QMD lane
- compact RLM pointer lane
- drop-first diagnostics lane

The hook output protocol stays unchanged: all lanes render into one
`additionalContext` string.

### 4.5 Retrieval Triage

Status: implemented first slice.

`hooks.recall_triage_enabled` adds a cheap no-LLM classifier before
UserPromptSubmit retrieval. It is deliberately conservative: only narrow
self-contained prompts such as acknowledgements, current time/date questions,
or short rewrite/format/translate tasks skip memory recall and RAG. Prompts with
memory/repo/action cues retrieve, and unknown prompts are `uncertain` and still
retrieve. `hooks.recall_triage_emit_diagnostics` can inject a tiny
`<recall-triage>` block for measurement. The lever remains default-off until a
prompt fixture proves false negatives are acceptable.

`simba eval triage [--path CASES.jsonl] [--json]` runs the first fixture. The
built-in smoke currently gates on zero false negatives and reports accuracy,
false positives, and per-case decisions, making the lever re-runnable before any
default-on discussion.

### 5. Memory-Quality Counters

Status: implemented.

The current usage/feedback sidecar now has explicit counters:

- `match_count`
- `inject_count`
- `use_count`
- `noise_count`
- `save_count`

Recall bumps match/inject, store bumps save, and explicit good/bad feedback bumps
use/noise. Simba does not infer "used" or "noise" from agent behavior.

### 5.1 Outcome-Driven Half-Life

Status: implemented first slice.

`memory.outcome_quality_weight` lets the decay pass fold `use_count` vs
`noise_count` into effective feedback before computing strength and dormancy.
The lever is default-off (`0.0`) so existing decay/ranking behavior is unchanged
until a recall/HaluMem/noise eval proves that outcome counters move stale or
noisy memories down without recall regression.

## Borrow Selectively

### 5.5 Anticipated Queries

Status: implemented first slice.

Simba now has an append-only `memory_anticipated_queries` sidecar for likely
future query phrasings. `/store` accepts `anticipatedQueries`, and
`simba memory store` accepts repeatable `--anticipated-query` or comma-separated
`--anticipated-queries`. The sidecar deduplicates, caps entries with
`memory.anticipated_query_max_per_memory`, and does not affect recall ranking
yet. The next gate is an FTS/expansion lane with held-out query lift and no
plain-query regression.

### 6. Structured Query Filters

Status: implemented.

The recall path supports a small query grammar that routes into existing filters
and metadata checks:

- `type:`
- `project:`
- `after:` / `before:`
- `tag:`
- `path:`
- `symbol:`

This is not a planner; filter tokens are stripped from the semantic query, pushed
to route filters when possible, and otherwise applied after retrieval.

### 7. General-Memory Temporal Provenance

Status: implemented for new writes.

The KG already carries richer temporal/provenance fields. General memories
now append sidecar metadata for new writes:

- occurred-at
- observed-at
- source file/span/url
- extraction agent/version
- source session/transcript id

`as_of` recall filters remain deferred.

### 8. SubtleMemory Readback Ceiling

Status: implemented.

`simba eval bench subtlememory --compare-readback` records an exact-session
readback ceiling beside normal semantic recall. `--driver-report PATH` writes
the per-case failure ledger that turns the benchmark into an implementation
queue. `--driver-loop PATH` runs baseline plus the built-in same-session
expansion sweep, writes a comparison artifact, picks the winner by contradictory
`recall@10`, emits a promotion gate with recall/MRR guard checks, and does not
mutate persistent config. The report preserves the normal recall metrics, adds
ceiling metrics, and records deltas plus gold-width diagnostics so SubtleMemory
failures can be split into:

- retrieval missed the target session
- target-session readback is available but answer synthesis collapsed the
  relation
- gold spans more turns than the cutoff `k`, making `bridge_recall@k` a ceiling
  artifact rather than a retrieval bug

The first driver-selected lever is default-off same-session expansion:
`memory.session_expansion_enabled`. On persona_0, the measured top2/weight2
setting moved overall recall@10 0.211 -> 0.226 and contradictory recall@10
0.218 -> 0.244. On held-out persona_1, the driver picked top1/weight2 and moved
overall recall@10 0.198 -> 0.211 and contradictory recall@10 0.166 -> 0.191,
with a small contradictory MRR drop that stays inside the gate. It remains
default-off pending broader held-out-persona and cross-bench regression gates.

Additional held-out pass: combined persona_1..3 selected top1/weight2 with gate
pass, moving overall recall@10 +0.015 and contradictory recall@10 +0.027 while
keeping contradictory MRR positive (+0.002) and overall MRR inside the guard
(-0.0002). Top2/weight2 remains very close and has stronger MRR in that combined
artifact, so the promotion decision should stay gate-driven rather than hardcode
the first single-persona winner. A cross-bench smoke over LoCoMo(1),
LongMemEval(5), and HotpotQA(20) produced zero delta for top1/top2 variants,
which is the expected no-op/no-regression result because those loaders do not
carry session-source metadata for same-session expansion to fold.

## Defer

- adaptive cutoff expansion beyond the existing score-adaptive truncation until
  a narrow factual eval proves a gap
- custom vector storage, PQ, or HNSW work until LanceDB is measured as the
  bottleneck
- hosted/public API hardening beyond loopback/body-size protections
- two-pass long-session summarization unless it is default-off and engine-gated
- graph/PPR retrieval unless a new metric reopens the case

## Landed Slices

Implemented:

- transcript message index sidecar and `simba sessions index/search`
- richer daemon `/health` and `codex-status` rendering
- benchmark provenance blocks appended to new result records
- append-only supersession audit, recall demotion, and trust-gated confirmation
- persistent judge log and default-off write-time conflict endpoint wiring
- active task snapshots and bounded hook injection
- context-injection budget lanes
- default-off retrieval triage for UserPromptSubmit
- recall-triage eval fixture and CLI gate
- quality counters
- default-off outcome-quality decay lever
- anticipated-query sidecar for store-time metadata
- structured query filters
- general-memory temporal/provenance sidecar for new writes
- default-off Codex extraction analysis traces
- negative-lesson trace events for failed extraction candidates
- SubtleMemory readback ceiling comparison
- default-off same-session expansion lever selected by the SubtleMemory driver

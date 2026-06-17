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

Simba already has write-time supersession. The next step is an append-only audit
sidecar that links old id -> new id with project path, memory type, similarity,
reason, provenance, and created-at timestamp. Recall should demote or redirect
superseded hits instead of silently losing lineage.

## Borrow Next

### 4. Context-Injection Budget Lanes

The guardian/rules context must not grow back into a large per-turn tax. Build a
single allocator with named lanes:

- protected guardian/rules lane
- capped active recall lane
- small doctrine/redirect lane
- bounded RAG/QMD lane
- compact RLM pointer lane
- drop-first diagnostics lane

Keep hook output protocol unchanged and snapshot-test the rendered context.

### 5. Memory-Quality Counters

Extend the current usage/feedback sidecar with explicit counters:

- `match_count`
- `inject_count`
- `use_count`
- `noise_count`
- `save_count`

Start with CLI/eval labels and explicit feedback. Do not infer "used" or
"noise" from agent behavior until it has a measured precision benefit.

## Borrow Selectively

### 6. Structured Query Filters

Add a small query grammar that routes into existing filters and FTS metadata:

- `type:`
- `project:`
- `after:` / `before:`
- `tag:`
- `path:`
- `symbol:`

Do not build a new planner for this.

### 7. General-Memory Temporal Provenance

The KG already carries richer temporal/provenance fields. General memories
should gain metadata for new writes first:

- occurred-at
- observed-at
- source file/span/url
- extraction agent/version
- source session/transcript id

Only add `as_of` recall filters after the metadata exists.

## Defer

- adaptive cutoff expansion beyond the existing score-adaptive truncation until
  a narrow factual eval proves a gap
- custom vector storage, PQ, or HNSW work until LanceDB is measured as the
  bottleneck
- hosted/public API hardening beyond loopback/body-size protections
- two-pass long-session summarization unless it is default-off and engine-gated
- graph/PPR retrieval unless a new metric reopens the case

## Current First Slice

This slice implements the first two borrow-first gates:

- richer daemon `/health` and `codex-status` rendering
- benchmark provenance blocks appended to new result records

Follow-up slices should land supersession audit, then context lanes.

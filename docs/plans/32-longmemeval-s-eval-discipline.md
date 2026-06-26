# 32 - LongMemEval-S Eval Discipline

Build a Hippo-style, claim-safe LongMemEval-S measurement path for Simba.

The immediate goal is not to add another recall lever. The goal is to make one
command produce a report that a friend, contributor, or skeptical reader can
trust: exact dataset hash, exact config, exact metric axis, reproducible command,
and explicit claim boundaries.

## Why This Exists

Simba already has useful eval infrastructure:

- `simba eval bench longmemeval`
- append-only `.simba/eval/results.jsonl`
- dataset/config/model provenance
- local QA judge support
- an existing LongMemEval loader that keeps session dates attached to turns

But Simba's current LongMemEval recall metric is turn-level evidence recall:
ranked ids look like `session_id#turn_index`, and gold ids are `has_answer`
turns. That is stricter and useful for internal debugging, but it is not the
same public axis used by gbrain's LongMemEval-S report.

The gbrain-style public axis is session-level R@5: did at least one retrieved
session id match `answer_session_ids` for the question's haystack?

This plan adds both axes:

- **turn evidence recall**: Simba's stricter internal metric
- **session recall**: leaderboard-compatible LongMemEval-S retrieval metric

It also separates two protocols:

- **per-haystack**: each question is retrieved against only its own
  `haystack_sessions`; this is the standard LongMemEval-S/gbrain axis
- **global-pool**: all LongMemEval-S sessions are pooled into one store; this is
  closer to real agent memory, where the user does not give us the right
  haystack at query time

## Gbrain 1:1 Parity Contract

The primary public comparison must match gbrain's published LongMemEval-S axis
before we interpret the number.

Source contract to mirror:

- dataset: `xiaowu0162/longmemeval`, `_s` split
- sample: full 500 questions
- haystack: per-question haystack, around 50 sessions per question
- metric: retrieval recall@5
- gold: `answer_session_ids`, not `has_answer` turn ids
- scoring: success if any ground-truth answer session appears in the top 5
  retrieved sessions
- adapter headline: `gbrain-hybrid`
- reference score: `97.60%` R@5, `488/500`
- no LLM in the headline retrieval loop
- expansion result: `gbrain-hybrid+expansion` also `97.60%`, so query expansion
  is a reported null on this benchmark

Simba's 1:1 lane must therefore report:

- `simba-session`: session-level R@5 on full LongMemEval-S per-haystack
- `simba-session+expansion` only if an explicitly comparable expansion lever is
  enabled; otherwise do not imply parity with gbrain expansion
- `simba-turn-evidence`: stricter internal evidence recall, shown separately
- `simba-global-pool`: agent-realistic stress result, shown separately

The report must place the gbrain reference and Simba 1:1 result in the same
table, but must not mix the stricter turn-evidence or global-pool numbers into
that table.

Example table shape:

| System | Protocol | Unit | K | N | R@5 | LLM in retrieval? | Notes |
|---|---|---|---:|---:|---:|---|---|
| gbrain-hybrid | per-haystack | session | 5 | 500 | 0.976 | no | published v0.28.8 |
| Simba | per-haystack | session | 5 | 500 | TBD | no/yes | exact config hash |

If Simba uses a local reranker or LLM-HyDE during retrieval, the table must mark
`LLM in retrieval?` truthfully. Do not claim a no-LLM comparison if any LLM arm
fires under the measured config.

## Claim Rules

Allowed only after the full gate passes:

- "Simba reaches X session-level R@5 on LongMemEval-S per-haystack, full 500
  questions, k=5, dataset SHA ...".
- "Simba beats / does not beat gbrain's published 97.60 session-level R@5 under
  this metric."
- "Simba's global-pool R@5 is X, which is the more agent-realistic stress test."

Forbidden before the gate:

- "Simba beats gbrain."
- "Simba is SOTA."
- "LongMemEval is solved."
- Any headline based on an `N=30` subset, oracle haystack, turn-level recall, or
  QA judge score without saying exactly that.

## Current Baselines To Preserve

These are context, not final claims:

- LongMemEval oracle turn-level recall@5 around `0.780` to `0.814`, depending on
  embedder/config.
- Real-haystack LongMemEval-S subset (`N=30`) recall@5 `0.776`, QA `0.567`,
  recall@10 `0.882`; useful directional smoke only.
- LoCoMo remains useful for broader recall and QA regressions, but it is not the
  gbrain comparison axis.

## Implementation Slice 1 - Session-Level Gold

### Files

- `src/simba/eval/dataset.py`
- `src/simba/eval/benchmarks/longmemeval.py`
- `tests/eval/benchmarks/test_longmemeval.py`

### Shape

Extend `EvalCase` with optional benchmark metadata:

```python
metadata: dict[str, object] = dataclasses.field(default_factory=dict)
```

The LongMemEval loader should set:

```python
metadata={
    "answer_session_ids": ["s1", "s2"],
    "haystack_session_ids": [...],
    "question_type": "...",
}
```

Do not replace `relevant_ids`. Keep turn-level `has_answer` ids exactly as they
are today so existing recall metrics remain stable.

### Tests First

Add tests that prove:

- `answer_session_ids` is preserved in `EvalCase.metadata`.
- `haystack_session_ids` is preserved in `EvalCase.metadata`.
- `question_date` still threads through unchanged.
- Existing `relevant_ids == ["session#turn"]` behavior is unchanged.
- Dataset serialization round-trips metadata if `Dataset.to_dict()` is used.

## Implementation Slice 1.5 - Gbrain Parity Fixture

### Files

- `src/simba/eval/longmemeval_parity.py` (new)
- `tests/eval/test_longmemeval_parity.py` (new)
- `src/simba/eval/longmemeval_report.py` (later slice consumes it)

### Shape

Create a small model for the public reference axis:

```python
GBRAIN_LONGMEMEVAL_S_REFERENCE = {
    "source": "gbrain-evals/docs/benchmarks/2026-05-07-longmemeval-s.md",
    "dataset": "xiaowu0162/longmemeval _s",
    "system": "gbrain-hybrid",
    "n": 500,
    "k": 5,
    "unit": "session",
    "protocol": "per-haystack",
    "recall@5": 0.976,
    "hits": 488,
    "llm_in_retrieval": False,
}
```

Also model `gbrain-vector`, `gbrain-keyword`, and
`gbrain-hybrid+expansion` as context rows, but only `gbrain-hybrid` is the
headline comparison.

The reference should be data, not prose embedded in the report template. This
lets claim gates and future updates compare against one canonical object.

### Tests First

Add tests that prove:

- gbrain reference row is session-level, per-haystack, k=5, n=500.
- gbrain reference R@5 is exactly `0.976`.
- `gbrain-hybrid+expansion` is represented as LLM-in-retrieval and a null vs
  `gbrain-hybrid`.
- report comparison code refuses to compare Simba turn-level or global-pool
  results against the gbrain headline row.

## Implementation Slice 2 - Session Recall Scorer

### Files

- `src/simba/eval/benchmarks/session_recall.py` (new)
- `src/simba/eval/benchmarks/run.py`
- `tests/eval/benchmarks/test_session_recall.py` (new)

### Shape

Add a small pure scorer that converts ranked turn ids to ranked session ids:

```python
def turn_id_to_session_id(memory_id: str) -> str:
    return memory_id.split("#", 1)[0]


def collapse_ranked_to_sessions(ids: list[str]) -> list[str]:
    ...


def score_session_recall(
    ranked_ids_by_case: dict[str, list[str]],
    gold_sessions_by_case: dict[str, list[str]],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, object]:
    ...
```

The collapse must deduplicate sessions while preserving first rank. If the same
session appears in ranked positions 1, 4, and 9, it counts once at rank 1.

### Tests First

Add tests that prove:

- `s1#0`, `s1#3`, `s2#0` collapses to `s1`, `s2`.
- R@5 succeeds when any gold answer session is in the first five collapsed
  sessions.
- MRR uses the first matching collapsed session rank.
- Multiple gold sessions are handled.
- Missing `answer_session_ids` yields an explicit skipped/invalid count, not a
  silent zero.

## Implementation Slice 3 - CLI Flags

### Files

- `src/simba/__main__.py`
- `tests/test_eval_bench.py`

### CLI

Extend `simba eval bench longmemeval` with:

```text
--unit turn|session|both        default: turn
--protocol per-haystack|global-pool|both  default: per-haystack
--report PATH                  optional locked markdown/json report
--claim-gate                   fail non-zero if the report is not claim-safe
```

`--unit turn` preserves current behavior.

`--unit session` appends a `session_recall` block to the result record. It may
still run the same retrieval path internally; scoring is the only difference.

`--unit both` records both:

```json
{
  "recall": {... turn evidence metrics ...},
  "session_recall": {... session-level metrics ...}
}
```

`--protocol per-haystack` preserves current one-dataset-per-question behavior.

`--protocol global-pool` builds a single pooled LongMemEval-S dataset:

- corpus: deduplicated turns from all haystack sessions
- cases: all questions
- case gold: turn gold for evidence recall, `answer_session_ids` for session
  recall

`--protocol both` runs both and records/report them separately. Do not mix their
metrics in one average.

### Tests First

Add CLI tests using stubs:

- `--unit session` passes through and writes `session_recall`.
- `--unit both` writes both `recall` and `session_recall`.
- `--protocol global-pool` calls the pooled loader path.
- `--protocol both` records `per_haystack` and `global_pool` as separate result
  blocks or separate report sections.
- Unsupported `--unit bad` and `--protocol bad` return `1`.

## Implementation Slice 4 - Global-Pool Loader

### Files

- `src/simba/eval/benchmarks/longmemeval.py`
- `tests/eval/benchmarks/test_longmemeval.py`

### Shape

Add:

```python
def load_longmemeval_global_pool(
    path: str | pathlib.Path,
    *,
    include_abstention: bool = False,
) -> list[Dataset]:
    ...
```

Return a one-item list with a single `Dataset`.

Important details:

- Deduplicate corpus ids by `session_id#turn_index`.
- Preserve the first observed content for duplicated sessions.
- Prefix each turn with its session date, using the existing zip of
  `haystack_dates` with `haystack_sessions`.
- Keep case `question_date`.
- Preserve `answer_session_ids` and `haystack_session_ids` in metadata.
- Drop unresolvable cases the same way the existing loader does, but count them
  in report provenance.

### Tests First

Add tests that prove:

- two questions sharing a session produce one corpus copy.
- cases remain separate.
- question dates remain per-case, not global.
- session metadata survives.
- abstention behavior matches the per-haystack loader.

## Implementation Slice 5 - Locked Report

### Files

- `src/simba/eval/longmemeval_report.py` (new)
- `tests/eval/test_longmemeval_report.py` (new)
- `docs/plans/README.md`

### Shape

The report writer should accept one or more result records and emit a markdown
file with:

- title and timestamp
- dataset path, SHA-256, file size, question count
- git SHA and dirty-worktree flag
- config hash and relevant memory/bench config summary
- embedder identity and dimension
- reranker identity
- answerer and judge identity if QA ran
- protocol sections:
  - per-haystack
  - global-pool
- unit sections:
  - session recall
  - turn evidence recall
  - QA, if run
- a dedicated "Gbrain 1:1 comparison" table that includes only:
  - gbrain reference session R@5
  - Simba per-haystack session R@5
- per-question-type table
- excluded/unresolved/abstention counts
- exact command
- "Allowed claims"
- "Forbidden claims"
- reproduction notes

The report must be generated from the appended result record, not manually
assembled from local variables. The JSONL remains the source of truth.

### Tests First

Add tests that prove:

- report includes dataset SHA and config hash.
- report includes both protocol names when both were run.
- report includes session-level R@5 when present.
- report includes the gbrain 1:1 table only when a claim-safe per-haystack
  session result is present.
- report does not put turn-evidence or global-pool rows in the gbrain 1:1 table.
- report refuses to render "beats gbrain" unless the full claim gate passes.
- report includes forbidden-claims text for subset or oracle runs.

## Implementation Slice 6 - Claim Gate

### Files

- `src/simba/eval/longmemeval_claims.py` (new)
- `tests/eval/test_longmemeval_claims.py` (new)

### Gate Inputs

The gate should inspect a result/report model, not parse markdown.

Required for a gbrain comparison claim:

- dataset is LongMemEval-S, not oracle
- full 500 questions attempted
- `k=5`
- session-level scoring present
- per-haystack protocol present
- comparison target is the canonical `gbrain-hybrid` reference row
- dataset hash recorded
- config hash recorded
- no dirty-worktree flag, unless explicitly marked as local-only exploratory
- no tuned-on-test override flag
- excluded/unresolved/abstention counts reported
- LLM-in-retrieval status reported

The gate should not require beating gbrain to pass as "claim-safe." It should
return two separate concepts:

```python
claim_safe: bool
beats_gbrain_9760: bool | None
```

This lets a report honestly say "claim-safe run, did not beat gbrain."

### Tests First

Add tests that prove:

- subset `N=30` is not claim-safe.
- oracle haystack is not claim-safe for gbrain comparison.
- turn-only scoring is not claim-safe for gbrain comparison.
- global-pool scoring is not claim-safe for the gbrain 1:1 comparison, even
  though it is valid as a separate stress result.
- full LongMemEval-S per-haystack session R@5 can be claim-safe.
- R@5 `0.9759` does not beat gbrain; `0.9761` does.
- a result with an LLM retrieval arm is claim-safe only if the comparison table
  marks that fact.

## Implementation Slice 7 - Full Run Commands

The final implementation should document and support these exact commands:

```bash
# Leaderboard-compatible retrieval axis.
uv run simba eval bench longmemeval \
  --path /path/to/longmemeval_s.json \
  --unit session \
  --protocol per-haystack \
  --k 5 \
  --abstention \
  --report .simba/eval/reports/longmemeval-s-per-haystack.md \
  --claim-gate

# Explicit gbrain 1:1 comparison report.
uv run simba eval bench longmemeval \
  --path /path/to/longmemeval_s.json \
  --unit session \
  --protocol per-haystack \
  --k 5 \
  --abstention \
  --report .simba/eval/reports/longmemeval-s-gbrain-1to1.md \
  --claim-gate

# Simba-realistic stress axis.
uv run simba eval bench longmemeval \
  --path /path/to/longmemeval_s.json \
  --unit both \
  --protocol global-pool \
  --k 5 \
  --abstention \
  --report .simba/eval/reports/longmemeval-s-global-pool.md

# Full dual report.
uv run simba eval bench longmemeval \
  --path /path/to/longmemeval_s.json \
  --unit both \
  --protocol both \
  --k 5 \
  --abstention \
  --report .simba/eval/reports/longmemeval-s-dual.md \
  --claim-gate
```

## Acceptance Criteria

- Existing LongMemEval turn-level recall behavior remains unchanged by default.
- `--unit session` reports session-level R@1/R@3/R@5/R@10/MRR.
- `--protocol global-pool` runs all questions against one deduplicated corpus.
- `--protocol both --unit both` produces a report with four retrieval sections:
  - per-haystack session recall
  - per-haystack turn evidence recall
  - global-pool session recall
  - global-pool turn evidence recall
- The gbrain 1:1 table compares only `gbrain-hybrid` vs Simba per-haystack
  session R@5 on full LongMemEval-S, k=5.
- The gbrain 1:1 table reports whether Simba used any LLM retrieval arm.
- Report includes dataset SHA, git SHA, dirty flag, config hash, model identities,
  excluded/unresolved counts, abstention counts, and exact command.
- Claim gate blocks gbrain comparison claims on subset/oracle/turn-only runs.
- Full test and ruff target passes for touched files.

## Verification Commands

```bash
uv run pytest \
  tests/eval/benchmarks/test_longmemeval.py \
  tests/eval/benchmarks/test_session_recall.py \
  tests/eval/test_longmemeval_parity.py \
  tests/eval/test_longmemeval_report.py \
  tests/eval/test_longmemeval_claims.py \
  tests/test_eval_bench.py -q

uv run ruff check \
  src/simba/eval/benchmarks/longmemeval.py \
  src/simba/eval/benchmarks/session_recall.py \
  src/simba/eval/longmemeval_parity.py \
  src/simba/eval/longmemeval_report.py \
  src/simba/eval/longmemeval_claims.py \
  src/simba/__main__.py \
  tests/eval/benchmarks/test_longmemeval.py \
  tests/eval/benchmarks/test_session_recall.py \
  tests/eval/test_longmemeval_parity.py \
  tests/eval/test_longmemeval_report.py \
  tests/eval/test_longmemeval_claims.py \
  tests/test_eval_bench.py
```

Optional full eval commands should be run only when the dataset path is present
and the user accepts the runtime cost.

## Open Design Questions

1. Should global-pool use every turn from every question haystack, or reconstruct
   a canonical unique-session corpus first and treat duplicated text collisions
   as dataset hygiene warnings?
2. Should claim-safe reports require a clean git worktree, or allow dirty local
   runs with a visible "not public-claim-safe" banner?
3. Should we add a separate "local-first comparison" table that excludes OpenAI
   embeddings and frontier judges, rather than trying to sit on the same axis as
   paid hosted systems?

## Strategic Read

This plan is valuable even if Simba does not beat gbrain. A failed but
claim-safe run is a win because it tells us exactly where the gap lives:

- session retrieval
- turn evidence retrieval
- per-haystack vs global-pool collapse
- answer generation
- abstention
- temporal / multi-session subtypes

That is the discipline we want friends and contributors to inherit.

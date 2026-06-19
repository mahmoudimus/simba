# 30 - AutoContext Trace Curator

Status: implemented second slice.

## Problem

Simba now records `simba codex-extract --run --trace` analysis artifacts, but
those traces are still raw implementation logs. They are useful to a developer,
yet too noisy to drive the next memory-quality loop directly.

The tempting wrong move is to rebuild the archived write-time fact index: parse
every trace into canonical structured facts and optimize storage around that
new database. That path already lost to Simba's store-raw, answer-time
reasoning design. The safer AutoContext-inspired move is a review layer:
convert extraction traces into compact, evidence-backed curator reports and
playbook candidates that a human or conservative later gate can promote through
the existing memory store.

## Target End State

`simba codex-curate` reads append-only analysis trace JSONL files and writes
append-only curator reports under `.simba/curator_runs/`.

The report answers:

- What candidate memories did the extraction run see?
- What source span and evidence justified each candidate?
- Which candidates were stored, duplicated, superseded, rejected, or failed?
- What repeated negative lessons or noisy patterns should change extraction?
- Which reviewable playbook summaries are worth promoting later?

The curator does not mutate memory state. It produces reviewable artifacts and,
optionally, exact `simba memory store ...` commands for a later reviewed pass.

## Non-Goals

- Do not introduce a canonical fact database.
- Do not auto-store curator output.
- Do not delete, rewrite, or compact existing analysis traces.
- Do not increase hook prompt context.
- Do not require hosted LLMs or external embedding services.
- Do not make curator output part of recall ranking until measured separately.

## Existing Substrate

`simba codex-extract --run --trace` already writes analysis JSONL events under
`.simba/analysis_runs` by default, with optional overrides through
`--trace-dir` and `codex.extraction_trace_dir`.

The first trace slice emits events such as:

- `run_started`
- `transcript_loaded`
- `candidate`
- `curator_decision`
- `store_result`
- `store_error`
- `negative_lesson`
- `run_completed`

Store outcomes already distinguish stored rows, duplicates, supersession,
pending confirmation, and errors. The curator should treat those outcomes as
evidence, not reinterpret storage decisions as new truth.

## Design

Add `src/simba/codex/curator.py` with small dataclasses and pure functions:

```python
@dataclass(frozen=True)
class TraceCandidate:
    index: int
    memory_type: str
    content: str
    context: str
    reason: str
    score: float | None
    source_span: str | None
    evidence: str | None
    decision: str | None
    store_status: str | None
    memory_id: str | None


@dataclass(frozen=True)
class CuratorReport:
    trace_path: Path
    session_id: str | None
    project_path: str | None
    transcript_path: str | None
    status: str
    candidates: tuple[TraceCandidate, ...]
    negative_lessons: tuple[str, ...]
    metrics: Mapping[str, int | float]
    suggested_actions: tuple[str, ...]
```

Core functions:

```python
def load_trace(path: Path) -> CuratorTrace: ...
def summarize_trace(trace: CuratorTrace) -> CuratorReport: ...
def write_markdown(report: CuratorReport, path: Path) -> Path: ...
def write_json(report: CuratorReport, path: Path) -> Path: ...
def find_latest_trace(trace_dir: Path) -> Path | None: ...
```

The parser should be tolerant:

- unknown event types are preserved in diagnostics, not fatal
- malformed JSONL rows become report warnings with line numbers
- missing optional fields render as empty cells
- failed or incomplete runs still produce a partial report

## CLI

Add a new command in `src/simba/__main__.py`:

```bash
simba codex-curate --latest
simba codex-curate --trace .simba/analysis_runs/<run>.jsonl
simba codex-curate --out .simba/curator_runs/<run>.md
simba codex-curate --json
```

Defaults:

- `--latest` reads the newest trace from `codex.extraction_trace_dir` or
  `.simba/analysis_runs`
- markdown writes to `.simba/curator_runs/<trace-stem>.md`
- `--json` writes `.simba/curator_runs/<trace-stem>.json`
- no mode calls `memory store`

Add config fields under the existing Codex config section:

```python
curator_report_dir: str = ""
curator_default_format: str = "markdown"
curator_min_candidate_score: float = 0.0
```

An empty report dir means `<cwd>/.simba/curator_runs`.

## Report Shape

Markdown reports should include:

- run metadata: session id, project path, transcript path, trace path, final
  status
- metrics: candidate count, stored count, duplicate count, superseded count,
  pending confirmation count, error count, negative lesson count
- accepted/stored candidates
- duplicates and supersession decisions
- pending confirmations
- store errors
- negative lessons
- weak or low-score candidates needing review
- suggested next actions

Every candidate row must carry a source span or an explicit `missing` marker.
Evidence should be short and local to the candidate; the report is a review
artifact, not a transcript copy.

## Playbook Candidate Output

The first slice should only summarize playbook candidates; it should not install
or activate them. A playbook candidate is a repeated pattern in the trace, such
as:

- a candidate type that repeatedly stores successfully
- a candidate reason that repeatedly fails storage
- a duplicate pattern that suggests an extractor should strengthen evidence
- a negative lesson that appears across multiple traces

Represent each playbook candidate as:

```json
{
  "kind": "extractor_pattern",
  "summary": "...",
  "evidence": ["trace-line-or-candidate-id"],
  "suggested_change": "...",
  "risk": "low|medium|high"
}
```

Cross-run aggregation belongs in the second implementation slice. The first
slice may group repeated patterns inside one trace only.

## Measure-First Gates

Before any default-on or auto-promotion discussion, measure the curator as a
review tool:

- `review_compression`: transcript tokens vs report tokens
- `candidate_precision`: reviewer-approved candidates / reviewed candidates
- `duplicate_rate`: duplicate outcomes / candidates
- `store_error_rate`: store errors / candidates
- `evidence_coverage`: candidates with source span and evidence / candidates
- `review_latency`: time to approve or reject all candidates in a report

The first shipped gate is a fixture-level correctness gate, not a quality claim.
Quality claims require sampled real traces plus reviewer labels.

## Tests

Create `tests/codex/test_curator.py`.

RED-first cases:

- `test_summarize_trace_groups_store_outcomes`
- `test_curator_keeps_negative_lessons_out_of_memory`
- `test_markdown_report_includes_evidence_and_source_span`
- `test_json_report_is_stable`
- `test_incomplete_trace_still_writes_partial_report`
- `test_unknown_events_are_reported_not_fatal`

Extend CLI coverage in `tests/test_main_cli.py`:

- `test_codex_curate_latest_uses_configured_trace_dir`
- `test_codex_curate_trace_writes_default_curator_dir`
- `test_codex_curate_json_writes_json_report`
- `test_codex_curate_does_not_call_memory_store`

## Acceptance Criteria

- `simba codex-curate --trace <trace>` writes a readable markdown report.
- `simba codex-curate --trace <trace> --json` writes stable JSON.
- `simba codex-curate --latest` selects the newest configured trace.
- Reports are append-only artifacts under `.simba/curator_runs` by default.
- Every candidate in the report links back to trace evidence or marks evidence
  as missing.
- No curator path mutates LanceDB, SQLite memory state, supersession state, or
  hook context.
- Config is exposed through `simba config get/set`.
- README and CHANGELOG document the command as default-off review tooling.

## Verification Commands

```bash
uv run pytest tests/codex/test_curator.py tests/test_main_cli.py -q
uv run ruff check src/simba/codex/curator.py src/simba/__main__.py tests/codex/test_curator.py tests/test_main_cli.py
```

For a real trace smoke:

```bash
simba codex-extract --run --trace
simba codex-curate --latest
```

## Later Slices

Second slice:

- append reviewer decisions to `.simba/curator_runs/<run>.review.jsonl`
- emit exact `simba memory store ...` commands for approved candidates
- compare reviewer labels against store outcomes
- aggregate repeated negative lessons across traces

Status: implemented for per-report review decisions and accepted-candidate
store command generation. Cross-report label aggregation remains deferred.

Third slice:

- generate candidate extractor playbooks from multiple reviewed reports
- add a measured promotion gate before any playbook affects extraction
- feed accepted playbooks back into documentation or extraction heuristics

Only after those gates pass should Simba consider any automated promotion from
curator output into persistent memory.

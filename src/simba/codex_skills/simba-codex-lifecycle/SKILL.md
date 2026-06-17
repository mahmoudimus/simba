---
name: simba-codex-lifecycle
description: Enforce Simba's Codex lifecycle routine for coding tasks. Use when starting or finishing implementation work in a Simba-enabled repo to run `simba codex-status` at start, handle any still-pending raw Codex transcript extraction, and run `simba codex-finalize` before final handoff.
---

# Simba Codex Lifecycle

Run this routine for implementation tasks in Simba-enabled repositories.

## Required Sequence

1. Recall relevant memory first (use the user request as query text):

```bash
simba codex-recall "<user request>"
```

2. Run status once per session:

```bash
simba codex-status
```

Do not run this on every prompt. Re-run only if:
- there was a long pause or major context shift,
- memory actions fail unexpectedly,
- you are about to finalize and want a fresh check.

3. Codex `PreCompact` is the primary automatic session-analysis path. If
   `codex-status` still reports a raw JSONL transcript as `pending_extraction`
   and you need to analyze it now, run the explicit storage path:

```bash
simba codex-extract --run
```

If that reports no candidates or fails, run the manual prompt path:

```bash
simba codex-extract
```

4. Before final handoff, run finalize:

```bash
simba codex-finalize
```

## Finalize Inputs

Prefer to include both response text and transcript path when available:

```bash
simba codex-finalize --response-file /path/to/response.txt --transcript /path/to/transcript.jsonl
```

If those files are not available, still run `simba codex-finalize` as a best-effort check.

If `simba` is not in PATH, use:

```bash
uvx --from /Users/mahmoud/src/ai/simba simba codex-status
```

## Output Discipline

- Mention whether `codex-recall` returned relevant memories and how they informed your approach.
- Mention in your final update that lifecycle checks were run.
- If you run `codex-extract --run`, mention the stored / duplicate / error
  counts. If extraction remains pending, run `codex-extract` and report that
  action.

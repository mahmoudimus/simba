---
name: memories-learn
description: Extract learnings from session transcripts and store in semantic memory database
allowed-tools: Task, Bash(simba *)
---

Check the dispatch mode:
```bash
simba config get hooks.learn_async
```

Resolve the transcript for THIS project (never the global `latest.json` — it is a
single symlink overwritten by whichever session compacted last, across all
projects, so it cross-wires sessions):
```bash
simba transcript pending --json
```
This prints the newest `pending_extraction` transcript whose `project_path` matches
the current working directory: `{transcript_path, session_id, project_path}` (or
`{}` + exit 1 if there is nothing to extract for this project — in that case stop,
there is no work to do). Use those three values below.

Build this Task prompt:
```
Read the transcript at <TRANSCRIPT_PATH> and extract learnings to store in the semantic memory database.

For each learning found, store it by running:
simba memory store --type <TYPE> --content "<LEARNING>" --context "<CONTEXT>" --confidence <SCORE> --session-source "<SESSION_ID>" --project-path "<PROJECT_PATH>"

LEARNING TYPES:
- WORKING_SOLUTION: Commands, code, or approaches that worked
- GOTCHA: Traps, counterintuitive behaviors, "watch out for this"
- PATTERN: Recurring architectural decisions or workflows
- DECISION: Explicit design choices with reasoning
- FAILURE: What didn't work and why
- PREFERENCE: User's stated preferences

RULES:
- Be specific - include actual commands, paths, error messages
- Confidence 0.95+ for explicitly confirmed, 0.85+ for strong evidence
- Skip generic programming knowledge Claude already knows
- Focus on user-specific infrastructure, preferences, workflows
- Keep content under 200 characters, use context field for details
- Preserve proper nouns, file paths, and identifiers verbatim — never replace them with generic words
- Preserve numeric precision: keep exact values exact; never weaken an exact number to a range or approximation
- Resolve relative dates to absolute ones (e.g. "yesterday" -> the actual date)

Extract 5-15 quality learnings.
```

Dispatch using the Task tool with subagent_type=memory-extractor:
- If hooks.learn_async is "true": set run_in_background=true (fire and forget)
- Otherwise: dispatch normally and wait for completion

After the extractor finishes (synchronous mode only), mark the transcript done so
it isn't re-extracted on the next run:
```bash
simba transcript mark-extracted <SESSION_ID>
```
(In async mode, skip this — the background agent owns completion.)

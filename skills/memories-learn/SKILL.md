---
name: memories-learn
description: >
  Parse session transcripts to extract key learnings (solutions, gotchas, patterns, decisions, failures, preferences)
  and store each as a typed entry in simba's semantic memory database via `simba memory store`.
  Use when the user wants to save insights from a conversation, remember session knowledge, update the knowledge base,
  extract notes from transcript history, or persist learnings for future recall.
trigger: conversation history, save insights, remember from session, extract learnings, store memories, knowledge base, session notes, persist knowledge, recall, remember
allowed-tools: Task, Bash(simba *)
---

## 1. Validate environment

```bash
# Check async dispatch preference
ASYNC=$(simba config get hooks.learn_async 2>/dev/null || echo "false")

# Locate the latest transcript
LATEST=~/.claude/transcripts/latest.json
if [ ! -f "$LATEST" ]; then echo "ERROR: $LATEST not found — no transcript to process"; exit 1; fi

# Extract transcript path, session ID, project path
TRANSCRIPT=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['path'])" "$LATEST")
SESSION_ID=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['sessionId'])" "$LATEST")
PROJECT_PATH=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('projectPath',''))" "$LATEST")

test -f "$TRANSCRIPT" && echo "OK transcript=$TRANSCRIPT session=$SESSION_ID project=$PROJECT_PATH" \
  || { echo "ERROR: transcript file not found at $TRANSCRIPT"; exit 1; }
```

## 2. Build extraction prompt

Construct the following prompt, substituting the values obtained above:

```
Read the transcript at <TRANSCRIPT_PATH> and extract 5-15 key learnings. For each, run:

simba memory store --type <TYPE> --content "<LEARNING>" --context "<CONTEXT>" --confidence <SCORE> --session-source "<SESSION_ID>" --project-path "<PROJECT_PATH>"

TYPES (pick the best fit):
- WORKING_SOLUTION: commands, code, or approaches that worked
- GOTCHA: traps, counterintuitive behaviors, surprises
- PATTERN: recurring architectural decisions or workflows
- DECISION: explicit design choices with reasoning
- FAILURE: what didn't work and why
- PREFERENCE: user's stated preferences or style choices

RULES:
- Include actual commands, paths, and error messages — no generic advice
- Confidence: 0.95+ if explicitly confirmed, 0.85+ if strongly implied
- Skip knowledge Claude already has; focus on project-specific details
- Keep content under 200 chars; put extra detail in the context field
```

## 3. Dispatch via Task tool

```
Task(
  description="<the extraction prompt above with real values substituted>",
  subagent_type="memory-extractor",
  run_in_background=<true if ASYNC is "true", else false>
)
```

If `run_in_background=true`, the extraction proceeds asynchronously — no need to wait.
Otherwise, wait for the Task to complete and verify output shows `simba memory store` calls succeeding.

## Error recovery

- If `simba memory store` returns a connection error, confirm the daemon is running: `simba server` or check `simba config get memory.port`.
- If the transcript JSON is malformed, fall back to reading the raw file and extracting learnings manually.

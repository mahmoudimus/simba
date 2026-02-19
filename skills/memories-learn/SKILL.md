---
name: memories-learn
description: Extract learnings from session transcripts and store in semantic memory database
allowed-tools: Task, Bash(simba *)
---

Check the dispatch mode:
```bash
simba config get hooks.learn_async
```

Then read ~/.claude/transcripts/latest.json to get the transcript path, session ID, and project path.

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

Extract 5-15 quality learnings.
```

Dispatch using the Task tool with subagent_type=memory-extractor:
- If hooks.learn_async is "true": set run_in_background=true (fire and forget)
- Otherwise: dispatch normally and wait for completion

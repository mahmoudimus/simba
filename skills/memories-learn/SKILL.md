---
name: memories-learn
description: Extract learnings from session transcripts and store in semantic memory database
disable-model-invocation: true
context: fork
agent: memory-extractor
allowed-tools: Read, Bash(curl *)
---

Read the metadata file at ~/.claude/transcripts/latest.json to get the transcript path, then read that transcript and extract learnings.

Store each learning via:
```bash
curl -X POST http://localhost:8741/store -H "Content-Type: application/json" -d '{"type": "<TYPE>", "content": "<LEARNING>", "context": "<CONTEXT>", "confidence": <SCORE>}'
```

Extract 5-15 quality learnings. Focus on solutions, gotchas, patterns, and user preferences.

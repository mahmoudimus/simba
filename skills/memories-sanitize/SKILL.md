---
name: memories-sanitize
description: Review recent memories and remove invalid or misleading ones from the semantic memory database
disable-model-invocation: true
context: fork
allowed-tools: Read, Bash(curl *)
---

# Memory Sanitization Procedure

Review memories and identify those that are **invalid**, **misleading**, or **outdated**. This applies to any work session - debugging, feature development, refactoring, exploration, etc.

## Step 1: List Memories

Query the daemon to get all memories:

```bash
curl -s http://localhost:8741/list
```

This returns JSON with all memories including:
- `id` - Memory ID (e.g., `mem_abc123`)
- `type` - GOTCHA, WORKING_SOLUTION, PATTERN, DECISION, FAILURE, PREFERENCE
- `content` - The learning content
- `context` - Additional context
- `confidence` - Confidence score (0-1)
- `createdAt` - ISO timestamp

## Step 2: Identify Invalid Memories

Review each memory and mark as INVALID if it:

### From Any Session Type
1. **Outdated information** - Was true but no longer applies after changes
2. **Superseded by better approach** - A newer, better solution exists
3. **Too specific/narrow** - Only applied to a removed/changed feature
4. **Incorrect generalization** - Specific case wrongly stated as general rule

### From Debugging Sessions
5. **Red herring** - An initial hypothesis that turned out to be wrong
6. **Wrong blame** - Points to code that was actually working correctly
7. **Non-issue** - Something suspected as a bug but wasn't

### From Feature Development
8. **Abandoned approach** - Design decision that was later reversed
9. **Prototype artifact** - Workaround that was replaced by proper implementation
10. **Incomplete understanding** - Early assumption corrected by later work

## Step 3: Delete Invalid Memories

For each invalid memory, delete it:

```bash
curl -s -X DELETE http://localhost:8741/memory/<MEMORY_ID>
```

## Step 4: Add Corrected Memories (Optional)

If an invalid memory should be replaced with a correct version:

```bash
curl -s -X POST http://localhost:8741/store \
  -H "Content-Type: application/json" \
  -d '{
    "type": "WORKING_SOLUTION",
    "content": "<CORRECT_LEARNING>",
    "context": "<CONTEXT>",
    "confidence": 0.95
  }'
```

## Memory Types Reference

| Type | Use For |
|------|---------|
| `GOTCHA` | Counterintuitive behaviors, traps, "watch out for this" |
| `WORKING_SOLUTION` | Commands, code, or approaches that worked |
| `PATTERN` | Recurring architectural decisions or workflows |
| `DECISION` | Explicit design choices with reasoning |
| `FAILURE` | What didn't work and why (useful to avoid repeating) |
| `PREFERENCE` | User's stated preferences |

## Examples

### Debugging Example
**Invalid**: "Reference equality vs .Equals() mismatch causes graph disconnection"
**Correct**: "Coordinate tolerance 1e-6 too small for reprojected coords; use 0.01m for EPSG:3857"

### Feature Development Example
**Invalid**: "Use polling for real-time updates in dashboard"
**Correct**: "Use WebSocket for real-time updates - polling caused excessive server load"

### Refactoring Example
**Invalid**: "ServiceLocator pattern works well for dependency injection"
**Correct**: "Constructor injection preferred - ServiceLocator made testing difficult"

## Tips

- Wait until work is complete before extracting memories
- During iterative development, early decisions often get revised
- Lower confidence scores for findings that might change
- Be specific in context field about file paths and scenarios
- Periodic sanitization prevents memory database pollution

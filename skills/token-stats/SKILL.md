---
name: token-stats
description: Show token economics comparing usage with turbo-search vs without. Demonstrates actual savings from search-first approach.
---

# /token-stats - Token Economics Dashboard

Show the token savings achieved by using search-first exploration vs blind file reading.

## Instructions

When the user invokes `/token-stats`, analyze token usage and display savings.

### 1. Gather Activity Data

```bash
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")
ACTIVITY_FILE="$REPO_ROOT/.simba/search/activity.log"

# Count files read this session
if [ -f "$ACTIVITY_FILE" ]; then
    echo "=== Session Activity ==="
    FILES_READ=$(grep -c "READ:" "$ACTIVITY_FILE" 2>/dev/null || echo "0")
    FILES_EDITED=$(grep -c "EDIT:" "$ACTIVITY_FILE" 2>/dev/null || echo "0")
    SEARCHES=$(grep -c "SEARCH:" "$ACTIVITY_FILE" 2>/dev/null || echo "0")
    echo "Files read: $FILES_READ"
    echo "Files edited: $FILES_EDITED"
    echo "Searches performed: $SEARCHES"
else
    echo "No activity log found for this session"
fi
```

### 2. Calculate Codebase Stats

```bash
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")

# Total files in codebase
TOTAL_FILES=$(rg --files "$REPO_ROOT" 2>/dev/null | wc -l | tr -d ' ')

# Estimate total tokens (rough: ~1.33 tokens per word, ~3 words per line)
TOTAL_LINES=$(rg -c '' "$REPO_ROOT" 2>/dev/null | awk -F: '{s+=$2} END {print s}')
ESTIMATED_TOTAL_TOKENS=$((TOTAL_LINES * 4 / 3))

echo ""
echo "=== Codebase Size ==="
echo "Total indexable files: $TOTAL_FILES"
echo "Total lines: $TOTAL_LINES"
echo "Estimated tokens: $ESTIMATED_TOTAL_TOKENS"
```

### 3. Query Memory for Historical Data

```bash
uv run python -m simba.search stats
```

### 4. Calculate and Display Economics

Based on the gathered data, calculate and present:

**Token Economics Model:**

| Scenario | Calculation | Typical Cost |
|----------|-------------|--------------|
| **Blind exploration** | Read 20+ files to find relevant code | ~50,000 tokens |
| **With turbo-search** | Search (50 tokens) + Read 3-5 targeted files | ~5,000 tokens |
| **Savings** | | **~90%** |

**Present this table to the user:**

```
TOKEN ECONOMICS DASHBOARD

  Codebase: [PROJECT_NAME]
  Total Files: [X] | Total Lines: [Y] | Est. Tokens: [Z]

                     THIS SESSION
  Searches performed:     [N]     (~50 tokens each)
  Files read (targeted):  [M]     (~1,000 tokens each)
  Files edited:           [K]

  ESTIMATED USAGE WITH PLUGIN:     ~[X] tokens
  ESTIMATED WITHOUT (blind read):  ~[Y] tokens

  SAVINGS: ~[Z]% ([Y-X] tokens saved)

                   HISTORICAL (ALL SESSIONS)
  Total sessions:         [N]
  Total files tracked:    [M]
  Knowledge entries:      [K]
  Facts stored:           [F]

  Cumulative savings:     ~[X] tokens
  (Based on [S] search-first explorations)
```

### 5. Token Calculation Logic

Use these estimates for calculations:

```
SEARCH_COST = 50              # tokens per qmd search
FILE_READ_COST = 1000         # avg tokens per file read
BLIND_EXPLORATION_FILES = 20  # files typically read without search
TARGETED_READ_FILES = 3       # files read with search-first approach

# With plugin
with_plugin = (SEARCHES * SEARCH_COST) + (FILES_READ * FILE_READ_COST)

# Without plugin (estimate)
without_plugin = BLIND_EXPLORATION_FILES * FILE_READ_COST

# Savings
savings_tokens = without_plugin - with_plugin
savings_percent = (savings_tokens / without_plugin) * 100
```

### 6. Pro Tips

End with actionable suggestions:

```
Pro Tips to Maximize Savings:
   - Use 'qmd search' before reading any file
   - Run /remember at session end to build context
   - Check /memory-stats for accumulated knowledge
   - The more you use it, the smarter it gets!
```

## Notes

- Estimates are based on typical Claude token encoding
- Actual savings vary based on codebase structure and task type
- Historical data requires using /remember consistently

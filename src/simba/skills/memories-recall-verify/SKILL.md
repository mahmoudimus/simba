---
name: memories-recall-verify
description: Self-correcting memory recall — when recalled memories are ambiguous or conflicting, re-query for the specific entity (or ask) before answering, and never fabricate when memory is insufficient
allowed-tools: Bash(simba *)
---

# Self-Correcting Recall

Use this when you are about to answer a question whose answer depends on stored
memory (preferences, decisions, facts, "what did we decide about X"), and the
memories you have are **ambiguous, conflicting, or incomplete**. The goal is to
ground the answer in the *right* memory — not the first plausible one — and to
admit when memory doesn't contain the answer.

## Step 1 — Recall

```bash
simba memory recall "<the user's question, as a statement>"
```

Each line is: `<id> [<TYPE>] (<similarity>) <content>`.

## Step 2 — Detect a problem

Inspect the recalled set and decide whether you can answer *directly*. You
**cannot** yet if any of these hold:

- **Ambiguous reference** — the question names a generic thing ("the API key",
  "the staging DB", "the meeting") and the memories describe **multiple distinct
  instances** of it. You must not blend facts across instances.
- **Conflicting values** — two memories give different values for the same
  attribute of the same subject.
- **Scope mismatch** — the memories are about a *different* target than the one
  asked about (right attribute, wrong entity).
- **Insufficient** — nothing recalled actually contains the asked-for value.

If none hold, answer directly from the recalled memory.

## Step 3 — Re-query (the self-correction)

For an ambiguous or scope-mismatched result, run a **narrower** recall naming
the specific entity and attribute:

```bash
simba memory recall "<specific entity> <attribute>"
```

Repeat once or twice with sharper terms. Broaden only if nothing returns:

```bash
simba memory recall --limit 8 "<broader phrasing>"
```

## Step 4 — Resolve conflicts by recency

When two memories conflict, prefer the fresher one:

- Recalled context tags the most recently created memory with
  `recency="newest"` and shows each memory's `created` date — prefer it.
- For knowledge-graph facts, the currently-valid edge is the one with no
  `valid_to`; check with `simba db facts` (it prints `occurred:` event dates).

## Step 5 — Answer or ask

- **Clear winner** → answer, citing the specific memory.
- **Still ambiguous after re-query** → ask the user to disambiguate
  ("Do you mean the *staging* or *prod* key?"). Do **not** guess.
- **Not in memory** → say so plainly ("I don't have that in memory"). Never
  fabricate a value to fill the gap.

## Verification before you answer

- The value you're about to give is attached to the **exact** entity asked about.
- You did not mix attributes/values across different entities.
- Partial or low-confidence answers are disclosed as such.

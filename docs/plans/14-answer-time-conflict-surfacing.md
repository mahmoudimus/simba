# 14 — Answer-time conflict surfacing (the measured #1 quality gap)

Branch: `feat/answer-time-conflict-surfacing` (base: `feat/borrow-subtlememory-toki`).
Status: **lever built (TDD, default-OFF, 13 tests), measured, verdict below.**

## Why
Step C (`docs/plans/13`) measured simba's clearest quality gap: on SubtleMemory's
**contradictory** slice, simba answers **0/20 even with perfect retrieval** — it
retrieves the conflicting facts but the *answerer* collapses the conflict (fabricates
one side 14/20, or abstains) instead of surfacing "needs confirmation". An **answer-time**
problem, not retrieval — and toki's write-time resolution layer structurally can't move it.

## What was built
`src/simba/memory/conflict.py` (default-OFF `memory.conflict_surfacing_enabled`,
`memory.conflict_surfacing_min_memories=2`):
- `detect_conflict(memories, query, *, llm_client)` — one LLM call: do any retrieved
  memories conflict in a way relevant to the question? Returns the named pair or None
  (fail-open).
- `surface_directive(conflict)` — a directive that **names the specific conflict** and
  instructs: surface it + state what must be confirmed, don't pick a side.
- `conflict_note(...)` — gated entry point ("" unless enabled + ≥min memories + a
  conflict is detected). Wired into `hooks/_memory_client.format_memories` (default-off
  ⇒ byte-identical output + zero latency when disabled).

## Measurements (deepseek-v4-flash answerer, deepseek-v4-pro judge, SubtleMemory)

**Probe — blanket always-on directive (n=18, 6/slice):** contradictory 0.00→0.167,
**complementary 0.167→0.00**, nuanced flat. Lesson: the answerer IS steerable, but a
*generic* directive is weak AND **harms** non-conflict slices (over-hedging) ⇒
detection-gating is mandatory; name the specific conflict.

**Gated detect-and-name lever (n=40: 20 contradictory + 20 complementary):**

| slice | base | lever | Δ | detection fire-rate |
|---|---|---|---|---|
| contradictory | 0.00 | 0.05 | +0.05 | **0.15** (3/20) |
| complementary | 0.15 | 0.15 | 0.00 | **0.00** (0/20) |

## Verdict
- **The architecture is right + safe.** Gating fired on 0/20 complementary ⇒ **zero
  harm** (fixes the blanket directive's −0.17). Detect→gate→name is the correct shape.
- **The bottleneck moved to DETECTION RECALL.** Single-pass LLM detection recognizes
  only ~15% of SubtleMemory's *deliberately latent* conflicts, so the lever rarely
  fires ⇒ contradictory only 0→1/20. When it fires, surfacing can work. The failure is
  no longer "answerer collapses" or "retrieval misses" — it's "detector doesn't notice
  the latent conflict."
- **Small honest win, near the single-pass ceiling.** Shipped **default-OFF** as a
  sound instrument (like entity-bridge / kg-ppr), not earned default-on.

## What would move it further (and the caveat)
A higher-recall detector — pairwise checks (O(k²) LLM calls, costly) or a reasoning
model — could lift fire-rate. But SubtleMemory's conflicts are *engineered* to be
subtle, so (a) gains are bounded and (b) tuning the detector to this dataset risks
benchmark-overfitting ([[eval-do-not-chase-1.0]]). Pursue only a **general** detector
improvement, measured on held-out conflicts, not a SubtleMemory-specific one. n here is
modest (20/slice); the qualitative finding (gating prevents harm; detection recall is
the wall) is unambiguous from the fire-rates.

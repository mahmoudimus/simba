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

---

## 2026-06-11 — pairwise REVERSES the detection-recall verdict (measured)

Probes (gitignored, count-depth worktree): `.simba/subtle_side_coverage.py`,
`.simba/subtle_conflict_ab.py`, `.simba/subtle_conflict_harm.py`,
`.simba/subtle_case_example.py` (worked example `0_related-3288f4bf72cf_0`).
Persona_0, flash answerer/detector, pro judge, k=10 contexts.

**Three corrections to the record first:**
1. **The bench was a NO-OP for this lever.** `simba eval bench --qa` never called
   `conflict_note` — a config-flip arm (contradictory 0.086→0.056) measured pure LLM
   noise. Fixed in this PR: `score_case` now mirrors the `format_memories` injection
   (gated, zero-cost when off), so the bench measures what ships.
2. **`bridge_recall@k = 0.0` was a metric artifact** (gold = ALL ~24-40 turns of both
   sessions, > k). The meaningful number is session-level side-coverage: **both
   conflict sides in top-10 in 50%** of contradictory cases, one side 39%, neither 11%.
   So half the slice is retrieval-bound; the other half is answer-time-addressable.
3. **"Detection recall is the wall" was a SINGLE-strategy artifact.** Pairwise
   (`conflict_detect_strategy=pairwise`, shipped in spec-15) removes it.

**Paired A/B over identical top-10 contexts (contradictory, n=36):**

| slice | n | acc OFF | acc ON | fire-rate | flagged pair spans both sessions |
|---|---|---|---|---|---|
| both sides retrieved | 18 | 0.111 | **0.944** | 0.944 | 0.944 |
| side missing | 18 | 0.056 | 0.111 | 0.444 | 0.000 |
| overall | 36 | 0.083 | **0.528** | 0.694 | 0.472 |

Zero contradictory cases broken by the directive. The worked example is fixed by it.

**Harm check on non-contradictory slices (paired; harm only possible where it fires):**
fire-rate 0.20 on complementary AND nuanced (8/40 fired vs single's 0.00) — pairwise
over-fires on engineered tensions that aren't contradictions. Of 8 fired+graded:
**1 harmed** (nuanced, right→wrong), 0 helped. Net on a full persona ≈ +16 contradictory
vs ≈ −2.6 non-contradictory cases: **strongly net positive, but NOT harm-free** —
the spec-14 "zero harm" property belonged to the single strategy's never-firing.

## Revised verdict
- Detect→gate→name was right; with pairwise the lever WORKS where co-retrieval allows
  (0.944 on the addressable slice). The remaining contradictory gap is **retrieval**
  (side-coverage), not detection or steering.
- **Stays default-OFF** — up to `conflict_detect_max_pairs`=45 detector calls per
  query is not hot-path-viable; write-time B2 (spec-15, shipped default-off) is the
  amortized path. When enabling answer-time surfacing, `pairwise` is the measured-good
  strategy; over-fire precision (0.20 on non-conflicts) is the refinement axis.
- An LLM-extract→fixed-Datalog-rules selection gate was harness-validated on the worked
  example (rules correct; weak step = value canonicalization) — parked as v2: the
  directive already saturates the addressable slice.

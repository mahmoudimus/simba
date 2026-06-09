# 15 — Write-time neuro-symbolic enrichment loop (Option B)

Branch base: `feat/eval-sota-and-memory-borrows` (PR #58). The strategic bet from the
LLM-bound analysis: the residual quality gaps (contradiction surfacing; reasoning over
retrieved evidence) are **answer-time LLM failures**, and answer-time is the worst place
to ask the LLM to do hard work (per-query, latency-bound, single-pass, no gold). simba
has a layer the answer-time LLM doesn't — the **write-time neuro-symbolic stack** (KG,
Z3, AGM, the toki resolution layer). Move the hard reasoning **off the answer-time hot
path into write-time**, where it is amortized (once per memory), batchable, latency-
insensitive (can use a stronger/slower model), and where structure already lives — then
**annotate recalled memories** so the answer-time LLM only has to *surface a handed
flag*, not *detect*.

## Crux measured (GO, bounded) — 2026-06-09
The conflict lever (`docs/plans/14`) hit a wall: answer-time single-pass detection over
top-k raw turns = **0.15** recall. Crux probe (`.simba/detect_ceiling_probe.py`,
deepseek-v4-flash, 40 SubtleMemory contradictory cases): handing the detector the **two
clean conflicting `facts` isolated** → **0.65** recall (4.3×). So the 15% was **burial +
extraction noise, not inherent undetectability**. Bounded: ceiling 0.65 (35% missed even
given clean facts) × surfacing-given-detection (~0.5–0.7, from the lever) ⇒ realistic
end-to-end contradictory ≈ **0.3–0.45 from 0.0**. A real win, not a solve.

## Two sub-levers (both validated as worthwhile by the crux)
- **(a) Selection / isolation** — comparing the conflicting *pair* (write-time: new
  memory vs each nearest neighbor) instead of all-at-once-buried-in-top-k. The bulk of
  the 0.15→0.65 gain. Belongs at write-time (amortized, O(neighbors) per store).
- **(b) Extraction quality** — the 0.65 was on *clean* post-extraction facts; raw turns
  are noisier, so real recall sits between 0.15 and 0.65. Better extraction → closer to
  the ceiling.

## Staged build (each a measured gate)
- **B1 — measure the mechanism end-to-end (cheap, answer-time pairwise).** Add a
  `pairwise` detection mode to `memory/conflict.py` (compare retrieved memories pairwise
  / new-vs-rest), measure SubtleMemory **contradictory QA** lift vs the lever's 1/20
  (and non-regression on complementary via the gate). Go/no-go for the engineering: does
  higher detection recall actually convert to answer accuracy, or does surfacing eat it?
- **B2 — the shippable architecture (write-time detection).** On store, compare a new
  memory against its nearest neighbors; persist detected conflicts as **edges / toki
  audit rows** (reuse `neuron/resolve_ops.py` + `kg/store.py`). Recall-time just **reads
  pre-computed conflicts** and annotates — *zero* answer-time detection latency. Default-off.
- **B3 — extraction density.** Improve write-time extraction (raw turns → clean facts)
  so pairwise-over-raw recall climbs toward the 0.65 ceiling. Borrow-adjacent: A-MEM
  neighbor-evolution, ReMe procedural ([[memory-systems-borrow-survey]]).

## Symbolic path (parallel, for the clean cases)
For *factual* contradictions (X lives in Boston vs Seattle), Z3/toki over extracted
triples detects **deterministically, no LLM** — already built. The LLM-pairwise path is
for the *latent* conflicts Z3 can't type. Use both; route by whether the conflict is
structurable.

## Kill criteria
- If **B1 end-to-end contradictory stays ≤0.1** (surfacing eats the detection gains), or
- if **write-time pairwise over *raw* memories can't beat 0.15**,
then B is bounded → stop, accept Option A (ride model improvements), record it next to
the conflict-lever verdict ([[answer-time-conflict-surfacing-lever]]).

## Measure on
SubtleMemory contradictory (primary) + HaluMem (forgetting/conflict) + non-regression on
complementary/nuanced (the gate) and the bundled recall sets (no latency/recall hit when
off). Honesty rules throughout ([[eval-do-not-chase-1.0]]); report flat/negative.

---

## B1 result — GO (2026-06-09)
Added a `pairwise` detection strategy (`memory.conflict_detect_strategy`, default
`single`; `conflict_detect_max_pairs=45`). End-to-end SubtleMemory **contradictory**
(n=30, k=8, deepseek-v4-flash answerer + detector, deepseek-v4-pro judge):

| arm | accuracy | detection fire-rate |
|---|---|---|
| no-lever | 0.033 | — |
| lever (single) | 0.267 | 0.267 |
| **lever (pairwise)** | **0.367** | **0.467** |

Within-run (same cases/k): **pairwise > single** (+0.10 acc, +0.20 fire), fire-rate
0.467 approaching the 0.65 clean-fact ceiling ⇒ remaining gap is extraction noise (B3).
Both ≫ the no-lever 0.033 baseline and ≫ the 0.1 kill threshold ⇒ the detect→surface
chain **converts** (better detection ⇒ better answers).

### B1 FIRMED (n=60, k=8) — verdict: proceed to B2
| arm | accuracy | fire-rate |
|---|---|---|
| no-lever | 0.033 | — |
| lever (single) | **0.267** | 0.333 |
| lever (pairwise) | **0.317** | 0.400 |

- **The headline win is the LEVER ITSELF: 0.033 → 0.267 (8×)** on contradictory. Single
  held at 0.267 across both n=30 and n=60 (not optimistic).
- **Pairwise adds a modest, real increment over single: +0.05 acc / +0.067 fire** (the
  n=30 +0.10 was a touch lucky). Pairwise is a *refinement*, not the main event.
- **This CORRECTS `docs/plans/14`**, which reported the lever at ~0.05 — that was an
  unlucky n=20 sample **at k=10**. The lever is k-sensitive: detection fire-rate climbs
  **0.15 (k=10 all-at-once) → 0.33 (k=8) → 0.40 (pairwise) → 0.65 (clean isolated pair)**.
  Fewer candidates fed to detection = less burial = higher recall.
- **B2 implication:** write-time detection (new memory vs a *few* nearest neighbors) sits
  at the low-burial end *naturally* — so it makes the lever (a) shippable (no answer-time
  latency) and (b) high-recall (few candidates) even with cheap single-pass detection;
  pairwise is an optional extra. Right-sized expectation: B2 ships ~0.27–0.32 contradictory
  (from 0.03), bounded by the 0.65 detection ceiling × surfacing.

## B2 — the shippable write-time architecture (next)
Answer-time pairwise is O(k²) LLM calls/query — fine for measurement, too slow to ship.
Move it to **write time**: on store, compare a new memory against its nearest neighbors
(O(neighbors), amortized), persist detected conflicts as edges / `neuron/resolve_ops.py`
audit rows; recall-time **reads** pre-computed conflicts and annotates (zero answer-time
detection latency); answer-time surfaces. Then B3 (extraction density → recall toward
the 0.65 ceiling).

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

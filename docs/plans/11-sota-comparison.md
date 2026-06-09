# 11 — SoTA comparison: one judge (DeepSeek-V4), same axis

Goal: replace "we think we're competitive" with a **measured** head-to-head. The
fix to the comparability trap (leaderboards graded by GPT-4o; our numbers by a
different judge) is **not** to match their old judge — it's to **re-judge everyone
on the same data with one strong judge**. DeepSeek-V4 is near-frontier, open, and
cheap → use it as the single grader for simba **and** the baselines.

> DeepSeek-V4 caveat (from its own report): V4-Pro-Max *trails* GPT-5.4/Gemini-3.1
> marginally and SoTA by ~3–6 months — it's the leading **open** model, not the top
> model. That's irrelevant for grading (near-frontier is plenty for correct/incorrect
> vs gold) and ideal for an open, reproducible judge.

## What we have (inventory, 2026-06-08)

- **Judge:** `deepseek-v4-pro` and `deepseek-v4-flash` both reachable via `llm-cli`.
  Use **v4-pro** as the grader (v4-flash for a cheap pre-pass / large N).
- **Baselines runnable locally** (source cloned in `~/src/ai/memory`, not yet
  installed): **`mem0`** (the most-cited LoCoMo baseline) and **`letta`** (MemGPT
  lineage). These we can run end-to-end → grade their answers with the same judge.
- **Published-only baselines:** Zep / Graphiti (cloud-leaning). **GPT-4o — the judge
  behind every published LoCoMo/LME number — is deprecated/unserved**, so those
  numbers are a **frozen, un-extendable axis**: we can't re-grade their predictions
  and can't calibrate a current judge against GPT-4o. Calibration bridge is dead.
  Published-only systems are **loose references only**; the sole real comparison is
  **reproduce + re-judge with v4**.
- **Fully-local judge fallback** (offline / zero-API only): **Qwen3-30B-A3B**
  (MoE, ≈GPT-4o-class, runs on Mac-MLX or the 4090). A capability *downgrade* from
  v4-pro — use only if API-free grading is required; otherwise v4-pro is stronger.
- **Datasets on hand:** `locomo10.json`, `longmemeval_oracle.json`,
  `HaluMem-Medium.jsonl`, `hotpot_dev_distractor_v1.json`. **Missing:**
  `longmemeval_s` (the hard full haystack — fetch for the real LME test).

## Plan

1. **Single judge = deepseek-v4-pro** (`judge.provider=llm-cli`, `judge.model=
   deepseek-v4-pro`). Answerer fixed across systems (deepseek-v4-pro or a pinned
   model) so only the *memory system* varies.
2. **Reproduce baselines** on LoCoMo: install `mem0` + `letta` in throwaway venvs,
   run their ingest→recall→answer, capture predictions.
3. **Grade all systems** (simba + mem0 + letta) with deepseek-v4 on identical
   (question, gold, predicted) triples → one leaderboard, one axis.
4. **Datasets:** LoCoMo (have), LongMemEval — fetch `_s` (hard) + keep oracle as
   upper bound, HaluMem (have). HotpotQA stays the multi-hop recall instrument.
5. **Published-only (Zep):** GPT-4o is gone → no calibration possible. Cite as a
   loose reference only, or reproduce it too — never a fake apples-to-apples.
6. **Report:** `BENCHMARKS.md` gains a "vs baselines (deepseek-v4 judge)" block;
   record per system + git SHA in `results.jsonl`.

## Cost / risk

- deepseek-v4 judging is cheap (API, ~cents per hundred triples). Main cost is the
  baselines' *answerer* calls when reproducing them.
- Risk: baselines may not reproduce their published setup exactly (versions, prompts)
  → note any deviation; the *same-judge* comparison is still fair even if a baseline
  underperforms its paper (we'd flag it, not hide it).
- Honesty guardrails as always: held-out splits, no tuning to the number,
  measured-negative reporting. See [[eval-do-not-chase-1.0]].

## Sequencing

A) Fetch `longmemeval_s`; wire `judge.model=deepseek-v4-pro`; re-baseline **simba**
   under the v4 judge (cheap, no new system). →
B) Stand up **mem0** on LoCoMo, grade with v4, first head-to-head. →
C) Add **letta**; add LongMemEval_s + HaluMem; (optional) Zep calibration bridge.

## Step A result (2026-06-08) — simba on the v4 axis

answerer=`deepseek-v4-flash`, judge=`deepseek-v4-pro` (single judge for everyone):
- **LoCoMo QA = 0.426** (n=122); recall@5=0.614 (unchanged, answerer-independent)
- **LongMemEval-oracle QA = 0.644** (n=180); recall@5=0.815

Lower than prior local-judge runs (LoCoMo 0.54 gpt-oss/Qwen; LME-oracle 0.79) —
**v4-pro is a stricter grader**. That's the whole point: one strict current judge,
same for every system. These are the comparable-axis baselines for Step B (mem0).

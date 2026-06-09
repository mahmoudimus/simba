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
- **Published-only baselines:** Zep / Graphiti (cloud-leaning, hard to self-host) →
  use their published numbers + a small **calibration subset** (same triples graded
  by both deepseek-v4 and GPT-4o) to place them on our axis with a known offset.
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
5. **Bridge published-only (Zep):** ~50–100-triple calibration set graded by both
   deepseek-v4 and GPT-4o → report the offset, place Zep's published number ±delta.
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

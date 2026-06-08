# 10 — HaluMem: the eval where forgetting & contradiction-resolution pay off

simba's evals (LoCoMo / LongMemEval / HotpotQA) all score **recall@k of stored
evidence on a fixed corpus** — they **reward retention and penalize forgetting**
(deleting/down-ranking anything can only lower recall@k). That's why simba's
**Phase-6 decay/dormant tier** and **Phase-7 contradiction-resolution** are shipped
but *unvalidated*: on a recall@k eval they can only look bad. We need a benchmark
with the **inverse pressure**. It exists.

## HaluMem (arXiv 2511.03506, IAAR-Shanghai / MemTensor, Nov 2025)

The first **operation-level memory-hallucination** benchmark. A "memory
hallucination" is one of four failure modes: **fabrication, error, conflict
(contradiction), omission**. Three independently-scored subtasks:

1. **Extraction** — extract correct memory points from dialogue *and resist
   injected distractors*. Metrics: Recall, **Target Precision** (penalizes
   extra/false memories), Accuracy, F1, **False-Memory-Resistance** (FMR =
   credit for *ignoring* distractor/false content).
2. **Updating** — when later dialogue contradicts/supersedes a stored fact, does
   the system overwrite the stale one? Metrics: Correct / **Hallucination Rate** /
   Omission Rate. **Keeping an outdated fact scores as a failure.**
3. **QA** — retrieve+answer without fabricating. Correct / Hallucination / Omission.

**Why this is the missing eval:** Target Precision + FMR + Hallucination-Rate +
the Updating subtask **structurally reward forgetting and contradiction-resolution**
— the exact opposite of recall@k. This is the benchmark on which Phase-6 / Phase-7
can finally show *measurable* value.

ReMe's published HaluMem-Medium: Memory Accuracy 94.06 / QA 88.78 / Memory
Integrity 67.72 (ProMem leads integrity 73.8) — LLM-judged, so simba's number will
be on its own judge; the point is the *delta* from our features, not the leaderboard.

## How it maps to simba

| HaluMem subtask/metric | simba feature it measures (that recall@k can't) |
|---|---|
| Target Precision, FMR, QA-Hallucination | **Phase-6 dormant tier** — stale memories excluded from recall → fewer false surfaces → higher precision / lower hallucination |
| Updating (overwrite-on-contradiction) | **Phase-7 contradiction-resolution** (Z3 UNSAT-core) + supersession — overwrite scores Correct; leaving it scores Omission |
| Extraction precision/FMR | simba's LLM extraction + dedup/supersession quality |

## Plan (implementer-ready)

- **P0 — fetch.** `scripts/fetch_benchmarks.sh`: add HaluMem (HF
  `IAAR-Shanghai/HaluMem`, **HaluMem-Medium** split first) into gitignored
  `.simba/benchmarks/`. **License CC-BY-NC-ND-4.0** → runtime download only, never
  vendor/redistribute a modified copy. Corpus is >1M tokens/user → **subsample
  users** (`--user-num`, mirror ReMe).
- **P1 — loader.** `src/simba/eval/benchmarks/halumem.py` — port the *pattern* (not
  the code) from `~/src/ai/memory/ReMe/benchmark/halumem/eval_reme.py`: schema is
  `persona_info`, `sessions[].dialogue`, `memory_points`, `questions[].evidence`.
  Map onto simba's loader/`Dataset` conventions where possible; HaluMem needs new
  structures for memory-points + update-events (it's not a recall@k shape).
- **P2 — three evaluators + metrics.** `MemoryExtraction`, `MemoryUpdating`,
  `MemoryQA`, emitting Target-Precision / FMR / Hallucination-Rate / Omission /
  Correct. Add these to `eval/metrics.py`. Extraction/Updating are **deterministic
  counts once the judge labels each point** → judge cost is bounded by
  #memory-points, not #queries×k.
- **P3 — judge.** Reuse the **local mlx judge** already wired on
  `feat/local-eval-llm` (Gemma) — HaluMem's judge is swappable (ReMe runs Qwen).
  No GPT-4o requirement; stays within simba's local-first rule. Cache verdicts
  (existing judge cache).
- **P4 — CLI.** `simba eval bench halumem [--user-num N] [--split medium|long]`
  (mirror locomo/longmemeval); append to `results.jsonl`; surface the new metrics in
  `BENCHMARKS.md`.
- **P5 — the measurements (the payoff).** Ablations that recall@k *cannot* show:
  - **dormant tier ON vs OFF** (`memory.dormant_filter_enabled`) → expect higher
    Target Precision / lower QA-Hallucination with dormancy ON.
  - **supersession / contradiction-resolution ON vs OFF** → expect higher Updating
    Correct, lower Omission.
  - This is the first place Phase-6/Phase-7 can earn their keep with a number.

## Tests (TDD)
- `tests/eval/benchmarks/test_halumem.py` — loader maps the JSONL schema; the three
  metrics compute correctly on a toy fixture (a fabricated memory lowers Target
  Precision; an un-overwritten contradiction scores Omission; a distractor ignored
  raises FMR). Judge mocked.

## Acceptance / why this matters
- **Win condition is different from every other spec:** not "higher recall@k," but
  a **measured improvement in precision / hallucination-rate / update-correctness
  when Phase-6/Phase-7 are ON.** If dormancy/contradiction-resolution *don't* move
  HaluMem, that's a real finding (they're cosmetic) — equally decision-grade.
- Strategic: this is simba's path to **"SOTA on a dimension nobody else measures
  well"** — verified, non-hallucinating, self-correcting memory — which is more
  defensible than the saturated recall@k race ([[eval-do-not-chase-1.0]]). It also
  completes Pillar 1 (fix the eval instrument) with the one benchmark that rewards
  what makes simba architecturally distinct.

## Caveats
- LLM judge required (not pure-offline) — but local/swappable; cache + subsample.
- CC-BY-NC-ND: internal eval only; do not redistribute the dataset.
- Same org (MemTensor) ships MemOS/MemRL/HaluMem — treat their leaderboard numbers
  as their-judge; report simba's own-judge deltas.

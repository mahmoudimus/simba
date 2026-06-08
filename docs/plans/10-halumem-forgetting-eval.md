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

---

## Build status (2026-06-07, branch `feat/halumem-eval`)

**Apparatus DONE + proven; full run blocked by local-LLM *serving* latency.**

Built + committed (TDD, ruff-clean):
- `eval/benchmarks/halumem.py` — loader + `aggregate_qa` (accuracy / hallucination_rate
  / omission_rate + boundary block). 5 tests.
- `eval/benchmarks/halumem_qa.py` — recall → answer → 3-way boundary-aware judge. 4 tests.
- `simba eval halumem` CLI + `bench.halumem_path/halumem_user_limit` + fetch wired.
- Dataset fetched: HaluMem-Medium (20 users / 33 MB).

**Pipeline proven end-to-end** (answerer → "Paris"; judge → valid `{"outcome":...}`;
unit tests green). The first run returned `graded=0, skipped=352` — NOT a HaluMem
bug; two infra issues uncovered:

1. **Silent-provider footgun (fixed):** `LlmClient.available()` returned True for the
   unsupported `mlx-vlm` provider → `complete()` silently returned `""` → all 352 Q
   skipped with no error. Fixed: `available()` validates known providers + warns
   (`fix(llm): unknown provider -> unavailable + warn`).
2. **The branch's local eval-LLM was misconfigured:** answerer `gemma-4-e4b` is a
   *VLM* checkpoint (weights prefixed `language_model.`) that `mlx_lm` can't load;
   only the 26b judge model loads. Set `llm.model_path` → the working
   `gemma-4-26b-a4b-it-4bit` (now answerer≈judge — note the self-grading caveat).

**The real blocker — serving latency (Pillar-1 gap):** simba's `mlx-lm` client spawns
`mlx_lm.generate` **per call**, reloading the model every time (~tens of seconds for
the 26b). 352 Q × 2 calls ⇒ hours. The local path does **not** escape the
[[eval-ablation-latency-trap]] with this design. **The fix is a persistent local
model server** (load once, serve many) — that's the true Pillar-1 unlock that makes
HaluMem *and* all local LLM-judged eval affordable. Until then a full run means
cloud (the 17s/call trap) or an overnight job.

**Next:** (a) build the persistent mlx server, then run HaluMem subsampled + the
Phase-6/7 ablation; OR (b) one bounded overnight/cloud run for a first baseline.
Also still TODO for the *ablation*: confirm `recall_adapter.build_retriever` applies
store-time supersession + the decay/dormant pass (else dormancy/supersession won't
actually be exercised by the eval ingest).

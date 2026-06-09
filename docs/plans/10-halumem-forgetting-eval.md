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

---

## First baseline (corrected) — 2026-06-07

**Serving blocker resolved.** The persistent `mlx-server` provider (commit
`b814f5b`) loads the model once and serves many completions — the Pillar-1 unlock.
Working config: `llm/judge.provider=mlx-server`,
`base_url=http://127.0.0.1:8082`, `model=mlx-community/Qwen3-4B-Instruct-2507-4bit`
(non-reasoning, ~0.3 s/call warm). 881 questions graded with **0 skipped**.

**A retrieval bug masked the first numbers.** The first 5-user run looked degenerate
— `accuracy 0.288 / omission 0.680 / Basic-Fact-Recall 0.037 / Boundary abstention
1.000` (the classic "answerer gets no context → abstains on everything"). Root cause
was **not** the recall stack but the eval ingest: HaluMem numbers `memory_points`
**per session** (index resets to 1 each session), so `_user_corpus` built ids as
`{uuid}_mp_{index}` — for one user with **718 points across 65 sessions** there were
only **74 unique indices**, colliding ~65× each. The gold point was masked in
`id2content`, and `_search` returns only the `id` (discarding the matched row's
content), so even a correct vector rank resolved to the wrong text. Fix (commit
`208b848`): fold the session index into the id → `{uuid}_s{si}_mp_{index}`.
Single-question instrumentation after the fix: the gold birth-date fact ranks **#0**
and is judged **correct** (was omission).

**Corrected baseline** (5 users, 881 Q, git_sha `208b848`,
answerer==judge Qwen3-4B — *self-grading caveat*):

| Category | n | accuracy | hallucination | omission |
|---|---|---|---|---|
| **OVERALL** | 881 | **0.662** | 0.085 | 0.253 |
| Memory Boundary (abstention) | 193 | 0.979 | 0.021 | — |
| Multi-hop Inference | 65 | 0.692 | 0.062 | |
| Memory Conflict | 182 | 0.648 | 0.049 | |
| Basic Fact Recall | 190 | 0.547 | 0.116 | |
| Generalization & Application | 201 | 0.547 | 0.104 | |
| **Dynamic Update** | 50 | **0.340** | **0.300** | 0.360 |

**The target this eval was built to find: Dynamic Update.** It is by far the worst
category — accuracy 0.340 and hallucination **0.300** (3–6× every other category).
Dynamic-Update questions ask for the *current* value of a fact that was explicitly
updated over time (e.g. `monthly_income 18000 → 22000`); a 0.300 hallucination rate
means simba surfaces the **stale** value and the answerer reports it. This is exactly
the failure mode **Phase-7 contradiction-resolution / supersession** is designed to
fix, and now there is a concrete number to move. (Memory Conflict at 0.049 halluc is
already healthy — intra-snapshot contradictions are handled; it's *temporal*
supersession that leaks.)

### The ablation is not yet wired (Phase-6/7 currently a no-op in eval)

Confirmed by reading the recall path: `recall_adapter._search` calls `hybrid_search`
**without `cwd`**, so `_filter_dormant` (Phase-6) never runs; and the corpus is
ingested flat via `recall_adapter.build_retriever` — no `memory_usage` rows, no
`run_decay_pass`, and no store-time supersession (Phase-7), even though HaluMem ships
the ground-truth `is_update` / `original_memories` signal. So toggling
`dormant_filter_enabled` / decay today changes nothing in this harness.

**Next experiment — oracle supersession ceiling first.** Before measuring simba's own
detection, mark the ground-truth-superseded points (`original_memories` of every
`is_update=True` point) as non-retrievable and re-run, measuring the **Dynamic
Update hallucination drop**. That establishes the *ceiling* of forgetting:
- if the ceiling is flat → forgetting can't help here, stop;
- if it moves → wire simba's real Phase-7 supersession at ingest and measure how much
  of the ceiling it captures (detection accuracy becomes the lever).

---

## Thinking-answerer + ceiling decomposition — 2026-06-07

**Setup.** Swapped the answerer to a *reasoning* model (`gpt-oss-20b-MXFP4-Q8`, MoE
3.6B-active) on a second mlx-server port, keeping the fast `Qwen3-4B-Instruct` judge
on 8083 (answerer ≠ judge). `llm/client.py` gained `_strip_reasoning()` to pull the
gpt-oss **harmony** `final` channel (dropping `analysis`) and Qwen `<think>` blocks —
verified on real output (commit `581e65e`). Result snapshots now record the
answerer + judge models (`00debfc`).

**The thinking answerer is more *honest*, not strictly better.** vs Qwen3-4B-Instruct
it abstains on conflicting / under-specified evidence instead of guessing: lower
hallucination everywhere, higher omission, a large win on Generalization & Application
(+0.184), and ~flat overall (0.66 → 0.61, within noise). The instruct model's higher
Dynamic-Update score was partly **lucky guessing** — it commits to a value where the
reasoner correctly declines because recency is unknowable from the context.

**Decisive experiment — forget vs oracle ceiling (gpt-oss, 5 users):**

| Category | baseline | forget (drop oracle-superseded) | **oracle (gold evidence)** |
|---|---|---|---|
| Dynamic Update (n=50) | acc 0.22 / hal 0.28 | acc 0.26 / hal 0.20 | **acc 0.86 / hal 0.00** |
| Memory Conflict (n=182) | acc 0.31 / hal 0.01 | acc 0.31 / hal 0.01 | acc 0.29 / hal 0.01 |

Two **different** failure modes — and neither is "retention vs forgetting":

1. **Dynamic Update is RETRIEVAL-bound.** Given the gold facts the reasoner nails it
   (**0.86 / zero hallucination**) — the answerer is not the bottleneck. Oracle
   *forgetting* helps only modestly (0.22→0.26; hal 0.28→0.20) because exact/normalized
   supersession-match catches just ~50 of ~150 stale points. **Root cause:**
   `build_answer_prompt` lists memory *content* with **no timestamps / ordering**, so
   "what is the current X?" is unanswerable from a bag of "updated A→B" events unless
   the current value is isolated. **The real lever is recency-aware retrieval /
   surfacing timestamps**, with Phase-7 forgetting a modest second-order help — NOT the
   headline it was scoped to be.
2. **Memory Conflict is NOT a memory defect.** Oracle ≈ baseline (~0.30) — perfect
   evidence doesn't move it. Instrumenting 6 questions showed two causes: (a) the local
   judge graded a bare **"No."** (correct polarity, gold "No, <elaboration>") as
   *omission* — it penalized the reasoner's terse style but not the instruct model's
   verbose one (fixed: judge prompt now credits polarity-matching Yes/No, commit
   `deb5480`); (b) gpt-oss abstains on questions embedding unverifiable dates ("on
   Sep 06, 2025") even when the core fact is present. Both are answer/grade issues, not
   retrieval/forgetting. The judge fix only moved MC +0.04 (within noise), so most MC
   omissions are genuine abstentions, not mis-grades.

**Measurement caveat.** 5-user LLM-judged HaluMem has **±0.04 run-to-run noise** and a
~0–3 % skip rate (empty answers / unjudgeable verdicts) — consistent with
[[eval-ablation-latency-trap]]. Only large gaps are trustworthy (the 0.22→0.86
Dynamic-Update oracle swing is real; ±0.04 aggregate deltas are not). Answerer ==
self-grading caveat does not apply here (separate judge model), but the judge is a
local 4B, not GPT-4 — absolute numbers are internal-only.

**Revised conclusion for spec 10.** The forgetting / Phase-7 thesis earns a *small*
real win on Dynamic-Update hallucination (≈ −0.08); the dominant, noise-robust lever
is **recency-aware retrieval** (carry timestamps into the candidate set + the answer
context). Memory Conflict is an answer-policy / judge-calibration axis, not a memory
axis. Next experiment to run: add recency/timestamps to the answer context (and/or a
recency tie-break in scoring) and re-measure Dynamic Update — that is where the 0.22→
0.86 headroom lives.

---

## Recency-instruction A/B — verdict: lever already shipped (2026-06-08)

The ceiling work pointed at recency. But the daemon's `format_memories` **already**
annotates injected memories with `created="<date>"` + a `recency="newest"` flag
(Phase-1). The open question for 0.7.0 was whether to *also* inject an explicit
recency-resolution **instruction** ("the most recent is the current truth"). The
prior gpt-oss lift (DU 0.22→0.68) conflated *adding dates* with *the instruction*,
so it didn't answer that.

**Clean A/B** (deepseek-chat answerer **and** judge — a capable non-Claude consumer;
same model both arms ⇒ judge bias is common-mode and cancels in the delta). Both
arms get the **identical** date-labelled + newest-flagged context; the only
difference is the instruction sentence:

| Category | A: dates+flag, no instruction | B: + instruction | Δ |
|---|---|---|---|
| Dynamic Update (n=50) | acc **0.720** / hal 0.220 | 0.660 / 0.300 | −0.060 |
| Memory Conflict (n=182) | acc **0.780** / hal 0.093 | 0.808 / 0.077 | +0.028 |
| **Pooled (n=232)** | acc **0.767** | 0.776 | **+0.009 (noise)** |

**Verdict:**
- The explicit instruction is a **no-op** for a capable consumer (pooled +0.009,
  inside ±0.04 noise; DU even dips). Claude ≥ deepseek, so it's a no-op there too.
- **The lever that matters — date labels + newest flag — is already in production**
  (`format_memories`). With *just* that annotation, deepseek hits 0.72 / 0.78,
  near the 0.86 oracle. The earlier gpt-oss jump was the **dates**, not the hint.
- So **recency-aware retrieval is validated and already shipped** — not new 0.7.0
  work. Don't add the instruction to `format_memories`.
- **Eval-fidelity fix applied:** `build_answer_prompt` carried the instruction
  (from the #55 recency change) that the product never injects — handing the
  answerer an extra hint the daemon doesn't. Dropped it (kept date labels +
  newest flag), so the benchmark measures what ships. The A/B shows removal is a
  no-op-to-slight-improvement.

**0.7.0 looks elsewhere** (the recency axis is closed).

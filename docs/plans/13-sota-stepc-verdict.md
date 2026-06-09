# 13 — Step C verdict: is simba state-of-the-art?

Synthesis of the full same-axis program (Step A/B in [[11-sota-comparison]] +
Step C here: letta, longmemeval_s real haystack, SubtleMemory contradiction QA).
Written to be the **opposite of hype**: it separates "architecturally competitive
+ robust + now auditable" from "measured leaderboard SoTA" — and is explicit about
what we **still cannot** claim.

**Same axis for every number below** (verified live from `.simba/config.toml`, not
`config list` which prints misleading dataclass defaults):
answerer = `llm-cli/deepseek-v4-flash`, judge = `llm-cli/deepseek-v4-pro`.

---

## 1. Same-axis leaderboard (LoCoMo QA, deepseek-v4-pro judge)

One judge, one answerer, only the *memory system* varies. Exact stack noted per row
— **the stack is load-bearing**, not a footnote.

| System | LoCoMo QA | N | Exact stack (the caveat) |
|---|---|---|---|
| **simba** | **0.426** | 122 | store-raw + hybrid (vector+BM25 RRF) retrieve; deepseek-v4-flash answerer |
| mem0-OSS | **~0.09–0.10** | 114 | OSS `mem0.Memory` + deepseek-v4-flash extraction + bge-small embed — **NOT** mem0's paper hosted+GPT-4 config |
| letta (MemGPT) | **n/a** | 0 | **can-not-reproduce** — v0.16.8 runtime DB is Postgres-only; no local-SQLite path; in-process LocalClient removed |

Published-only references (a **frozen, un-extendable axis** — every one was graded by
**GPT-4o, which is deprecated/unserved**, so we can neither re-grade their predictions
nor calibrate v4 against GPT-4o): mem0 ~66%, ReMe 86.23, Zep — **loose references only**,
never apples-to-apples with the table above.

### letta — documented can-not-reproduce (not a loss, not a win)
letta v0.16.8 **cannot** be driven library-style on local SQLite. Its runtime DB layer
(`letta/server/db.py`) unconditionally builds
`create_async_engine(convert_to_async_uri(settings.letta_pg_uri))` →
`postgresql+asyncpg://letta:letta@localhost:5432/letta`, with **no sqlite branch**.
`DatabaseChoice.SQLITE` + the `sqlite`/`sqlite-vec` extra are honored **only** by
`alembic/env.py` (migrations), never by the runtime session factory. The old in-process
`LocalClient` API is gone; the only documented paths are the `letta_client` REST SDK
against a running `letta server` (needs Postgres+pgvector via `compose.yaml`/`LETTA_PG_URI`)
or the hosted Letta API (`LETTA_API_KEY`). Both violate the strict no-Postgres/no-Docker
bound → **reproduced=false, stopped per bound**. letta joins the published-only bucket
alongside Zep. (Positive note for a future Postgres-backed attempt: letta has a native
`deepseek` provider, and the mem0 v4 answer+grade harness at
`memory/mem0/.locomo_v4_run/` is directly reusable for the second half.)

**What the leaderboard does and does not say:** simba > mem0-OSS by ~0.33 on identical
inputs — but this is **architecture robustness to a substituted LLM**, not a ceiling
comparison. mem0's collapse is its **store-time fact-extraction** mangling LoCoMo temporal
facts off GPT-4 (hallucinated dates, summarized-away specifics). simba stores raw memories
and retrieves, so it is *insensitive* to the answerer/extractor swap. To compare *ceilings*
we'd need mem0's real GPT-4 config — impractical (GPT-4o deprecated, hosted needs a paid key).

---

## 2. Real-haystack longmemeval_s (the hard test, not the oracle upper bound)

Ran simba on the **full distractor haystack** longmemeval_s (~498 turns/question median,
~17.6k turns over the subset), **N=30** stratified 5-per-category across all 6 question
types. Wall-clock ~19.5 min, warm caches, same axis.

| Metric | longmemeval_s (N=30) | oracle ref (prior) | drop |
|---|---|---|---|
| recall@5 | 0.776 | 0.815 | **-0.039** |
| QA accuracy | 0.567 | 0.644 | **-0.077** |
| recall@10 | 0.882 | — | — |
| recall@1 | 0.384 | — | — |
| mrr | 0.711 | — | — |

**Read:** the distractor haystack costs only ~4 pts recall@5 and ~8 pts QA vs the oracle
upper bound — a **modest, directionally-real drop, NOT a collapse**. simba's retrieval
holds up against hundreds of distractor turns. Hard buckets: single-session-preference
(QA 0.2, r@5 0.53) and multi-session (QA 0.4, r@5 0.61); knowledge-update is
retrieval-OK / answer-hard (r@5 0.93, QA 0.4).

**Honesty caveats (do not over-read the delta):** (1) **abstention was not measured** —
all 5 `_abs` questions have 0 has_answer turns so the loader drops them; reported
abstention.n=0 means "zero cases scored," not "0% accuracy." (2) N=30 is a **fresh
stratified subsample**, NOT the same questions as the oracle baseline, so the oracle→_s
delta carries subsample noise at 5/category — treat magnitudes as approximate, not exact
same-question diffs.

---

## 3. SubtleMemory — is conflict-surfacing a strength or a weakness?

**Plainly: a weakness, and a CLEAN NEGATIVE — at the answer layer, not the retrieval layer.**
This firms the directional gate at **n=20** contradictory cases (was n=2–6).

| Slice | N | system acc | oracle (perfect gold-id) acc |
|---|---|---|---|
| **contradictory** | **20** | **0.000** | **0.050** |
| complementary | 4 | 0.25 | 0.00 |
| nuanced | 4 | 0.25 | 0.00 |

system-vs-oracle gap on the contradictory headline = **+0.050** (oracle barely better →
**essentially flat**). **Perfect retrieval does not fix it.** The contradiction is destroyed
at **answer generation**, not recall.

**Failure classification (oracle path, the cleanest read since retrieval is perfect, n=20):**
surfaced-correctly = 1; **COLLAPSE = 14** (fabricated-one-side 8 + needs-confirm-miss 6);
blanket-abstained = 5. The dominant mode is **collapse** (confidently picks one side / fills
the form on the decisive field), **not** abstention. Judge-strictness checked (the HaluMem
lesson) — the judge is **not** over-strict: the one "surfaced" case is a genuine true positive,
the failures genuinely collapse (gold "CLARIFY opposite winter preferences" → model "HOLD:
Quebec City Winter Pass" — picked a specific option despite the word HOLD), and blanket
"I don't know" is correctly failed (gold wanted the specific clarifying question).

**Lever implication:** a toki-style **resolution/retrieval** layer cannot move this — the
bottleneck is the deepseek answerer collapsing subtle latent contradictions. The only lever
that could help acts at **answer-generation time** (force conflict detection / refusal-with-
clarification), not at retrieval. (Contrast slices n=4 are noise-level; not the headline.)

---

## 4. The verdict: is simba demonstrated SoTA?

**No — not as a measured leaderboard claim. Yes — as an architecturally competitive,
robust, now-auditable system on the same axis.** Keep these strictly separate.

**What IS demonstrated (measured, same axis):**
- **Robust retrieval under a substituted (non-frontier) LLM.** With the *same* deepseek-v4
  stack, simba (0.426 LoCoMo) holds the facts mem0-OSS (~0.09) loses — because store-raw +
  hybrid retrieve is insensitive to the answerer swap while extract-at-store is only as good
  as its extraction LLM. This is a real, defensible **architecture** finding.
- **Robust to a real distractor haystack.** longmemeval_s costs only ~4 pts r@5 / ~8 pts QA
  vs the oracle upper bound — no collapse against hundreds of distractor turns.
- **Auditable.** Every number is on one open, reproducible judge (deepseek-v4-pro), one
  answerer, with the exact stack recorded per system — the comparability trap (mixed judges)
  is closed for the systems we can actually run.

**What is NOT demonstrated (we cannot claim it — say so):**
- **Not a leaderboard SoTA win.** The only same-axis head-to-head we completed is vs
  **mem0-OSS** (a deliberately-handicapped non-paper config) and a **can-not-reproduce**
  letta. That is a 1-of-N comparison against a weakened baseline — **not** a podium.
- **No comparison to published SoTA exists, and the gap is structural, not lazy.** mem0 ~66%,
  ReMe 86.23, Zep were all **GPT-4o-judged**; GPT-4o is **deprecated/unserved**, so their
  predictions can't be re-graded and v4 can't be calibrated against GPT-4o. The bridge to the
  published axis is **dead** — those numbers are a loose reference forever, not a target we
  can honestly beat or lose to.
- **Conflict-surfacing is a measured weakness**, not a strength: contradictory QA 0/20, and
  perfect retrieval doesn't fix it (answer-layer collapse).
- **Ceilings unknown.** mem0's real GPT-4 config and letta's Postgres-backed run were both
  out of bound — we measured *floors under a shared stack*, not *ceilings*.

**One-line honest read:** simba is **architecturally ahead of the OSS baselines we could run,
robust to weaker LLMs and to real distractor haystacks, and fully auditable on a single open
judge — but it is not a demonstrated leaderboard SoTA, the path to the published GPT-4o-judged
SoTA is closed by deprecation, and subtle-contradiction surfacing is a measured answer-layer
weakness, not a win.**

---

## Appendix — exact commands (Step C)

```bash
# config verification (same axis)
.venv/bin/python -m simba config get llm.provider   # llm-cli
.venv/bin/python -m simba config get llm.model      # deepseek-v4-flash
.venv/bin/python -m simba config get judge.provider # llm-cli
.venv/bin/python -m simba config get judge.model    # deepseek-v4-pro

# longmemeval_s, real haystack, N=30 stratified
.venv/bin/python -m simba eval bench longmemeval \
  --path .simba/benchmarks/lme_s_strat.json --qa --abstention --json

# SubtleMemory contradiction QA, n=20 contradictory + 4+4 contrast
.venv/bin/python .simba/subtle_contra.py --contra 20 --other 4 --personas 6 --k 10 \
  --out .simba/subtle_contra_out.json

# letta can-not-reproduce evidence (Postgres-only runtime)
cd ~/src/ai/memory/letta && rg sqlite letta/server/db.py   # (no runtime sqlite branch)
```

Cross-refs: [[11-sota-comparison]] (Step A/B), [[eval-do-not-chase-1.0]],
[[eval-judge-one-server-default]], [[halumem-is-the-forgetting-eval]].

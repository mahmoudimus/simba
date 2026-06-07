# simba — session handoff (2026-06-07)

A complete handoff for continuing the **memory/recall + neuro-symbolic** work on
simba. Read this top-to-bottom, then use the **Handoff prompt** at the very end to
boot a fresh context.

---

## 0. Current state (facts)

- **Branch:** `main` @ `84dd1c4`. Clean working tree (modulo gitignored
  `.simba/`).
- **Version:** `0.5.0` (pyproject + `uv.lock`).
- **Released to PyPI** (`simba-ai`) + GitHub Releases: `0.4.0` and `0.5.0` (both
  this session), atop `0.3.0`.
- **Live recall config (the product defaults):**
  - `memory.embedding_model = bge-large-en-v1.5`, `embedding_dims = 1024`
    (was nomic-embed-text/768 — **0.5.0 breaking change**, needs `simba memory reembed`)
  - `memory.rrf_k = 20` (was 60)
  - `memory.llm_rerank_enabled = True` (default-on, non-blocking on the daemon)
  - hybrid recall (vector + BM25 RRF), intent-aware floors (0.28 broad / 0.35
    precise), composite scoring on, Phase-6 decay on.
- **No open PRs.** All work merged.

---

## 1. What we achieved this session

In order:

1. **Authored 5 implementer-ready specs** (`docs/plans/01`–`05`) for the remaining
   roadmap, then **implemented all five in parallel** (5 worktrees / agents) and
   **integrated them** into `main` (PRs #41–#45): the eval program, configurable
   judge + baselines, HyDE+IRCoT, decay/forgetting (Phase 6), reflection (Phase 5)
   + neuro-symbolic distillation (Phase 7) + ops hardening. ~1330 tests green.
2. **Ran the benchmark program** and committed the first real `BENCHMARKS.md`
   baseline (PR #46).
3. **Launched the recall-excellence program** (`docs/plans/07`) and landed two
   **measured recall wins**: `rrf_k 60→20` (#49) and the **bge-large embedder**
   default (#50, breaking).
4. **Shipped two releases**: `0.4.0` (non-breaking: eval program + Phases 4–7 +
   rrf_k) and `0.5.0` (breaking: bge default).
5. **Captured a dedicated multi-hop plan** (`docs/plans/06`) — evidence-gated,
   lead = retrieval-time GraphRAG.

**Recall scoreboard (stacked, this session):**

| bench | r@5 before → after | r@10 | notable |
|---|---|---|---|
| LoCoMo (1977 Q) | **0.573 → 0.614** | 0.682 → 0.724 | open-domain 0.211→0.314, multi-hop 0.305→0.336, single-hop up |
| LongMemEval oracle (470 Q) | **0.780 → 0.814** | 0.900 → 0.911 | multi-session 0.667→0.687 |

The weak axes (multi-hop, open-domain) moved most; single-hop never regressed.

---

## 2. Repo map — every markdown file & plan

**Implementation specs (`docs/plans/`, tracked — `docs/` is otherwise gitignored):**
- [`README.md`](README.md) — index of all plans + the standing discipline.
- [`01-eval-bench-infra.md`](01-eval-bench-infra.md) — `simba eval bench` /
  `leaderboard`, results store, CI smoke. **SHIPPED** (#41).
- [`02-judge-baselines.md`](02-judge-baselines.md) — separate `judge` section,
  baselines, abstention, latency. **SHIPPED** (#43).
- [`03-hyde-ircot.md`](03-hyde-ircot.md) — LLM HyDE + answer-time IRCoT.
  **SHIPPED** (#44); IRCoT is **eval-only** (not productionized).
- [`04-decay-forgetting.md`](04-decay-forgetting.md) — Phase 6 decay/feedback.
  **SHIPPED** (#42), default-on, but the **usage/feedback eval fixture is unrun**.
- [`05-reflection-neurosymbolic-ops.md`](05-reflection-neurosymbolic-ops.md) —
  Phase 5 reflection + Phase 7 neuro-symbolic + ops. **SHIPPED** (#45); Phase 7 is
  **scaffolded + unit-tested but empirically unproven**.
- [`06-multihop.md`](06-multihop.md) — **NOT STARTED.** Evidence-gated multi-hop
  phase; decision gate filled → **lead Track B** (retrieval-time GraphRAG: PPR +
  community detection over `kg_edges`). Track A (IRCoT productionization) deferred;
  Track C = Phase 7.
- [`07-recall-excellence.md`](07-recall-excellence.md) — **IN PROGRESS.** The
  5-pillar program. Pillar 3 (embedder) + Pillar 5 (fusion) **done**; contains the
  full sweep + bake-off result tables.

**Repo-root docs (tracked):**
- [`BENCHMARKS.md`](../../BENCHMARKS.md) — generated leaderboard (current = bge +
  rrf_k_20: LoCoMo r@5 0.614, LongMemEval 0.814). Regenerate via
  `uv run simba eval leaderboard`. Carries a methodology-caveats footer.
- [`CHANGELOG.md`](../../CHANGELOG.md) — 0.5.0 / 0.4.0 / 0.3.0 / … (Keep-a-Changelog).
- [`roadmap.md`](../../roadmap.md) — refreshed with a v0.4.0 status block at the
  top; historical notes below it. **Multi-hop = active frontier.** (Note: the
  body below the status block predates this session and is partly stale.)
- [`README.md`](../../README.md) — project README (renders on PyPI).
- [`CLAUDE.md`](../../CLAUDE.md) + [`.claude/rules/CORE_INSTRUCTIONS.md`](../../.claude/rules/CORE_INSTRUCTIONS.md)
  — project rules (pure Python, append-only storage, all config via `simba config`,
  end responses with `[✓ rules]`). CORE_INSTRUCTIONS is force-added despite
  `.claude` being gitignored.

**Personal memory (NOT in git — `~/.claude/projects/-Users-mahmoud-src-ai-simba/memory/`):**
`MEMORY.md` is the index. Key entries to read on pickup:
- `roadmap-specs-merged.md` — the running state-of-the-roadmap (most important).
- `bge-large-biggest-recall-lever.md` — the embedder win + the saturated-eval lesson.
- `eval-ablation-latency-trap.md` — why per-query cloud ablations are impractical here.
- `eval-do-not-chase-1.0.md` — never tune to saturate; measure deltas on held-out data.
- `multihop-is-reasoning-not-retrieval.md`, `reranker-is-the-multihop-win.md`,
  `kg-into-recall-cooccurrence-negative.md`, `graph-lib-eval-borrow-not-vendor.md`
  — the hard-won multi-hop lessons that gate Track B.
- `simba-release-process.md` — the release pipeline + the `on:release` gotcha.
- Operational: `gh-github-token-shadows-mahmoudimus.md`,
  `git-push-https-via-gh-when-ssh-agent-empty.md`, `zsh-git-commit-backtick-trap.md`,
  `rg-is-not-grep-flag-usage.md`.

---

## 3. Testing — what & how

- **Unit/integration:** `uv run pytest` (~1330+ tests). Real LanceDB on `tmp_path`
  fixtures (no mock-DB classes); `httpx.AsyncClient` for routes; fakes/monkeypatch
  for the embedder + LLM. **TDD throughout** (RED → GREEN): new modules were
  test-first (e.g. `tests/memory/test_dim_guard.py`).
- **Lint/format:** `uv run ruff check src/ tests/` + `uv run ruff format` (88 cols,
  pathlib not os.path, TYPE_CHECKING for annotation-only imports).
- **Recall benchmarks (the product metric):**
  `uv run simba eval bench locomo|longmemeval [--qa] [--per N] [--json] [--baseline]`
  → recall@k (+ optional LLM-judge QA), appended to `.simba/eval/results.jsonl`;
  `uv run simba eval leaderboard` renders `BENCHMARKS.md`.
  - **Local & fast** for recall@k (embeddings cached by model id; no cloud).
  - QA layer uses the **cloud** `llm-cli` → DeepSeek (answerer `deepseek-v4-flash`,
    judge `deepseek-reasoner`).
- **How levers were validated:** measured-delta vs the committed baseline **on two
  datasets** (tune on LoCoMo, confirm neutral-or-better on LongMemEval — cross-
  dataset generalization beats single-split). Every win re-confirmed before
  flipping a default.
- **Datasets:** `scripts/fetch_benchmarks.sh` → `.simba/benchmarks/` (gitignored).
  LoCoMo from GitHub; LongMemEval from HF — **files are extensionless**
  (`.../resolve/main/longmemeval_oracle`, no `.json` — the script's `.json` URL
  404s; override `LME_ORACLE_URL`).

---

## 4. Decisions we MADE (with rationale)

- **Implemented the 5 specs in parallel via worktrees**, then integrated with real
  merge commits resolving conflicts on shared files (`__main__.py`, `judge.py`,
  `scheduler.py`, `config.py`). Faster than sequential; conflicts were tractable.
- **Eval scripts deleted, folded into `simba eval bench`** — `run_qa.py` /
  `run_longmemeval.py` flags (`--baseline`/`--abstention`/`--cache`/separate judge)
  now live on the CLI (resolution of the #41↔#43 design clash).
- **`_maybe_decay` made fail-open** (matching its scheduler siblings) — fixed an
  order-dependent test + a real robustness gap.
- **rrf_k 60→20** — measured win, non-breaking, shipped in 0.4.0.
- **Default embedder → bge-large-en-v1.5** — biggest measured lever; shipped as the
  **breaking** 0.5.0 with a dim-mismatch guard (`vector_db.check_embedding_dim`).
- **Two releases, not one** — 0.4.0 (non-breaking) then 0.5.0 (breaking bge), so the
  reembed migration is clearly version-gated.
- **Multi-hop lead = Track B** (retrieval-time GraphRAG), per the (inconclusive)
  IRCoT ablation + the cost reality.
- **Judge `deepseek-r1` → `deepseek-reasoner`** locally (r1 isn't a valid model on
  this box).

## 5. Decisions we did NOT make / deferred

- **IRCoT productionization (Track A):** left **eval-only**. Open product question:
  does simba *answer* or only *retrieve*? (Recommendation in `06`: iterative-recall
  assist, not answering.) Needs a scaled (≥30 multi-hop) run to justify.
- **Full lever ablations** (reranker / HyDE on recall): **not run** — ~17s/call
  cloud latency makes per-query ablations a 6–8 h job. The wiring exists so they
  *can* be run (overnight / faster local model).
- **Full `longmemeval_s`** haystack: not run (only the oracle = upper bound).
- **Nomic-Q8** as a non-breaking middle option: rejected in favor of bge.
- **Phase 7 validation:** scaffolded, not empirically validated.
- **Internal eval builder refinement:** known-saturated, not yet fixed.

## 6. What worked WELL

- **The measurement flywheel.** Once `BENCHMARKS.md` + recall@k were cheap and
  local, levers became measured deltas, not guesses. rrf_k and bge were both found
  + confirmed this way.
- **Cross-dataset confirmation** caught nothing bad and gave confidence (bge won on
  both; rrf_k won on one, neutral on the other).
- **Parallel worktree implementation** of the 5 specs → fast, and the integration
  conflicts were all mechanical/tractable.
- **TDD + the dim-mismatch guard** made the breaking embedder change safe.

## 7. What needs improvement / DIDN'T work well

- **Cloud LLM latency (~17s/call)** is the dominant constraint. It made QA/ablation
  iteration painful and a full reranker recall ablation infeasible interactively.
  → **Add a fast local eval LLM (mlx-lm)** (Pillar 1).
- **Small-n LLM-judged ablations are noise.** The IRCoT probe at n=8 was
  inconclusive (single-hop drifted ±1 case despite IRCoT not touching it). → ≥30
  per category, or prefer recall@k.
- **The internal real-corpus eval saturates** (recall@1=1.0) — it can't
  discriminate, and it produced the wrong "nomic ≈ Qwen3" verdict that the
  discriminating external eval overturned. → Pillar 1 fix.
- **Config-shadowing trap:** `simba config set` writing defaults back into
  `.simba/config.toml` *shadows* later dataclass-default changes (bit us twice with
  rrf_k and embed defaults). When changing a default, **remove the override from the
  config file**, don't just rely on the dataclass.
- **`BENCHMARKS.md` leaderboard delta** is just "latest-two-by-group" — noisy after
  a config sweep appends many runs. The committed numbers (latest) are right; the
  delta column can be misleading.
- **C1 (KG co-occurrence into recall) was a negative** — recorded so we don't repeat
  it; Track B must be PPR/community, not naive co-occurrence.

---

## 8. What's LEFT / roadmap (priority order)

The cheap fusion wins are harvested. Remaining program (`docs/plans/07` pillars):

1. **Pillar 2 — Track B: retrieval-time GraphRAG** (`docs/plans/06`). PPR +
   community detection over `kg_edges`, fold into candidates before rescore;
   **local + recall@k-measurable** (no cloud latency). The multi-hop swing.
   Includes the throwaway-KG-in-bench harness (B4) so it's measurable on corpora
   with no KG. **← recommended next.**
2. **Pillar 1 — fix the eval instrument.** (a) harder internal query-gen so the
   real-corpus eval discriminates; (b) a **fast local eval LLM** (mlx-lm) to kill
   the 17s/call trap; (c) run **full `longmemeval_s`**.
3. **Full lever ablations** (reranker/HyDE/IRCoT) once #2 makes them cheap.
4. **Pillar 4 — prove the feedback flywheel** (Phase 6): run its usage/feedback
   eval fixture; earn `decay`/feedback default-on with evidence.
5. **Track A / IRCoT productionization** (only if a scaled IRCoT run wins).
6. **Phase 7 validation** — contradiction-injection fixture + proof-carrying recall
   coverage + KG density. The long-game differentiator ("provable from F under R").
7. **Pillar 5 continuous** — more cheap fusion sweeps (intent floors, scoring
   weights) on the bge baseline as time allows.
8. **Housekeeping:** finish refreshing `roadmap.md` body (the section below the
   status block is partly stale).

---

## 9. Where to pick up next (concrete)

**Start Pillar 2 / Track B** (`docs/plans/06-multihop.md`, section "Track B"):
1. `git checkout main && git pull` (at `84dd1c4` / v0.5.0).
2. `uv sync`; confirm baseline: `uv run simba eval bench locomo` → r@5 ≈ 0.614.
   (Datasets: `bash scripts/fetch_benchmarks.sh` if `.simba/benchmarks/` is empty;
   fix LongMemEval URL — extensionless.)
3. New branch `feat/graphrag-recall`. TDD, in order:
   - `src/simba/kg/community.py` — label propagation (~40 LOC, pure Python).
   - `src/simba/kg/ppr.py` — personalized PageRank over `kg_edges`.
   - fold PPR-ranked neighbor memory-ids into candidates in `hybrid.py` after
     `rrf_fuse`, before `composite_rescore`; config `kg_ppr_enabled` etc.
   - `recall_adapter.build_retriever`: optional **throwaway KG** from the corpus
     (regex extractor) so it's measurable on LoCoMo/LongMemEval.
4. Measure recall@k delta vs the 0.614/0.814 baseline (esp. **multi-hop**), on both
   datasets. Default-off until the delta earns default-on. If the throwaway KG is
   too sparse to help, **that is the finding** (→ invest in extraction density).

**Operational gotchas** (see memories): use `env -u GITHUB_TOKEN gh auth switch
--user mahmoudimus` before any `gh`/push; `git -m` backticks run as commands in zsh
(use `-F file`); never `rg -rln` (`-r` is `--replace` in rg). Releases: tag `v*` →
`release.yml`; then **manually** `gh workflow run deploy.yml --ref vX.Y.Z` (a
token-created Release does NOT trigger `on:release`); `vars.PUBLISH_TO_PYPI=true`.

---

## 10. Handoff prompt (paste into a fresh context)

> You're continuing work on **simba** (`/Users/mahmoud/src/ai/simba`), a pure-Python
> local-first memory + neuro-symbolic plugin for Claude Code / Codex. Read
> `docs/plans/HANDOFF.md` first, then `docs/plans/README.md`, `docs/plans/06-multihop.md`,
> `docs/plans/07-recall-excellence.md`, `BENCHMARKS.md`, and `roadmap.md`. Also read
> the personal-memory index at
> `~/.claude/projects/-Users-mahmoud-src-ai-simba/memory/MEMORY.md` and the entries
> `roadmap-specs-merged`, `bge-large-biggest-recall-lever`, `eval-ablation-latency-trap`,
> `eval-do-not-chase-1.0`, `multihop-is-reasoning-not-retrieval`.
>
> State: `main` @ v0.5.0; recall defaults = bge-large-en-v1.5 (1024-d) + rrf_k=20 +
> reranker on. LoCoMo recall@5 0.614, LongMemEval 0.814. Phases 4–7 + the eval
> program are shipped; recall-excellence Pillars 3 & 5 are done.
>
> **Your task: execute Pillar 2 / Track B — retrieval-time GraphRAG** (PPR +
> community detection over `kg_edges`), per `docs/plans/06-multihop.md` "Track B".
> It's the multi-hop swing and is local + recall@k-measurable (avoid the
> ~17s/call cloud-LLM ablation trap). TDD every module; measure the recall@k delta
> (esp. multi-hop) vs the 0.614/0.814 baseline on BOTH LoCoMo and LongMemEval;
> default-off until the delta earns default-on; never tune to saturate
> (`eval-do-not-chase-1.0`). Do NOT repeat C1's naive co-occurrence
> (`kg-into-recall-cooccurrence-negative`). Build a throwaway KG from the bench
> corpus so it's measurable; if the KG is too sparse to help, that's the finding →
> pivot to extraction density. All config via `simba config` (@configurable);
> append-only storage; ruff-clean; end every response with `[✓ rules]`.
>
> Confirm you've read the handoff + memories, then propose the Track B build order
> before writing code.

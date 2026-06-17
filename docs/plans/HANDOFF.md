# simba — session handoff (2026-06-17)

Handoff for continuing simba after the **specs 25–28 + 0.11.0 release** session.
Read top-to-bottom, then use the **Handoff prompt** at the very end to boot a
fresh context.

---

## 0. Current state (facts)

- **Branch:** `main`. Clean tree (modulo gitignored `.simba/`). **No open PRs** —
  all merged.
- **Version:** `0.11.0` (pyproject + `uv.lock`). **Released to PyPI (`simba-ai`)
  + GitHub Release `v0.11.0`** this session (verified live on PyPI). Prior
  0.8.0–0.10.0 earlier.
- **Every new lever this release defaults OFF** → with defaults, hook output is
  byte-identical to 0.10.0.
- **Dogfood config** (this repo's `.simba/config.toml`, **gitignored / local-only**,
  so CI is unaffected): `[hooks] redirect_mode="rewrite"`, `pitfall_gate_enabled=true`,
  `engagement_marker_enabled=true`, `reasoning_verify_enabled=true`. Local LLM
  provider + judge wired via `llm-cli`.
- **⚠️ A long-running daemon predating specs 26/27/28 must be restarted**
  (`simba server`) to load the new code (daemon-served paths: `/hook/context`,
  `/hook/message_end`, `project_scopes` recall). Subprocess hooks (Claude/Codex)
  already use the new code via `uv run -m simba.hooks.<name>`.

## 1. What we shipped this session (all on `main`, all default-OFF)

| PR | What | Config lever |
|---|---|---|
| #79 | **`memory.max_content_length` = single source of truth** for the content cap. `simba.memory.config.resolve_max_content_length()` drives enforcement + every "keep content under N" guidance prompt + the truncations (`rules_cli`, `post_tool_use` auto-rule, `sync/indexer`). | `memory.max_content_length` (default 200) |
| #80 | **spec 25** — conditional guardian `SIMBA:core` re-injection (skip when prior response had `[✓ rules]`; inject on first-prompt/post-compaction/decay; fail-open). Per-session flag: `guardian/signal_flag.py`. | `hooks.guardian_signal_gated` |
| #81 | **CI hardening** — job `timeout-minutes: 20` + per-test `pytest-timeout --timeout=120 --timeout-method=thread` in `scripts/checks.sh`. | — |
| #82 | **spec 26** — hierarchical (ancestor-prefix) project recall (`vector_db.search_memories` + `fts.search` scope-membership; client computes chain via `find_repo_root`; `projectPath` normalized on store). **+ spec 28** — intent-primed doctrine (`doctrine/` cosine matcher, no LLM) + mandated `simba preflight` gate (`PreToolUse` blocks a mutating tool with no preflight this turn; `guardian/preflight_flag.py`). | `memory.hierarchical_recall` (+`_include_global`); `hooks.intent_priming_enabled` / `preflight_mandate_enabled` / `_risk_only` |
| #83/#84 | **spec 27** — reasoning-layer verification. Tier 1: simba-emitted `🦁☑` ledger at `UserPromptSubmit`, echo-verified at `Stop` (`hooks/engagement.py`, `guardian/engagement_flag.py`). `Stop`/`SubagentStop` (newly wired) doctrine-verify → `block_reason` → `decision:block` (`hooks/reasoning_verify.py`, `subagent_stop.py`). Tier 2 (pi): `context` re-inject + `message_end` annotate (`hooks/context.py`, `message_end.py`, `pi/extension/simba.ts`). #84 = review fixes (capture-on-block, `Array.isArray` guard, test isolation). | `hooks.engagement_marker_enabled` / `reasoning_verify_enabled` |
| #85 | **Global no-real-GGUF-load guard** — autouse fixture in `tests/conftest.py` (reranker accessors raise → fail-open; `gguf`-marked tests exempt) so CI never reaches Hugging Face. | — |
| #86 | **0.11.0 release** (CHANGELOG + version + uv.lock). | — |

**Local (gitignored) doctrine gates:** per-project guards (a generated-file
hand-edit guard, a docker-test-runner redirect rewrite, hierarchical recall + a
one-time fact re-tag to the repo-root scope) were applied to two of the user's
**private** projects via their gitignored local `.simba/` config — those live
outside this repo and are intentionally not detailed here.

## 2. Tests & how we ran them

- **Full suite:** `uv run --no-sync pytest -q --timeout=120 --timeout-method=thread`
  → **2124 passed, 3 skipped** (clingo optional). ruff: `uv run --no-sync ruff check src/ tests/`.
- **Characterization** pins byte-identical Claude/Codex hook output with every lever
  OFF (the load-bearing invariant for all hook changes).
- **Hierarchical recall MEASURED** (A/B on the production stack, recall@k OFF vs ON,
  gold at child + ancestor/global distractors): throwaway harness under `/tmp`
  (LoCoMo full-query + a bounded longmemeval_s replicate). Result: **situational** —
  LoCoMo ~9:1 ratio ≈ −0.7pp, longmemeval_s ~15:1 −4 to −9pp recall@k; inheritance
  works (0 → ~0.74). Stays default-OFF. (See memory: `hierarchical-recall-dilution-measured`.)
- **CI per PR:** watched, then **explicitly verified the check conclusion**
  (`gh pr checks <N>` = pass / `mergeStateStatus` CLEAN) before merge — see Decisions.

## 3. Decisions made

- **Content cap:** don't raise the default; make `max_content_length` the single
  source of truth so all guidance hydrates from it (pivoted from "raise to 750").
- **Hierarchical recall:** measured → situational lever, **stays default-OFF**
  globally; enabled per-project where the ancestor:child noise ratio is low.
- **One private project's data:** re-tagged its curated facts up to the repo-root
  scope (option A); left the bulk under a legacy checkout path as-is.
- **A per-project PR-review gate: DEFERRED** → pursue **affordance over prohibition**
  (a `simba pr-review`-style skill primed via intent) rather than a redirect-deny;
  revisit once spec-27/28 machinery exists (noted in spec 28).
- **CORE rules graduate OUT when gated** — added as a CORE rule in `CORE_INSTRUCTIONS.md`.
- **Release flow:** branch → PR → verify-conclusion → merge → tag `v*` → `release.yml`
  cuts the Release → **manual** `gh workflow run deploy.yml --ref v<ver>` for PyPI.

## 4. Decisions NOT made / open questions

- **pi Tier-2 (spec 27 M2):** the `context` re-injection appends an **unregistered
  custom-role** message — pi's `convertToLlm` may DROP it, making that tier a silent
  no-op. **UNVERIFIED** against the pi runtime; flagged inline in `simba.ts`. pi-only +
  default-OFF, so staged-not-live.
- **Re-homing one project's legacy-path facts** onto its current scope (left as-is).
- **Graduating any spec-25/27/28 lever** to default-ON — all unmeasured; would need an
  A/B like spec 26's.

## 5. What worked / needs improvement / didn't work

- **Worked:** subagent-driven spec implementation (one implementer per spec, isolated
  worktree); the **default-OFF + byte-identical** discipline; the **CI timeout guard**
  (caught a real ~2h pytest hang → fast fail); **verify-conclusion-before-merge** (caught
  #85's ruff E501); dogfooding spec 27 live (the `🦁☑` marker fires every turn).
- **Needs improvement:** **`gh pr checks --watch` exit code ≠ check conclusion** — it
  exited 0 on a FAILED check (merged #83 on a flaky red). ALWAYS verify conclusion
  explicitly (memory: `gh-pr-checks-watch-exit-vs-conclusion`). Tests read the **ambient
  `.simba/config.toml`** (the dogfood levers broke 3 "off by default" tests) — fixed by
  promoting the GGUF guard to global + pinning the 3 to `HooksConfig()`, but the pattern
  recurs whenever a dogfood lever flips on. The `🦁☑` marker reports **`top 0.00`** on
  task-notification turns (low-signal recall) — decide if it should show `idle` there.
- **Didn't work (then fixed):** #83 merged on a flaky **HF-429** (a test reached the real
  bge-reranker) → fixed by the global no-HF guard (#85). The combined ruff+suite log hid
  the E501 (read only the suite tail) → #85 needed a follow-up lint fix.

## 6. What's left / roadmap

- **Activate spec 27 fully:** restart the daemon; **verify the pi `context` tier (M2)**
  against the pi runtime (does `convertToLlm` keep custom-role messages? if not, switch
  to a converted/user-role shape).
- **Measure** the spec-25/27/28 levers if pursuing default-ON graduation.
- **PR-review affordance** (a `simba pr-review`-style skill) — the deferred item.
- **Marker polish:** `🦁☑ top 0.00` on non-prompt turns → `idle`.
- Older roadmap (see memories / specs): blog draft, KU/0.7.1, reader levers, multi-hop
  revisit at true 5M fullwiki.

## 7. Where to pick up

The cleanest next thing is **activating + de-risking spec 27** (restart daemon → verify
pi Tier-2 M2). Everything else is opt-in measurement or the deferred PR-review affordance.

**Operational gotchas (memories):** `env -u GITHUB_TOKEN gh auth switch --user mahmoudimus`
before any `gh`/push (active acct 401s); push over SSH; **verify the check CONCLUSION,
not the `--watch` exit code, before merging**; `git -m` backticks run as commands in zsh
(use `-F`); never `rg -rln` (`-r` is `--replace`); releases — tag `v*` → `release.yml`,
then **manual** `gh workflow run deploy.yml --ref vX.Y.Z` (`vars.PUBLISH_TO_PYPI=true`).
**Keep private project names / local paths out of this public repo** (commits, docs, specs).

---

## 8. Handoff prompt (paste into a fresh context)

> You're continuing work on **simba** (a pure-Python local-first memory/reasoning
> plugin for coding agents: Claude Code + Codex + pi). `main` is at **0.11.0**
> (released to PyPI this session). Read `docs/plans/HANDOFF.md` first, then
> `docs/plans/README.md` + the relevant `docs/plans/NN-*.md` specs (25–28), and the
> personal-memory index `~/.claude/projects/-Users-mahmoud-src-ai-simba/memory/MEMORY.md`
> (key entries: `hierarchical-recall-dilution-measured`, `gh-pr-checks-watch-exit-vs-conclusion`,
> `pi-harness-support-spec23`, `sota-levers-graduate-to-default-on`).
>
> Context: this session shipped specs 25–28 + the `max_content_length` single source of
> truth + CI hardening, all behind **default-OFF** levers (byte-identical hook output with
> defaults). Hierarchical recall (spec 26) was **measured** as a situational lever (stays
> default-OFF). Conventions: pure Python under `src/simba/` (the one TS file is the pi
> bridge `simba.ts`); all config via `@configurable`; default-OFF for unmeasured/situational
> levers; **byte-identical Claude/Codex output is the load-bearing invariant** for hook
> changes; tests must mock daemon/LLM/model (a real model load flaked CI on HF — now
> globally guarded); **end every response with `[✓ rules]`**; **no Claude attribution in
> commits/PRs**; **never put private project names / local paths in this public repo**;
> `gh pr create` needs `gh auth switch --user mahmoudimus`; push over SSH; **verify the
> check CONCLUSION (not the `--watch` exit) before merging**.
>
> Top priority: **activate + de-risk spec 27** — (1) restart the daemon so daemon-served
> paths load the new code, (2) **verify the pi `context` re-injection (spec 27 M2)** — the
> bridge appends an unregistered custom-role message that pi's `convertToLlm` may drop (a
> silent no-op); confirm against the pi runtime and switch to a converted/user-role shape
> if needed. Then: optionally measure the spec-25/27/28 levers for graduation, build the
> deferred PR-review *affordance* skill, and polish the `🦁☑ top 0.00` → `idle` case.
> See HANDOFF §4–6 for the full open list. Confirm you've read the handoff + memories,
> then propose the plan before writing code.

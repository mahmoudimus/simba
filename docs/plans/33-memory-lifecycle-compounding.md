# Spec 33 — Memory lifecycle: retention, decay, and promotion

**Status**: phases 0–2 implemented (2026-07-03, branch `feat/spec33-memory-lifecycle`); phases 3–6 pending
**Depends on**: spec 26 (hierarchical scopes), spec 27 (engagement marker), spec 29 (borrow roadmap: quality counters, budget lanes, supersession audit)

**Implementation status (2026-07-03)**:

- ✅ Phase 0 — MaintenanceScheduler heartbeat (shadow), decay/hygiene
  `dry_run`, `POST /maintenance/run`, `/stats.lastMaintenance`,
  match/inject split + `POST /recall/ack`, `simba memory maintain`.
- ✅ Phase 1 — usage signals behind `hooks.usage_signals_enabled`:
  gate-fire use, Stop citation detection (entropy terms), session noise
  sweep, `memory_usage.last_used`, `post_feedback`/`ack_injected`.
- ✅ Phase 2 — `memory.decay_type_multipliers` (type-aware half-lives),
  rule-TTL refresh (`hooks.rule_ttl_refresh_enabled`, freshness =
  max(createdAt, lastUsedAt); hygiene honors `last_used`), recall
  surfaces `lastUsedAt`, `memory.store_budget_per_session` throttle.
- ⏳ Phase 3 (identity/user lane), 4 (adjudicator/episodes), 5 (promotion
  queue/nudges), 6 (graveyards/bridge) — not started.

Everything ships default-OFF/shadow per the graduation policy. To dogfood:
`simba config set memory.maintenance_apply true`, `simba config set
hooks.usage_signals_enabled true`, `simba config set hooks.recall_ack_enabled
true`, `simba config set hooks.rule_ttl_refresh_enabled true`, then
`simba memory maintain` to watch a pass.

## 0. The audit that motivates this

A live audit of the running system (2026-07-03, daemon uptime ~3d, corpus 8,530
memories since 2026-01-31) found that **every lifecycle mechanism is already
built, most are default-ON in config, and almost none of them execute**:

| Mechanism | Built | Config | Actually runs | Evidence |
|---|---|---|---|---|
| Usage sidecar (`memory_usage`) | ✅ | — | partially | 5,731 rows; `access_count` bumps on recall |
| Strength decay (`decay.py`) | ✅ | `decay_enabled=True` | **never** | strength min=avg=max=1.0 across all rows |
| Dormancy filter (`hybrid.py`) | ✅ | `dormant_filter_enabled=True` | no-op | 0 dormant rows to filter |
| Feedback (`POST /feedback`) | ✅ | weights configured | **never called** | 0 rows with `feedback_score≠0`; no callers in hooks |
| Quality counters (use/noise) | ✅ | — | **never bumped** | `use_count=0, noise_count=0` everywhere; `match==inject` (bumped on same line) |
| Hygiene (rule pruning) | ✅ | `hygiene_scheduler_enabled=True` | **never** | lives inside SyncScheduler |
| Supersession | ✅ | `supersede_enabled=True` | **yes** — the one closed loop | 605 events; recall demotes superseded hits |
| Supersession adjudication | ❌ | — | — | 166 `pending_confirmation`, 0 ever confirmed/rejected |
| Episodes | ✅ | `episodes.enabled=True` | stalled | last `episode_jobs` row 2026-06-08 |
| Rule TTL | read-side only | `rule_max_age_days=14` | yes | filters by `created_at`, **ignores usage** |

**Root cause, one sentence**: the daemon's only background loop is
`SyncScheduler`, started in `server.py:_start_sync_scheduler` only when
`sync_interval > 0` — and the default is `0`, so decay AND hygiene (both hosted
inside `run_once`) have never fired. `/health` confirms: `sync.enabled=false`.

Corpus-shape consequences (as of the audit):

- **41% dead tail**: 3,524 of 8,530 memories have never been recalled once
  (2,799 have no usage row at all; 725 more have `access_count=0`).
  69% have been recalled ≤2 times, ever.
- **Extreme concentration**: the top memory (one project's docker-test
  TOOL_RULE) holds 17,286 of 73,317 total accesses (24%); the top 100 memories
  (1.2% of corpus) hold 55%. Rule-gate lookups route through `/recall`, so
  gate probes dominate counts.
- **Unbounded inflow**: +1,212 memories in July 1–3 (~400/day) with no decay,
  no capacity caps, no feedback. Write-heavy, read-light, feedback-none.
- **Fragmented identity**: strict exact-path scoping splits the largest
  project into 4 invisible shards (5,674 / 273 / 165 / 127 across worktrees);
  the rules table mixes raw paths and project-id hashes.
- **Disjoint brains**: the curated Claude-Code auto-memory layer (84 files,
  MEMORY.md) contains gotchas (e.g. the `push.default=upstream` worktree trap)
  that have **zero** counterpart in the daemon corpus — unsearchable from
  codex/pi/cli.
- **Dead scaffolding**: `kg_edges=0`, `facts=0`, `anticipated_queries=0`,
  `doctrine_triggers=0`, `judge_log=0`, `sessions=0`; legacy stores
  (turbo-search `simba.search` 0/0/0, tailor `reflections.jsonl` frozen Feb 7,
  `graphify-out` stale Jun 16) still have live skills/readers pointing at them.
- **Extraction is opt-in and pending**: latest codex transcript (Jun 27) sits
  `pending_extraction`; 117 exported Claude transcripts have no automatic
  extraction path at all.

The design goal is NOT new machinery. It is: **connect three broken wires
(clock, feedback, promotion), unify identity, and retire dead layers** — then
measure whether sessions actually compound.

## 1. Principles

1. **Loop-closure over construction.** Prefer wiring existing modules
   (`decay.py`, `usage.py`, `supersession.py`, engagement per-turn record) to
   adding anything new.
2. **Dormancy, not deletion.** Forgetting = reversible demotion (flag in the
   mutable sidecar). LanceDB stays append-only; deletion only via sanitize
   with a tombstone. HaluMem is the over-retention guard.
3. **Default-OFF until MEASURED** (repo policy). New behavior ships behind
   config, runs in shadow mode first, flips default only with a cited A/B.
4. **Signals before policy.** Decay and promotion are only as good as the
   usage ledger; fixing the counters precedes everything else.
5. **Knowledge flows toward determinism.** The compounding ladder ends in
   gates and config defaults (existing graduation policies), not in a larger
   corpus. Corpus size is a cost, not a KPI.

## 2. Part 1 — The heartbeat (decouple maintenance from sync)

The clock that never ticked.

- New knob: `memory.maintenance_interval_hours` (default `24`, `0` = off).
- Daemon `lifespan` starts a dedicated `MaintenanceScheduler` independent of
  `sync_interval`, running: (a) `run_decay_pass`, (b) `run_hygiene_pass`,
  (c) supersession adjudication (Part 4), (d) health-report snapshot (Part 7).
- Also run once at daemon startup (after a grace delay, e.g. 5 min) so a
  laptop that never stays up 24h still gets passes.
- Batch/buffer all sidecar writes; never write LanceDB from maintenance
  (the 37GB version-bloat lesson: per-recall LanceDB writes are how we got
  a 25,183-version table).
- `simba memory maintain --run` = manual trigger; prints the DecayResult /
  hygiene summary. Everything the scheduler does must be runnable by hand.

## 3. Part 2 — A truthful usage ledger

Today `match_count == inject_count` (bumped on the same line in
`routes.py:178-179`) and `use/noise` are never written. Redefine and wire:

| Counter | Meaning | Writer |
|---|---|---|
| `match` | returned by search (post-fusion, pre-trim) | daemon `/recall` (as today) |
| `inject` | actually placed into a hook's `additionalContext` after lane/budget trim | hook posts an ack: `POST /recall/ack {ids}` (batched, fire-and-forget) |
| `use` | evidence the model consumed it | Stop hook (below) + gate fires |
| `noise` | injected, never used by session end | Stop/SessionEnd sweep |

**`use` detection, v1 (deterministic, no LLM):**

1. **Gate fires are uses.** A PreToolUse rule/pitfall/redirect hit on a memory
   bumps `use=1` and refreshes `last_accessed`. (The docker-test rule's 17k
   probes prove gates are the highest-value consumers.)
2. **Citation overlap.** The engagement per-turn record (spec 27) already
   holds this turn's injected memory ids. At Stop, check the response for
   distinctive terms from each injected memory (reuse the
   `entropy_terms`/exact-boost machinery: rare-word + identifier-shape terms
   only). Overlap ≥ 2 distinctive terms → `use=1, feedback +0.3`.
3. **Session-end noise sweep.** Ids injected ≥2 times this session with no
   `use` → `noise+1, feedback −0.1` (weak, deliberately asymmetric).

**Explicit feedback surfaces:**

- `simba memory feedback <id> good|bad` (CLI, exists as endpoint already).
- `/memories-sanitize` gains a mode that emits `bad` feedback / sets dormant
  instead of delete-only.

Config: `memory.usage_signals_enabled` (default off → shadow-log only),
`hooks.recall_ack_enabled`.

## 4. Part 3 — Decay policy (type-aware, usage-refreshed)

Machinery unchanged (`compute_strength`: recency × reinforcement × feedback);
policy becomes type-aware via half-life multipliers on
`decay_half_life_days=30`:

| Type | Multiplier | Effective half-life | Rationale |
|---|---|---|---|
| EPISODE | 0.5× | 15d | session digests age fastest |
| FAILURE | 1× | 30d | fixed bugs stop mattering; v2: auto-dormant when the referenced fix commit lands |
| GOTCHA / WORKING_SOLUTION | 1.5× | 45d | operational knowledge, medium shelf life |
| PATTERN / DECISION | 4× | 120d | architectural facts |
| PREFERENCE (project) | 12× | 360d | slow |
| PREFERENCE (user scope, Part 4) | exempt | — | only sanitize can demote the user model |
| TOOL_RULE | no strength decay | TTL below | gates live and die by firing |

**Rule TTL fix**: `rule_max_age_days` read-window switches from `created_at`
to `max(created_at, last_accessed)`. Today a rule dies 14 days after creation
*no matter how often it fired* — "use it or re-learn it" becomes "use it and
keep it". Junk rules (raw `ls: No such file...` captures) age out; the docker
rule lives as long as it fires. Hygiene (now actually running) hard-prunes
rules unseen for `tool_rule_max_age_days=30`.

**Dormancy**: threshold stays 0.1; the filter is already wired in
`hybrid.py`. Ship with `memory.dormancy_shadow_mode=True`: compute strength,
set flags, but **log** what would have been hidden instead of hiding it.
Flip the filter live only after the Part 7 guards pass.

**Capacity caps** (existing `_apply_capacity_cap`, currently 0/off): per
(type, project) — GOTCHA/WORKING_SOLUTION 750, PATTERN/DECISION 400,
EPISODE 100. Caps demote weakest-first to dormant; nothing is deleted.
At ~400 stores/day this is the only thing standing between the corpus and
six figures.

**Inflow throttle**: per-session store budget
`memory.store_budget_per_session` (default 0=off; measured target ~25
non-EPISODE stores). Over-capture is real: auto-learn stored raw
`ModuleNotFoundError` output as TOOL_RULEs.

## 5. Part 4 — Retention lanes and identity

**Identity normalization** (prerequisite for everything cross-session):

- Normalize scope to **repo root**: resolve worktrees (`git rev-parse
  --git-common-dir`) so `.worktrees/*` stores land on the main repo scope.
  One-time migration folds the existing shards (the audited project's
  273+165+127 rejoin its 5,674).
- One keying scheme: `project_id` hash everywhere (the rules table currently
  mixes raw paths and hashes, which is why `simba rule list` shows nothing
  for this repo while `--all` shows simba-path rows).

**Scope lanes**: `user` | `project` (| `session` implicit via EPISODE).

- PREFERENCE rows about the *human* (not the codebase) get `scope=user` at
  extraction time (one added question in the extraction prompt) and live
  globally.
- Recall gains a **user lane**: 1 slot, high floor (`min_similarity ≥ 0.55`),
  queried across all projects. This is the "smarter about ME" fix — today the
  user model is 348 PREFERENCE rows, 4% of corpus, almost all project
  directives, recallable only inside the project that learned them.
- Existing hierarchical-recall dilution measurement (LoCoMo −0.7pp, LME
  −4..−9pp) applied to *ancestor:child noise*; a 1-slot curated user lane is
  a different animal — measure it separately before judging it by that
  result.

**Supersession adjudication** (the unread inbox): maintenance pass resolves
`pending_confirmation` rows — LLM single-pass judge (existing conflict
plumbing) confirms/rejects; confirmed → old memory set dormant (today it is
only annotated/demoted in ranking); unadjudicated after 30d → newest-wins
(`lww` operator already exists in neuron resolution ops).

## 6. Part 5 — The promotion ladder

Knowledge moves toward cheaper, more deterministic layers; every hop earns
its per-turn token cost. Demotion mirrors every promotion.

```
L0 transcript (rlm, lossless)
 └─ extraction (per-session, non-optional visibility)
L1 episodic memory (daemon corpus)
 └─ consolidation (episodes + supersession-confirm + near-dup merge)
L2 consolidated fact (survives sanitize; linked; trust-scored)
 └─ promotion queue (usage-triggered, human-approved)
L3 rule layer (TOOL_RULE/redirect · CORE capsule · CLAUDE.md · skill · auto-memory file)
 └─ existing graduation policies
L4 deterministic gate / config default        ← cheapest, no tokens per turn
```

**L0→L1**: parity with codex's `pending_extraction` state for Claude Code —
PreCompact/SessionEnd enqueues, SessionStart nudges ("1 transcript pending
extraction, run `simba codex-extract`"). Auto-extract stays opt-in per
policy, but *pending must be visible* or it never happens (latest codex
transcript has sat pending since Jun 27).

**L1→L2**: re-arm episodes (`auto_on_precompact` is on but last ran Jun 8 —
diagnose and fix in Phase 4); weekly consolidation merges near-dup clusters
using the supersession event log as its queue.

**L2→L3 promotion triggers** (checked by maintenance, drafts to a queue —
never auto-applied):

- `use_count ≥ 3` across ≥ 2 distinct sessions AND `noise/use < 0.5`
  → draft: imperative-with-INSTEAD shape → TOOL_RULE/redirect;
    behavioral guidance → CLAUDE.md / CORE-capsule candidate;
    multi-step procedure → skill draft (improve-skill path).
- PREFERENCE(user) recalled in ≥ 2 distinct projects → global user-model
  entry (auto-memory file + global CLAUDE.md candidate).
- Surfaced at SessionStart: "3 promotion candidates (`simba memory promote
  --review`)". Approval is human (`--to rule|core|claude-md|skill`).

**L3→L4**: unchanged — the two existing graduation policies (SoTA levers
flip default-ON when measured; CORE rules graduate OUT when a gate exists).

**Demotions**: gates that never fire in 60d → demoted back to L1 memory
(hygiene logs the demotion); capsule rules with gates → removed (existing
policy); L1 dormant → excluded from recall but greppable
(`simba memory list --include-dormant`).

**Layer roles & the bridge** (fix the disjoint brains):

- Auto-memory (`~/.claude/projects/.../memory/`) = curated L2/L3. Daemon =
  L1 + L2 bulk.
- One-way mirror: auto-memory files (and their updates) are stored into the
  daemon with `trust_source=curated` provenance (dedup at 0.92 makes re-runs
  idempotent). The `push.default` trap becomes findable from codex/pi/cli.
- Promotion queue drafts *into* auto-memory format, so the curated layer
  grows from measured usage instead of only from manual effort.

## 7. Part 6 — Retire the graveyards

- `simba.search` project-memory (0 sessions / 0 knowledge / 0 facts): delete
  the module or repoint the `memory-stats` skill at daemon `/stats`. A live
  skill returning zeros is worse than no skill.
- tailor `reflections.jsonl` (frozen Feb 7): archive; the `reflections` DB
  table (1,112 rows) is the successor — give its reflection pass a slot in
  the weekly maintenance run and feed its repeat-cluster output into the
  promotion queue (a normalized error recurring across ≥2 sessions is a rule
  candidate by definition).
- `graphify-out` (Jun 16, 164MB): refresh on a cadence or drop it from
  skill-trigger paths; a stale code graph confidently answers wrong.
- Empty scaffolding (`kg_edges`, `facts`, `anticipated_queries`,
  `doctrine_triggers`): each either gets a measurement plan this quarter or
  its tables/config are removed. No dead knobs.
- Curated-layer supersession: MEMORY.md contradictions (LLM-reranker
  "default-on" vs "retired") get the same rule as the corpus — newest wins,
  the loser's index line is edited, not appended alongside.

## 8. Part 7 — Measurement: is it compounding?

`simba memory health` (CLI + weekly maintenance snapshot + one SessionStart
line). KPIs, each re-runnable from the sidecar/reflections tables:

| KPI | Definition | Direction |
|---|---|---|
| Utilization | `use>0` fraction of injected memories | ↑ |
| Dead tail | never-recalled fraction of corpus | ↓ |
| Repeat-failure rate | normalized errors recurring in ≥2 sessions ≥3d apart | ↓ (the single best "is it learning" number) |
| Noise ratio | noise / inject | ↓ |
| Active corpus | non-dormant count | plateaus while total grows |
| Promotion throughput | candidates drafted → approved / week | > 0 |
| Rule survival | rules alive >30d via refresh (not re-learning) | ↑ |
| User-model reach | user-scope memories recalled cross-project | > 0 |

**Guards** (must not regress when flipping defaults): LME-S / LoCoMo recall@k
flat; HaluMem improves (it penalizes over-retention — dormancy should help);
recall p50 latency flat.

**A/B before default-ON** (SoTA policy): dormancy shadow log vs live filter
on the eval suites; decay+feedback on/off on a 2-week live window comparing
repeat-failure rate and utilization.

## 9. Config surface (all new knobs)

```
[memory]
maintenance_interval_hours = 24      # 0 = off; the heartbeat
usage_signals_enabled = false        # Part 2 (shadow until measured)
dormancy_shadow_mode = true          # Part 3
decay_type_multipliers = "EPISODE:0.5,FAILURE:1,GOTCHA:1.5,WORKING_SOLUTION:1.5,PATTERN:4,DECISION:4,PREFERENCE:12"
store_budget_per_session = 0         # 0 = off
supersession_adjudication_enabled = false
promotion_queue_enabled = false
user_lane_enabled = false
user_lane_min_similarity = 0.55

[hooks]
recall_ack_enabled = false           # inject-ack (match/inject split)
citation_use_detection = false       # Stop-hook use signal
```

## 10. Rollout phases

0. **Counters + heartbeat, shadow.** Split match/inject, start
   MaintenanceScheduler logging DecayResult only. Zero behavior change.
1. **Signals live.** Gate-fire `use`, citation `use`, session-end `noise`,
   feedback CLI. Ledger becomes truthful; still no ranking change
   (strength already feeds scoring at weight 0.4 — it starts moving now,
   which IS a ranking change: measure it).
2. **Decay live, dormancy shadow → live.** Type multipliers, rule-TTL
   refresh fix, capacity caps. Flip dormant filter after guards pass.
3. **Identity.** Worktree→root + path→id migration; user lane.
4. **Consolidation.** Supersession adjudicator; episode re-arm; reflections
   pass into maintenance.
5. **Promotion queue + SessionStart nudges** (pending extraction, candidates,
   health line).
6. **Graveyard cleanup** (Part 6) + auto-memory bridge.

Each phase lands default-OFF, gets measured, and flips per the graduation
policy — this spec's mechanisms are subject to the same ladder they
implement.

## Appendix: audit queries (re-runnable)

```bash
curl -s localhost:8741/stats                          # corpus, types, clientHits
sqlite3 .simba/simba.db "SELECT count(*), sum(access_count>0), sum(access_count),
  max(access_count), sum(dormant), min(strength)||'/'||avg(strength)||'/'||max(strength),
  sum(feedback_score!=0), sum(match_count), sum(inject_count), sum(use_count),
  sum(noise_count) FROM memory_usage;"
sqlite3 .simba/simba.db "SELECT status, count(*) FROM memory_supersessions GROUP BY status;"
sqlite3 .simba/simba.db "SELECT status, count(*) FROM episode_jobs GROUP BY status;"
sqlite3 .simba/memory/memory_fts.db "SELECT project_path, count(*) FROM memory_fts
  GROUP BY project_path ORDER BY 2 DESC LIMIT 12;"
uv run simba codex-status                             # extraction pending state
```

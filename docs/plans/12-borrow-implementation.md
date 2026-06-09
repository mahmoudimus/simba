# 12 — Borrow implementation report (SubtleMemory + toki resolution)

Branch: `feat/borrow-subtlememory-toki` (base: `feat/sota-deepseek-judge`)
Status: **full suite green (1516 passed), ruff clean, every new feature default-OFF / no-op.**

This spec closes out the cross-system borrow program from spec 08
(`08-borrow-survey.md`). It implements two graded borrows — a **relational /
contradiction benchmark** (SubtleMemory) and a **write-time contradiction-
resolution layer** (adapted from *toki*) — plus three cheap borrows
(corpus-doctor detection harness, arousal-modulated decay, a KG-edge `proof`
lineage carry). The headline up front: the toki resolution layer is a **write-
time correctness win, not a retrieval-utility win** (this matches toki's own
§6 finding), and SubtleMemory confirms the contradiction slice is the hardest
relational axis but the *answer-surfacing* signal it measures needs an LLM
judge to read — the recall arm alone does not capture it. Nothing here is on by
default; the branch is safe to review as pure additive surface.

---

## 1. What was built

### 1.1 SubtleMemory benchmark (instrument, default-OFF)

`src/simba/eval/benchmarks/subtlememory.py` (+ `tests/eval/benchmarks/test_subtlememory.py`, 14 tests).

SubtleMemory is a *relational / contradiction* memory eval: the hard signal is
not raw recall but whether the system surfaces the right **combination** of
memories when they relate — **complementary** (combine / any-one), **nuanced**
(a time/context boundary decides which applies), and **contradictory** (an
unresolved conflict that should be *surfaced*, not silently collapsed).
`contradictory` is the headline differentiator between memory systems.

- Loader maps one persona dir (`persona_{0..9}/`) → one simba `Dataset`,
  mirroring `halumem.py` / `locomo.py`: each dialogue turn → a `Memory`
  keyed `{session_id}_{turn_index}`; each QA pair → an `EvalCase` whose gold is
  every turn of the case's target sessions, `intent` = `relation_type`,
  `note` = `relation_subtype`. Cases with no resolvable gold are dropped (not
  scoreable).
- `aggregate_by_relation` mirrors `halumem.aggregate_qa`: overall + per-slice
  accuracy, with a dedicated `contradictory` block flagged `is_headline`.
- Wired into the CLI: `simba eval bench subtlememory [--persona-limit L] ...`.
  Config in `bench_config.py`: `subtlememory_path`, `subtlememory_persona_limit`
  (defaults to 1 persona for cheap smoke runs; ~100 cases / ~2.5k turns each).

### 1.2 toki contradiction-resolution layer (Phase-7, default-OFF)

`src/simba/neuron/resolve_ops.py` (704 LOC, + `tests/neuron/test_resolve_ops.py`,
22 tests) and the REVISE integration in `src/simba/neuron/revise.py`
(+ `test_revise_operators.py` 4 tests, `test_resolution_anomaly_probe.py` 6 tests).
Schema: `src/simba/neuron/schema.py` adds two append-only tables
(`kg_audit_resolutions`, `neuron_judge_log`).

Adapts toki's bitemporal operator algebra and keyed judge-log discipline to
simba's SQLite + KG-store substrate (pure Python; no DuckDB, no K-semiring
polynomials — a simplified JSON provenance merge stands in until simba tracks
write-event tokens). Four typed operators return `(winner, AuditRecord)` as
**pure functions** (the operator never mutates state; the loser is the caller's
to stamp dormant — mirroring toki):

| Operator | id | Selection key |
|----------|------|--------------|
| Last-Writer-Wins | `lww` | most-recent `(valid_from, edge_id)` |
| Evidence | `evi` | highest `confidence`, LWW tie-break |
| AwaitConfirm | `await` | oracle/judge vote, logged & replayed |
| PerRule | `rule` | declarative rule, logged & replayed |

It is the write-time **correctness** layer, structurally closing three
write-time anomalies (the N1/N2/N3 model carried from toki):

- **N1 (replay-inconsistency)** — the keyed, append-only `neuron_judge_log`
  records a verdict under `(r_key, theta)` **before** the operator commits
  (the H1 ordering invariant), so a crash/reload replays the *same* winner:
  re-judging cannot flip it.
- **N2 (belief-drift skew)** — the merged provenance carries **both**
  conflicting facts' lineage (the merge dominates each summand), so the loser
  is reconstructable and no partition can silently drift.
- **N3 (audit erasure)** — every resolution appends an `AuditRecord` to
  `kg_audit_resolutions` preserving the loser's object, edge id, and merged
  provenance, so the superseded fact is always recoverable.

`revise.py` dispatches to the typed-operator path only when
`neuron.resolution_ops_enabled` is True; otherwise the legacy entrenchment-only
dormancy path runs **unchanged**. Conflicts are detected with simba's existing
Z3 verifier relation (antonym / same-pred on shared `(subject, object)`
endpoints + overlapping belief-time), so the operator path resolves exactly the
witnesses the verifier flagged as the UNSAT core.

### 1.3 Cheap borrows

- **corpus-doctor inject→detect→score harness** —
  `src/simba/eval/corpus_doctor.py` (+ `test_corpus_doctor.py`, 17 tests). A
  pure, deterministic Phase-7 eval instrument that synthetically corrupts a
  KG-edge corpus with typed contradictions (`antonym`, `temporal_overlap`,
  `duplicate` — the injection vocabulary mirrors `z3_verify`), runs an arbitrary
  `detect_fn`, and scores precision/recall/F1 against the known injections.
  Config-gated (`CorpusDoctorConfig.enabled = False` → returns zeroed metrics),
  so it costs nothing in CI until turned on.
- **arousal-modulated decay rate** — `memory/strength.py` + `memory/decay.py`.
  A multiplier raises the time-decay factor to a power
  (`d ** arousal_decay_multiplier`): `< 1.0` = slower decay (more arousal /
  importance, retains longer), `> 1.0` = faster decay. Default `1.0` is an
  **exact mathematical no-op** (`d ** 1.0 == d`) and the gate
  (`memory.arousal_decay_enabled`) is False.
- **KG-edge `proof` lineage carry** — `kg/store.py` adds a `proof` column;
  `revise._fetch_edges` carries it into the fact so the operator path's
  provenance merge dominates both summands (this is what makes the N2
  reconstruction invariant real — without it the merge is hollow). The legacy
  path ignores `proof`, so it is a no-op when resolution ops are off.

---

## 2. Measured results (honest, including the flat axes)

### 2.1 SubtleMemory — recall arm, persona_0 (`--baseline`, no LLM judge)

`simba eval bench subtlememory --persona-limit 1 --baseline`
(default recall stack: bge-large + RRF k=20 + reranker on):

| slice | n | recall@5 | recall@10 | ndcg@5 | mrr |
|-------|---|----------|-----------|--------|-----|
| overall | 141 | 0.133 | 0.242 | 0.711 | 0.815 |
| complementary | 33 | 0.131 | 0.246 | 0.695 | 0.790 |
| **contradictory** (headline) | 36 | 0.134 | 0.233 | **0.636** | **0.741** |
| nuanced | 72 | 0.133 | 0.244 | 0.756 | 0.864 |

**The contradictory-slice gap is real but modest, and it is a *ranking* gap, not
a recall gap.** Raw recall@k is nearly flat across slices (each gold spans every
turn of a multi-turn session, so recall@k is dominated by session length, not
relation type). The discriminating signal is **ndcg@5 / mrr**, where
`contradictory` is the lowest of the three slices (ndcg@5 0.636 vs 0.756 nuanced;
mrr 0.741 vs 0.864) — the system ranks the conflicting evidence *lower*, exactly
the failure mode SubtleMemory is designed to expose.

**Caveat / what the recall arm cannot see:** SubtleMemory's true headline is
*answer surfacing* — does the system present the unresolved conflict rather than
collapse to one side? That is a QA-judge axis (`--qa`), not a retrieval-rank
axis, and it requires an LLM judge. Per `eval-ablation-latency-trap`, a full
QA judge run over the contradictory slice was **not** executed here (cloud
~17s/call; local judge is the next step). So the recall numbers above are a
*pre-screen*, not the verdict. **This is the honest limit of the measurement.**

### 2.2 toki resolution layer — N1/N2/N3 correctness probe (before/after)

`tests/neuron/test_resolution_anomaly_probe.py` runs the **same** planted
contradiction through REVISE twice — flag OFF (legacy baseline) and ON
(operator path) — and asserts the anomaly is **present** in the baseline and
**absent** once enabled. All 6 probes pass:

| anomaly | OFF (baseline) | ON (operator path) |
|---------|----------------|--------------------|
| **N3** audit erasure | loser stamped dormant, **0 audit rows** → identity unrecoverable | exactly **1 loser-preserving** audit row (object / edge id / strategy) |
| **N2** belief-drift | no merged lineage exists at all | merge **dominates both** summands (`src-old` + `src-new` both reachable) |
| **N1** replay-inconsistency | re-judging the same pair **flips** the winner (NYC→LA) across reloads | keyed judge-log replays the **same** committed verdict every reload |

This is the toki borrow's actual payoff: it **closes the gap, measured as a
flag flip**, not a tuned number. The win is structural (auditability,
deterministic replay, reconstructable supersession), independent of retrieval.

### 2.3 toki resolution layer — retrieval-utility axis: EXPECTED FLAT

Consistent with toki §6 ("end-to-end retrieval shows no significant gain"),
the resolution layer is **not a recall lever** and was not expected to move
recall@k. It changes *which* fact is retained on a write conflict and preserves
an audit trail — it does not change candidate ranking. No utility delta is
claimed. **The correctness win stands on its own; the utility axis is honestly
flat.**

### 2.4 corpus-doctor — detector coverage (z3 verifier as `detect_fn`)

End-to-end `run_corpus_doctor_eval` on a 5-edge KG-edge corpus, z3-verifier
detector (`build_z3_script` → `verify_z3` → `_parse_output`), seed 42, 5
injected contradictions:

| kinds | tp | fp | fn | precision | recall | F1 |
|-------|----|----|----|-----------|--------|----|
| antonym | 2 | 0 | 3 | **1.00** | 0.40 | 0.57 |
| antonym,temporal_overlap | 2 | 0 | 3 | 1.00 | 0.40 | 0.57 |
| duplicate | 2 | 0 | 3 | 1.00 | 0.40 | 0.57 |

The z3 detector is **precise (no false positives)** but partial-recall on a tiny
corpus: recall is bounded by how many injected pairs the antonym / same-pred
exclusion model covers and by corpus size (5 edges → few eligible source edges,
so several draws collide / are skipped). The harness *works* and quantifies
detector coverage; tuning the detector's recall is future work, not this branch.

---

## 3. What is default-OFF / no-op (safe-to-review surface)

| Feature | Gate (config) | Default | No-op guarantee |
|---------|---------------|---------|-----------------|
| toki resolution ops | `neuron.resolution_ops_enabled` | `False` | REVISE keeps legacy entrenchment-only path verbatim |
| toki judge-log | `neuron.judge_log_enabled` | `False` | no rows written; AwaitConfirm/PerRule unused |
| corpus-doctor harness | `corpus_doctor.enabled` | `False` | `run_corpus_doctor_eval` returns zeroed `DetectionMetrics` |
| arousal-modulated decay | `memory.arousal_decay_enabled` + `arousal_decay_multiplier=1.0` | `False` / `1.0` | `d ** 1.0 == d` (exact mathematical no-op) |
| SubtleMemory benchmark | additive CLI dataset | n/a | new code path only reached via `simba eval bench subtlememory` |
| KG-edge `proof` carry | additive column | n/a | only read by `_to_fact` on the operator path (off by default) |

No existing default was changed. The recall stack defaults
(bge-large + RRF k=20 + reranker-on) are untouched. All knobs are reachable via
`simba config get/set <section>.<key>` (no hidden constants, no env-var-only
config).

---

## 4. Commit list

`git log --oneline feat/sota-deepseek-judge..HEAD`:

```
feat(eval): SubtleMemory relational/contradiction benchmark (default-off instrument)
feat(neuron): typed contradiction-resolution operators + audit + judge-log replay (Phase-7, default-off)
feat(memory): arousal-modulated decay rate (default no-op, config-gated)
feat(eval): corpus_doctor inject/detect/score harness (Phase-7 instrument)
docs(plans): 12 — borrow implementation report (SubtleMemory + toki resolution)
```

Branch test footprint: **115 new tests** across 8 files
(`test_subtlememory.py` 14, `test_corpus_doctor.py` 17, `test_resolve_ops.py` 22,
`test_revise_operators.py` 4, `test_resolution_anomaly_probe.py` 6,
`test_store.py` 27, `test_decay.py` 8, `test_strength.py` 17). Full repo suite:
**1516 passed**, ruff clean.

TDD note: the final `proof`-lineage carry was verified by the regression test
`test_revise_with_operators_preserves_loser_provenance` and the N2 probe — with
the carry reverted, both fail for the right reason (the provenance merge has
empty summands, `{"winner":{"raw":""},"loser":{"raw":""}}`), and pass once
restored.

---

## 5. What a human should review / decide before merge

1. **toki resolution ops — turn it on for write correctness?** The N1/N2/N3 wins
   are structural and measured (§2.2), but the layer is **utility-flat** (§2.3).
   Decide whether write-time auditability + deterministic conflict replay is
   worth enabling by default in the neuro-symbolic REVISE path, or whether it
   stays an opt-in for users who need a recoverable supersession trail. If
   enabled, choose the default operator (`lww` vs `evi`) deliberately — `lww`
   trusts recency, `evi` trusts confidence.

2. **SubtleMemory contradictory slice — run the QA judge.** §2.1 is a recall
   pre-screen; the real verdict (does the system *surface* the conflict) needs
   a `--qa` LLM-judge pass. Decide judge + budget (local mlx judge to avoid the
   ~17s/call cloud trap; see `eval-judge-one-server-default`,
   `eval-ablation-latency-trap`). Until that runs, **do not** claim a
   contradiction-handling win or loss — the recall gap is suggestive only.

3. **SubtleMemory gold granularity.** Gold is currently *every turn* of a case's
   target sessions, which makes recall@k session-length-dominated and washes out
   slice differences (§2.1). If the benchmark is to drive ranking decisions, a
   human should decide whether to tighten gold to the specific evidence turns
   (the loader supports this — `facts` are parsed but not yet used as gold).

4. **`subtlememory_path` default is an absolute machine-local path**
   (`/Users/mahmoud/src/ai/memory/SubtleMemory/data/subtlememory`). Fine for
   local eval, but it should not ship as a portable default — decide whether to
   blank it (like other bench paths that require `--path`) before any release.

5. **corpus-doctor detector recall** (§2.4) is partial; the harness is sound but
   the z3 detector misses a fraction of injected pairs on small corpora. Decide
   whether that is acceptable as a coverage instrument or whether the detector
   warrants tuning (out of scope for this branch).

6. **Schema migration.** Two new append-only tables
   (`kg_audit_resolutions`, `neuron_judge_log`) and a `proof` column on
   `kg_edges` are created on demand by `schema.py`. Confirm the on-demand
   migration path is acceptable (no destructive ALTER; additive only) for
   existing `.simba/simba.db` stores.

7. **Borrows deliberately *not* taken** (recorded in `08-borrow-survey.md` and
   the memory index): entity-bridge multi-hop (measured negative), Hebbian
   graph-topology learning, procedural memory + reconsolidation, proactive urge
   queue. None are in this branch; confirm that scoping is intended.

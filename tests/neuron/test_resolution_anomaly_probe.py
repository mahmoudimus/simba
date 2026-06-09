"""Anomaly before/after probe: does the resolution layer CLOSE N1/N2/N3?

This is the honest correctness measurement for the toki borrow. Each probe runs
the SAME planted contradiction through the REVISE integration twice — once with
``neuron.resolution_ops_enabled`` OFF (the legacy entrenchment-only baseline) and
once ON (the typed-operator path) — and asserts the anomaly is PRESENT in the
baseline and ABSENT once the layer is enabled.

  N3 (audit erasure): legacy REVISE stamps the loser dormant but writes NO audit
      row, so the superseded fact's identity (object / edge id / merged lineage)
      is NOT recoverable. The operator path appends a loser-preserving audit row.
  N2 (belief-drift / as-of reconstruction): legacy REVISE leaves no merged
      provenance carrying BOTH conflicting facts' lineage, so a partition can
      drift. The operator path's merge dominates both summands.
  N1 (replay-inconsistency): with no keyed judge-log, an oracle re-judge of the
      same pair can FLIP the winner across reloads. The keyed append-only
      judge-log replays the SAME committed verdict deterministically.

The probe asserts the gap, not a tuned number: flip the config flag, the anomaly
opens/closes. Utility (retrieval accuracy) is a separate, expected-flat axis —
see the SubtleMemory measurement; the correctness win stands on its own.
"""

from __future__ import annotations

import time

import pytest

import simba.db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    return db_path


def _seed_conflict(project_path: str = "/proj") -> list[int]:
    """Plant the canonical antonym contradiction (A uses B / A does_not_use B)."""
    import simba.kg.store

    simba.kg.store.kg_add("A", "uses", "B", "src-old", project_path=project_path)
    time.sleep(1.01)  # distinct valid_from so LWW has a strict winner
    simba.kg.store.kg_add(
        "A", "does_not_use", "B", "src-new", project_path=project_path
    )
    with simba.db.connect():
        return [
            e.id
            for e in simba.kg.store.KgEdge.select().where(
                simba.kg.store.KgEdge.project_path == project_path
            )
        ]


def _audit_count(project_path: str = "/proj") -> int:
    import simba.neuron.schema

    _ = simba.neuron.schema
    with simba.db.connect() as db:
        return db.execute_sql(
            "SELECT COUNT(*) FROM kg_audit_resolutions WHERE project_path=?",
            (project_path,),
        ).fetchone()[0]


# ── N3: audit erasure (baseline ANOMALY → closed) ───────────────────────────


def test_n3_baseline_anomaly_loser_not_recoverable_when_disabled(tmp_db) -> None:
    """Resolution OFF: the loser is stamped dormant but NO audit row preserves
    its identity — the superseded fact is unrecoverable from any audit trail."""
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    ids = _seed_conflict()
    cfg = NeuronConfig(revise_enabled=True, resolution_ops_enabled=False)
    result = revise_unsat_core(ids, project_path="/proj", cfg=cfg)

    # The legacy path still supersedes (one edge goes dormant) ...
    assert len(result.dormant_edge_ids) == 1
    # ... but the anomaly is present: no audit row, loser identity erased.
    assert _audit_count() == 0

    import simba.neuron.resolve_ops as ops

    assert ops.query_audit(loser_edge_id=result.dormant_edge_ids[0]) == []


def test_n3_anomaly_closed_when_enabled(tmp_db) -> None:
    """Resolution ON: the same supersession appends a loser-preserving audit
    row, so the superseded fact (object / edge id / lineage) is recoverable."""
    import simba.neuron.resolve_ops as ops
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    ids = _seed_conflict()
    cfg = NeuronConfig(
        revise_enabled=True,
        resolution_ops_enabled=True,
        resolution_default_operator="lww",
    )
    result = revise_unsat_core(ids, project_path="/proj", cfg=cfg)

    assert len(result.dormant_edge_ids) == 1
    # Anomaly closed: exactly one audit row recovers the loser.
    assert _audit_count() == 1
    rows = ops.query_audit(loser_edge_id=result.dormant_edge_ids[0])
    assert len(rows) == 1
    assert rows[0]["loser_edge_id"] == result.dormant_edge_ids[0]
    assert rows[0]["strategy_id"] == "lww"


# ── N2: belief-drift / merged-provenance reconstruction ─────────────────────


def test_n2_baseline_anomaly_no_merged_lineage_when_disabled(tmp_db) -> None:
    """Resolution OFF: no audit row means no merged provenance binds the two
    conflicting facts' lineage — a partition can silently drift (N2)."""
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    ids = _seed_conflict()
    revise_unsat_core(
        ids,
        project_path="/proj",
        cfg=NeuronConfig(revise_enabled=True, resolution_ops_enabled=False),
    )
    # No merged-lineage record exists at all (the anomaly).
    assert _audit_count() == 0


def test_n2_anomaly_closed_merge_dominates_both_when_enabled(tmp_db) -> None:
    """Resolution ON: the merged provenance dominates BOTH conflicting facts'
    lineage, so the loser is reconstructable and no partition can drift."""
    import simba.neuron.resolve_ops as ops
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    ids = _seed_conflict()
    result = revise_unsat_core(
        ids,
        project_path="/proj",
        cfg=NeuronConfig(
            revise_enabled=True,
            resolution_ops_enabled=True,
            resolution_default_operator="lww",
        ),
    )
    rows = ops.query_audit(loser_edge_id=result.dormant_edge_ids[0])
    assert len(rows) == 1
    merge = rows[0]["provenance_merge"]
    # The merge carries both summands (winner + loser provenance).
    assert ops.provenance_dominates("src-old", merge)
    assert ops.provenance_dominates("src-new", merge)


# ── N1: replay-inconsistency (oracle re-judge cannot flip) ──────────────────


def test_n1_baseline_anomaly_rejudge_can_flip_without_log(tmp_db) -> None:
    """Resolution OFF (no keyed judge-log): re-running an oracle over the same
    pair across a 'reload' can FLIP the winner — there is no committed verdict
    to replay, so the resolution is non-deterministic (the N1 anomaly)."""
    import simba.neuron.resolve_ops as ops

    f1 = {
        "edge_id": 1,
        "subject": "A",
        "predicate": "location",
        "object": "NYC",
        "valid_from": "2024-01-01",
        "valid_to": "9999-12-31",
        "confidence": 0.8,
    }
    f2 = {
        "edge_id": 2,
        "subject": "A",
        "predicate": "location",
        "object": "LA",
        "valid_from": "2024-06-01",
        "valid_to": "9999-12-31",
        "confidence": 0.8,
    }
    r_key = ops.r_key(f1, f2)
    theta = "prompt_v1+seed=7"

    # No verdict was committed -> the durable log is empty, so a crash/reload has
    # nothing to replay (the resolution is not pinned).
    assert ops.query_judge_verdicts(r_key, theta) == []

    # A first oracle votes 0 (incumbent NYC); a reloaded oracle votes 1 (LA).
    # Without a keyed log, BOTH execute and the elected winner flips.
    w_first, _ = ops.resolve_await(f1, f2, judge_callback=lambda a, b: 0)
    w_reload, _ = ops.resolve_await(f1, f2, judge_callback=lambda a, b: 1)
    assert w_first["object"] == "NYC"
    assert w_reload["object"] == "LA"  # the anomaly: re-judge flipped it

    with pytest.raises(ValueError):
        ops.replay_from_log(f1, f2, r_key, theta)  # nothing to replay


def test_n1_anomaly_closed_log_replays_same_verdict(tmp_db) -> None:
    """Resolution ON (judge-logged): the verdict committed to the keyed
    append-only log replays the SAME winner after a reload — re-judging cannot
    flip it (the H1 ordering invariant)."""
    import simba.neuron.resolve_ops as ops

    f1 = {
        "edge_id": 1,
        "subject": "A",
        "predicate": "location",
        "object": "NYC",
        "valid_from": "2024-01-01",
        "valid_to": "9999-12-31",
        "confidence": 0.8,
    }
    f2 = {
        "edge_id": 2,
        "subject": "A",
        "predicate": "location",
        "object": "LA",
        "valid_from": "2024-06-01",
        "valid_to": "9999-12-31",
        "confidence": 0.8,
    }
    r_key = ops.r_key(f1, f2)
    theta = "prompt_v1+seed=7"

    # Commit the witnessed verdict BEFORE the operator commit (H1 ordering).
    ops.record_judge_verdict(r_key, theta, vote=0, winner_edge_id=1)

    # Reload #1 and reload #2 both replay the committed vote -> same winner,
    # even if a fresh re-judge would have voted the other way.
    w1, _ = ops.replay_from_log(f1, f2, r_key, theta)
    w2, _ = ops.replay_from_log(f1, f2, r_key, theta)
    assert w1["object"] == "NYC"
    assert w2["object"] == "NYC"
    assert w1["edge_id"] == w2["edge_id"] == 1

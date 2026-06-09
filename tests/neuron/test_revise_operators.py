"""Phase 7 REVISE integration with typed resolution operators (default-OFF).

When ``neuron.resolution_ops_enabled`` is False (the default), revise_unsat_core
uses the legacy entrenchment-only dormancy path (unchanged behaviour). When
enabled, it resolves each conflict via the typed operator, records an
append-only audit row (N3 recoverability), and stamps the loser dormant.
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


def _seed_conflict():
    import simba.kg.store

    simba.kg.store.kg_add("A", "uses", "B", "test", project_path="/proj")
    time.sleep(1.01)
    simba.kg.store.kg_add("A", "does_not_use", "B", "test", project_path="/proj")
    with simba.db.connect():
        ids = [
            e.id
            for e in simba.kg.store.KgEdge.select().where(
                simba.kg.store.KgEdge.project_path == "/proj"
            )
        ]
    return ids


def test_revise_default_off_uses_legacy_path(tmp_db) -> None:
    """With resolution_ops_enabled False (default), no audit rows are written."""
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    ids = _seed_conflict()
    cfg = NeuronConfig(revise_enabled=True)
    assert cfg.resolution_ops_enabled is False

    result = revise_unsat_core(ids, project_path="/proj", cfg=cfg)
    assert len(result.dormant_edge_ids) == 1

    with simba.db.connect() as db:
        count = db.execute_sql(
            "SELECT COUNT(*) FROM kg_audit_resolutions WHERE project_path=?",
            ("/proj",),
        ).fetchone()[0]
    assert count == 0


def test_revise_with_operators_records_audit(tmp_db) -> None:
    """With operators enabled, the loser is dormant AND recoverable from audit."""
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
    assert len(result.retained_edge_ids) == 1

    # Loser is stamped dormant (append-only: the row still exists).
    with simba.db.connect():
        from simba.kg.store import KgEdge

        dormant_e = KgEdge.get_by_id(result.dormant_edge_ids[0])
        assert dormant_e.dormant == 1
        assert dormant_e.valid_to is not None

    # N3: the loser is recoverable from the audit trail.
    audit_rows = ops.query_audit(
        loser_edge_id=result.dormant_edge_ids[0], project_path="/proj"
    )
    assert len(audit_rows) >= 1
    assert audit_rows[0]["strategy_id"] == "lww"


def test_revise_with_operators_preserves_loser_provenance(tmp_db) -> None:
    """N2/N3: the audit row's merged provenance must dominate BOTH conflicting
    edges' source lineage — the loser is reconstructable, no partition drifts.

    Regression: ``_fetch_edges`` must carry the edge ``proof`` into the fact so
    ``_to_fact`` populates provenance; otherwise the merge is hollow (empty
    summands) and the N2 reconstruction invariant silently fails.
    """
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
    rows = ops.query_audit(
        loser_edge_id=result.dormant_edge_ids[0], project_path="/proj"
    )
    assert len(rows) == 1
    merge = rows[0]["provenance_merge"]
    # Both summands' source lineage is reachable in the merge (the "test" proof
    # _seed_conflict passes to kg_add for both edges).
    assert ops.provenance_dominates("test", merge)


def test_revise_disabled_returns_empty(tmp_db) -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.revise import revise_unsat_core

    result = revise_unsat_core(
        [1, 2],
        project_path="/proj",
        cfg=NeuronConfig(revise_enabled=False, resolution_ops_enabled=True),
    )
    assert result.dormant_edge_ids == []

"""Phase 7 contradiction-resolution operators — correctness contract (PRIMARY).

These tests encode the write-time CORRECTNESS contract borrowed from toki's
bitemporal anomaly defences, NOT a retrieval-utility claim (per toki §6:
end-to-end retrieval shows no significant utility gain; the value is write-time
freedom from N1/N2/N3):

  N1 (replay-inconsistency): a verdict re-read from the keyed append-only
      judge-log replays the SAME winner — re-judging cannot flip it.
  N2 (belief-drift / as-of reconstruction): the loser is reconstructable from
      the audit row's merged provenance, so no partition can drift; the merge
      dominates BOTH conflicting facts' lineage.
  N3 (audit erasure): the losing fact (object, edge id, lineage) is recoverable
      from the append-only audit row after supersession.

Plus operator dispatch (lww/evi/await/rule) and the n-ary resolve_conflict_set
fold that equals the binary operator at n == 2.
"""

from __future__ import annotations

import pytest

import simba.db


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    return db_path


def _fact(edge_id, obj, *, conf=0.8, vfrom="2024-01-01", prov=""):
    return {
        "edge_id": edge_id,
        "subject": "Alice",
        "predicate": "location",
        "object": obj,
        "confidence": conf,
        "valid_from": vfrom,
        "valid_to": "9999-12-31",
        "occurred_at": vfrom,
        "provenance": prov,
    }


# ── Contradiction predicate ─────────────────────────────────────────────────


def test_is_contradiction_true_for_same_sp_diff_object() -> None:
    from simba.neuron.resolve_ops import is_contradiction

    assert is_contradiction(_fact(1, "NYC"), _fact(2, "LA")) is True


def test_is_contradiction_false_for_same_object() -> None:
    from simba.neuron.resolve_ops import is_contradiction

    assert is_contradiction(_fact(1, "NYC"), _fact(2, "NYC")) is False


def test_is_contradiction_false_for_disjoint_belief_time() -> None:
    from simba.neuron.resolve_ops import is_contradiction

    f1 = _fact(1, "NYC")
    f1["valid_to"] = "2024-02-01"
    f2 = _fact(2, "LA", vfrom="2024-03-01")
    assert is_contradiction(f1, f2) is False


# ── Operator dispatch (lww / evi) ───────────────────────────────────────────


def test_resolve_lww_picks_most_recent() -> None:
    from simba.neuron.resolve_ops import resolve_lww

    older = _fact(1, "NYC", vfrom="2024-01-01")
    newer = _fact(2, "LA", vfrom="2024-06-01")
    winner, audit = resolve_lww(older, newer)
    assert winner["object"] == "LA"
    assert winner["edge_id"] == 2
    assert audit.loser_object == "NYC"
    assert audit.loser_edge_id == 1
    assert audit.strategy_id == "lww"


def test_resolve_evidence_picks_higher_confidence() -> None:
    from simba.neuron.resolve_ops import resolve_evidence

    low = _fact(1, "NYC", conf=0.5)
    high = _fact(2, "LA", conf=0.95)
    winner, audit = resolve_evidence(low, high)
    assert winner["object"] == "LA"
    assert audit.loser_object == "NYC"
    assert audit.strategy_id == "evi"


def test_resolve_await_uses_callback() -> None:
    from simba.neuron.resolve_ops import resolve_await

    f1 = _fact(1, "NYC")
    f2 = _fact(2, "LA")
    winner, audit = resolve_await(f1, f2, judge_callback=lambda a, b: 0)
    assert winner["object"] == "NYC"
    assert audit.strategy_id == "await"
    assert audit.judge_verdict is not None


def test_resolve_rule_uses_policy_oracle() -> None:
    from simba.neuron.resolve_ops import resolve_rule

    f1 = _fact(1, "NYC")
    f2 = _fact(2, "LA")
    winner, audit = resolve_rule(f1, f2, policy_oracle=lambda a, b: 1)
    assert winner["object"] == "LA"
    assert audit.strategy_id == "rule"


def test_operators_reject_non_contradiction() -> None:
    from simba.neuron.resolve_ops import resolve_lww

    with pytest.raises(ValueError):
        resolve_lww(_fact(1, "NYC"), _fact(2, "NYC"))


def test_await_rejects_out_of_range_vote() -> None:
    from simba.neuron.resolve_ops import resolve_await

    with pytest.raises(ValueError):
        resolve_await(_fact(1, "NYC"), _fact(2, "LA"), judge_callback=lambda a, b: 7)


def test_resolve_by_strategy_id_dispatch() -> None:
    from simba.neuron.resolve_ops import resolve_by_strategy

    winner, audit = resolve_by_strategy(
        _fact(1, "NYC", vfrom="2024-01-01"),
        _fact(2, "LA", vfrom="2024-06-01"),
        strategy_id="lww",
    )
    assert winner["object"] == "LA"
    assert audit.strategy_id == "lww"


def test_resolve_by_strategy_unknown_raises() -> None:
    from simba.neuron.resolve_ops import resolve_by_strategy

    with pytest.raises(ValueError):
        resolve_by_strategy(_fact(1, "NYC"), _fact(2, "LA"), strategy_id="bogus")


# ── N-ary conflict set fold ─────────────────────────────────────────────────


def test_resolve_conflict_set_lww_argmax() -> None:
    from simba.neuron.resolve_ops import resolve_conflict_set

    facts = [
        _fact(1, "NYC", vfrom="2024-01-01"),
        _fact(2, "LA", vfrom="2024-06-01"),
        _fact(3, "SF", vfrom="2024-03-01"),
    ]
    winner, audit = resolve_conflict_set(facts, strategy_id="lww")
    assert winner["object"] == "LA"  # latest valid_from
    assert audit.strategy_id == "lww"


def test_resolve_conflict_set_n2_equals_binary() -> None:
    from simba.neuron.resolve_ops import resolve_conflict_set, resolve_lww

    a = _fact(1, "NYC", vfrom="2024-01-01")
    b = _fact(2, "LA", vfrom="2024-06-01")
    bin_winner, bin_audit = resolve_lww(a, b)
    set_winner, set_audit = resolve_conflict_set([a, b], strategy_id="lww")
    assert bin_winner["object"] == set_winner["object"]
    assert bin_audit.loser_edge_id == set_audit.loser_edge_id


def test_resolve_conflict_set_requires_two() -> None:
    from simba.neuron.resolve_ops import resolve_conflict_set

    with pytest.raises(ValueError):
        resolve_conflict_set([_fact(1, "NYC")], strategy_id="lww")


def test_resolve_conflict_set_requires_pairwise_contradiction() -> None:
    from simba.neuron.resolve_ops import resolve_conflict_set

    facts = [_fact(1, "NYC"), _fact(2, "LA"), _fact(3, "NYC")]
    with pytest.raises(ValueError):
        resolve_conflict_set(facts, strategy_id="lww")


# ── N3: audit erasure is impossible ─────────────────────────────────────────


def test_n3_loser_recoverable_from_audit_row(tmp_db) -> None:
    """N3 defence: after resolution + supersession, the loser is fully
    recoverable from the append-only audit row (object, edge id, lineage)."""
    import simba.neuron.resolve_ops as ops
    import simba.neuron.schema  # registers audit schema

    _ = simba.neuron.schema
    f1 = _fact(1, "NYC", prov="src1")
    f2 = _fact(2, "LA", conf=0.95, prov="src2")
    winner, audit = ops.resolve_evidence(f1, f2)
    assert winner["object"] == "LA"

    audit_id = ops.record_audit(audit, project_path="/proj")
    assert audit_id is not None

    rows = ops.query_audit(loser_edge_id=1, project_path="/proj")
    assert len(rows) == 1
    row = rows[0]
    assert row["loser_object"] == "NYC"
    assert row["loser_edge_id"] == 1
    assert row["winner_object"] == "LA"
    # provenance merge dominates BOTH the winner and the loser lineage (N3).
    assert "src1" in row["provenance_merge"]
    assert "src2" in row["provenance_merge"]


def test_n3_audit_trail_is_append_only(tmp_db) -> None:
    """Recording the same loser twice (re-resolution) appends, never clobbers."""
    import simba.neuron.resolve_ops as ops
    import simba.neuron.schema

    _ = simba.neuron.schema
    _w, audit1 = ops.resolve_lww(
        _fact(1, "NYC", vfrom="2024-01-01"), _fact(2, "LA", vfrom="2024-06-01")
    )
    ops.record_audit(audit1, project_path="/proj", system_time="2024-06-01T00:00:00Z")
    ops.record_audit(audit1, project_path="/proj", system_time="2024-07-01T00:00:00Z")
    rows = ops.query_audit(loser_edge_id=1, project_path="/proj")
    assert len(rows) == 2


# ── N2: merged provenance dominates both conflicting facts ──────────────────


def test_n2_merged_provenance_dominates_both() -> None:
    """N2 defence: the merged provenance carries BOTH facts' lineage so no
    partition can silently drift — the loser is reconstructable from the merge."""
    from simba.neuron.resolve_ops import provenance_dominates, resolve_evidence

    f1 = _fact(1, "NYC", prov="src1")
    f2 = _fact(2, "LA", conf=0.95, prov="src2")
    _winner, audit = resolve_evidence(f1, f2)
    assert provenance_dominates("src1", audit.provenance_merge)
    assert provenance_dominates("src2", audit.provenance_merge)


# ── N1: keyed judge-log replay is deterministic ─────────────────────────────


def test_n1_judge_log_replays_same_verdict(tmp_db) -> None:
    """N1 defence: a verdict committed to the keyed append-only judge-log
    replays the SAME vote after a crash/reload — re-judging cannot flip it."""
    import simba.neuron.resolve_ops as ops
    import simba.neuron.schema

    _ = simba.neuron.schema
    f1 = _fact(1, "NYC")
    f2 = _fact(2, "LA")
    r_key = ops.r_key(f1, f2)
    theta = "prompt_v1+seed=7"

    # Record the witnessed verdict BEFORE the operator commit (H1 ordering).
    ops.record_judge_verdict(r_key, theta, vote=0, winner_edge_id=1)

    # Simulate a crash/reload: re-query the same (r_key, theta).
    verdicts = ops.query_judge_verdicts(r_key, theta)
    assert len(verdicts) == 1
    assert verdicts[0].vote == 0
    assert verdicts[0].winner_edge_id == 1

    # Replaying from the log reproduces the identical winner — no re-judge.
    replay_winner, _audit = ops.replay_from_log(f1, f2, r_key, theta)
    assert replay_winner["edge_id"] == 1
    assert replay_winner["object"] == "NYC"


def test_n1_judge_log_is_append_only_and_ordered(tmp_db) -> None:
    """Multiple verdicts under one key preserve arrival order (append-only)."""
    import simba.neuron.resolve_ops as ops
    import simba.neuron.schema

    _ = simba.neuron.schema
    r_key = "(1,2)"
    theta = "t"
    ops.record_judge_verdict(r_key, theta, vote=0)
    ops.record_judge_verdict(r_key, theta, vote=1)
    verdicts = ops.query_judge_verdicts(r_key, theta)
    assert [v.vote for v in verdicts] == [0, 1]
    assert verdicts[0].seq < verdicts[1].seq


def test_n1_record_verdict_rejects_bad_vote(tmp_db) -> None:
    import simba.neuron.resolve_ops as ops
    import simba.neuron.schema

    _ = simba.neuron.schema
    with pytest.raises(ValueError):
        ops.record_judge_verdict("(1,2)", "t", vote=5)


def test_n1_query_unknown_key_returns_empty(tmp_db) -> None:
    import simba.neuron.resolve_ops as ops
    import simba.neuron.schema

    _ = simba.neuron.schema
    assert ops.query_judge_verdicts("(99,100)", "t") == []

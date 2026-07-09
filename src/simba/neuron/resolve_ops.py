"""Phase 7 typed contradiction-resolution operators (borrowed from toki).

Adapts toki's bitemporal operator algebra (LWW / Evidence / AwaitConfirm /
PerRule) and its keyed judge-log discipline to simba's SQLite + KG-store
substrate. Pure-Python; no DuckDB, no K-semiring polynomials (simba uses a
simplified JSON provenance merge until it tracks write-event tokens).

The module is the write-time CORRECTNESS layer, NOT a retrieval-utility lever
(per toki §6, end-to-end retrieval shows no significant gain). Its value is
structural freedom from three write-time anomalies:

  N1 (replay-inconsistency) — the keyed, append-only ``neuron_judge_log``
      records a verdict under ``(r_key, theta)`` BEFORE the operator commits,
      so a crash/reload replays the SAME winner: re-judging cannot flip it.
  N2 (belief-drift skew) — the merged provenance carries BOTH conflicting
      facts' lineage (the merge dominates each summand), so the loser is
      reconstructable and no partition can silently drift.
  N3 (audit erasure) — every resolution emits an ``AuditRecord`` appended to
      ``kg_audit_resolutions`` preserving the loser's object, edge id, and
      merged provenance, so the superseded fact is always recoverable.

Operators are PURE functions returning ``(winner, AuditRecord)``; the loser is
the caller's to stamp dormant (mirrors toki: the operator never mutates state).
"""

from __future__ import annotations

import json
import logging
import time
import typing
from dataclasses import dataclass

logger = logging.getLogger("simba.neuron.resolve_ops")

_SELECTION_STRATEGIES = ("lww", "evi")
_ORACLE_STRATEGIES = ("await", "rule")
_ALL_STRATEGIES = _SELECTION_STRATEGIES + _ORACLE_STRATEGIES


# ── Contradiction predicate (mirrors toki §3.1) ─────────────────────────────


def is_contradiction(
    f1: typing.Mapping[str, typing.Any], f2: typing.Mapping[str, typing.Any]
) -> bool:
    """Return True iff ``f1`` and ``f2`` contradict.

    Same ``(subject, predicate)``, different ``object``, with overlapping
    closed-open belief-time periods ``[valid_from, valid_to)``.
    """
    return (
        f1.get("subject") == f2.get("subject")
        and f1.get("predicate") == f2.get("predicate")
        and f1.get("object") != f2.get("object")
        and (f1.get("valid_from") or "") < (f2.get("valid_to") or "")
        and (f2.get("valid_from") or "") < (f1.get("valid_to") or "")
    )


# ── Audit + judge-log records ───────────────────────────────────────────────


@dataclass
class AuditRecord:
    """Loser-preserving audit tuple emitted by every resolution (N3 defence).

    The merged provenance dominates both the winner and the loser lineage, so
    the superseded fact is always recoverable from the audit trail. ``strategy_id``
    names the operator (``lww`` / ``evi`` / ``await`` / ``rule``); ``judge_verdict``
    references the judge-log key for oracle-driven resolutions.
    """

    subject: str
    predicate: str
    winner_object: str
    loser_object: str
    winner_edge_id: int
    loser_edge_id: int
    valid_from: str
    valid_to: str
    occurred_at: str | None
    system_time: str
    provenance_merge: str
    strategy_id: str
    confidence_winner: float
    confidence_loser: float
    judge_verdict: str | None = None


@dataclass(frozen=True)
class JudgeVerdictRecord:
    """One witnessed verdict read back from the keyed append-only judge-log."""

    seq: int | None = None
    r_key: str = ""
    theta: str = ""
    vote: int = 0
    winner_edge_id: int | None = None
    system_time: str = ""


def _now() -> str:
    """Return current UTC time as an ISO-8601 ``Z`` string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Provenance merge (simplified; toki uses K[X,T] polynomials) ─────────────


def _decode_prov(p: str) -> typing.Any:
    if not p:
        return {"raw": ""}
    try:
        return json.loads(p) if p.startswith("{") else {"raw": p}
    except (json.JSONDecodeError, TypeError):
        return {"raw": p}


def _merge_provenance(p_winner: str, p_loser: str) -> str:
    """Merge winner + loser provenance into one JSON object preserving both.

    Simplified stand-in for toki's K-semiring sum: the result carries both
    summands, so it dominates each (the N2/N3 reachability invariant). When
    simba adopts write-event tokens this becomes a proper polynomial merge.
    """
    merged = {"winner": _decode_prov(p_winner), "loser": _decode_prov(p_loser)}
    return json.dumps(merged, separators=(",", ":"))


def _merge_provenance_all(provenances: typing.Sequence[str]) -> str:
    """N-ary generalisation of :func:`_merge_provenance`.

    The winner is the first element; all losers follow. The merge carries
    every summand so it dominates each (provenance-completeness for an n-ary
    conflict set). Raises on an empty family (rule 02: no silent fallback).
    """
    if not provenances:
        raise ValueError("_merge_provenance_all requires at least one provenance")
    merged = {
        "winner": _decode_prov(provenances[0]),
        "losers": [_decode_prov(p) for p in provenances[1:]],
    }
    return json.dumps(merged, separators=(",", ":"))


def provenance_dominates(p_old: str, p_merge: str) -> bool:
    """Return True iff ``p_old``'s lineage is reachable inside ``p_merge``.

    Sound for the simplified JSON merge: a summand dominates iff its serialised
    form appears in the merged JSON (winner / loser / losers slot). This is the
    N2/N3 reachability check the audit row must satisfy for every loser.
    """
    needle = json.dumps(_decode_prov(p_old), separators=(",", ":"))
    return needle in p_merge


# ── Shared emission (dual-row: stamped winner + audit) ──────────────────────


def _emit_resolution(
    winner: typing.Mapping[str, typing.Any],
    loser: typing.Mapping[str, typing.Any],
    strategy_id: str,
    system_time: str,
    *,
    prov_merge: str | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Stamp the winner with fresh system_time + merged provenance and emit
    the loser-preserving :class:`AuditRecord`."""
    if prov_merge is None:
        prov_merge = _merge_provenance(
            str(winner.get("provenance") or ""),
            str(loser.get("provenance") or ""),
        )

    winner_out = dict(winner)
    winner_out["system_time"] = system_time
    winner_out["resolution_strategy_id"] = strategy_id
    winner_out["provenance"] = prov_merge

    audit = AuditRecord(
        subject=winner["subject"],
        predicate=winner["predicate"],
        winner_object=winner["object"],
        loser_object=loser["object"],
        winner_edge_id=winner["edge_id"],
        loser_edge_id=loser["edge_id"],
        valid_from=winner.get("valid_from") or "",
        valid_to=winner.get("valid_to") or "",
        occurred_at=winner.get("occurred_at"),
        system_time=system_time,
        provenance_merge=prov_merge,
        strategy_id=strategy_id,
        confidence_winner=float(winner.get("confidence") or 0.8),
        confidence_loser=float(loser.get("confidence") or 0.8),
    )
    return winner_out, audit


def _require_contradiction(
    f1: typing.Mapping[str, typing.Any],
    f2: typing.Mapping[str, typing.Any],
    op: str,
) -> None:
    if not is_contradiction(f1, f2):
        raise ValueError(
            f"{op}: facts must contradict "
            f"({f1.get('subject')!r}, {f1.get('predicate')!r}, "
            f"{f1.get('object')!r}) vs ({f2.get('object')!r})"
        )


# ── Selection keys (single source of truth: binary AND n-ary) ───────────────


def _lww_key(fact: typing.Mapping[str, typing.Any]) -> tuple[str, int]:
    """LWW: arg max over ``(valid_from, edge_id)`` — most recent wins."""
    return (fact.get("valid_from") or "", int(fact.get("edge_id") or 0))


def _evidence_key(
    fact: typing.Mapping[str, typing.Any],
) -> tuple[float, str, int]:
    """Evidence: arg max over ``(confidence, valid_from, edge_id)``."""
    return (
        float(fact.get("confidence") or 0.0),
        fact.get("valid_from") or "",
        int(fact.get("edge_id") or 0),
    )


_KeyFn = typing.Callable[[typing.Mapping[str, typing.Any]], typing.Any]
_SELECTION_KEYS: dict[str, _KeyFn] = {
    "lww": _lww_key,
    "evi": _evidence_key,
}


# ── The four operators (pure functions) ─────────────────────────────────────


def resolve_lww(
    incumbent: typing.Mapping[str, typing.Any],
    challenger: typing.Mapping[str, typing.Any],
    *,
    system_time: str | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Last-writer-wins: the most recent ``valid_from`` wins (toki RC)."""
    _require_contradiction(incumbent, challenger, "resolve_lww")
    system_time = system_time or _now()
    winner, loser = max(
        ((incumbent, challenger), (challenger, incumbent)),
        key=lambda pair: _lww_key(pair[0]),
    )
    return _emit_resolution(winner, loser, "lww", system_time)


def resolve_evidence(
    incumbent: typing.Mapping[str, typing.Any],
    challenger: typing.Mapping[str, typing.Any],
    *,
    system_time: str | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Evidence-weighted: the higher ``confidence`` wins (toki SI)."""
    _require_contradiction(incumbent, challenger, "resolve_evidence")
    system_time = system_time or _now()
    winner, loser = max(
        ((incumbent, challenger), (challenger, incumbent)),
        key=lambda pair: _evidence_key(pair[0]),
    )
    return _emit_resolution(winner, loser, "evi", system_time)


def resolve_await(
    incumbent: typing.Mapping[str, typing.Any],
    challenger: typing.Mapping[str, typing.Any],
    *,
    judge_callback: typing.Callable[
        [typing.Mapping[str, typing.Any], typing.Mapping[str, typing.Any]], int
    ],
    system_time: str | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """AwaitConfirm: an external callback votes 0 (incumbent) or 1 (challenger)."""
    _require_contradiction(incumbent, challenger, "resolve_await")
    system_time = system_time or _now()
    vote = judge_callback(incumbent, challenger)
    if vote not in (0, 1):
        raise ValueError(f"judge_callback must return 0 or 1; got {vote!r}")
    winner, loser = (incumbent, challenger) if vote == 0 else (challenger, incumbent)
    winner_out, audit = _emit_resolution(winner, loser, "await", system_time)
    audit.judge_verdict = r_key(incumbent, challenger)
    return winner_out, audit


def resolve_rule(
    incumbent: typing.Mapping[str, typing.Any],
    challenger: typing.Mapping[str, typing.Any],
    *,
    policy_oracle: typing.Callable[
        [typing.Mapping[str, typing.Any], typing.Mapping[str, typing.Any]], int
    ],
    system_time: str | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """PerRule: a policy oracle votes 0 (incumbent) or 1 (challenger) (toki SR)."""
    _require_contradiction(incumbent, challenger, "resolve_rule")
    system_time = system_time or _now()
    vote = policy_oracle(incumbent, challenger)
    if vote not in (0, 1):
        raise ValueError(f"policy_oracle must return 0 or 1; got {vote!r}")
    winner, loser = (incumbent, challenger) if vote == 0 else (challenger, incumbent)
    winner_out, audit = _emit_resolution(winner, loser, "rule", system_time)
    audit.judge_verdict = r_key(incumbent, challenger)
    return winner_out, audit


def resolve_by_strategy(
    incumbent: typing.Mapping[str, typing.Any],
    challenger: typing.Mapping[str, typing.Any],
    *,
    strategy_id: str,
    system_time: str | None = None,
    judge_callback: typing.Callable[..., int] | None = None,
    policy_oracle: typing.Callable[..., int] | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Dispatch to the named operator (``lww`` / ``evi`` / ``await`` / ``rule``).

    Raises ``ValueError`` on an unknown strategy or a missing oracle (rule 02:
    no silent fallback).
    """
    if strategy_id == "lww":
        return resolve_lww(incumbent, challenger, system_time=system_time)
    if strategy_id == "evi":
        return resolve_evidence(incumbent, challenger, system_time=system_time)
    if strategy_id == "await":
        if judge_callback is None:
            raise ValueError("resolve_by_strategy: 'await' requires judge_callback")
        return resolve_await(
            incumbent,
            challenger,
            judge_callback=judge_callback,
            system_time=system_time,
        )
    if strategy_id == "rule":
        if policy_oracle is None:
            raise ValueError("resolve_by_strategy: 'rule' requires policy_oracle")
        return resolve_rule(
            incumbent, challenger, policy_oracle=policy_oracle, system_time=system_time
        )
    raise ValueError(
        f"resolve_by_strategy: unknown strategy {strategy_id!r}; "
        f"expected one of {_ALL_STRATEGIES}"
    )


def resolve_pair_unchecked(
    incumbent: typing.Mapping[str, typing.Any],
    challenger: typing.Mapping[str, typing.Any],
    *,
    strategy_id: str,
    system_time: str | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Resolve a pair already established as conflicting by the CALLER's model.

    toki's :func:`is_contradiction` is a same-``(subject, predicate)`` /
    different-``object`` predicate. simba's verifier flags a different topology:
    antonym predicate pairs (``uses`` / ``does_not_use``) or a repeated predicate
    on a shared ``(subject, object)`` with overlapping belief-time. This entry
    point skips the toki object-mismatch guard so the simba supersession path can
    drive the same LWW / Evidence selection key + audit emission over its own
    conflict witnesses, without misclassifying them as non-contradictions.

    Only the selection strategies (``lww`` / ``evi``) are supported here; oracle
    strategies must go through the judge-logged :func:`resolve_await` /
    :func:`resolve_rule` so the N1 keyed-log ordering holds.
    """
    if strategy_id not in _SELECTION_KEYS:
        raise ValueError(
            f"resolve_pair_unchecked supports only {_SELECTION_STRATEGIES}; "
            f"got {strategy_id!r}"
        )
    system_time = system_time or _now()
    key = _SELECTION_KEYS[strategy_id]
    winner, loser = max(
        ((incumbent, challenger), (challenger, incumbent)),
        key=lambda pair: key(pair[0]),
    )
    return _emit_resolution(winner, loser, strategy_id, system_time)


# ── N-ary conflict-set fold (n == 2 equals the binary operator) ─────────────


def _require_pairwise_contradiction(
    facts: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> None:
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            if not is_contradiction(facts[i], facts[j]):
                raise ValueError(
                    "resolve_conflict_set requires a pairwise-contradicting set; "
                    f"members ({facts[i].get('edge_id')!r}, "
                    f"{facts[j].get('edge_id')!r}) do not contradict"
                )


def resolve_conflict_set(
    facts: typing.Sequence[typing.Mapping[str, typing.Any]],
    *,
    strategy_id: str,
    system_time: str | None = None,
    judge_callback: typing.Callable[
        [typing.Sequence[typing.Mapping[str, typing.Any]]], int
    ]
    | None = None,
    policy_oracle: typing.Callable[
        [typing.Sequence[typing.Mapping[str, typing.Any]]], int
    ]
    | None = None,
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Resolve an n-ary conflict set (``n >= 2``) under one strategy.

    LWW / Evidence are order-independent folds (``arg max`` on the selection
    key). AwaitConfirm / PerRule ask an oracle for an index into the canonically
    ordered set. The winner inherits the merge of EVERY member's provenance, so
    the audit row dominates every loser. For ``n == 2`` the result matches the
    binary operator bit-for-bit (same key + same merge).
    """
    if len(facts) < 2:
        raise ValueError(
            f"resolve_conflict_set requires at least two facts; got {len(facts)}"
        )
    _require_pairwise_contradiction(facts)
    system_time = system_time or _now()

    if strategy_id in _SELECTION_KEYS:
        winner = max(facts, key=_SELECTION_KEYS[strategy_id])
    elif strategy_id == "await":
        if judge_callback is None:
            raise ValueError("resolve_conflict_set: 'await' requires judge_callback")
        ordered = sorted(facts, key=lambda f: int(f.get("edge_id") or 0))
        idx = judge_callback(ordered)
        if not (isinstance(idx, int) and 0 <= idx < len(ordered)):
            raise ValueError(f"await index {idx!r} out of range [0, {len(ordered)})")
        winner = ordered[idx]
    elif strategy_id == "rule":
        if policy_oracle is None:
            raise ValueError("resolve_conflict_set: 'rule' requires policy_oracle")
        ordered = sorted(facts, key=lambda f: int(f.get("edge_id") or 0))
        idx = policy_oracle(ordered)
        if not (isinstance(idx, int) and 0 <= idx < len(ordered)):
            raise ValueError(f"rule index {idx!r} out of range [0, {len(ordered)})")
        winner = ordered[idx]
    else:
        raise ValueError(
            f"resolve_conflict_set: unknown strategy {strategy_id!r}; "
            f"expected one of {_ALL_STRATEGIES}"
        )

    winner_id = winner["edge_id"]
    losers = [f for f in facts if f["edge_id"] != winner_id]
    prov_merge = _merge_provenance_all(
        [str(winner.get("provenance") or "")]
        + [str(f.get("provenance") or "") for f in losers]
    )
    # Representative loser = max edge_id, so the audit row identity is stable.
    representative = max(losers, key=lambda f: int(f.get("edge_id") or 0))
    winner_out, audit = _emit_resolution(
        winner, representative, strategy_id, system_time, prov_merge=prov_merge
    )
    if strategy_id in _ORACLE_STRATEGIES:
        audit.judge_verdict = r_key_set(facts)
    return winner_out, audit


# ── Keyed judge-log (N1 defence) ────────────────────────────────────────────


def r_key(
    f1: typing.Mapping[str, typing.Any], f2: typing.Mapping[str, typing.Any]
) -> str:
    """Canonical, order-independent judge-log key for a contradicting pair."""
    return repr(tuple(sorted((int(f1["edge_id"]), int(f2["edge_id"])))))


def r_key_set(facts: typing.Sequence[typing.Mapping[str, typing.Any]]) -> str:
    """Canonical, order-independent judge-log key for an n-ary conflict set."""
    return repr(tuple(sorted(int(f["edge_id"]) for f in facts)))


def record_judge_verdict(
    r_key_value: str,
    theta: str,
    vote: int,
    *,
    winner_edge_id: int | None = None,
    system_time: str | None = None,
) -> None:
    """Append one witnessed verdict to ``neuron_judge_log`` BEFORE the operator
    commits its state mutation (the H1 ordering invariant: N1 defence).

    Append-only: a row is never overwritten; ``seq`` totally orders arrivals.
    Raises on a non-binary vote (rule 02: no silent coercion).
    """
    import simba.db
    import simba.neuron.schema  # registers neuron_judge_log

    _ = simba.neuron.schema
    if vote not in (0, 1):
        raise ValueError(f"vote must be 0 or 1; got {vote!r}")
    system_time = system_time or _now()
    with simba.db.connect() as db:
        db.execute_sql(
            "INSERT INTO neuron_judge_log "
            "(r_key, theta, vote, winner_edge_id, system_time) "
            "VALUES (?, ?, ?, ?, ?)",
            (r_key_value, theta, vote, winner_edge_id, system_time),
        )


def query_judge_verdicts(r_key_value: str, theta: str) -> list[JudgeVerdictRecord]:
    """Return all verdicts for ``(r_key, theta)`` in arrival order (replay path).

    Empty list if the key was never recorded. This is the durable surface a
    crash-recovery re-queries to re-apply the committed vote deterministically.
    """
    import simba.db
    import simba.neuron.schema

    _ = simba.neuron.schema
    with simba.db.connect() as db:
        cursor = db.execute_sql(
            "SELECT seq, r_key, theta, vote, winner_edge_id, system_time "
            "FROM neuron_judge_log WHERE r_key = ? AND theta = ? ORDER BY seq ASC",
            (r_key_value, theta),
        )
        return [
            JudgeVerdictRecord(
                seq=row[0],
                r_key=row[1],
                theta=row[2],
                vote=row[3],
                winner_edge_id=row[4],
                system_time=row[5],
            )
            for row in cursor.fetchall()
        ]


def replay_from_log(
    f1: typing.Mapping[str, typing.Any],
    f2: typing.Mapping[str, typing.Any],
    r_key_value: str,
    theta: str,
    *,
    strategy_id: str = "await",
) -> tuple[dict[str, typing.Any], AuditRecord]:
    """Replay an oracle verdict recovered from the judge-log (N1 defence).

    Re-runs the named oracle operator with a constant oracle returning the
    committed vote, reproducing the originally-elected winner. Read-only on the
    log. Raises if ``(r_key, theta)`` was never recorded.
    """
    verdicts = query_judge_verdicts(r_key_value, theta)
    if not verdicts:
        raise ValueError(f"replay_from_log: no verdict at ({r_key_value!r}, {theta!r})")
    vote = verdicts[-1].vote
    if strategy_id == "await":
        return resolve_await(f1, f2, judge_callback=lambda a, b: vote)
    if strategy_id == "rule":
        return resolve_rule(f1, f2, policy_oracle=lambda a, b: vote)
    raise ValueError(
        f"replay_from_log: strategy must be 'await' or 'rule'; got {strategy_id!r}"
    )


# ── Append-only audit trail (N3 defence) ────────────────────────────────────


def record_audit(
    audit: AuditRecord,
    *,
    project_path: str | None = None,
    system_time: str | None = None,
) -> int:
    """Append an :class:`AuditRecord` to ``kg_audit_resolutions`` (N3 defence).

    Returns the new row id. The loser's object, edge id, and merged provenance
    are preserved so the superseded fact is always recoverable. Append-only:
    repeated resolutions of the same loser become distinct rows keyed by
    ``(loser_edge_id, system_time)``.
    """
    import simba.db
    import simba.neuron.schema  # registers kg_audit_resolutions

    _ = simba.neuron.schema
    if project_path is None:
        project_path = simba.db.resolve_project_id()
    system_time = system_time or audit.system_time or _now()
    with simba.db.connect() as db:
        cursor = db.execute_sql(
            "INSERT INTO kg_audit_resolutions "
            "(subject, predicate, winner_object, loser_object, "
            "winner_edge_id, loser_edge_id, valid_from, valid_to, occurred_at, "
            "system_time, provenance_merge, strategy_id, confidence_winner, "
            "confidence_loser, judge_verdict, project_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                audit.subject,
                audit.predicate,
                audit.winner_object,
                audit.loser_object,
                audit.winner_edge_id,
                audit.loser_edge_id,
                audit.valid_from,
                audit.valid_to,
                audit.occurred_at,
                system_time,
                audit.provenance_merge,
                audit.strategy_id,
                audit.confidence_winner,
                audit.confidence_loser,
                audit.judge_verdict,
                project_path,
                _now(),
            ),
        )
        return cursor.lastrowid


def query_audit(
    *,
    loser_edge_id: int | None = None,
    project_path: str | None = None,
) -> list[dict[str, typing.Any]]:
    """Return audit rows (the N3 recovery surface), newest-first.

    Filter by ``loser_edge_id`` to recover a specific superseded fact, and/or
    by ``project_path`` for scoping.
    """
    import simba.db
    import simba.neuron.schema

    _ = simba.neuron.schema
    clauses: list[str] = []
    params: list[typing.Any] = []
    if loser_edge_id is not None:
        clauses.append("loser_edge_id = ?")
        params.append(loser_edge_id)
    if project_path is not None:
        clauses.append("project_path = ?")
        params.append(project_path)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    cols = [
        "id",
        "subject",
        "predicate",
        "winner_object",
        "loser_object",
        "winner_edge_id",
        "loser_edge_id",
        "valid_from",
        "valid_to",
        "occurred_at",
        "system_time",
        "provenance_merge",
        "strategy_id",
        "confidence_winner",
        "confidence_loser",
        "judge_verdict",
        "project_path",
        "created_at",
    ]
    with simba.db.connect() as db:
        cursor = db.execute_sql(
            f"SELECT {', '.join(cols)} FROM kg_audit_resolutions{where} "
            "ORDER BY id DESC",
            tuple(params),
        )
        return [dict(zip(cols, row, strict=True)) for row in cursor.fetchall()]


__all__ = [
    "AuditRecord",
    "JudgeVerdictRecord",
    "is_contradiction",
    "provenance_dominates",
    "query_audit",
    "query_judge_verdicts",
    "r_key",
    "r_key_set",
    "record_audit",
    "record_judge_verdict",
    "replay_from_log",
    "resolve_await",
    "resolve_by_strategy",
    "resolve_conflict_set",
    "resolve_evidence",
    "resolve_lww",
    "resolve_pair_unchecked",
    "resolve_rule",
]

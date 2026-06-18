"""Tests for same-session candidate expansion."""

from __future__ import annotations

import simba.memory.session_expand as sx


def _rec(mid: str, session: str, score: float = 0.0) -> dict:
    return {
        "id": mid,
        "content": mid,
        "sessionSource": session,
        "rrf_score": score,
    }


def test_seed_sessions_preserves_first_seen_order() -> None:
    seeds = sx.seed_sessions(
        [_rec("a", "s1"), _rec("b", "s1"), _rec("c", "s2"), _rec("d", "s3")],
        top_sessions=2,
    )

    assert seeds == ["s1", "s2"]


def test_fold_session_records_adds_missing_and_boosts_existing() -> None:
    fused = [_rec("a", "s1", 0.05), _rec("b", "s2", 0.03)]
    out = sx.fold_session_records(
        fused,
        [_rec("a", "s1"), _rec("c", "s1")],
        rrf_k=20,
        weight=1.0,
    )

    assert {r["id"] for r in out} == {"a", "b", "c"}
    assert out[0]["id"] == "a"
    assert next(r for r in out if r["id"] == "c")["rrf_score"] > 0.0

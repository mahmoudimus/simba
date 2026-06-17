"""Tests for the intent-priming injection builder (spec 28 Phase B)."""

from __future__ import annotations

import simba.doctrine.priming as priming
import simba.doctrine.store as store


def _doctrine(did, doctrine, triggers, embeddings, *, risk_tier=False, rules=()):
    return store.Doctrine(
        id=did,
        doctrine=doctrine,
        triggers=list(triggers),
        trigger_embeddings=[list(e) for e in embeddings],
        risk_tier=risk_tier,
        applicable_rules=list(rules),
        project_path="",
    )


class TestPrime:
    def test_matched_doctrine_is_injected(self) -> None:
        pr = _doctrine(
            "pr",
            "Use the worktree skill for PR review.",
            ["review PR"],
            [[1.0, 0.0]],
            risk_tier=True,
            rules=["redirect: git show pr-N -> worktree"],
        )
        result = priming.prime(
            "please review PR #42",
            doctrines=[pr],
            embed_fn=lambda _t: [1.0, 0.0],
            min_similarity=0.55,
            max_doctrines=3,
        )
        assert "intent-priming" in result.text
        assert "Use the worktree skill" in result.text
        assert "git show pr-N" in result.text  # applicable gate listed
        assert result.risk_primed is True

    def test_no_match_is_empty(self) -> None:
        pr = _doctrine("pr", "PR doctrine", ["review PR"], [[1.0, 0.0]])
        result = priming.prime(
            "totally unrelated request",
            doctrines=[pr],
            embed_fn=lambda _t: [0.0, 1.0],  # orthogonal
            min_similarity=0.55,
        )
        assert result.text == ""
        assert result.risk_primed is False

    def test_non_risk_match_does_not_arm(self) -> None:
        d = _doctrine("d", "a non-risk note", ["note about X"], [[1.0, 0.0]])
        result = priming.prime(
            "note about X please",
            doctrines=[d],
            embed_fn=lambda _t: [1.0, 0.0],
            min_similarity=0.55,
        )
        assert result.text != ""  # still primed (advisory)
        assert result.risk_primed is False  # but not a risk-tier prime

    def test_caps_injected_doctrines(self) -> None:
        ds = [
            _doctrine(f"d{i}", f"doctrine {i}", [f"t{i}"], [[1.0, 0.0]])
            for i in range(5)
        ]
        result = priming.prime(
            "t0 t1 t2 t3 t4",
            doctrines=ds,
            embed_fn=lambda _t: [1.0, 0.0],
            min_similarity=0.55,
            max_doctrines=2,
        )
        # at most 2 doctrine bodies injected
        assert result.text.count("doctrine ") == 2

    def test_empty_doctrines_no_embed(self) -> None:
        calls: list[str] = []
        result = priming.prime(
            "anything",
            doctrines=[],
            embed_fn=lambda t: calls.append(t) or [1.0],
            min_similarity=0.55,
        )
        assert result.text == ""
        assert calls == []

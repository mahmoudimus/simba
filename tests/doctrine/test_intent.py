"""Tests for the cheap doctrine intent matcher (spec 28 Phase A).

The matcher is a pure cosine match over precomputed trigger embeddings — no LLM,
no network. The hot-path embed of the prompt is injected as ``embed_fn`` so tests
never touch a real embedder.
"""

from __future__ import annotations

import simba.doctrine.intent as intent
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


class TestCosine:
    def test_identical_vectors_score_one(self) -> None:
        assert intent._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors_score_zero(self) -> None:
        assert intent._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_is_safe(self) -> None:
        assert intent._cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestMatchDoctrine:
    def test_review_pr_prompt_matches_pr_trigger(self) -> None:
        # "review PR" prompt embeds near the PR-review trigger, far from the other.
        pr = _doctrine(
            "pr",
            "Use the worktree skill for PR review; never review-in-place.",
            ["review PR", "PR review"],
            [[1.0, 0.0]],
            risk_tier=True,
        )
        drizzle = _doctrine(
            "drz",
            "Regenerate init-schema via the docker script; never hand-edit.",
            ["regenerate init-schema", "drizzle migration"],
            [[0.0, 1.0]],
            risk_tier=True,
        )
        prompt_vec = [0.98, 0.02]  # near the PR trigger
        matches = intent.match_doctrine(prompt_vec, [pr, drizzle], min_similarity=0.55)
        assert [m.doctrine.id for m in matches] == ["pr"]
        assert matches[0].similarity > 0.9

    def test_unrelated_prompt_matches_nothing(self) -> None:
        pr = _doctrine("pr", "PR review doctrine", ["review PR"], [[1.0, 0.0]])
        prompt_vec = [0.0, 1.0]  # orthogonal to the only trigger
        matches = intent.match_doctrine(prompt_vec, [pr], min_similarity=0.55)
        assert matches == []

    def test_max_over_multiple_triggers(self) -> None:
        # A doctrine matches on its BEST trigger, not the first.
        d = _doctrine(
            "d",
            "multi-trigger doctrine",
            ["alpha", "beta"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        matches = intent.match_doctrine([0.0, 1.0], [d], min_similarity=0.55)
        assert len(matches) == 1
        assert matches[0].similarity == 1.0

    def test_sorted_by_similarity_desc(self) -> None:
        a = _doctrine("a", "A", ["a"], [[1.0, 0.0]])
        b = _doctrine("b", "B", ["b"], [[0.7, 0.7]])
        matches = intent.match_doctrine([1.0, 0.0], [a, b], min_similarity=0.3)
        assert [m.doctrine.id for m in matches] == ["a", "b"]


class TestClassify:
    def test_classify_embeds_prompt_once_and_matches(self) -> None:
        calls: list[str] = []

        def fake_embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0]

        pr = _doctrine("pr", "PR doctrine", ["review PR"], [[1.0, 0.0]], risk_tier=True)
        matches = intent.classify(
            "please review PR #42",
            [pr],
            embed_fn=fake_embed,
            min_similarity=0.55,
        )
        assert len(calls) == 1  # one embed call, no per-trigger embedding on hot path
        assert [m.doctrine.id for m in matches] == ["pr"]

    def test_classify_empty_doctrine_list_no_embed(self) -> None:
        calls: list[str] = []

        def fake_embed(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0]

        matches = intent.classify("anything", [], embed_fn=fake_embed)
        assert matches == []
        assert calls == []  # nothing to match -> never pays the embed

    def test_classify_fail_open_on_embed_error(self) -> None:
        def boom(text: str) -> list[float]:
            raise RuntimeError("embed down")

        pr = _doctrine("pr", "PR doctrine", ["review PR"], [[1.0, 0.0]])
        # Any embed failure returns no matches (priming is advisory; never crash).
        assert intent.classify("review PR", [pr], embed_fn=boom) == []

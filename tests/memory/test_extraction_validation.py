"""Tests for the write-time extraction validation gate (Eywa-style).

A pure, deterministic check that an LLM-extracted claim is grounded in its source
span before it is promoted to a stored belief: hard-value (number/date) preservation,
support overlap, and polarity. No LLM. Source is the *supporting span*, not a whole
transcript. Borrowed from Eywa (arXiv 2605.30771).
"""

from __future__ import annotations

from simba.memory.extraction_validation import validate_extraction


class TestHardValue:
    def test_hallucinated_amount_fails(self) -> None:
        r = validate_extraction(
            "I paid $3,750 for the laptop", "I paid $2,750 for the laptop last week"
        )
        assert not r.ok and "hard_value" in r.failed

    def test_present_amount_passes(self) -> None:
        r = validate_extraction(
            "the trip lasted 8 days", "we camped and the trip lasted 8 days total"
        )
        assert r.ok

    def test_hallucinated_date_fails(self) -> None:
        r = validate_extraction(
            "baked sourdough on 2023-05-16", "baked sourdough on 2023-05-20"
        )
        assert not r.ok and "hard_value" in r.failed

    def test_comma_and_plain_number_normalize_equal(self) -> None:
        r = validate_extraction(
            "reached 12000 points", "she reached 12,000 points in the program"
        )
        assert r.ok


class TestSupportOverlap:
    def test_unsupported_claim_fails(self) -> None:
        r = validate_extraction(
            "the user adopted a golden retriever puppy",
            "we discussed quarterly cloud migration budgets",
        )
        assert not r.ok and "support" in r.failed

    def test_faithful_subset_passes(self) -> None:
        r = validate_extraction(
            "user is leading the cloud migration project",
            "I am currently leading the cloud migration project at my company",
        )
        assert r.ok


class TestPolarity:
    def test_invented_negation_fails(self) -> None:
        r = validate_extraction(
            "the user does not like espresso",
            "honestly I really like espresso in the morning",
        )
        assert not r.ok and "polarity" in r.failed

    def test_matching_polarity_passes(self) -> None:
        r = validate_extraction(
            "the user likes espresso", "I really like espresso in the morning"
        )
        assert r.ok

    def test_can_disable_polarity_check(self) -> None:
        r = validate_extraction(
            "the user does not like espresso",
            "I really like espresso in the morning",
            check_polarity=False,
        )
        # still must pass the other checks; polarity no longer fails it
        assert "polarity" not in r.failed


class TestGateControl:
    def test_disabled_always_ok(self) -> None:
        r = validate_extraction(
            "paid $9999 on 1999-01-01", "unrelated text", enabled=False
        )
        assert r.ok and r.failed == []

    def test_multiple_failures_collected(self) -> None:
        r = validate_extraction("user did not pay $5000", "we talked about the weather")
        assert not r.ok and len(r.failed) >= 2

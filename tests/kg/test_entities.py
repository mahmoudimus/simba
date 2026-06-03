"""Tests for KG entity normalization + resolution."""

from __future__ import annotations

import simba.kg.entities as ent


class TestNormalize:
    def test_lowercases_and_trims(self) -> None:
        assert ent.normalize_entity("  GitHub  ") == "github"

    def test_strips_leading_article(self) -> None:
        assert ent.normalize_entity("the GITHUB_TOKEN") == "github_token"
        assert ent.normalize_entity("a daemon") == "daemon"
        assert ent.normalize_entity("an episode") == "episode"

    def test_strips_surrounding_quotes_and_backticks(self) -> None:
        assert ent.normalize_entity('"peewee"') == "peewee"
        assert ent.normalize_entity("`kg_add`") == "kg_add"

    def test_strips_possessive_and_trailing_punct(self) -> None:
        assert ent.normalize_entity("Claude's") == "claude"
        assert ent.normalize_entity("daemon.") == "daemon"

    def test_collapses_internal_whitespace(self) -> None:
        assert ent.normalize_entity("memory   daemon") == "memory daemon"

    def test_keeps_identifier_underscores(self) -> None:
        # code identifiers must stay distinct from space-separated phrases
        assert ent.normalize_entity("GITHUB_TOKEN") == "github_token"
        assert ent.normalize_entity("github token") == "github token"

    def test_empty_stays_empty(self) -> None:
        assert ent.normalize_entity("   ") == ""


class TestResolve:
    def test_exact_normalized_match_returns_existing_canonical(self) -> None:
        existing = ["GITHUB_TOKEN", "memory daemon"]
        # "the github_token" normalizes to the same key as "GITHUB_TOKEN"
        assert ent.resolve("the GITHUB_TOKEN", existing) == "GITHUB_TOKEN"

    def test_no_match_returns_cleaned_input(self) -> None:
        # unknown entity: returned trimmed (canonical display), not normalized
        assert ent.resolve("  Brand New Thing  ", ["github"]) == "Brand New Thing"

    def test_embedding_merge_when_above_threshold(self) -> None:
        # "Bob" and "Robert" don't share a normalized key; an injected embedder
        # makes them similar enough to merge.
        vecs = {"robert": [1.0, 0.0], "bob": [0.99, 0.14]}

        def embed(name: str) -> list[float]:
            return vecs[ent.normalize_entity(name)]

        out = ent.resolve("Bob", ["Robert"], embed=embed, threshold=0.9)
        assert out == "Robert"

    def test_embedding_below_threshold_stays_distinct(self) -> None:
        vecs = {"robert": [1.0, 0.0], "alice": [0.0, 1.0]}

        def embed(name: str) -> list[float]:
            return vecs[ent.normalize_entity(name)]

        out = ent.resolve("Alice", ["Robert"], embed=embed, threshold=0.9)
        assert out == "Alice"

    def test_empty_existing_returns_cleaned_input(self) -> None:
        assert ent.resolve("Thing", []) == "Thing"

"""Tests for entropy-gated exact-term matching (src/simba/memory/entropy_terms.py).

Trigram FTS collides high-entropy tokens (50815 -> 508/081/815 overlaps 50806/50858).
The fix: route high-information query tokens to a WHOLE-WORD exact match instead of
trigrams. "Information" = corpus df-surprisal (-log2(df/N)) — NOT char-entropy, which
wrongly flags common all-distinct words like "debug". Unseen tokens (df=0) score
maximal, which is the novel-error-code case.
"""

from __future__ import annotations

import math

from simba.memory.entropy_terms import (
    exact_boost,
    high_entropy_terms,
    is_lexically_novel,
    surprisal,
)


class TestLexicalNovelty:
    def test_codes_symbols_camelcase_allcaps_are_novel(self) -> None:
        for t in (
            "50815",
            "verify.cpp",
            "%var_1DC",
            "sub_7FFD",
            "TigressNotEqualSignBitRule",
            "INTERR",
            "Hodur",
            "tigress",
        ):
            assert is_lexically_novel(t), t

    def test_common_english_words_are_not_novel(self) -> None:
        # clearly-common (zipf >= ~4): borderline tech words (debug 2.86, operand 2.19)
        # are intentionally left to the corpus-df gate, not the lexical gate.
        for t in ("how", "internal", "error", "control", "the", "block", "consistency"):
            assert not is_lexically_novel(t), t


class TestConjunctionGate:
    def test_corpus_common_marker_dropped_even_though_shape_novel(self) -> None:
        # INTERR is shape-novel (all-caps) but at df=191 it is everywhere -> dropped;
        # 50815 is novel AND rare -> kept. Neither gate alone would do this.
        terms = high_entropy_terms(
            "INTERR 50815", {"interr": 191, "50815": 4}, n_docs=5000, min_bits=6.0
        )
        assert terms == ["50815"]


class TestSurprisal:
    def test_rare_beats_common(self) -> None:
        assert surprisal(1, 1000) > surprisal(500, 1000)

    def test_unseen_is_maximal(self) -> None:
        # df=0 (novel code) must outscore any seen term
        assert surprisal(0, 1000) > surprisal(1, 1000)

    def test_monotonic_decreasing_in_df(self) -> None:
        vals = [surprisal(df, 1000) for df in (0, 1, 10, 100, 999)]
        assert vals == sorted(vals, reverse=True)

    def test_is_in_bits(self) -> None:
        # df≈N -> ~0 bits; smoothing keeps it small but >=0
        assert 0 <= surprisal(1000, 1000) < 1.0
        assert surprisal(1, 1000) > math.log2(1000) - 2


class TestHighEntropyTerms:
    def _dfmap(self):
        # 50815 rare (df 4), interr common-ish (191), 'debug/how/to' very common
        return {
            "50815": 4,
            "interr": 191,
            "debug": 1500,
            "how": 4000,
            "to": 4900,
            "error": 1200,
        }

    def test_keeps_rare_code_drops_common_words(self) -> None:
        terms = high_entropy_terms(
            "INTERR 50815 internal error how to debug",
            self._dfmap(),
            n_docs=5000,
            min_bits=6.0,
        )
        assert "50815" in terms
        for w in ("debug", "how", "to", "error", "interr"):
            assert w not in terms

    def test_unseen_token_is_high_entropy(self) -> None:
        # a code not in the corpus (df=0) must be picked up
        terms = high_entropy_terms(
            "hit INTERR 59999 today", self._dfmap(), n_docs=5000, min_bits=6.0
        )
        assert "59999" in terms

    def test_symbols_and_paths_survive_tokenization(self) -> None:
        terms = high_entropy_terms(
            "crash in verify.cpp at %var_1DC", {}, n_docs=5000, min_bits=6.0
        )
        assert "verify.cpp" in terms and "%var_1dc" in terms

    def test_empty_when_no_query(self) -> None:
        assert high_entropy_terms("", {}, n_docs=10, min_bits=6.0) == []


class TestExactBoost:
    def _pool(self):
        return [
            {"id": str(i), "content": c}
            for i, c in enumerate(
                [
                    "generic INTERR about wrong successor set 50860",  # 0
                    "INTERR 50858 bidirectional edge",  # 1
                    "another 50806 succ count note",  # 2
                    "INTERR 50815: binary rule nested drops operand",  # 3 (target)
                ]
            )
        ]

    def test_exact_match_pinned_to_top(self) -> None:
        out = exact_boost(self._pool(), ["50815"])
        assert out[0]["id"] == "3"

    def test_preserves_order_among_non_matches(self) -> None:
        out = exact_boost(self._pool(), ["50815"])
        assert [m["id"] for m in out[1:]] == ["0", "1", "2"]

    def test_whole_word_only_no_substring(self) -> None:
        pool = [
            {"id": "a", "content": "code 508150 unrelated"},
            {"id": "b", "content": "the real INTERR 50815 here"},
        ]
        out = exact_boost(pool, ["50815"])
        assert out[0]["id"] == "b"  # 508150 must NOT count as a match

    def test_case_insensitive_and_context_field(self) -> None:
        pool = [
            {"id": "x", "content": "noise"},
            {"id": "y", "content": "c", "context": "see Verify.CPP"},
        ]
        out = exact_boost(pool, ["verify.cpp"])
        assert out[0]["id"] == "y"

    def test_no_terms_is_identity(self) -> None:
        pool = self._pool()
        assert exact_boost(pool, []) == pool

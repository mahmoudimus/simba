"""Tests for the DimMem-style dimensional schema (src/simba/memory/dimensions.py).

Typed write-side fields (time / location / reason / purpose / keywords) that make
aggregation DETERMINISTIC: tag a memory at write time, then filter/count by field at
read time instead of re-individuating in-head. Borrowed from DimMem (arXiv 2605.15759).
"""
from __future__ import annotations

from simba.memory.dimensions import (
    Dimensions,
    extract_dimensions,
    filter_by,
    from_blob,
    matches,
    parse_dimensions,
    to_blob,
)


class TestExtractDeterministic:
    def test_pulls_iso_date_and_keywords(self) -> None:
        d = extract_dimensions("Acquired a peace lily from the nursery on 2023-05-06")
        assert d.time == "2023-05-06"
        assert "lily" in d.keywords or "peace" in d.keywords

    def test_normalizes_slash_date(self) -> None:
        assert extract_dimensions("baked bread on 2023/05/16").time == "2023-05-16"

    def test_bare_year(self) -> None:
        assert extract_dimensions("graduated in 2019").time == "2019"

    def test_no_date_is_none(self) -> None:
        assert extract_dimensions("I like espresso").time is None

    def test_location_reason_purpose_are_llm_only(self) -> None:
        d = extract_dimensions("Acquired a peace lily on 2023-05-06")
        assert d.location is None and d.reason is None and d.purpose is None


class TestParseLlmReply:
    def test_parses_full_object(self) -> None:
        d = parse_dimensions(
            '{"time":"2023-05-06","location":"nursery",'
            '"reason":"gift","purpose":"decor","keywords":["plant","lily"]}'
        )
        assert d.time == "2023-05-06" and d.location == "nursery"
        assert d.reason == "gift" and d.keywords == ["plant", "lily"]

    def test_tolerates_fence_and_missing_fields(self) -> None:
        d = parse_dimensions('```json\n{"time": "2019"}\n```')
        assert d.time == "2019" and d.location is None and d.keywords == []

    def test_garbage_is_empty_dimensions(self) -> None:
        assert parse_dimensions("not json") == Dimensions()


class TestMatches:
    def test_time_window(self) -> None:
        d = Dimensions(time="2023-05-06")
        assert matches(d, time_start="2023-05-01", time_end="2023-05-31")
        assert not matches(d, time_start="2023-06-01", time_end="2023-06-30")

    def test_undated_is_excluded_from_a_window(self) -> None:
        assert not matches(Dimensions(), time_start="2023-05-01", time_end="2023-05-31")

    def test_keyword(self) -> None:
        d = Dimensions(keywords=["plant", "lily"])
        assert matches(d, keyword="lily")
        assert not matches(d, keyword="bread")

    def test_no_criteria_matches(self) -> None:
        assert matches(Dimensions())


class TestFilterBy:
    def test_filters_records_in_window(self) -> None:
        recs = [
            {"id": "1", "dims": Dimensions(time="2023-05-06")},
            {"id": "2", "dims": Dimensions(time="2023-04-01")},
            {"id": "3", "dims": Dimensions()},  # undated
        ]
        out = filter_by(recs, lambda r: r["dims"],
                        time_start="2023-05-01", time_end="2023-05-31")
        assert [r["id"] for r in out] == ["1"]


class TestBlobRoundTrip:
    def test_roundtrip(self) -> None:
        d = Dimensions(time="2023-05-06", location="nursery", keywords=["plant"])
        assert from_blob(to_blob(d)) == d

    def test_blob_embeds_in_context_and_parses_back(self) -> None:
        ctx = "free-text context." + to_blob(Dimensions(time="2019"))
        assert from_blob(ctx).time == "2019"

    def test_no_blob_returns_empty(self) -> None:
        assert from_blob("just plain context") == Dimensions()

"""Tests for narrative-date resolution (src/simba/sync/dates.py)."""

from __future__ import annotations

from simba.sync.dates import resolve_occurred_at

_CREATED = "2025-06-15T12:00:00Z"


class TestAbsoluteDates:
    def test_iso_date(self) -> None:
        assert resolve_occurred_at("decided on 2025-03-01 to switch") == "2025-03-01"

    def test_month_day_year(self) -> None:
        assert resolve_occurred_at("shipped March 5, 2024") == "2024-03-05"

    def test_day_month_year(self) -> None:
        assert resolve_occurred_at("met on 5 April 2023") == "2023-04-05"

    def test_month_year_uses_first_of_month(self) -> None:
        assert resolve_occurred_at("started in March 2025") == "2025-03-01"

    def test_iso_takes_priority_over_month_name(self) -> None:
        # An exact ISO date wins over a looser month-year mention.
        out = resolve_occurred_at("around March 2025 (exactly 2025-03-14)")
        assert out == "2025-03-14"

    def test_invalid_iso_ignored(self) -> None:
        assert resolve_occurred_at("port 2025-13-99 is bogus") is None


class TestRelativeDates:
    def test_yesterday(self) -> None:
        assert resolve_occurred_at("fixed it yesterday", created_at=_CREATED) == (
            "2025-06-14"
        )

    def test_n_days_ago(self) -> None:
        assert resolve_occurred_at("3 days ago we merged", created_at=_CREATED) == (
            "2025-06-12"
        )

    def test_last_week(self) -> None:
        assert resolve_occurred_at("last week's outage", created_at=_CREATED) == (
            "2025-06-08"
        )

    def test_today(self) -> None:
        assert resolve_occurred_at("done today", created_at=_CREATED) == "2025-06-15"

    def test_relative_without_created_at_is_none(self) -> None:
        assert resolve_occurred_at("fixed it yesterday") is None


class TestNoDate:
    def test_no_date_returns_none(self) -> None:
        assert resolve_occurred_at("use ruff for linting", created_at=_CREATED) is None

    def test_empty_text(self) -> None:
        assert resolve_occurred_at("", created_at=_CREATED) is None

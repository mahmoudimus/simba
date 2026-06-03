"""Narrative-date resolution for fact extraction (event time / ``occurred_at``).

Best-effort, dependency-free: pull an absolute date out of free text, or resolve
a simple relative phrase ("yesterday", "3 days ago") against the memory's
storage time.  Returns an ISO ``YYYY-MM-DD`` string, or ``None`` when no date is
found.  Used to populate the KG's bitemporal ``occurred_at`` axis.
"""

from __future__ import annotations

import calendar
import contextlib
import datetime
import re

# Month name/abbr -> number (january..december + jan..dec).
_MONTHS: dict[str, int] = {}
for _i, _name in enumerate(calendar.month_name):
    if _name:
        _MONTHS[_name.lower()] = _i
for _i, _name in enumerate(calendar.month_abbr):
    if _name:
        _MONTHS[_name.lower()] = _i

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_DAY_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b")
_DAY_MONTH_YEAR_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b")
_MONTH_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{4})\b")
_N_AGO_RE = re.compile(r"\b(\d+)\s+(day|week|month)s?\s+ago\b", re.IGNORECASE)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30}


def _iso(year: int, month: int, day: int) -> str | None:
    """Return an ISO date string if (year, month, day) is a real calendar date."""
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def _parse_created(created_at: str | None) -> datetime.date | None:
    """Parse a memory ``createdAt`` (ISO, possibly ``Z``-suffixed) to a date."""
    if not created_at:
        return None
    with contextlib.suppress(ValueError):
        return datetime.datetime.fromisoformat(
            created_at.strip().replace("Z", "+00:00")
        ).date()
    with contextlib.suppress(ValueError):
        return datetime.date.fromisoformat(created_at[:10])
    return None


def resolve_occurred_at(text: str, *, created_at: str | None = None) -> str | None:
    """Resolve a narrative date in ``text`` to an ISO ``YYYY-MM-DD`` string.

    Tries absolute forms first (ISO, ``Month DD, YYYY``, ``DD Month YYYY``,
    ``Month YYYY`` → first of month), then relative phrases ("yesterday",
    "N days/weeks/months ago", "last week/month", "today") resolved against
    ``created_at``.  Returns ``None`` when nothing is found.
    """
    text = text or ""

    if (m := _ISO_RE.search(text)) and (iso := _iso(int(m[1]), int(m[2]), int(m[3]))):
        return iso

    if m := _MONTH_DAY_YEAR_RE.search(text):
        mon = _MONTHS.get(m[1].lower())
        if mon and (iso := _iso(int(m[3]), mon, int(m[2]))):
            return iso

    if m := _DAY_MONTH_YEAR_RE.search(text):
        mon = _MONTHS.get(m[2].lower())
        if mon and (iso := _iso(int(m[3]), mon, int(m[1]))):
            return iso

    if m := _MONTH_YEAR_RE.search(text):
        mon = _MONTHS.get(m[1].lower())
        if mon and (iso := _iso(int(m[2]), mon, 1)):
            return iso

    base = _parse_created(created_at)
    if base is None:
        return None

    low = text.lower()
    if m := _N_AGO_RE.search(low):
        days = int(m[1]) * _UNIT_DAYS[m[2].lower()]
        return (base - datetime.timedelta(days=days)).isoformat()
    if "yesterday" in low:
        return (base - datetime.timedelta(days=1)).isoformat()
    if "last week" in low:
        return (base - datetime.timedelta(days=7)).isoformat()
    if "last month" in low:
        return (base - datetime.timedelta(days=30)).isoformat()
    if "today" in low:
        return base.isoformat()
    return None

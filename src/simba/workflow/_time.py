"""Internal ISO-8601 clock helpers (UTC, ``...Z`` suffix).

Every time-dependent op in the engine accepts an injectable ``now`` ISO
string; these helpers parse/format/offset it deterministically so tests never
touch the wall clock. (Private — not part of the public 6-file surface.)
"""

from __future__ import annotations

import datetime as _dt

_FMT = "%Y-%m-%dT%H:%M:%SZ"


def now() -> str:
    """Return the current UTC time as an ISO ``...Z`` string."""
    return _dt.datetime.now(_dt.UTC).strftime(_FMT)


def parse(value: str) -> _dt.datetime:
    """Parse an ISO ``...Z`` string into a tz-aware UTC datetime."""
    return _dt.datetime.strptime(value, _FMT).replace(tzinfo=_dt.UTC)


def add_seconds(value: str, seconds: float) -> str:
    """Return ``value`` shifted forward by ``seconds`` (ISO ``...Z``)."""
    shifted = parse(value) + _dt.timedelta(seconds=seconds)
    return shifted.strftime(_FMT)


def resolve(value: str | None) -> str:
    """Return ``value`` if given, else the real clock — the injection seam."""
    return value if value is not None else now()

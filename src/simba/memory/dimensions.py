"""DimMem-style dimensional schema for stored memories (arXiv 2605.15759).

A counting question is ``individuate ∘ aggregate``. Answer-time individuation over raw
text is interpretation-bound (measured: a model reads "projects I led" 2..7 ways). The
*write-side* alternative is to tag each memory with typed DIMENSIONS — time / location /
reason / purpose / keywords — so aggregation becomes a DETERMINISTIC filter over fields
(count events in a time window, group by keyword) instead of a re-individuation in-head.

This module is pure: a :class:`Dimensions` schema, a deterministic extractor (date +
keywords — what's reliably parseable with no LLM), an LLM-reply parser (when a richer
extractor fills location/reason/purpose), deterministic :func:`matches` /
:func:`filter_by` for aggregation, and a context-blob (de)serializer so dimensions ride
in the existing ``context`` field (no schema migration). Off by default.
"""
from __future__ import annotations

import dataclasses
import json
import re

import simba.memory.keywords

# ISO (2023-05-06), slash (2023/05/06), or a bare 4-digit year. First match wins.
_DATE = re.compile(r"\b(\d{4})[-/](\d{2})[-/](\d{2})\b|\b(19|20)\d{2}\b")
_BLOB_RE = re.compile(r"⟦dims⟧(\{.*?\})", re.DOTALL)
_BLOB_PREFIX = "\n\n⟦dims⟧"


@dataclasses.dataclass
class Dimensions:
    time: str | None = None          # normalized: "yyyy-mm-dd" or "yyyy"
    location: str | None = None
    reason: str | None = None
    purpose: str | None = None
    keywords: list[str] = dataclasses.field(default_factory=list)


def _first_date(text: str) -> str | None:
    m = _DATE.search(text or "")
    if not m:
        return None
    if m.group(1):  # yyyy[-/]mm[-/]dd
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return m.group(0)  # bare year


def extract_dimensions(text: str, *, max_keywords: int = 8) -> Dimensions:
    """Deterministic extraction (no LLM): the first date + salient keywords.

    location / reason / purpose are left None — they need an LLM (see
    :func:`parse_dimensions`); the date + keyword dimensions are the reliable subset
    that already enables time-window and keyword aggregation.
    """
    return Dimensions(
        time=_first_date(text),
        keywords=simba.memory.keywords.focus_terms(text or "", max_terms=max_keywords),
    )


def _opt_str(v: object) -> str | None:
    s = str(v).strip() if isinstance(v, str) else ""
    return s or None


def parse_dimensions(text: str) -> Dimensions:
    """Parse an LLM reply (possibly fenced / with prose) into :class:`Dimensions`.

    Any non-JSON / missing field is tolerated — a hard failure yields empty Dimensions
    (fail-open), so a malformed extraction never blocks a store."""
    text = (text or "").strip()
    lb, rb = text.find("{"), text.rfind("}")
    if lb == -1 or rb <= lb:
        return Dimensions()
    try:
        data = json.loads(text[lb:rb + 1])
    except (json.JSONDecodeError, ValueError):
        return Dimensions()
    if not isinstance(data, dict):
        return Dimensions()
    kw = data.get("keywords")
    keywords = [str(x).strip() for x in kw if str(x).strip()] \
        if isinstance(kw, list) else []
    return Dimensions(
        time=_opt_str(data.get("time")),
        location=_opt_str(data.get("location")),
        reason=_opt_str(data.get("reason")),
        purpose=_opt_str(data.get("purpose")),
        keywords=keywords,
    )


def matches(
    dims: Dimensions,
    *,
    time_start: str | None = None,
    time_end: str | None = None,
    keyword: str | None = None,
    location: str | None = None,
) -> bool:
    """Deterministic aggregation predicate. A time window EXCLUDES undated memories
    (they are not provably in-window — the possible-worlds-honest default). String
    date comparison is valid because dates are normalized big-endian."""
    if time_start is not None or time_end is not None:
        if dims.time is None:
            return False
        if time_start is not None and dims.time < time_start:
            return False
        if time_end is not None and dims.time > time_end:
            return False
    kws = {k.lower() for k in dims.keywords}
    if keyword is not None and keyword.lower() not in kws:
        return False
    loc = (dims.location or "").lower()
    return location is None or loc == location.lower()


def filter_by(records: list, get_dims, **criteria) -> list:
    """Filter ``records`` to those whose dimensions (via ``get_dims``) match."""
    return [r for r in records if matches(get_dims(r), **criteria)]


def to_blob(dims: Dimensions) -> str:
    """Serialize dimensions to a compact tagged blob appended to a ``context`` field
    (no schema migration). Only non-empty fields are written."""
    payload = {k: v for k, v in dataclasses.asdict(dims).items() if v}
    return f"{_BLOB_PREFIX}{json.dumps(payload, separators=(',', ':'))}⟧"


def from_blob(context: str) -> Dimensions:
    """Recover dimensions embedded in a ``context`` field, or empty if none present."""
    m = _BLOB_RE.search(context or "")
    return parse_dimensions(m.group(1)) if m else Dimensions()

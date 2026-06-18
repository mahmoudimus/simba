"""Tiny structured-filter grammar for recall queries."""

from __future__ import annotations

import dataclasses
import json
import re
import typing

import simba.memory.vector_db

_TOKEN_RE = re.compile(
    r"(?P<key>type|project|after|before|tag|path|symbol):"
    r"(?P<value>\"[^\"]+\"|'[^']+'|\S+)",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class ParsedQuery:
    query: str
    route_filters: dict[str, typing.Any]
    post_filters: dict[str, typing.Any]


def _clean_value(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw


def parse(query: str) -> ParsedQuery:
    """Extract supported ``key:value`` filters and return the cleaned query."""
    route_filters: dict[str, typing.Any] = {}
    post_filters: dict[str, typing.Any] = {}

    def _replace(match: re.Match[str]) -> str:
        key = match.group("key").lower()
        value = _clean_value(match.group("value"))
        if key == "type":
            route_filters["types"] = [
                part.strip().upper()
                for part in value.split(",")
                if part.strip()
            ]
        elif key == "project":
            route_filters["projectPath"] = (
                simba.memory.vector_db.normalize_project_path(value)
            )
        elif key in {"after", "before", "tag", "path", "symbol"}:
            post_filters[key] = value
        return " "

    cleaned = _TOKEN_RE.sub(_replace, query)
    cleaned = " ".join(cleaned.split())
    return ParsedQuery(
        query=cleaned or query,
        route_filters=route_filters,
        post_filters=post_filters,
    )


def _tags(record: dict[str, typing.Any]) -> list[str]:
    raw = record.get("tags") or "[]"
    if isinstance(raw, list):
        return [str(t) for t in raw]
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [str(t) for t in parsed] if isinstance(parsed, list) else []


def apply(
    records: list[dict[str, typing.Any]], post_filters: dict[str, typing.Any]
) -> list[dict[str, typing.Any]]:
    """Apply metadata filters that cannot be pushed into the retrieval arms."""
    if not post_filters:
        return records
    out: list[dict[str, typing.Any]] = []
    tag = str(post_filters.get("tag", "")).lower()
    path = str(post_filters.get("path", "")).lower()
    symbol = str(post_filters.get("symbol", "")).lower()
    after = str(post_filters.get("after", ""))
    before = str(post_filters.get("before", ""))
    for rec in records:
        created = str(rec.get("createdAt") or "")
        haystack = f"{rec.get('content', '')}\n{rec.get('context', '')}".lower()
        if after and created and created[:10] < after:
            continue
        if before and created and created[:10] > before:
            continue
        if tag and tag not in {t.lower() for t in _tags(rec)}:
            continue
        if path and path not in haystack:
            continue
        if symbol and symbol not in haystack:
            continue
        out.append(rec)
    return out

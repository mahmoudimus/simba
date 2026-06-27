"""Local location-containment ratifier for envelope membership.

This module is deliberately narrower than a geocoder. It answers whether a
listed evidence location is contained by a question target such as
``United States`` using a provenance-carrying containment graph.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import json
import pathlib
import re
import typing

DEFAULT_LOCATION_PATH = pathlib.Path(__file__).parent / "datasets" / (
    "location_containment.json"
)
MAX_RATIFICATION_DEPTH = 6
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclasses.dataclass(frozen=True)
class LocationRatification:
    source_location: str
    target_location: str
    ratified: bool
    path: tuple[str, ...]
    provenance: tuple[str, ...]
    reason: str


@dataclasses.dataclass(frozen=True)
class _LocationRecord:
    location_id: str
    label: str
    aliases: tuple[str, ...]
    parent_ids: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def terms(self) -> tuple[str, ...]:
        return _unique_terms((self.label, *self.aliases))


@dataclasses.dataclass(frozen=True)
class _LocationGraph:
    records: dict[str, _LocationRecord]
    term_index: dict[str, tuple[str, ...]]

    def lookup(self, phrase: str) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for variant in _phrase_variants(phrase):
            for location_id in self.term_index.get(variant, ()):
                if location_id in seen:
                    continue
                seen.add(location_id)
                ids.append(location_id)
        return tuple(ids)

    def neighbors(self, location_id: str) -> tuple[str, ...]:
        record = self.records.get(location_id)
        if record is None:
            return ()
        return tuple(parent for parent in record.parent_ids if parent in self.records)


def ratify_location_containment(
    source_location: str,
    target_location: str,
    *,
    location_path: str | pathlib.Path | None = None,
) -> LocationRatification:
    source = _canonical_phrase(source_location)
    target = _canonical_phrase(target_location)
    if not source or not target:
        return LocationRatification(
            source_location=source_location,
            target_location=target_location,
            ratified=False,
            path=(),
            provenance=(),
            reason="empty_location",
        )
    if source == target:
        return LocationRatification(
            source_location=source_location,
            target_location=target_location,
            ratified=True,
            path=(source,),
            provenance=(),
            reason="same_location",
        )

    graph = _load_locations(str(pathlib.Path(location_path or DEFAULT_LOCATION_PATH)))
    source_ids = graph.lookup(source)
    target_ids = set(graph.lookup(target))
    if not source_ids:
        return LocationRatification(
            source_location=source_location,
            target_location=target_location,
            ratified=False,
            path=(),
            provenance=(),
            reason="source_location_not_found",
        )
    if not target_ids:
        return LocationRatification(
            source_location=source_location,
            target_location=target_location,
            ratified=False,
            path=(),
            provenance=(),
            reason="target_location_not_found",
        )

    ratified_path = _shortest_path(
        graph,
        source_ids=source_ids,
        target_ids=target_ids,
    )
    if not ratified_path:
        return LocationRatification(
            source_location=source_location,
            target_location=target_location,
            ratified=False,
            path=(),
            provenance=(),
            reason="no_containment_path",
        )
    return LocationRatification(
        source_location=source_location,
        target_location=target_location,
        ratified=True,
        path=tuple(graph.records[item].label for item in ratified_path),
        provenance=tuple(
            provenance
            for item in ratified_path
            for provenance in graph.records[item].provenance
        ),
        reason="location_containment",
    )


def _shortest_path(
    graph: _LocationGraph,
    *,
    source_ids: tuple[str, ...],
    target_ids: set[str],
) -> tuple[str, ...]:
    queue: collections.deque[tuple[str, tuple[str, ...]]] = collections.deque(
        (source_id, (source_id,)) for source_id in source_ids
    )
    visited: set[str] = set()
    while queue:
        location_id, path = queue.popleft()
        if location_id in visited:
            continue
        visited.add(location_id)
        if location_id in target_ids:
            return path
        if len(path) - 1 >= MAX_RATIFICATION_DEPTH:
            continue
        for neighbor_id in graph.neighbors(location_id):
            if neighbor_id not in visited:
                queue.append((neighbor_id, (*path, neighbor_id)))
    return ()


@functools.cache
def _load_locations(path: str) -> _LocationGraph:
    location_path = pathlib.Path(path)
    if not location_path.exists():
        return _LocationGraph(records={}, term_index={})
    payload = json.loads(location_path.read_text(encoding="utf-8"))
    records: dict[str, _LocationRecord] = {}
    term_members: dict[str, list[str]] = collections.defaultdict(list)
    if not isinstance(payload, list):
        return _LocationGraph(records={}, term_index={})

    for item in payload:
        if not isinstance(item, dict):
            continue
        record = _record_from_payload(item)
        if record is None:
            continue
        records[record.location_id] = record

    for location_id, record in records.items():
        for term in record.terms:
            for variant in _phrase_variants(term):
                term_members[variant].append(location_id)

    return _LocationGraph(
        records=records,
        term_index={
            term: tuple(sorted(set(location_ids)))
            for term, location_ids in term_members.items()
        },
    )


def _record_from_payload(payload: dict[str, typing.Any]) -> _LocationRecord | None:
    location_id = str(payload.get("id", "")).strip()
    label = str(payload.get("label", "")).strip()
    if not location_id or not label:
        return None
    return _LocationRecord(
        location_id=location_id,
        label=label,
        aliases=_json_string_tuple(payload.get("aliases", [])),
        parent_ids=_json_string_tuple(payload.get("parents", [])),
        provenance=_json_string_tuple(payload.get("provenance", [])),
    )


def _json_string_tuple(value: typing.Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _phrase_variants(phrase: str) -> tuple[str, ...]:
    canonical = _canonical_phrase(phrase)
    if not canonical:
        return ()
    variants = {canonical}
    if canonical in {"us", "u s", "usa", "u s a"}:
        variants.add("united states")
    if canonical == "united states of america":
        variants.add("united states")
    return tuple(sorted(variants))


def _unique_terms(values: typing.Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        term = _canonical_phrase(value)
        if not term or term in seen:
            continue
        seen.add(term)
        out.append(term)
    return tuple(out)


def _canonical_phrase(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.lower()))

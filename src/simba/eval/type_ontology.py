"""Local type-subsumption ratifier for recursive fact compilation.

This module treats the offline lexicon as a T-Box. Evidence facts can assert
local sortals such as ``sortal(new_pair_1, boots)``; this ratifier decides
whether that type is compatible with a question target such as ``clothing``.
It does not mint or rewrite evidence entities.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import json
import pathlib
import re
import typing

DEFAULT_LEXICON_PATH = pathlib.Path(".simba/lexicon/nltk_lexicon.jsonl")
MAX_RATIFICATION_DEPTH = 8
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BRIDGE_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
}


@dataclasses.dataclass(frozen=True)
class TypeRatification:
    source_type: str
    target_type: str
    ratified: bool
    path: tuple[str, ...]
    provenance: tuple[str, ...]
    reason: str


@dataclasses.dataclass(frozen=True)
class _ConceptRecord:
    concept_id: str
    provider: str
    provider_ref: str
    label: str
    aliases: tuple[str, ...]
    parent_ids: tuple[str, ...]
    definition: str

    @property
    def terms(self) -> tuple[str, ...]:
        return _unique_terms((self.label, _provider_lemma(self.provider_ref)))


@dataclasses.dataclass(frozen=True)
class _LexiconGraph:
    records: dict[str, _ConceptRecord]
    term_index: dict[str, tuple[str, ...]]

    def lookup(self, phrase: str) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for variant in _phrase_variants(phrase):
            for concept_id in self.term_index.get(variant, ()):
                if concept_id in seen:
                    continue
                seen.add(concept_id)
                ids.append(concept_id)
        return tuple(ids)

    def neighbors(self, concept_id: str) -> tuple[str, ...]:
        record = self.records.get(concept_id)
        if record is None:
            return ()
        neighbors: list[str] = []
        for parent_id in record.parent_ids:
            if parent_id in self.records:
                neighbors.append(parent_id)
        for term in record.terms:
            for sibling_id in self.term_index.get(term, ()):
                sibling = self.records.get(sibling_id)
                if (
                    sibling_id != concept_id
                    and sibling is not None
                    and _sense_bridge_is_compatible(record, sibling)
                ):
                    neighbors.append(sibling_id)
        return tuple(sorted(set(neighbors)))


def ratify_type_subsumption(
    source_type: str,
    target_type: str,
    *,
    lexicon_path: str | pathlib.Path | None = None,
) -> TypeRatification:
    """Return whether ``source_type`` is a subtype/sense-compatible target type."""

    source = _canonical_phrase(source_type)
    target = _canonical_phrase(target_type)
    if not source or not target:
        return TypeRatification(
            source_type=source_type,
            target_type=target_type,
            ratified=False,
            path=(),
            provenance=(),
            reason="empty_type",
        )
    if source == target:
        return TypeRatification(
            source_type=source_type,
            target_type=target_type,
            ratified=True,
            path=(source,),
            provenance=(),
            reason="same_type",
        )

    path = pathlib.Path(lexicon_path or DEFAULT_LEXICON_PATH)
    graph = _load_lexicon(str(path))
    source_ids = graph.lookup(source)
    target_ids = set(graph.lookup(target))
    if not source_ids:
        return TypeRatification(
            source_type=source_type,
            target_type=target_type,
            ratified=False,
            path=(),
            provenance=(),
            reason="source_type_not_found",
        )
    if not target_ids:
        return TypeRatification(
            source_type=source_type,
            target_type=target_type,
            ratified=False,
            path=(),
            provenance=(),
            reason="target_type_not_found",
        )

    ratified_path = _shortest_path(
        graph,
        source_ids=source_ids,
        target_ids=target_ids,
    )
    if not ratified_path:
        return TypeRatification(
            source_type=source_type,
            target_type=target_type,
            ratified=False,
            path=(),
            provenance=(),
            reason="no_subsumption_path",
        )
    return TypeRatification(
        source_type=source_type,
        target_type=target_type,
        ratified=True,
        path=ratified_path,
        provenance=ratified_path,
        reason="ontology_subsumption",
    )


def _shortest_path(
    graph: _LexiconGraph,
    *,
    source_ids: tuple[str, ...],
    target_ids: set[str],
) -> tuple[str, ...]:
    queue: collections.deque[tuple[str, tuple[str, ...]]] = collections.deque(
        (source_id, (source_id,)) for source_id in source_ids
    )
    visited: set[str] = set()
    while queue:
        concept_id, path = queue.popleft()
        if concept_id in visited:
            continue
        visited.add(concept_id)
        if concept_id in target_ids:
            return path
        if len(path) - 1 >= MAX_RATIFICATION_DEPTH:
            continue
        for neighbor_id in graph.neighbors(concept_id):
            if neighbor_id not in visited:
                queue.append((neighbor_id, (*path, neighbor_id)))
    return ()


@functools.cache
def _load_lexicon(path: str) -> _LexiconGraph:
    lexicon_path = pathlib.Path(path)
    records: dict[str, _ConceptRecord] = {}
    term_members: dict[str, list[str]] = collections.defaultdict(list)
    if not lexicon_path.exists():
        return _LexiconGraph(records={}, term_index={})

    with lexicon_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("kind") != "concept":
                continue
            record = _record_from_payload(payload)
            if record is None:
                continue
            records[record.concept_id] = record

    for concept_id, record in records.items():
        for term in record.terms:
            for variant in _phrase_variants(term):
                term_members[variant].append(concept_id)

    return _LexiconGraph(
        records=records,
        term_index={
            term: tuple(sorted(set(concept_ids)))
            for term, concept_ids in term_members.items()
        },
    )


def _record_from_payload(payload: dict[str, typing.Any]) -> _ConceptRecord | None:
    concept_id = str(payload.get("id", "")).strip()
    provider = str(payload.get("provider", "")).strip()
    provider_ref = str(payload.get("provider_ref", "")).strip()
    label = str(payload.get("label", "")).strip()
    if not concept_id or not provider or not provider_ref or not label:
        return None
    if provider == "wordnet" and ".n." not in provider_ref:
        return None
    aliases = _json_string_tuple(payload.get("aliases_json"))
    parent_ids = tuple(
        _full_concept_id(provider, parent)
        for parent in _json_string_tuple(payload.get("parents_json"))
    )
    return _ConceptRecord(
        concept_id=concept_id,
        provider=provider,
        provider_ref=provider_ref,
        label=label,
        aliases=aliases,
        parent_ids=parent_ids,
        definition=str(payload.get("definition", "")),
    )


def _json_string_tuple(value: typing.Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ()
    else:
        parsed = value
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item).strip() for item in parsed if str(item).strip())


def _full_concept_id(provider: str, provider_ref: str) -> str:
    if ":" in provider_ref:
        return provider_ref
    return f"{provider}:{provider_ref}"


def _provider_lemma(provider_ref: str) -> str:
    lemma = provider_ref.split(".", 1)[0]
    return lemma.replace("_", " ")


def _sense_bridge_is_compatible(
    left: _ConceptRecord,
    right: _ConceptRecord,
) -> bool:
    if set(left.parent_ids) & set(right.parent_ids):
        return True
    left_tokens = _semantic_tokens(left.definition)
    right_tokens = _semantic_tokens(right.definition)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    overlap = len(left_tokens & right_tokens)
    return overlap >= 2 and overlap / min(len(left_tokens), len(right_tokens)) >= 0.5


def _semantic_tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(value.lower())
        if token not in _BRIDGE_STOPWORDS
    }


def _phrase_variants(phrase: str) -> tuple[str, ...]:
    canonical = _canonical_phrase(phrase)
    if not canonical:
        return ()
    variants = {canonical}
    words = canonical.split()
    if words:
        last = words[-1]
        for singular in _singular_variants(last):
            variants.add(" ".join((*words[:-1], singular)))
    if canonical == "clothes":
        variants.add("clothing")
    return tuple(sorted(variants))


def _singular_variants(word: str) -> tuple[str, ...]:
    variants = {word}
    if word.endswith("ies") and len(word) > 3:
        variants.add(f"{word[:-3]}y")
    if word.endswith("es") and len(word) > 3:
        variants.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        variants.add(word[:-1])
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

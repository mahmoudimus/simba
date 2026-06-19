"""Typed world/domain lexicon providers for executable ambiguity evals."""

from __future__ import annotations

import dataclasses
import functools
import re
import typing


@dataclasses.dataclass(frozen=True)
class LexiconConcept:
    id: str
    label: str
    aliases: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()
    provider_refs: tuple[str, ...] = ()
    match_patterns: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class LexiconFrame:
    id: str
    label: str
    lexical_units: tuple[str, ...] = ()
    provider_refs: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ConceptMatch:
    concept_id: str
    phrase: str
    provider_refs: tuple[str, ...]


class LexiconProvider(typing.Protocol):
    def concepts(self) -> tuple[LexiconConcept, ...]: ...

    def frames(self) -> tuple[LexiconFrame, ...]: ...


class OfflineSeedProvider:
    """Deterministic seed graph recorded from public world-lexicon resources.

    This provider is intentionally small: it gives local evals a stable T-Box-like
    vocabulary while the provider boundary stays compatible with Wikidata,
    WordNet, FrameNet, PropBank, or project-local graph providers.
    """

    def concepts(self) -> tuple[LexiconConcept, ...]:
        return (
            LexiconConcept(
                id="loyalty_points",
                label="loyalty points",
                aliases=("points", "reward points", "loyalty program points"),
                parent_ids=("quantity",),
                provider_refs=("schema:QuantitativeValue",),
            ),
            LexiconConcept(
                id="musical_instrument",
                label="musical instrument",
                aliases=(
                    "instrument",
                    "instruments",
                    "guitar",
                    "piano",
                    "drum set",
                    "ukulele",
                    "violin",
                ),
                parent_ids=("artifact",),
                provider_refs=("wikidata:Q34379", "wordnet:instrumentality.n.03"),
            ),
            LexiconConcept(
                id="scale_model_kit",
                label="model kit",
                aliases=("model kit", "kit", "scale model"),
                parent_ids=("artifact", "product"),
                provider_refs=("wikidata:model kit", "wordnet:model.n.07"),
                match_patterns=(r"\b\d+/\d+\s+scale\b",),
            ),
            LexiconConcept(
                id="project_work",
                label="project",
                aliases=("project", "feature", "team effort"),
                parent_ids=("planned_process",),
                provider_refs=("wordnet:undertaking.n.01",),
            ),
            LexiconConcept(
                id="baking_event",
                label="baking event",
                aliases=("bake", "baked", "baking", "recipe"),
                parent_ids=("event", "cooking"),
                provider_refs=("framenet:Apply_heat", "wordnet:bake.v.02"),
            ),
        )

    def frames(self) -> tuple[LexiconFrame, ...]:
        return (
            LexiconFrame(
                id="threshold_lookup",
                label="threshold lookup",
                lexical_units=("need", "redeem", "total"),
                provider_refs=("schema:QuantitativeValue",),
            ),
            LexiconFrame(
                id="current_possession",
                label="current possession",
                lexical_units=("own", "have", "had", "my", "mine"),
                provider_refs=("framenet:Possession",),
            ),
            LexiconFrame(
                id="worked_or_bought",
                label="worked on or bought",
                lexical_units=(
                    "worked on",
                    "working on",
                    "started working on",
                    "bought",
                    "finished",
                    "got",
                    "picked up",
                ),
                provider_refs=("framenet:Commerce_buy", "framenet:Work"),
            ),
            LexiconFrame(
                id="leadership_role",
                label="leadership role",
                lexical_units=("led", "leading", "lead"),
                provider_refs=("framenet:Leadership",),
            ),
            LexiconFrame(
                id="baking_event",
                label="baking event",
                lexical_units=("bake", "baked", "baking"),
                provider_refs=("framenet:Apply_heat",),
            ),
        )


class WorldLexicon:
    def __init__(self, providers: tuple[LexiconProvider, ...]) -> None:
        concepts: dict[str, LexiconConcept] = {}
        frames: dict[str, LexiconFrame] = {}
        for provider in providers:
            for concept in provider.concepts():
                concepts[concept.id] = concept
            for frame in provider.frames():
                frames[frame.id] = frame
        self._concepts = concepts
        self._frames = frames

    def concept(self, concept_id: str) -> LexiconConcept | None:
        return self._concepts.get(concept_id)

    def frame(self, frame_id: str) -> LexiconFrame | None:
        return self._frames.get(frame_id)

    def terms_for_concepts(self, concept_ids: tuple[str, ...]) -> tuple[str, ...]:
        terms: list[str] = []
        for concept_id in concept_ids:
            concept = self.concept(concept_id)
            if concept is None:
                continue
            terms.append(concept.label)
            terms.extend(concept.aliases)
        return _unique_terms(terms)

    def patterns_for_concepts(self, concept_ids: tuple[str, ...]) -> tuple[str, ...]:
        patterns: list[str] = []
        for concept_id in concept_ids:
            concept = self.concept(concept_id)
            if concept is not None:
                patterns.extend(concept.match_patterns)
        return tuple(patterns)

    def lexical_units_for_frame(self, frame_id: str) -> tuple[str, ...]:
        frame = self.frame(frame_id)
        return frame.lexical_units if frame is not None else ()

    def resolve_concepts(self, text: str) -> tuple[ConceptMatch, ...]:
        low = text.lower()
        matches: list[ConceptMatch] = []
        for concept in self._concepts.values():
            terms = (concept.label, *concept.aliases)
            if any(_term_in_text(term, low) for term in terms):
                matches.append(
                    ConceptMatch(
                        concept_id=concept.id,
                        phrase=concept.label,
                        provider_refs=concept.provider_refs,
                    )
                )
                continue
            if any(re.search(pattern, low) for pattern in concept.match_patterns):
                matches.append(
                    ConceptMatch(
                        concept_id=concept.id,
                        phrase=concept.label,
                        provider_refs=concept.provider_refs,
                    )
                )
        return tuple(matches)


@functools.cache
def default_world_lexicon() -> WorldLexicon:
    return WorldLexicon((OfflineSeedProvider(),))


def _unique_terms(terms: typing.Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        cleaned = re.sub(r"\s+", " ", term.lower()).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return tuple(out)


def _term_in_text(term: str, text: str) -> bool:
    cleaned = re.escape(term.lower().strip())
    if not cleaned:
        return False
    return bool(re.search(rf"\b{cleaned}s?\b", text))

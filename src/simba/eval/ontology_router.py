"""LLM-style micro-schema routing to remote ontology ratifiers."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import re
import typing

import simba.eval.ambiguity_fail18 as ambiguity_fail18

DEFAULT_OUT = pathlib.Path(".simba/lexicon/candidates.jsonl")
PROMPT_VERSION = "seeded-graph-v6-variant-normalized"


@dataclasses.dataclass(frozen=True)
class MicroConcept:
    id: str
    label: str
    domain: str
    aliases: tuple[str, ...] = ()
    source_hints: tuple[str, ...] = ()
    purpose: str = ""


@dataclasses.dataclass(frozen=True)
class MicroFrame:
    id: str
    label: str
    lexical_units: tuple[str, ...] = ()
    roles: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class MicroEdge:
    source: str
    relation: str
    target: str
    evidence: str = ""


@dataclasses.dataclass(frozen=True)
class MicroSchema:
    question: str
    concepts: tuple[MicroConcept, ...]
    frames: tuple[MicroFrame, ...] = ()
    edges: tuple[MicroEdge, ...] = ()
    provenance: str = "deterministic-fallback"


@dataclasses.dataclass(frozen=True)
class SourceRoute:
    source: str
    reason: str
    endpoint: str


@dataclasses.dataclass(frozen=True)
class RatificationHit:
    concept_id: str
    source: str
    ok: bool
    matched: str = ""
    provider_ref: str = ""
    error: str = ""


@dataclasses.dataclass(frozen=True)
class RatificationReport:
    schema: MicroSchema
    routes: dict[str, tuple[SourceRoute, ...]]
    hits: tuple[RatificationHit, ...]
    cache_status: str = "miss"


@dataclasses.dataclass(frozen=True)
class ReliabilityComponent:
    id: str
    stage: str
    score: float
    reason: str


@dataclasses.dataclass(frozen=True)
class RequiredConceptReliability:
    concept_id: str
    effective_reliability: float
    components: tuple[ReliabilityComponent, ...]

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "concept_id": self.concept_id,
            "effective_reliability": self.effective_reliability,
            "components": [
                dataclasses.asdict(component) for component in self.components
            ],
        }


@dataclasses.dataclass(frozen=True)
class Fail18OntologyItem:
    question_id: str
    question: str
    failure_mode: str
    answer_type: str
    gold_numeric: int | None
    schema: MicroSchema
    routes: dict[str, tuple[SourceRoute, ...]]
    hits: tuple[RatificationHit, ...]
    cache_status: str = ""

    def to_dict(self) -> dict[str, typing.Any]:
        ratified_concepts = _ratified_concept_ids(self.hits)
        material = [
            concept.id
            for concept in self.schema.concepts
            if _is_material_concept(concept)
        ]
        required = _required_answer_concept_ids(self.schema, self.answer_type)
        reliability = _required_concept_reliabilities(self.schema, required, self.hits)
        return {
            "question_id": self.question_id,
            "question": self.question,
            "failure_mode": self.failure_mode,
            "answer_type": self.answer_type,
            "gold_numeric": self.gold_numeric,
            "schema": _schema_to_dict(self.schema),
            "routes": {
                key: [dataclasses.asdict(route) for route in routes]
                for key, routes in self.routes.items()
            },
            "hits": [dataclasses.asdict(hit) for hit in self.hits],
            "cache_status": self.cache_status,
            "ratified_concepts": sorted(ratified_concepts),
            "material_concepts": material,
            "ratified_material_concepts": sorted(
                concept_id for concept_id in material if concept_id in ratified_concepts
            ),
            "required_answer_concepts": required,
            "ratified_required_answer_concepts": sorted(
                concept_id for concept_id in required if concept_id in ratified_concepts
            ),
            "required_answer_complete": bool(required)
            and all(concept_id in ratified_concepts for concept_id in required),
            "required_answer_reliability": [item.to_dict() for item in reliability],
        }


@dataclasses.dataclass(frozen=True)
class Fail18OntologySummary:
    total: int
    provenance: str
    remote: bool
    rows_with_any_hit: int
    rows_with_material_hit: int
    rows_with_required_hit: int
    rows_with_required_complete: int
    material_concepts: int
    ratified_material_concepts: int
    required_concepts: int
    ratified_required_concepts: int
    required_reliability_min: float
    required_reliability_avg: float
    results: tuple[Fail18OntologyItem, ...]

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "total": self.total,
            "provenance": self.provenance,
            "remote": self.remote,
            "rows_with_any_hit": self.rows_with_any_hit,
            "rows_with_material_hit": self.rows_with_material_hit,
            "rows_with_required_hit": self.rows_with_required_hit,
            "rows_with_required_complete": self.rows_with_required_complete,
            "material_concepts": self.material_concepts,
            "ratified_material_concepts": self.ratified_material_concepts,
            "required_concepts": self.required_concepts,
            "ratified_required_concepts": self.ratified_required_concepts,
            "required_reliability_min": self.required_reliability_min,
            "required_reliability_avg": self.required_reliability_avg,
            "results": [item.to_dict() for item in self.results],
        }


_SOURCE_ENDPOINTS = {
    "schema": "https://schema.org/docs/jsonldcontext.json",
    "qudt": "https://qudt.org/vocab/unit",
    "agrovoc": "https://agrovoc.fao.org/browse/rest/v1/agrovoc/search",
    "unesco": "https://vocabularies.unesco.org/sparql",
    "getty": "https://vocab.getty.edu/sparql.json",
}

_DOMAIN_SOURCES = {
    "quantity": ("qudt", "schema"),
    "unit": ("qudt", "schema"),
    "threshold": ("qudt", "schema"),
    "time": ("qudt", "schema"),
    "event": ("schema", "getty", "unesco"),
    "activity": ("schema", "getty"),
    "product": ("schema", "agrovoc"),
    "action": ("schema",),
    "food": ("agrovoc", "schema"),
    "agriculture": ("agrovoc",),
    "environment": ("agrovoc", "unesco"),
    "biomedical": ("bioportal", "mesh", "obo"),
    "culture": ("getty", "unesco", "schema"),
    "object": ("getty", "schema"),
    "place": ("getty", "schema"),
    "organization": ("schema", "unesco"),
    "marketing": ("schema",),
    "education": ("unesco", "schema"),
    "social": ("unesco", "schema"),
    "generic": ("schema", "lov"),
}


def build_synthesis_prompt(question: str, corpus_snippets: tuple[str, ...] = ()) -> str:
    snippets = "\n".join(f"- {snippet}" for snippet in corpus_snippets[:8])
    return f"""Generate a minimal ontology micro-schema graph for this question.

Treat the question as the seed vertex. Process the corpus snippets with the
answer computation in mind:
- concept nodes are typed entities, quantities, events, units, places, or
  constraints needed to answer this question.
- ontology source routes are external ratification edges, not proof by LLM.
- schema edges connect the question seed to answer-bearing and constraint nodes.
- mark each concept purpose as one of:
  answer_bearing, constraint, evidence, incidental, meta.

Use answer_bearing only for nodes that the executable answer needs to count,
sum, threshold, filter, or look up. Do not mark person/user/count/how-many as
answer_bearing. Do not mark time words as answer_bearing unless the answer is a
duration or amount of time.

Return strict JSON:
{{
  "concepts": [
    {{
      "id": "snake_case",
      "label": "...",
      "domain": "...",
      "aliases": [],
      "purpose": "answer_bearing|constraint|evidence|incidental|meta"
    }}
  ],
  "frames": [
    {{"id": "snake_case", "label": "...", "lexical_units": [], "roles": {{}}}}
  ],
  "edges": [
    {{
      "source": "question",
      "relation": "counts|sums|filters_by|looks_up|constrains|mentions",
      "target": "concept_id",
      "evidence": "short text span"
    }}
  ]
}}

Use domains such as quantity, unit, threshold, product, action, food,
biomedical, culture, object, place, time, education, social, generic.

Question: {question}
Corpus snippets:
{snippets}
"""


def propose_micro_schema(question: str) -> MicroSchema:
    """Deterministic fallback for the LLM proposal step."""
    low = question.lower()
    concepts: list[MicroConcept] = []
    frames: list[MicroFrame] = []

    if "point" in low or "redeem" in low:
        concepts.append(
            MicroConcept(
                id="loyalty_points",
                label="loyalty points",
                domain="quantity",
                aliases=("points", "reward points", "quantitative value"),
                source_hints=("qudt", "schema"),
                purpose="answer_bearing",
            )
        )
        frames.append(
            MicroFrame(
                id="threshold_lookup",
                label="threshold lookup",
                lexical_units=("need", "redeem", "total"),
                roles={"amount": "loyalty_points"},
            )
        )
    if "skincare" in low or "product" in low:
        concepts.append(
            MicroConcept(
                id="skincare_product",
                label="skincare product",
                domain="product",
                aliases=("skin care product", "cosmetic product", "product"),
                source_hints=("schema", "agrovoc"),
                purpose="constraint",
            )
        )
    if re.search(r"\bbak(?:e|ed|ing)\b", low):
        concepts.append(
            MicroConcept(
                id="baking_event",
                label="baking event",
                domain="food",
                aliases=("bake", "baked", "baking"),
                source_hints=("agrovoc", "schema"),
                purpose="answer_bearing",
            )
        )
        frames.append(
            MicroFrame(
                id="cooking_creation",
                label="cooking creation",
                lexical_units=("bake", "baked", "baking"),
                roles={"created_food": "food"},
            )
        )
    if "model kit" in low or "model kits" in low:
        concepts.append(
            MicroConcept(
                id="model_kit",
                label="model kit",
                domain="product",
                aliases=("scale model", "kit", "product"),
                source_hints=("schema", "getty"),
                purpose="answer_bearing",
            )
        )
    if "instrument" in low:
        concepts.append(
            MicroConcept(
                id="musical_instrument",
                label="musical instrument",
                domain="object",
                aliases=("instrument", "guitar", "piano"),
                source_hints=("schema", "getty"),
                purpose="answer_bearing",
            )
        )
    if "project" in low:
        concepts.append(
            MicroConcept(
                id="project_work",
                label="project",
                domain="generic",
                aliases=("project", "work plan"),
                source_hints=("schema", "lov"),
                purpose="answer_bearing",
            )
        )
    if not concepts:
        concepts.append(
            MicroConcept(
                id="query_subject",
                label=_subject_label(question),
                domain="generic",
                source_hints=("schema", "lov"),
                purpose="answer_bearing",
            )
        )

    return MicroSchema(
        question=question,
        concepts=tuple(_dedupe_concepts(concepts)),
        frames=tuple(frames),
    )


def propose_micro_schema_with_llm(
    question: str,
    corpus_snippets: tuple[str, ...] = (),
) -> MicroSchema:
    try:
        import simba.llm.client

        client = simba.llm.client.get_client()
        if not client.available():
            return propose_micro_schema(question)
        raw = client.complete_json(build_synthesis_prompt(question, corpus_snippets))
        schema = _schema_from_json(question, raw)
        return schema if schema is not None else propose_micro_schema(question)
    except Exception:
        return propose_micro_schema(question)


def build_report(
    question: str,
    *,
    corpus_snippets: tuple[str, ...] = (),
    use_llm: bool = False,
    remote: bool = False,
    timeout: float = 12.0,
    cache_path: pathlib.Path | None = None,
    append: bool = False,
) -> RatificationReport:
    if use_llm and cache_path is not None:
        cached = load_cached_report(
            question,
            corpus_snippets,
            cache_path,
            remote=remote,
        )
        if cached is not None:
            return cached
        cached_schema = load_cached_schema(question, corpus_snippets, cache_path)
        if cached_schema is not None:
            cached_schema = normalize_schema(cached_schema)
            report = dataclasses.replace(
                ratify_schema(cached_schema, remote=remote, timeout=timeout),
                cache_status="schema-cache",
            )
            append_report(
                report,
                cache_path,
                corpus_snippets=corpus_snippets,
                remote=remote,
            )
            return report

    schema = (
        propose_micro_schema_with_llm(question, corpus_snippets)
        if use_llm
        else propose_micro_schema(question)
    )
    schema = normalize_schema(schema)
    report = dataclasses.replace(
        ratify_schema(schema, remote=remote, timeout=timeout),
        cache_status="miss" if use_llm else "deterministic",
    )
    if append or (use_llm and cache_path is not None):
        append_report(
            report,
            cache_path or DEFAULT_OUT,
            corpus_snippets=corpus_snippets,
            remote=remote,
        )
    return report


def _schema_from_json(question: str, raw: typing.Any) -> MicroSchema | None:
    if not isinstance(raw, dict):
        return None
    concepts: list[MicroConcept] = []
    for item in raw.get("concepts", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        concept_id = str(item.get("id") or _schema_id(label)).strip()
        domain = str(item.get("domain") or "generic").strip().lower()
        if not label or not concept_id:
            continue
        aliases = tuple(str(alias) for alias in item.get("aliases", []) if alias)
        purpose = str(item.get("purpose") or item.get("role") or "").strip().lower()
        source_hints = tuple(
            str(source) for source in item.get("source_hints", []) if source
        ) or _DOMAIN_SOURCES.get(domain, ("schema", "lov"))
        concepts.append(
            MicroConcept(
                id=concept_id,
                label=label,
                domain=domain,
                aliases=aliases,
                source_hints=source_hints,
                purpose=purpose,
            )
        )
    frames: list[MicroFrame] = []
    for item in raw.get("frames", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        frame_id = str(item.get("id") or _schema_id(label)).strip()
        if not label or not frame_id:
            continue
        roles = item.get("roles", {})
        frames.append(
            MicroFrame(
                id=frame_id,
                label=label,
                lexical_units=tuple(
                    str(unit) for unit in item.get("lexical_units", []) if unit
                ),
                roles=roles if isinstance(roles, dict) else {},
            )
        )
    edges: list[MicroEdge] = []
    for item in raw.get("edges", []):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or item.get("from") or "").strip()
        target = str(item.get("target") or item.get("to") or "").strip()
        relation = str(item.get("relation") or item.get("predicate") or "").strip()
        if not source or not target or not relation:
            continue
        edges.append(
            MicroEdge(
                source=source,
                relation=relation,
                target=target,
                evidence=str(item.get("evidence") or "").strip(),
            )
        )
    if not concepts:
        return None
    return MicroSchema(
        question=question,
        concepts=tuple(_dedupe_concepts(concepts)),
        frames=tuple(frames),
        edges=tuple(edges),
        provenance="llm",
    )


def normalize_schema(schema: MicroSchema) -> MicroSchema:
    concepts: dict[str, MicroConcept] = {}
    target_map: dict[str, list[str]] = {}
    generated_edges: list[MicroEdge] = []
    for concept in schema.concepts:
        normalized = _normalize_concept(concept, schema.question)
        target_map[concept.id] = [item.id for item in normalized]
        if len(normalized) > 1:
            for item in normalized:
                generated_edges.append(
                    MicroEdge(
                        source=concept.id,
                        relation="decomposes_to",
                        target=item.id,
                        evidence=concept.label,
                    )
                )
        for item in normalized:
            concepts[item.id] = _merge_concept(concepts.get(item.id), item)

    edges = _normalize_edges(schema.edges, target_map, concepts)
    edges.extend(generated_edges)
    return dataclasses.replace(
        schema,
        concepts=tuple(concepts.values()),
        edges=tuple(_dedupe_edges(edges)),
        provenance=f"{schema.provenance}+normalized"
        if "+normalized" not in schema.provenance
        else schema.provenance,
    )


def _normalize_concept(
    concept: MicroConcept, question: str
) -> tuple[MicroConcept, ...]:
    text = _concept_text(concept)
    purpose = concept.purpose

    if _is_count_measure_concept(text):
        return (dataclasses.replace(concept, purpose="meta"),)
    if (
        _asks_for_time_amount(question)
        and _is_activity_constraint(text)
        and not _is_day_duration_concept(text)
        and not _is_hour_duration_concept(text)
    ):
        return (dataclasses.replace(concept, purpose="constraint"),)
    if _is_threshold_question(question) and _is_reward_product_concept(text):
        return (dataclasses.replace(concept, purpose="constraint"),)
    if _is_clothing_concept(text):
        return (
            _concept(
                "clothing_item",
                "clothing item",
                "object",
                ("clothing", "clothes", "garment", "apparel", "wearable"),
                purpose,
                ("getty", "schema"),
            ),
        )
    if _is_musical_instrument_concept(text):
        return (
            _concept(
                "musical_instrument",
                "musical instrument",
                "object",
                ("instrument", "guitar", "piano", "drum"),
                purpose,
                ("getty", "schema"),
            ),
        )
    if _is_model_kit_concept(text):
        return (
            _concept(
                "model_kit",
                "model kit",
                "product",
                ("scale model", "toy kit", "model", "kit", "plastic model"),
                purpose,
                ("schema", "getty"),
            ),
        )
    if _is_music_release_concept(text):
        return (
            _concept(
                "music_release",
                "music release",
                "object",
                ("music album", "album", "EP", "extended play", "music recording"),
                purpose,
                ("schema", "getty"),
            ),
        )
    if _is_wedding_attendance_concept(text):
        return (
            _concept(
                "wedding",
                "wedding",
                "event",
                ("marriage ceremony", "nuptials", "wedding event"),
                purpose,
                ("schema", "getty"),
            ),
            _concept(
                "attendance",
                "attendance",
                "action",
                ("attend", "attended", "participation"),
                "constraint",
                ("schema",),
            ),
        )
    if _is_wedding_concept(text):
        return (
            _concept(
                "wedding",
                "wedding",
                "event",
                ("marriage ceremony", "nuptials", "wedding event"),
                purpose,
                ("schema", "getty"),
            ),
        )
    if _is_baking_concept(text):
        return (
            _concept(
                "baking_event",
                "baking event",
                "food",
                ("baking", "bake", "baked", "baked good", "cooking"),
                purpose,
                ("agrovoc", "schema"),
            ),
        )
    if _is_points_threshold_concept(text):
        point_purpose = (
            "answer_bearing" if _is_threshold_question(question) else purpose
        )
        return (
            _concept(
                "loyalty_points",
                "loyalty points",
                "quantity",
                ("points", "point", "reward points", "Sephora points"),
                point_purpose,
                ("qudt", "schema"),
            ),
            _concept(
                "redemption_threshold",
                "redemption threshold",
                "threshold",
                ("minimum points", "points required", "points needed"),
                "constraint",
                ("qudt", "schema"),
            ),
        )
    if _is_day_duration_concept(text):
        out = [
            _concept(
                "day",
                "day",
                "unit",
                ("days", "duration day", "calendar day"),
                purpose,
                ("qudt", "schema"),
            )
        ]
        if "hawaii" in text:
            out.append(
                _concept("hawaii", "Hawaii", "place", ("Hawai'i",), "constraint")
            )
        if "nyc" in text or "new york" in text:
            out.append(
                _concept(
                    "new_york_city",
                    "New York City",
                    "place",
                    ("NYC", "New York"),
                    "constraint",
                )
            )
        return tuple(out)
    if _is_hour_duration_concept(text):
        hour_purpose = "answer_bearing" if _asks_for_time_amount(question) else purpose
        return (
            _concept(
                "hour",
                "hour",
                "unit",
                ("hours", "hrs", "h"),
                hour_purpose,
                ("qudt", "schema"),
            ),
        )
    if _is_money_concept(text):
        return (
            _concept(
                "money",
                "money",
                "quantity",
                ("monetary amount", "funds", "amount raised", "currency"),
                purpose,
                ("schema", "qudt"),
            ),
        )
    if _is_people_reached_concept(text):
        return (
            _concept(
                "people_reached",
                "people reached",
                "quantity",
                (
                    "audience reach",
                    "reach",
                    "impressions",
                    "people reached",
                    "people audience",
                    "interaction counter",
                ),
                purpose,
                ("schema",),
            ),
        )
    if _is_art_event_concept(text):
        return (
            _concept(
                "art_event",
                "art event",
                "event",
                ("event", "art exhibition", "exhibition", "cultural event"),
                purpose,
                ("schema", "getty", "unesco"),
            ),
        )
    if _is_cuisine_concept(text):
        return (
            _concept(
                "cuisine",
                "cuisine",
                "food",
                ("recipe cuisine", "culinary tradition", "food culture"),
                purpose,
                ("schema", "agrovoc", "unesco"),
            ),
        )
    if _is_furniture_concept(text):
        return (
            _concept(
                "furniture_item",
                "furniture item",
                "object",
                ("furniture", "furnishing", "piece of furniture"),
                purpose,
                ("schema", "getty"),
            ),
        )
    if _is_citrus_concept(text):
        return (
            _concept(
                "citrus_fruit",
                "citrus fruit",
                "food",
                ("citrus", "citrus fruit type", "lemon", "lime", "orange"),
                purpose,
                ("agrovoc", "schema"),
            ),
        )
    if _is_project_concept(text):
        return (
            _concept(
                "project",
                "project",
                "generic",
                ("projects", "work project"),
                purpose,
                ("schema",),
            ),
        )
    return (concept,)


def _concept(
    concept_id: str,
    label: str,
    domain: str,
    aliases: tuple[str, ...] = (),
    purpose: str = "",
    source_hints: tuple[str, ...] = (),
) -> MicroConcept:
    return MicroConcept(
        id=concept_id,
        label=label,
        domain=domain,
        aliases=aliases,
        source_hints=source_hints or _DOMAIN_SOURCES.get(domain, ("schema", "lov")),
        purpose=purpose,
    )


def _merge_concept(old: MicroConcept | None, new: MicroConcept) -> MicroConcept:
    if old is None:
        return new
    aliases = tuple(dict.fromkeys((*old.aliases, *new.aliases)))
    hints = tuple(dict.fromkeys((*old.source_hints, *new.source_hints)))
    purpose = (
        "answer_bearing"
        if "answer_bearing" in {old.purpose, new.purpose}
        else old.purpose or new.purpose
    )
    return MicroConcept(
        id=old.id,
        label=old.label or new.label,
        domain=old.domain if old.domain != "generic" else new.domain,
        aliases=aliases,
        source_hints=hints,
        purpose=purpose,
    )


def _normalize_edges(
    edges: tuple[MicroEdge, ...],
    target_map: dict[str, list[str]],
    concepts: dict[str, MicroConcept],
) -> list[MicroEdge]:
    out: list[MicroEdge] = []
    for edge in edges:
        sources = target_map.get(edge.source, [edge.source])
        targets = target_map.get(edge.target, [edge.target])
        for source in sources:
            for target in targets:
                relation = _normalized_relation(edge.relation, concepts.get(target))
                out.append(
                    MicroEdge(
                        source=source,
                        relation=relation,
                        target=target,
                        evidence=edge.evidence,
                    )
                )
    return out


def _normalized_relation(relation: str, target: MicroConcept | None) -> str:
    if target is None or target.purpose == "answer_bearing":
        return relation
    if relation in {"counts", "sums", "looks_up"}:
        return "constrains"
    return relation


def _dedupe_edges(edges: list[MicroEdge]) -> list[MicroEdge]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[MicroEdge] = []
    for edge in edges:
        key = (edge.source, edge.relation, edge.target, edge.evidence)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _is_clothing_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:items?_of_clothing|clothing|clothes|garment|apparel)\b",
            text,
        )
    )


def _is_model_kit_concept(text: str) -> bool:
    return bool(re.search(r"\b(?:model_kit|model kits?|scale model|toy kit)\b", text))


def _is_count_measure_concept(text: str) -> bool:
    return bool(re.search(r"\b(?:[a-z_]+_count|count|number of [a-z ]+)\b", text))


def _is_musical_instrument_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:owned_musical_instrument|musical_instrument|"
            r"musical instruments?)\b",
            text,
        )
    )


def _is_music_release_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:music_album_or_ep|music album|extended play|\bep\b)\b",
            text,
        )
    )


def _is_wedding_attendance_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:wedding_attendance|attended weddings?|weddings? attended|"
            r"marriage ceremony attendance)\b",
            text,
        )
    )


def _is_wedding_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:weddings?|marriage ceremony|nuptials)\b",
            text,
        )
    )


def _is_baking_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:baking_event|baking_action|baked_product|bake|baking|baked)\b",
            text,
        )
    )


def _is_points_threshold_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:points_threshold|points_required|points_needed|loyalty points?|"
            r"reward points?|sephora points?)\b",
            text,
        )
    )


def _is_reward_product_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:free_skincare_product|skincare product|skin care product|"
            r"free product|reward product)\b",
            text,
        )
    )


def _is_day_duration_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:total_days|total_camping_days|days?_in_|day|days|"
            r"duration day)\b",
            text,
        )
    )


def _is_hour_duration_concept(text: str) -> bool:
    return bool(re.search(r"(?:\bhours?\b|\bhrs?\b|total_hours|hours?_)", text))


def _is_money_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:total_money_raised|total_raised|money|funds|amount raised)\b",
            text,
        )
    )


def _is_people_reached_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:people_reached|total_people_reached|total_reach|"
            r"audience reach|people reached|audience size|impressions)\b",
            text,
        )
    )


def _is_art_event_concept(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:art[-_ ]related event|art event|art exhibition|"
            r"cultural event)\b",
            text,
        )
    )


def _is_cuisine_concept(text: str) -> bool:
    return bool(re.search(r"\b(?:cuisine|culinary tradition|food culture)\b", text))


def _is_furniture_concept(text: str) -> bool:
    return bool(re.search(r"\b(?:furniture_item|furniture|furnishing)\b", text))


def _is_citrus_concept(text: str) -> bool:
    return bool(re.search(r"\b(?:citrus_fruit_type|citrus fruit|citrus)\b", text))


def _is_project_concept(text: str) -> bool:
    return bool(re.search(r"\b(?:project_work|projects?|work project)\b", text))


def _is_activity_constraint(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:jogging|yoga|camping_trip|camping|travel|trip|"
            r"acquisition_event)\b",
            text,
        )
    )


def route_schema(schema: MicroSchema) -> dict[str, tuple[SourceRoute, ...]]:
    routes: dict[str, tuple[SourceRoute, ...]] = {}
    for concept in schema.concepts:
        source_ids = _route_sources_for_concept(concept)
        routes[concept.id] = tuple(
            SourceRoute(
                source=source,
                reason=f"domain={concept.domain}",
                endpoint=_SOURCE_ENDPOINTS.get(source, ""),
            )
            for source in source_ids
        )
    return routes


def probe_fail18(
    manifest_path: pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    *,
    corpus_path: pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    use_llm: bool = False,
    remote: bool = False,
    append: bool = False,
    out: pathlib.Path = DEFAULT_OUT,
    timeout: float = 12.0,
    limit: int = 0,
    cache: bool = True,
) -> Fail18OntologySummary:
    rows = ambiguity_fail18.load_manifest(manifest_path)
    corpus_by_id = _fail18_corpus_by_id(corpus_path)
    results: list[Fail18OntologyItem] = []
    for row in rows[: limit or None]:
        question = str(row.get("question", ""))
        snippets = _fail18_snippets(corpus_by_id.get(str(row["question_id"])))
        report = build_report(
            question,
            corpus_snippets=snippets,
            use_llm=use_llm,
            remote=remote,
            timeout=timeout,
            cache_path=out if cache and use_llm else None,
            append=append,
        )
        results.append(
            Fail18OntologyItem(
                question_id=str(row["question_id"]),
                question=question,
                failure_mode=str(row.get("failure_mode", "")),
                answer_type=ambiguity_fail18.classify_answer_type(row),
                gold_numeric=ambiguity_fail18.numeric_gold(row),
                schema=report.schema,
                routes=report.routes,
                hits=report.hits,
                cache_status=report.cache_status,
            )
        )
    return _summarize_fail18_probe(results, use_llm=use_llm, remote=remote)


def ratify_schema(
    schema: MicroSchema,
    *,
    remote: bool = False,
    timeout: float = 12.0,
) -> RatificationReport:
    routes = route_schema(schema)
    hits: list[RatificationHit] = []
    for concept in schema.concepts:
        terms = (concept.label, *concept.aliases)
        for route in routes[concept.id]:
            if route.source not in _SOURCE_ENDPOINTS:
                hits.append(
                    RatificationHit(
                        concept_id=concept.id,
                        source=route.source,
                        ok=False,
                        error="source not implemented in trial client",
                    )
                )
                continue
            if not remote:
                hits.append(
                    RatificationHit(
                        concept_id=concept.id,
                        source=route.source,
                        ok=False,
                        error="remote disabled",
                    )
                )
                continue
            hit = _ratify_terms(route.source, terms, timeout=timeout)
            hits.append(dataclasses.replace(hit, concept_id=concept.id))
    return RatificationReport(schema=schema, routes=routes, hits=tuple(hits))


def load_cached_report(
    question: str,
    corpus_snippets: tuple[str, ...],
    path: pathlib.Path = DEFAULT_OUT,
    *,
    remote: bool,
    prompt_version: str = PROMPT_VERSION,
) -> RatificationReport | None:
    row = _latest_cached_row(
        question,
        corpus_snippets,
        path,
        prompt_version=prompt_version,
        remote=remote,
    )
    if row is None:
        return None
    report = _report_from_cache_row(row)
    return dataclasses.replace(report, cache_status="report-cache") if report else None


def load_cached_schema(
    question: str,
    corpus_snippets: tuple[str, ...],
    path: pathlib.Path = DEFAULT_OUT,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> MicroSchema | None:
    row = _latest_cached_row(
        question,
        corpus_snippets,
        path,
        prompt_version=prompt_version,
    )
    if row is None:
        return None
    schema = _schema_from_json(question, row.get("schema", {}))
    if schema is None:
        return None
    provenance = str(row.get("schema", {}).get("provenance") or schema.provenance)
    return dataclasses.replace(schema, provenance=f"{provenance}-cache")


def append_report(
    report: RatificationReport,
    path: pathlib.Path = DEFAULT_OUT,
    *,
    corpus_snippets: tuple[str, ...] = (),
    prompt_version: str = PROMPT_VERSION,
    remote: bool | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = _cache_key(report.schema.question, corpus_snippets, prompt_version)
    row = {
        **key,
        "remote": remote,
        "trust_tier": "ratified" if any(hit.ok for hit in report.hits) else "proposed",
        "schema": _schema_to_dict(report.schema),
        "routes": {
            key: [dataclasses.asdict(route) for route in routes]
            for key, routes in report.routes.items()
        },
        "ratification_hits": [dataclasses.asdict(hit) for hit in report.hits],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _latest_cached_row(
    question: str,
    corpus_snippets: tuple[str, ...],
    path: pathlib.Path,
    *,
    prompt_version: str,
    remote: bool | None = None,
) -> dict[str, typing.Any] | None:
    if not path.exists():
        return None
    key = _cache_key(question, corpus_snippets, prompt_version)
    latest: dict[str, typing.Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(row.get(name) != value for name, value in key.items()):
                continue
            if remote is not None and row.get("remote") is not remote:
                continue
            latest = row
    return latest


def _report_from_cache_row(row: dict[str, typing.Any]) -> RatificationReport | None:
    schema_raw = row.get("schema", {})
    question = str(schema_raw.get("question") or "")
    schema = _schema_from_json(question, schema_raw)
    if schema is None:
        return None
    provenance = str(schema_raw.get("provenance") or schema.provenance)
    schema = dataclasses.replace(schema, provenance=f"{provenance}-cache")
    routes: dict[str, tuple[SourceRoute, ...]] = {}
    for concept_id, raw_routes in row.get("routes", {}).items():
        routes[str(concept_id)] = tuple(
            SourceRoute(
                source=str(item.get("source", "")),
                reason=str(item.get("reason", "")),
                endpoint=str(item.get("endpoint", "")),
            )
            for item in raw_routes
            if isinstance(item, dict)
        )
    hits = tuple(
        RatificationHit(
            concept_id=str(item.get("concept_id", "")),
            source=str(item.get("source", "")),
            ok=bool(item.get("ok")),
            matched=str(item.get("matched", "")),
            provider_ref=str(item.get("provider_ref", "")),
            error=str(item.get("error", "")),
        )
        for item in row.get("ratification_hits", [])
        if isinstance(item, dict)
    )
    return RatificationReport(schema=schema, routes=routes, hits=hits)


def _ratify_terms(
    source: str, terms: tuple[str, ...], *, timeout: float
) -> RatificationHit:
    import httpx

    try:
        if source == "schema":
            return _ratify_schema_org(terms, timeout=timeout)
        if source == "qudt":
            return _ratify_text_source(
                source, _SOURCE_ENDPOINTS[source], terms, timeout
            )
        if source == "agrovoc":
            return _ratify_agrovoc(terms, timeout=timeout)
        if source == "unesco":
            return _ratify_unesco(terms, timeout=timeout)
        if source == "getty":
            return _ratify_getty(terms, timeout=timeout)
    except httpx.HTTPError as exc:
        return RatificationHit("", source, False, error=str(exc))
    return RatificationHit("", source, False, error="unknown source")


def _ratify_schema_org(terms: tuple[str, ...], *, timeout: float) -> RatificationHit:
    import httpx

    resp = httpx.get(_SOURCE_ENDPOINTS["schema"], timeout=timeout)
    resp.raise_for_status()
    context = resp.json().get("@context", {})
    keys = {str(key).lower(): str(key) for key in context}
    for term in terms:
        normalized = _schema_key(term)
        if normalized.lower() in keys:
            matched = keys[normalized.lower()]
            return RatificationHit(
                concept_id="",
                source="schema",
                ok=True,
                matched=matched,
                provider_ref=f"https://schema.org/{matched}",
            )
    return RatificationHit("", "schema", False, error="no schema.org term match")


def _ratify_text_source(
    source: str, url: str, terms: tuple[str, ...], timeout: float
) -> RatificationHit:
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    text = resp.text.lower()
    for term in terms:
        if term.lower() in text:
            return RatificationHit(
                concept_id="",
                source=source,
                ok=True,
                matched=term,
                provider_ref=url,
            )
    return RatificationHit("", source, False, error=f"no {source} text match")


def _ratify_agrovoc(terms: tuple[str, ...], *, timeout: float) -> RatificationHit:
    import httpx

    for term in terms:
        resp = httpx.get(
            _SOURCE_ENDPOINTS["agrovoc"],
            params={"query": term, "lang": "en"},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.text.lower()
        if term.lower() in body:
            return RatificationHit(
                concept_id="",
                source="agrovoc",
                ok=True,
                matched=term,
                provider_ref=str(resp.url),
            )
    return RatificationHit("", "agrovoc", False, error="no AGROVOC search match")


def _ratify_unesco(terms: tuple[str, ...], *, timeout: float) -> RatificationHit:
    import httpx

    for term in terms:
        escaped = term.replace('"', '\\"')
        query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?c ?label WHERE {{
  ?c skos:prefLabel ?label .
  FILTER(lang(?label) = "en")
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{escaped}")))
}} LIMIT 1
"""
        try:
            resp = httpx.get(
                _SOURCE_ENDPOINTS["unesco"],
                params={"query": query, "format": "application/sparql-results+json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
        except (httpx.HTTPError, ValueError):
            continue
        if bindings:
            row = bindings[0]
            label = row.get("label", {}).get("value", term)
            uri = row.get("c", {}).get("value", "")
            return RatificationHit(
                concept_id="",
                source="unesco",
                ok=True,
                matched=label,
                provider_ref=uri,
            )
    return RatificationHit("", "unesco", False, error="no UNESCO SPARQL match")


def _ratify_getty(terms: tuple[str, ...], *, timeout: float) -> RatificationHit:
    import httpx

    last_error = ""
    for term in terms:
        escaped = term.replace('"', '\\"')
        query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?c ?label WHERE {{
  ?c (skos:prefLabel|skos:altLabel) ?label .
  FILTER(lang(?label) = "en")
  FILTER(CONTAINS(LCASE(STR(?label)), LCASE("{escaped}")))
}} LIMIT 1
"""
        try:
            resp = httpx.get(
                _SOURCE_ENDPOINTS["getty"],
                params={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
            continue
        if bindings:
            row = bindings[0]
            label = row.get("label", {}).get("value", term)
            uri = row.get("c", {}).get("value", "")
            return RatificationHit(
                concept_id="",
                source="getty",
                ok=True,
                matched=label,
                provider_ref=uri,
            )
    return RatificationHit(
        "", "getty", False, error=last_error or "no Getty SPARQL match"
    )


def _schema_to_dict(schema: MicroSchema) -> dict[str, typing.Any]:
    return {
        "question": schema.question,
        "provenance": schema.provenance,
        "concepts": [dataclasses.asdict(concept) for concept in schema.concepts],
        "frames": [dataclasses.asdict(frame) for frame in schema.frames],
        "edges": [dataclasses.asdict(edge) for edge in schema.edges],
    }


def _question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _corpus_snippet_hash(corpus_snippets: tuple[str, ...]) -> str:
    payload = json.dumps(list(corpus_snippets), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_key(
    question: str,
    corpus_snippets: tuple[str, ...],
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, str]:
    return {
        "question_hash": _question_hash(question),
        "corpus_snippet_hash": _corpus_snippet_hash(corpus_snippets),
        "prompt_version": prompt_version,
    }


def _schema_key(term: str) -> str:
    parts = re.findall(r"[a-z0-9]+", term.lower())
    return "".join(part.capitalize() for part in parts)


def _schema_id(term: str) -> str:
    parts = re.findall(r"[a-z0-9]+", term.lower())
    return "_".join(parts)


def _subject_label(question: str) -> str:
    terms = [
        token
        for token in re.findall(r"[a-z][a-z-]{2,}", question.lower())
        if token not in {"how", "many", "what", "which", "have", "need", "does"}
    ]
    return " ".join(terms[:3]) or "query subject"


def _dedupe_concepts(concepts: list[MicroConcept]) -> list[MicroConcept]:
    seen: set[str] = set()
    out: list[MicroConcept] = []
    for concept in concepts:
        if concept.id in seen:
            continue
        seen.add(concept.id)
        out.append(concept)
    return out


def _route_sources_for_concept(concept: MicroConcept) -> tuple[str, ...]:
    source_ids: list[str] = []
    for source in (
        *concept.source_hints,
        *_DOMAIN_SOURCES.get(concept.domain, ("schema", "lov")),
        *_label_sources(concept),
    ):
        if source not in source_ids:
            source_ids.append(source)
    return tuple(source_ids)


def _label_sources(concept: MicroConcept) -> tuple[str, ...]:
    terms = " ".join((concept.label, *concept.aliases)).lower()
    sources: list[str] = []
    if re.search(r"\b(?:model|kit|instrument|guitar|piano|drum)\b", terms):
        sources.append("getty")
    if re.search(r"\b(?:point|points|threshold|minimum|required)\b", terms):
        sources.append("qudt")
    if re.search(r"\b(?:bake|baking|food|recipe|meal)\b", terms):
        sources.append("agrovoc")
    if re.search(r"\b(?:holiday|school|education|social)\b", terms):
        sources.append("unesco")
    return tuple(sources)


def _is_material_concept(concept: MicroConcept) -> bool:
    if concept.domain in {"action", "generic"}:
        return False
    return concept.id not in {"user", "person", "count"}


def _required_answer_concept_ids(schema: MicroSchema, answer_type: str) -> list[str]:
    explicit = [
        concept.id
        for concept in schema.concepts
        if concept.purpose == "answer_bearing" and not _is_meta_concept(concept)
    ]
    if explicit:
        return explicit

    required = [
        concept.id
        for concept in schema.concepts
        if _is_required_answer_concept(concept, schema.question, answer_type)
    ]
    return required or [
        concept.id
        for concept in schema.concepts
        if _is_material_concept(concept)
        and not _is_incidental_for_answer(concept, schema.question, answer_type)
    ]


def _required_concept_reliabilities(
    schema: MicroSchema,
    required: list[str],
    hits: tuple[RatificationHit, ...],
) -> tuple[RequiredConceptReliability, ...]:
    concepts = {concept.id: concept for concept in schema.concepts}
    return tuple(
        _required_concept_reliability(concepts[concept_id], schema, hits)
        for concept_id in required
        if concept_id in concepts
    )


def _required_concept_reliability(
    concept: MicroConcept,
    schema: MicroSchema,
    hits: tuple[RatificationHit, ...],
) -> RequiredConceptReliability:
    components = (
        _proposal_component(concept, schema),
        _source_ratification_component(concept, hits),
        _evidence_span_component(concept, schema),
        _eval_delta_component(concept),
    )
    return RequiredConceptReliability(
        concept_id=concept.id,
        effective_reliability=min(component.score for component in components),
        components=components,
    )


def _proposal_component(
    concept: MicroConcept, schema: MicroSchema
) -> ReliabilityComponent:
    if concept.purpose == "answer_bearing":
        return ReliabilityComponent(
            id="proposal.answer_bearing",
            stage="abduction",
            score=0.70,
            reason="LLM or deterministic schema marked concept as answer_bearing",
        )
    if schema.provenance == "deterministic-fallback":
        return ReliabilityComponent(
            id="proposal.deterministic_required",
            stage="abduction",
            score=0.65,
            reason="deterministic fallback inferred required answer concept",
        )
    return ReliabilityComponent(
        id="proposal.inferred_required",
        stage="abduction",
        score=0.60,
        reason="required answer concept inferred from schema and answer type",
    )


def _source_ratification_component(
    concept: MicroConcept,
    hits: tuple[RatificationHit, ...],
) -> ReliabilityComponent:
    matching = [hit for hit in hits if hit.concept_id == concept.id]
    ok = [hit for hit in matching if hit.ok]
    if ok:
        sources = ",".join(sorted({hit.source for hit in ok}))
        return ReliabilityComponent(
            id="source.ratified",
            stage="deduction",
            score=0.90,
            reason=f"required concept ratified by {sources}",
        )
    if matching:
        sources = ",".join(sorted({hit.source for hit in matching}))
        return ReliabilityComponent(
            id="source.unratified",
            stage="deduction",
            score=0.35,
            reason=f"required concept had no successful source hit via {sources}",
        )
    return ReliabilityComponent(
        id="source.not_routed",
        stage="deduction",
        score=0.30,
        reason="required concept had no ontology source route",
    )


def _evidence_span_component(
    concept: MicroConcept, schema: MicroSchema
) -> ReliabilityComponent:
    for edge in schema.edges:
        if edge.target == concept.id and edge.evidence:
            return ReliabilityComponent(
                id="evidence.edge_span",
                stage="deduction",
                score=0.85,
                reason=f"graph edge carries evidence span: {edge.evidence[:80]}",
            )
    if _concept_mentioned_in_question(concept, schema.question):
        return ReliabilityComponent(
            id="evidence.question_span",
            stage="deduction",
            score=0.65,
            reason="concept label or alias appears in the question",
        )
    return ReliabilityComponent(
        id="evidence.unspanned",
        stage="deduction",
        score=0.40,
        reason="concept lacks an explicit graph evidence span",
    )


def _eval_delta_component(concept: MicroConcept) -> ReliabilityComponent:
    return ReliabilityComponent(
        id="eval.delta_unmeasured",
        stage="induction",
        score=0.50,
        reason=(
            "ontology probe is diagnostic-only; no answer-path eval delta "
            f"measured for {concept.id}"
        ),
    )


def _concept_mentioned_in_question(concept: MicroConcept, question: str) -> bool:
    low = question.lower()
    return any(
        re.search(rf"\b{re.escape(term.lower())}s?\b", low)
        for term in (concept.label, *concept.aliases)
        if term
    )


def _is_required_answer_concept(
    concept: MicroConcept, question: str, answer_type: str
) -> bool:
    if _is_meta_concept(concept):
        return False
    if answer_type == "threshold_lookup":
        return concept.domain in {"quantity", "threshold", "unit"} and re.search(
            r"\b(?:point|points|threshold|required|minimum|needed|loyalty)\b",
            _concept_text(concept),
        )
    if answer_type == "temporal_event_count":
        return not _is_time_or_amount_concept(concept) and concept.domain in {
            "action",
            "culture",
            "food",
            "object",
            "product",
            "social",
        }
    if answer_type in {
        "canonical_entity_count",
        "current_inventory",
        "role_filtered_count",
    }:
        return not _is_incidental_for_answer(concept, question, answer_type)
    if _asks_for_time_amount(question):
        return concept.domain in {"quantity", "unit", "time"} and re.search(
            r"\b(?:day|days|hour|hours|money|amount|duration|total)\b",
            _concept_text(concept),
        )
    return not _is_incidental_for_answer(concept, question, answer_type)


def _is_meta_concept(concept: MicroConcept) -> bool:
    label = concept.label.strip().lower()
    concept_id = concept.id.strip().lower()
    if concept_id in {"user", "person", "count", "quantity", "number"}:
        return True
    return label in {
        "person",
        "user",
        "self",
        "count",
        "number",
        "quantity",
        "number of times",
    }


def _is_incidental_for_answer(
    concept: MicroConcept, question: str, answer_type: str
) -> bool:
    if _is_meta_concept(concept):
        return True
    text = _concept_text(concept)
    if concept.domain == "place" and answer_type != "threshold_lookup":
        return True
    if _is_time_or_amount_concept(concept) and not _asks_for_time_amount(question):
        return True
    if concept.domain == "generic" and concept.purpose != "answer_bearing":
        return True
    if concept.domain == "action" and answer_type not in {
        "temporal_event_count",
        "canonical_entity_count",
    }:
        return True
    return bool(re.search(r"\b(?:last|past|current|time period)\b", text))


def _is_time_or_amount_concept(concept: MicroConcept) -> bool:
    text = _concept_text(concept)
    return concept.domain in {"quantity", "unit", "time"} and bool(
        re.search(
            r"\b(?:day|days|hour|hours|week|weeks|month|months|year|years|"
            r"time|duration|period|quantity|number|count)\b",
            text,
        )
    )


def _asks_for_time_amount(question: str) -> bool:
    low = question.lower()
    return bool(
        re.search(r"\bhow many\s+(?:days|hours|weeks|months|years)\b", low)
        or re.search(r"\bhow much\s+(?:money|time)\b", low)
        or "total traveling" in low
    )


def _is_threshold_question(question: str) -> bool:
    low = question.lower()
    return "points" in low and "redeem" in low


def _concept_text(concept: MicroConcept) -> str:
    terms = (concept.id, concept.label, concept.domain, *concept.aliases)
    return " ".join(terms).lower()


def _ratified_concept_ids(hits: tuple[RatificationHit, ...]) -> set[str]:
    return {hit.concept_id for hit in hits if hit.ok}


def _summarize_fail18_probe(
    results: list[Fail18OntologyItem], *, use_llm: bool, remote: bool
) -> Fail18OntologySummary:
    rows_with_any_hit = 0
    rows_with_material_hit = 0
    rows_with_required_hit = 0
    rows_with_required_complete = 0
    material_total = 0
    ratified_material_total = 0
    required_total = 0
    ratified_required_total = 0
    required_reliabilities: list[float] = []
    for item in results:
        ratified = _ratified_concept_ids(item.hits)
        if ratified:
            rows_with_any_hit += 1
        material = [
            concept.id
            for concept in item.schema.concepts
            if _is_material_concept(concept)
        ]
        material_total += len(material)
        ratified_material = [
            concept_id for concept_id in material if concept_id in ratified
        ]
        ratified_material_total += len(ratified_material)
        if ratified_material:
            rows_with_material_hit += 1
        required = _required_answer_concept_ids(item.schema, item.answer_type)
        required_total += len(required)
        required_reliabilities.extend(
            record.effective_reliability
            for record in _required_concept_reliabilities(
                item.schema,
                required,
                item.hits,
            )
        )
        ratified_required = [
            concept_id for concept_id in required if concept_id in ratified
        ]
        ratified_required_total += len(ratified_required)
        if ratified_required:
            rows_with_required_hit += 1
        if required and len(ratified_required) == len(required):
            rows_with_required_complete += 1
    return Fail18OntologySummary(
        total=len(results),
        provenance="llm" if use_llm else "deterministic-fallback",
        remote=remote,
        rows_with_any_hit=rows_with_any_hit,
        rows_with_material_hit=rows_with_material_hit,
        rows_with_required_hit=rows_with_required_hit,
        rows_with_required_complete=rows_with_required_complete,
        material_concepts=material_total,
        ratified_material_concepts=ratified_material_total,
        required_concepts=required_total,
        ratified_required_concepts=ratified_required_total,
        required_reliability_min=(
            min(required_reliabilities) if required_reliabilities else 0.0
        ),
        required_reliability_avg=(
            round(sum(required_reliabilities) / len(required_reliabilities), 4)
            if required_reliabilities
            else 0.0
        ),
        results=tuple(results),
    )


def _fail18_corpus_by_id(
    path: pathlib.Path,
) -> dict[str, dict[str, typing.Any]]:
    if not path.exists():
        return {}
    return {str(row["question_id"]): row for row in ambiguity_fail18.load_corpus(path)}


def _fail18_snippets(row: dict[str, typing.Any] | None) -> tuple[str, ...]:
    if row is None:
        return ()
    snippets: list[str] = []
    for session in row.get("haystack_sessions", []):
        messages = session.get("messages", []) if isinstance(session, dict) else session
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            text = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
            if text:
                snippets.append(text[:500])
            if len(snippets) >= 8:
                return tuple(snippets)
    return tuple(snippets)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try remote ontology ratification for a generated micro-schema."
    )
    parser.add_argument("question", nargs="*", help="Question to synthesize.")
    parser.add_argument("--llm", action="store_true", help="Use LLM schema proposal.")
    parser.add_argument("--remote", action="store_true", help="Run remote lookups.")
    parser.add_argument("--append", action="store_true", help="Append JSONL result.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Candidate log path.")
    parser.add_argument("--no-cache", action="store_true", help="Disable JSONL cache.")
    parser.add_argument("--fail18", action="store_true", help="Probe all fail18 cases.")
    parser.add_argument("--path", default="", help="fail18 manifest path.")
    parser.add_argument("--corpus", default="", help="fail18 corpus path.")
    parser.add_argument("--limit", type=int, default=0, help="Limit fail18 probe rows.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Remote request timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fail18:
        summary = probe_fail18(
            pathlib.Path(args.path) if args.path else ambiguity_fail18.DEFAULT_MANIFEST,
            corpus_path=(
                pathlib.Path(args.corpus)
                if args.corpus
                else ambiguity_fail18.DEFAULT_CORPUS
            ),
            use_llm=args.llm,
            remote=args.remote,
            append=args.append,
            out=pathlib.Path(args.out),
            timeout=args.timeout,
            limit=max(0, int(args.limit)),
            cache=not args.no_cache,
        )
        payload = summary.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(
            "fail18 ontology probe "
            f"({summary.provenance}, remote={summary.remote}): "
            f"required_complete={summary.rows_with_required_complete}/"
            f"{summary.total}; "
            f"required_hit={summary.rows_with_required_hit}/{summary.total}; "
            f"required_concepts={summary.ratified_required_concepts}/"
            f"{summary.required_concepts} ratified; "
            f"required_reliability_min={summary.required_reliability_min:.2f}; "
            f"required_reliability_avg={summary.required_reliability_avg:.2f}; "
            f"rows_with_material_hit={summary.rows_with_material_hit}/{summary.total}; "
            f"material_concepts={summary.ratified_material_concepts}/"
            f"{summary.material_concepts} ratified"
        )
        for item in summary.results:
            data = item.to_dict()
            concepts = ", ".join(
                f"{concept['id']}:{concept['domain']}"
                for concept in data["schema"]["concepts"]
            )
            ratified = ", ".join(data["ratified_material_concepts"]) or "-"
            required = ", ".join(data["required_answer_concepts"]) or "-"
            required_ratified = (
                ", ".join(data["ratified_required_answer_concepts"]) or "-"
            )
            reliability = {
                item["concept_id"]: item["effective_reliability"]
                for item in data["required_answer_reliability"]
            }
            print(
                f"  {item.question_id} type={item.answer_type} "
                f"cache={data['cache_status'] or '-'} "
                f"required={required} ratified_required={required_ratified} "
                f"reliability={reliability} "
                f"ratified_material={ratified} concepts=[{concepts}]"
            )
        return 0

    question = " ".join(args.question).strip()
    if not question:
        question = "How many points do I need to redeem a free skincare product?"
    report = build_report(
        question,
        use_llm=args.llm,
        remote=args.remote,
        cache_path=pathlib.Path(args.out) if args.llm and not args.no_cache else None,
        append=args.append,
    )
    print(
        json.dumps(
            {
                "schema": _schema_to_dict(report.schema),
                "routes": {
                    key: [dataclasses.asdict(route) for route in routes]
                    for key, routes in report.routes.items()
                },
                "hits": [dataclasses.asdict(hit) for hit in report.hits],
                "cache_status": report.cache_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

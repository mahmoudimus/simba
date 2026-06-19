from __future__ import annotations

import json

import simba.eval.ontology_router as ontology_router


def test_micro_schema_routes_threshold_product_question() -> None:
    schema = ontology_router.propose_micro_schema(
        "How many points do I need to earn to redeem a free skincare product?"
    )

    concept_ids = {concept.id for concept in schema.concepts}
    routes = ontology_router.route_schema(schema)

    assert {"loyalty_points", "skincare_product"} <= concept_ids
    assert {route.source for route in routes["loyalty_points"]} == {
        "qudt",
        "schema",
    }
    assert {route.source for route in routes["skincare_product"]} == {
        "schema",
        "agrovoc",
    }


def test_micro_schema_routes_culture_object_question() -> None:
    schema = ontology_router.propose_micro_schema(
        "How many musical instruments do I currently own?"
    )

    routes = ontology_router.route_schema(schema)

    assert "musical_instrument" in routes
    assert [route.source for route in routes["musical_instrument"]] == [
        "schema",
        "getty",
    ]


def test_schema_org_ratifier_matches_context(monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"@context": {"Product": "https://schema.org/Product"}}

    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: _Resp(),
    )

    hit = ontology_router._ratify_schema_org(("product",), timeout=1)

    assert hit.ok is True
    assert hit.matched == "Product"
    assert hit.provider_ref == "https://schema.org/Product"


def test_append_report_writes_candidate_jsonl(tmp_path) -> None:
    schema = ontology_router.propose_micro_schema("How many model kits?")
    report = ontology_router.RatificationReport(
        schema=schema,
        routes=ontology_router.route_schema(schema),
        hits=(
            ontology_router.RatificationHit(
                concept_id="model_kit",
                source="schema",
                ok=True,
                matched="Product",
                provider_ref="https://schema.org/Product",
            ),
        ),
    )
    out = tmp_path / "candidates.jsonl"

    ontology_router.append_report(report, out)

    text = out.read_text(encoding="utf-8")
    assert "model_kit" in text
    assert '"trust_tier": "ratified"' in text
    assert f'"prompt_version": "{ontology_router.PROMPT_VERSION}"' in text
    assert '"corpus_snippet_hash"' in text


def test_cached_report_round_trips_latest_jsonl_row(tmp_path) -> None:
    schema = ontology_router.propose_micro_schema("How many model kits?")
    first = ontology_router.RatificationReport(
        schema=schema,
        routes=ontology_router.route_schema(schema),
        hits=(
            ontology_router.RatificationHit(
                concept_id="model_kit",
                source="schema",
                ok=False,
                error="old miss",
            ),
        ),
    )
    second = ontology_router.RatificationReport(
        schema=schema,
        routes=ontology_router.route_schema(schema),
        hits=(
            ontology_router.RatificationHit(
                concept_id="model_kit",
                source="getty",
                ok=True,
                matched="toy kits",
            ),
        ),
    )
    out = tmp_path / "candidates.jsonl"
    snippets = ("I bought a model kit.",)

    ontology_router.append_report(first, out, corpus_snippets=snippets, remote=True)
    ontology_router.append_report(second, out, corpus_snippets=snippets, remote=True)

    cached = ontology_router.load_cached_report(
        "How many model kits?",
        snippets,
        out,
        remote=True,
    )

    assert cached is not None
    assert cached.cache_status == "report-cache"
    assert cached.hits[0].ok is True
    assert cached.hits[0].source == "getty"


def test_cached_report_respects_prompt_version(tmp_path) -> None:
    schema = ontology_router.propose_micro_schema("How many model kits?")
    report = ontology_router.RatificationReport(
        schema=schema,
        routes=ontology_router.route_schema(schema),
        hits=(),
    )
    out = tmp_path / "candidates.jsonl"
    ontology_router.append_report(
        report,
        out,
        prompt_version="old-prompt",
        remote=False,
    )

    cached = ontology_router.load_cached_report(
        "How many model kits?",
        (),
        out,
        remote=False,
    )

    assert cached is None


def test_build_report_uses_cached_schema_without_llm(monkeypatch, tmp_path) -> None:
    schema = ontology_router.propose_micro_schema("How many model kits?")
    cached = ontology_router.RatificationReport(
        schema=schema,
        routes=ontology_router.route_schema(schema),
        hits=(),
    )
    out = tmp_path / "candidates.jsonl"
    ontology_router.append_report(cached, out, remote=False)

    def _raise(*args, **kwargs):
        raise AssertionError("LLM should not be called on schema cache hit")

    def _ratify(schema, **kwargs):
        return ontology_router.RatificationReport(
            schema=schema,
            routes=ontology_router.route_schema(schema),
            hits=(),
        )

    monkeypatch.setattr(ontology_router, "propose_micro_schema_with_llm", _raise)
    monkeypatch.setattr(ontology_router, "ratify_schema", _ratify)

    report = ontology_router.build_report(
        "How many model kits?",
        use_llm=True,
        remote=True,
        cache_path=out,
    )

    assert report.cache_status == "schema-cache"
    assert report.schema.concepts[0].id == "model_kit"


def test_build_report_normalizes_cached_schema(monkeypatch, tmp_path) -> None:
    schema = ontology_router.MicroSchema(
        question="How many days did I spend in Hawaii?",
        concepts=(
            ontology_router.MicroConcept(
                id="days_spent_in_hawaii",
                label="days spent in Hawaii",
                domain="time",
                purpose="answer_bearing",
            ),
        ),
    )
    cached = ontology_router.RatificationReport(
        schema=schema,
        routes=ontology_router.route_schema(schema),
        hits=(),
    )
    out = tmp_path / "candidates.jsonl"
    ontology_router.append_report(cached, out, remote=False)

    def _raise(*args, **kwargs):
        raise AssertionError("LLM should not be called on schema cache hit")

    def _ratify(schema, **kwargs):
        return ontology_router.RatificationReport(
            schema=schema,
            routes=ontology_router.route_schema(schema),
            hits=(),
        )

    monkeypatch.setattr(ontology_router, "propose_micro_schema_with_llm", _raise)
    monkeypatch.setattr(ontology_router, "ratify_schema", _ratify)

    report = ontology_router.build_report(
        "How many days did I spend in Hawaii?",
        use_llm=True,
        remote=True,
        cache_path=out,
    )

    concepts = {concept.id: concept for concept in report.schema.concepts}
    assert report.cache_status == "schema-cache"
    assert concepts["day"].purpose == "answer_bearing"
    assert concepts["hawaii"].purpose == "constraint"
    assert ontology_router._required_answer_concept_ids(
        report.schema, "count"
    ) == ["day"]


def test_normalize_schema_decomposes_duration_places() -> None:
    schema = ontology_router.MicroSchema(
        question="How many total days did I spend in Hawaii and NYC?",
        concepts=(
            ontology_router.MicroConcept(
                id="total_days_in_hawaii_and_nyc",
                label="total days in Hawaii and NYC",
                domain="time",
                purpose="answer_bearing",
            ),
        ),
        edges=(
            ontology_router.MicroEdge(
                source="question",
                relation="counts",
                target="total_days_in_hawaii_and_nyc",
                evidence="total days",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    concepts = {concept.id: concept for concept in normalized.concepts}
    edge_relations = {
        (edge.relation, edge.target) for edge in normalized.edges
    }

    assert {"day", "hawaii", "new_york_city"} <= set(concepts)
    assert concepts["day"].purpose == "answer_bearing"
    assert concepts["hawaii"].purpose == "constraint"
    assert concepts["new_york_city"].purpose == "constraint"
    assert ("counts", "day") in edge_relations
    assert ("constrains", "hawaii") in edge_relations
    assert ("constrains", "new_york_city") in edge_relations
    assert ontology_router._required_answer_concept_ids(
        normalized, "count"
    ) == ["day"]


def test_normalize_schema_decomposes_wedding_attendance_idempotently() -> None:
    schema = ontology_router.MicroSchema(
        question="How many weddings have I attended this year?",
        concepts=(
            ontology_router.MicroConcept(
                id="wedding_attendance",
                label="wedding attendance",
                domain="social",
                purpose="answer_bearing",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    renormalized = ontology_router.normalize_schema(normalized)

    assert [concept.id for concept in normalized.concepts] == [
        "wedding",
        "attendance",
    ]
    assert [concept.id for concept in renormalized.concepts] == [
        "wedding",
        "attendance",
    ]
    assert ontology_router._required_answer_concept_ids(
        normalized, "count"
    ) == ["wedding"]


def test_normalize_schema_demotes_time_amount_activities() -> None:
    schema = ontology_router.MicroSchema(
        question="How many hours of jogging and yoga did I do last week?",
        concepts=(
            ontology_router.MicroConcept(
                id="hour",
                label="hour",
                domain="unit",
                purpose="answer_bearing",
            ),
            ontology_router.MicroConcept(
                id="jogging",
                label="jogging",
                domain="activity",
                purpose="answer_bearing",
            ),
            ontology_router.MicroConcept(
                id="yoga",
                label="yoga",
                domain="activity",
                purpose="answer_bearing",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    concepts = {concept.id: concept for concept in normalized.concepts}

    assert concepts["hour"].purpose == "answer_bearing"
    assert concepts["jogging"].purpose == "constraint"
    assert concepts["yoga"].purpose == "constraint"
    assert ontology_router._required_answer_concept_ids(
        normalized, "count"
    ) == ["hour"]


def test_normalize_schema_canonicalizes_model_kit_aliases() -> None:
    schema = ontology_router.MicroSchema(
        question="How many model kits have I worked on or bought?",
        concepts=(
            ontology_router.MicroConcept(
                id="scale_model",
                label="scale model",
                domain="object",
                aliases=("plastic kit",),
                purpose="answer_bearing",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    routes = ontology_router.route_schema(normalized)
    concept = normalized.concepts[0]

    assert concept.id == "model_kit"
    assert "toy kit" in concept.aliases
    assert [route.source for route in routes["model_kit"]] == [
        "schema",
        "getty",
        "agrovoc",
    ]


def test_normalize_schema_routes_threshold_question_to_points() -> None:
    schema = ontology_router.MicroSchema(
        question=(
            "How many points do I need to earn to redeem a free skincare "
            "product at Sephora?"
        ),
        concepts=(
            ontology_router.MicroConcept(
                id="free_skincare_product",
                label="free skincare product",
                domain="product",
                purpose="answer_bearing",
            ),
            ontology_router.MicroConcept(
                id="points_required",
                label="points required",
                domain="threshold",
                purpose="constraint",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    concepts = {concept.id: concept for concept in normalized.concepts}

    assert concepts["free_skincare_product"].purpose == "constraint"
    assert concepts["loyalty_points"].purpose == "answer_bearing"
    assert concepts["redemption_threshold"].purpose == "constraint"
    assert ontology_router._required_answer_concept_ids(
        normalized, "threshold_lookup"
    ) == ["loyalty_points"]


def test_normalize_schema_canonicalizes_total_reach() -> None:
    schema = ontology_router.MicroSchema(
        question="What was the total number of people reached?",
        concepts=(
            ontology_router.MicroConcept(
                id="total_reach",
                label="Total Reach",
                domain="quantity",
                aliases=("people reached", "audience size"),
                purpose="answer_bearing",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    concept = normalized.concepts[0]

    assert concept.id == "people_reached"
    assert "people audience" in concept.aliases
    assert "interaction counter" in concept.aliases


def test_normalize_schema_canonicalizes_owned_musical_instrument() -> None:
    schema = ontology_router.MicroSchema(
        question="How many musical instruments do I currently own?",
        concepts=(
            ontology_router.MicroConcept(
                id="owned_musical_instrument",
                label="Owned Musical Instrument",
                domain="object",
                purpose="answer_bearing",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)

    assert [concept.id for concept in normalized.concepts] == [
        "musical_instrument"
    ]
    assert ontology_router._required_answer_concept_ids(
        normalized, "current_inventory"
    ) == ["musical_instrument"]


def test_normalize_schema_canonicalizes_hyphenated_art_event() -> None:
    schema = ontology_router.MicroSchema(
        question="How many different art-related events did I attend?",
        concepts=(
            ontology_router.MicroConcept(
                id="art_event",
                label="art-related event",
                domain="event",
                purpose="answer_bearing",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    concept = normalized.concepts[0]

    assert concept.id == "art_event"
    assert "event" in concept.aliases
    assert ontology_router._required_answer_concept_ids(
        normalized, "count"
    ) == ["art_event"]


def test_normalize_schema_canonicalizes_composed_hours_answer() -> None:
    schema = ontology_router.MicroSchema(
        question="How many hours of jogging and yoga did I do last week?",
        concepts=(
            ontology_router.MicroConcept(
                id="total_hours_jogging_yoga",
                label="Total hours of jogging and yoga",
                domain="quantity",
                purpose="constraint",
            ),
            ontology_router.MicroConcept(
                id="hour",
                label="hour",
                domain="unit",
                purpose="incidental",
            ),
        ),
    )

    normalized = ontology_router.normalize_schema(schema)
    concepts = {concept.id: concept for concept in normalized.concepts}

    assert list(concepts) == ["hour"]
    assert concepts["hour"].purpose == "answer_bearing"
    assert ontology_router._required_answer_concept_ids(
        normalized, "count"
    ) == ["hour"]


def test_schema_from_json_normalizes_llm_candidate() -> None:
    schema = ontology_router._schema_from_json(
        "How many products?",
        {
            "concepts": [
                {
                    "id": "customer_product",
                    "label": "customer product",
                    "domain": "product",
                    "aliases": ["item"],
                    "purpose": "answer_bearing",
                }
            ],
            "frames": [
                {
                    "id": "purchase",
                    "label": "purchase",
                    "lexical_units": ["bought"],
                    "roles": {"goods": "customer_product"},
                }
            ],
            "edges": [
                {
                    "source": "question",
                    "relation": "counts",
                    "target": "customer_product",
                    "evidence": "How many products",
                }
            ],
        },
    )

    assert schema is not None
    assert schema.provenance == "llm"
    assert schema.concepts[0].source_hints == ("schema", "agrovoc")
    assert schema.concepts[0].purpose == "answer_bearing"
    assert schema.frames[0].roles == {"goods": "customer_product"}
    assert schema.edges[0].relation == "counts"


def test_route_schema_augments_llm_model_kit_with_getty() -> None:
    schema = ontology_router._schema_from_json(
        "How many model kits?",
        {
            "concepts": [
                {
                    "id": "model_kit",
                    "label": "Model Kit",
                    "domain": "product",
                    "aliases": ["scale model"],
                }
            ]
        },
    )

    assert schema is not None
    routes = ontology_router.route_schema(schema)

    assert [route.source for route in routes["model_kit"]] == [
        "schema",
        "agrovoc",
        "getty",
    ]


def test_probe_fail18_summarizes_without_remote(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    corpus = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question": "How many model kits have I worked on or bought?",
                    "failure_mode": "D_underextraction_readerfixed",
                    "gold_answer": "5",
                    "gold_count": None,
                    "clingo_certain": 1,
                    "clingo_possible": 6,
                }
            ]
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "haystack_sessions": [
                        [
                            {
                                "role": "user",
                                "content": "I bought a Tamiya model kit.",
                            }
                        ]
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = ontology_router.probe_fail18(
        manifest,
        corpus_path=corpus,
        remote=False,
    )

    assert summary.total == 1
    assert summary.rows_with_material_hit == 0
    assert summary.material_concepts == 1
    assert summary.results[0].answer_type == "canonical_entity_count"
    assert summary.results[0].schema.concepts[0].id == "model_kit"
    assert summary.results[0].to_dict()["required_answer_concepts"] == [
        "model_kit"
    ]


def test_required_score_ignores_incidental_year_hit() -> None:
    schema = ontology_router._schema_from_json(
        "How many weddings have I attended in this year?",
        {
            "concepts": [
                {
                    "id": "wedding",
                    "label": "Wedding",
                    "domain": "social",
                    "aliases": ["marriage ceremony"],
                },
                {
                    "id": "year",
                    "label": "Year",
                    "domain": "unit",
                    "aliases": ["calendar year"],
                },
            ]
        },
    )
    assert schema is not None
    item = ontology_router.Fail18OntologyItem(
        question_id="q",
        question=schema.question,
        failure_mode="",
        answer_type="count",
        gold_numeric=3,
        schema=schema,
        routes={},
        hits=(
            ontology_router.RatificationHit(
                concept_id="year",
                source="schema",
                ok=True,
            ),
        ),
    )

    data = item.to_dict()

    assert data["required_answer_concepts"] == ["wedding"]
    assert data["ratified_required_answer_concepts"] == []
    assert data["required_answer_complete"] is False


def test_required_score_honors_llm_answer_bearing_purpose() -> None:
    schema = ontology_router._schema_from_json(
        "How many hours of jogging and yoga did I do last week?",
        {
            "concepts": [
                {
                    "id": "hour",
                    "label": "Hour",
                    "domain": "unit",
                    "purpose": "answer_bearing",
                },
                {
                    "id": "last_week",
                    "label": "Last week",
                    "domain": "time",
                    "purpose": "constraint",
                },
            ]
        },
    )

    assert schema is not None
    assert ontology_router._required_answer_concept_ids(schema, "count") == ["hour"]


def test_required_score_keeps_typed_quantity_answer_node() -> None:
    schema = ontology_router.propose_micro_schema(
        "How many points do I need to earn to redeem a free skincare product?"
    )

    assert ontology_router._required_answer_concept_ids(
        schema, "threshold_lookup"
    ) == ["loyalty_points"]


def test_required_reliability_uses_gamma_style_min_gate() -> None:
    schema = ontology_router._schema_from_json(
        "How many model kits have I worked on or bought?",
        {
            "concepts": [
                {
                    "id": "model_kit",
                    "label": "Model Kit",
                    "domain": "product",
                    "aliases": ["scale model"],
                    "purpose": "answer_bearing",
                }
            ],
            "edges": [
                {
                    "source": "question",
                    "relation": "counts",
                    "target": "model_kit",
                    "evidence": "model kits",
                }
            ],
        },
    )
    assert schema is not None
    item = ontology_router.Fail18OntologyItem(
        question_id="q",
        question=schema.question,
        failure_mode="",
        answer_type="canonical_entity_count",
        gold_numeric=5,
        schema=schema,
        routes={},
        hits=(
            ontology_router.RatificationHit(
                concept_id="model_kit",
                source="getty",
                ok=True,
                matched="toy kits",
            ),
        ),
    )

    reliability = item.to_dict()["required_answer_reliability"][0]

    assert reliability["concept_id"] == "model_kit"
    assert reliability["effective_reliability"] == 0.50
    assert [component["stage"] for component in reliability["components"]] == [
        "abduction",
        "deduction",
        "deduction",
        "induction",
    ]
    assert reliability["components"][-1]["id"] == "eval.delta_unmeasured"


def test_required_reliability_caps_unratified_source() -> None:
    schema = ontology_router._schema_from_json(
        "How many weddings have I attended in this year?",
        {
            "concepts": [
                {
                    "id": "weddings",
                    "label": "Weddings",
                    "domain": "social",
                    "purpose": "answer_bearing",
                }
            ]
        },
    )
    assert schema is not None
    item = ontology_router.Fail18OntologyItem(
        question_id="q",
        question=schema.question,
        failure_mode="",
        answer_type="count",
        gold_numeric=3,
        schema=schema,
        routes={},
        hits=(
            ontology_router.RatificationHit(
                concept_id="weddings",
                source="schema",
                ok=False,
                error="no schema.org term match",
            ),
        ),
    )

    reliability = item.to_dict()["required_answer_reliability"][0]

    assert reliability["effective_reliability"] == 0.35
    assert reliability["components"][1]["id"] == "source.unratified"

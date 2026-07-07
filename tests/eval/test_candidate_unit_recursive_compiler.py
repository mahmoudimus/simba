from __future__ import annotations

import json
import pathlib

from simba.eval import candidate_unit_recursive_compiler as compiler


def _fact(
    predicate: str,
    arguments: dict[str, object],
    *,
    fact_id: str,
    session: str = "evidence_001",
    span: str = "",
    case_id: str = "case",
) -> compiler.RecursiveFact:
    return compiler.RecursiveFact(
        case_id=case_id,
        evidence_session_id=session,
        fact_id=fact_id,
        predicate=predicate,
        arguments=arguments,
        evidence_span=span,
        confidence=0.9,
    )


def _write_test_lexicon(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "nltk_lexicon.jsonl"
    rows = [
        _lexicon_record(
            "wordnet:boot.n.01",
            "boot.n.01",
            "boot",
            aliases=("boot",),
            parents=("footwear.n.02",),
            definition="footwear that covers the whole foot and lower leg",
        ),
        _lexicon_record(
            "wordnet:footwear.n.02",
            "footwear.n.02",
            "footgear",
            aliases=("footgear", "footwear"),
            parents=("covering.n.02",),
            definition="covering for a person's feet",
        ),
        _lexicon_record(
            "wordnet:footwear.n.01",
            "footwear.n.01",
            "footwear",
            aliases=("footwear",),
            parents=("clothing.n.01",),
            definition="clothing worn on a person's feet",
        ),
        _lexicon_record(
            "wordnet:clothing.n.01",
            "clothing.n.01",
            "article of clothing",
            aliases=("article of clothing", "clothing"),
            parents=(),
            definition="a covering designed to be worn on a person's body",
        ),
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _lexicon_record(
    concept_id: str,
    provider_ref: str,
    label: str,
    *,
    aliases: tuple[str, ...],
    parents: tuple[str, ...],
    definition: str,
) -> dict[str, object]:
    return {
        "id": concept_id,
        "provider": "wordnet",
        "kind": "concept",
        "provider_ref": provider_ref,
        "label": label,
        "aliases_json": json.dumps(list(aliases)),
        "parents_json": json.dumps(list(parents)),
        "definition": definition,
    }


def test_classify_question_extracts_action_obligation_intent() -> None:
    intent = compiler.classify_question(
        "How many items of clothing do I need to pick up or return from a store?"
    )

    assert intent.kind == "count_action_obligation"
    assert intent.aggregation == "count_distinct"
    assert intent.target_terms == ("clothing",)
    assert intent.action_terms == ("pick_up", "return")


def test_compile_action_obligations_merges_replacement_placeholder() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "new pair",
                "verb": "pick up",
                "status": "pending",
                "location": "",
            },
            fact_id="f1",
            span="I still need to pick up the new pair",
        ),
        _fact(
            "sortal",
            {
                "entity": "pair_1",
                "type": "boots",
                "source": "evidence",
                "antecedent": "",
                "licensed_by": "a pair of boots",
            },
            fact_id="f2a",
            span="exchanged a pair of boots",
        ),
        _fact(
            "sortal",
            {
                "entity": "new pair",
                "type": "boots",
                "source": "bridging",
                "antecedent": "pair_1",
                "licensed_by": "contrastive new",
            },
            fact_id="f2b",
            span="exchanged a pair of boots ... pick up the new pair",
        ),
        _fact(
            "distinct",
            {
                "a": "new pair",
                "b": "pair_1",
                "reason": "contrastive new marks a replacement token",
            },
            fact_id="f2c",
            span="exchanged a pair of boots ... pick up the new pair",
        ),
        _fact(
            "relation",
            {
                "source": "new pair",
                "relation": "replaces",
                "target": "pair_1",
            },
            fact_id="f2d",
            span="exchanged a pair of boots ... pick up the new pair",
        ),
        _fact(
            "object_type",
            {"entity": "new pair", "type": "clothing"},
            fact_id="f2e",
            span="new pair",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "returnable boots",
                "verb": "return",
                "status": "needed",
                "location": "Zara",
            },
            fact_id="f3",
            span="I need to return some boots to Zara",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "dry cleaning for the navy blue blazer",
                "verb": "pick up",
                "status": "pending",
                "location": "dry cleaning",
            },
            fact_id="f4",
            span="pick up my dry cleaning for the navy blue blazer",
        ),
        _fact(
            "object_type",
            {"entity": "returnable boots", "type": "clothing"},
            fact_id="f5",
            span="boots",
        ),
        _fact(
            "object_type",
            {"entity": "navy blue blazer", "type": "clothing"},
            fact_id="f6",
            span="navy blue blazer",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["computed_answer"] == 3.0
    labels = [
        unit["label"] for unit in row["candidate_units"] if unit["status"] == "included"
    ]
    assert labels == [
        "new pair of boots",
        "returnable boots",
        "dry cleaning for the navy blue blazer",
    ]
    assert row["recursive_fact_consistency"] == {"issue_count": 0, "issues": []}
    assert row["symbol_namespace"] == {"issue_count": 0, "issues": []}


def test_compile_action_obligations_does_not_saturate_from_coreference() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "new pair",
                "verb": "pick up",
                "status": "pending",
                "location": "",
            },
            fact_id="f1",
            span="I still need to pick up the new pair",
        ),
        _fact(
            "coreference",
            {
                "entity": "new pair",
                "same_as": "new pair of boots",
                "reason": "new pair refers to the exchanged boots",
            },
            fact_id="f2",
            span="exchanged a pair of boots ... pick up the new pair",
        ),
        _fact(
            "object_type",
            {"entity": "boots", "type": "clothing"},
            fact_id="f3",
            span="boots",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["computed_answer"] == 0.0
    assert row["candidate_units"] == []
    assert row["compiled_recursive_fact_count"] == 0
    assert row["quarantined_recursive_fact_count"] == len(facts)
    assert row["recursive_fact_consistency"]["issues"] == [
        {
            "issue": "bridging_coreference_misuse",
            "fact_id": "f2",
            "evidence_session_id": "evidence_001",
            "reason": (
                "coreference appears to saturate a relational noun instead of "
                "asserting same-token identity"
            ),
        }
    ]
    assert row["parse_errors"] == []
    assert row["compiler_warnings"] == ["quarantined_evidence_session:evidence_001:1"]


def test_compile_uses_offline_ontology_for_target_type_ratification(
    monkeypatch,
    tmp_path: pathlib.Path,
) -> None:
    monkeypatch.setattr(
        compiler.type_ontology,
        "DEFAULT_LEXICON_PATH",
        _write_test_lexicon(tmp_path),
    )
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "new pair",
                "verb": "pick up",
                "status": "pending",
                "location": None,
            },
            fact_id="action",
            span="I still need to pick up the new pair",
        ),
        _fact(
            "object_type",
            {"entity": "pair_1", "type": "boots"},
            fact_id="old_pair_type",
            span="exchanged a pair of boots",
        ),
        _fact(
            "sortal",
            {
                "entity": "new pair",
                "type": "boots",
                "source": "bridging",
                "antecedent": "pair_1",
                "licensed_by": "the new pair",
            },
            fact_id="new_pair_type",
            span="I still need to pick up the new pair",
        ),
        _fact(
            "distinct",
            {
                "a": "new pair",
                "b": "pair_1",
                "reason": "contrastive new marks a replacement token",
            },
            fact_id="distinct",
            span="exchanged a pair of boots ... pick up the new pair",
        ),
        _fact(
            "relation",
            {
                "source": "new pair",
                "relation": "replaces",
                "target": "pair_1",
            },
            fact_id="replaces",
            span="exchanged a pair of boots ... pick up the new pair",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["computed_answer"] == 1.0
    assert row["symbol_namespace"] == {"issue_count": 0, "issues": []}
    assert [
        unit["label"] for unit in row["candidate_units"] if unit["status"] == "included"
    ] == ["new pair of boots"]


def test_compile_flags_identity_distinct_conflicts() -> None:
    facts = [
        _fact(
            "coreference",
            {
                "entity": "replacement pair",
                "same_as": "exchanged pair",
                "reason": "bad identity claim",
            },
            fact_id="same",
        ),
        _fact(
            "distinct",
            {
                "a": "replacement pair",
                "b": "exchanged pair",
                "reason": "replacement is not the returned token",
            },
            fact_id="different",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["computed_answer"] == 0.0
    assert row["candidate_units"] == []
    assert row["compiled_recursive_fact_count"] == 0
    assert row["quarantined_recursive_fact_count"] == len(facts)
    assert row["recursive_fact_consistency"]["issues"] == [
        {
            "issue": "identity_distinct_conflict",
            "fact_id": "same",
            "conflicting_fact_id": "different",
            "evidence_session_id": "evidence_001",
            "reason": "same pair is asserted as both same_as and distinct",
        }
    ]
    assert row["parse_errors"] == []
    assert row["compiler_warnings"] == ["quarantined_evidence_session:evidence_001:1"]


def test_compile_flags_identity_replaces_conflicts() -> None:
    facts = [
        _fact(
            "coreference",
            {
                "entity": "replacement pair",
                "same_as": "exchanged pair",
                "reason": "bad identity claim",
            },
            fact_id="same",
        ),
        _fact(
            "relation",
            {
                "source": "replacement pair",
                "relation": "replaces",
                "target": "exchanged pair",
            },
            fact_id="replaces",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["computed_answer"] == 0.0
    assert row["candidate_units"] == []
    assert row["compiled_recursive_fact_count"] == 0
    assert row["quarantined_recursive_fact_count"] == len(facts)
    assert row["recursive_fact_consistency"]["issues"] == [
        {
            "issue": "identity_distinct_conflict",
            "fact_id": "same",
            "conflicting_fact_id": "replaces",
            "evidence_session_id": "evidence_001",
            "reason": "same pair is asserted as both same_as and distinct",
        }
    ]
    assert row["parse_errors"] == []
    assert row["compiler_warnings"] == ["quarantined_evidence_session:evidence_001:1"]


def test_compile_quarantines_entity_type_namespace_collisions() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "new pair",
                "verb": "pick up",
                "status": "pending",
                "location": "",
            },
            fact_id="action",
            span="pick up the new pair",
        ),
        _fact(
            "sortal",
            {
                "entity": "new pair",
                "type": "boots",
                "source": "bridging",
                "antecedent": "boots",
                "licensed_by": "new pair",
            },
            fact_id="sortal",
            span="new pair",
        ),
        _fact(
            "relation",
            {
                "source": "new pair",
                "relation": "replaces",
                "target": "boots",
            },
            fact_id="replace",
            span="new pair",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["computed_answer"] == 0.0
    assert row["candidate_units"] == []
    assert row["compiled_recursive_fact_count"] == 0
    assert row["quarantined_recursive_fact_count"] == len(facts)
    assert row["quarantined_evidence_sessions"][0]["evidence_session_id"] == (
        "evidence_001"
    )
    assert row["symbol_namespace"]["issues"] == [
        {
            "issue": "symbol_sort_collision",
            "reason": (
                "same bare symbol is used in both entity and type positions "
                "within one evidence namespace"
            ),
            "namespace": "evidence_001",
            "symbol": "boots",
            "uses": {
                "entity": [
                    {
                        "argument": "antecedent",
                        "fact_id": "sortal",
                        "predicate": "sortal",
                        "value": "boots",
                    },
                    {
                        "argument": "target",
                        "fact_id": "replace",
                        "predicate": "relation",
                        "value": "boots",
                    },
                ],
                "type": [
                    {
                        "argument": "type",
                        "fact_id": "sortal",
                        "predicate": "sortal",
                        "value": "boots",
                    }
                ],
            },
        }
    ]
    assert row["parse_errors"] == []
    assert row["compiler_warnings"] == ["quarantined_evidence_session:evidence_001:1"]


def test_compile_uses_clean_sessions_after_quarantining_bad_sessions() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "blue blazer",
                "verb": "pick up",
                "status": "pending",
                "location": "dry cleaner",
            },
            fact_id="good_action",
            session="evidence_good",
            span="pick up my blue blazer",
        ),
        _fact(
            "object_type",
            {"entity": "blue blazer", "type": "clothing"},
            fact_id="good_type",
            session="evidence_good",
            span="blue blazer",
        ),
        _fact(
            "sortal",
            {
                "entity": "new pair",
                "type": "boots",
                "source": "bridging",
                "antecedent": "boots",
                "licensed_by": "new pair",
            },
            fact_id="bad_sortal",
            session="evidence_bad",
            span="new pair",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["computed_answer"] == 1.0
    assert row["compiled_recursive_fact_count"] == 2
    assert row["quarantined_recursive_fact_count"] == 1
    assert row["candidate_units"][0]["label"] == "blue blazer"


def test_compile_scopes_symbol_namespace_to_each_evidence_session() -> None:
    facts = [
        _fact(
            "sortal",
            {"entity": "pair_1", "type": "boots"},
            fact_id="type",
            session="evidence_001",
        ),
        _fact(
            "relation",
            {"source": "new_pair_1", "relation": "replaces", "target": "boots"},
            fact_id="relation",
            session="evidence_002",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["parse_status"] == "parsed"
    assert row["symbol_namespace"] == {"issue_count": 0, "issues": []}


def test_compile_omits_open_null_arguments_from_fact_rendering() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "blue blazer",
                "verb": "pick up",
                "location": None,
                "status": "",
            },
            fact_id="action",
        ),
        _fact(
            "object_type",
            {"entity": "blue blazer", "type": "clothing"},
            fact_id="type",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    rendered_action = next(fact for fact in row["facts"] if fact.startswith("action("))
    assert "location=" not in rendered_action
    assert "status=" not in rendered_action
    assert "none" not in rendered_action


def test_compile_reports_inclusion_mutation_sensitivity() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "blue blazer",
                "verb": "pick up",
                "status": "pending",
                "location": "dry cleaner",
            },
            fact_id="included",
            span="pick up my blue blazer",
        ),
        _fact(
            "object_type",
            {"entity": "blue blazer", "type": "clothing"},
            fact_id="type_1",
            span="blue blazer",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "books",
                "verb": "pick up",
                "status": "pending",
                "location": "library",
            },
            fact_id="excluded",
            span="pick up library books",
        ),
        _fact(
            "object_type",
            {"entity": "books", "type": "book"},
            fact_id="type_2",
            span="books",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many items of clothing do I need to pick up or return from a store?"
        ),
        facts=facts,
    )

    assert row["computed_answer"] == 1.0
    sensitivity = row["inclusion_mutation_sensitivity"]
    assert sensitivity["single_flip_total"] == 2
    assert sensitivity["single_flip_answer_unchanged_count"] == 0
    assert sensitivity["balanced_swap_total"] == 1
    assert sensitivity["balanced_swap_answer_unchanged_count"] == 1
    assert sensitivity["aggregate_score_insensitive_to_balanced_swaps"] is True


def test_compile_baking_events_dedupes_and_excludes_generic_cooking() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "salads",
                "verb": "made",
                "status": "completed",
            },
            fact_id="f1",
            span="made some salads",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "whole wheat baguette",
                "verb": "made",
                "status": "completed",
            },
            fact_id="f2",
            span="made a delicious whole wheat baguette",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "new bread recipe using sourdough starter",
                "verb": "tried",
                "status": "completed",
            },
            fact_id="f3",
            span="tried out a new bread recipe using sourdough starter",
        ),
        _fact(
            "event",
            {
                "event": "baking sourdough bread",
                "type": "baking",
                "participants": "user",
                "status": "completed",
            },
            fact_id="f4",
            span="a new bread recipe using sourdough starter",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question="How many times did I bake something in the past two weeks?",
        facts=facts,
    )

    assert row["computed_answer"] == 2.0
    excluded = [
        unit["reason_code"]
        for unit in row["candidate_units"]
        if unit["status"] == "excluded"
    ]
    assert "not_baked_good" in excluded


def test_compile_charity_sum_dedupes_and_excludes_non_charity_benefit() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "charity bake sale",
                "verb": "helped raise",
                "status": "completed",
            },
            fact_id="f1",
            span="helped raise money at a charity bake sale",
        ),
        _fact(
            "value",
            {
                "entity": "charity bake sale",
                "attribute": "amount raised",
                "value": "1000",
                "unit": "USD",
            },
            fact_id="f2",
            span="raised $1,000 for the local children's hospital",
        ),
        _fact(
            "value",
            {
                "entity": "charity bake sale fundraising",
                "attribute": "amount raised",
                "value": "over 1000",
                "unit": "USD",
            },
            fact_id="f3",
            span="helped raise over $1,000 for the local children's hospital",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "music benefit concert",
                "verb": "helped organize",
                "status": "completed",
            },
            fact_id="f4",
            session="evidence_002",
            span="helped organize a music benefit concert",
        ),
        _fact(
            "value",
            {
                "entity": "music benefit concert",
                "attribute": "amount raised",
                "value": "5000",
                "unit": "USD",
            },
            fact_id="f5",
            session="evidence_002",
            span="raised $5,000 for the local music education program",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question="How much money did I raise for charity in total?",
        facts=facts,
    )

    assert row["computed_answer"] == 1000.0
    excluded = [
        unit["label"] for unit in row["candidate_units"] if unit["status"] == "excluded"
    ]
    assert excluded == ["music benefit concert"]


def test_action_obligation_paraphrase_uses_type_and_action_synonyms() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "linen jacket",
                "verb": "collect",
                "status": "pending",
                "location": "tailor shop",
            },
            fact_id="f1",
            span="still need to collect the linen jacket from the tailor shop",
        ),
        _fact(
            "action",
            {
                "subject": "user",
                "object": "red shoes",
                "verb": "send back",
                "status": "needed",
                "location": "shoe shop",
            },
            fact_id="f2",
            span="need to send back the red shoes to the shoe shop",
        ),
        _fact(
            "object_type",
            {"entity": "linen jacket", "type": "apparel"},
            fact_id="f3",
            span="linen jacket",
        ),
        _fact(
            "object_type",
            {"entity": "red shoes", "type": "garments"},
            fact_id="f4",
            span="red shoes",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question=(
            "How many garments do I still need to collect or send back from a shop?"
        ),
        facts=facts,
    )

    assert row["compiler_intent"]["kind"] == "count_action_obligation"
    assert row["computed_answer"] == 2.0


def test_attended_event_paraphrase_requires_user_action_edge() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "Ava and Kim ceremony",
                "verb": "went to",
                "status": "completed",
                "location": "garden",
            },
            fact_id="f1",
            span="I went to Ava and Kim's ceremony in the garden",
        ),
        _fact(
            "event",
            {
                "event": "Ava and Kim wedding",
                "type": "marriage ceremony",
                "participants": "Ava, Kim",
                "status": "completed",
            },
            fact_id="f2",
            span="Ava and Kim's ceremony",
        ),
        _fact(
            "event",
            {
                "event": "Gurkha wedding",
                "type": "wedding",
                "participants": "Gurkhas",
                "status": "completed",
            },
            fact_id="f3",
            span="Gurkha wedding customs",
            session="evidence_002",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question="How many marriage ceremonies did I go to this year?",
        facts=facts,
    )

    assert row["compiler_intent"]["kind"] == "count_attended_events"
    assert row["computed_answer"] == 1.0
    excluded = [
        unit["label"] for unit in row["candidate_units"] if unit["status"] == "excluded"
    ]
    assert excluded == ["Gurkha wedding"]


def test_charity_sum_excludes_amount_not_raised_by_user() -> None:
    facts = [
        _fact(
            "action",
            {
                "subject": "user",
                "object": "charity gala",
                "verb": "attended",
                "status": "completed",
            },
            fact_id="f1",
            span="I attended the charity gala",
        ),
        _fact(
            "value",
            {
                "entity": "charity gala",
                "attribute": "amount raised",
                "value": "500",
                "unit": "USD",
            },
            fact_id="f2",
            span="the charity gala raised $500",
        ),
    ]

    row = compiler.compile_case(
        case_id="case",
        question="How much money did I raise for charity in total?",
        facts=facts,
    )

    assert row["computed_answer"] == 0.0
    assert (
        row["candidate_units"][0]["reason_code"] == "not_question_charity_fundraising"
    )

"""Unit tests for the deterministic typed-fact envelope engine.

The provider boundary is not crossed here: every function under test is pure.
Fixtures mirror the two validated cases -- gpt4_d84a3211 (money sum, $185
collapse) and 5a7937c8 (day count).
"""

from __future__ import annotations

import typing

from simba.eval import candidate_unit_envelope as env


def fact(
    predicate: str,
    arguments: dict[str, typing.Any],
    *,
    span: str = "span",
    session: str = "s1",
    span_ok: bool = True,
) -> dict[str, typing.Any]:
    return {
        "predicate": predicate,
        "arguments": arguments,
        "evidence_span": span,
        "_session": session,
        "_span_ok": span_ok,
    }


def gpt4_facts() -> list[dict[str, typing.Any]]:
    """helmet $120 (date in another session), chain $25 + lights $40 (dates via
    the tune-up event), rack planned (no value). Mirrors gpt4_d84a3211."""
    return [
        fact("object_type", {"entity": "helmet_1", "type": "helmet"}, session="s1"),
        fact(
            "value",
            {"entity": "helmet_1", "attribute": "price", "value": "120", "unit": "USD"},
            session="s1",
        ),
        fact(
            "relation",
            {
                "source": "helmet_1",
                "relation": "purchased_at",
                "target": "bike_shop_downtown",
            },
            session="s1",
        ),
        fact("time", {"entity": "helmet_1", "date": "April 10th"}, session="s2"),
        fact("object_type", {"entity": "chain_1", "type": "chain"}, session="s2"),
        fact(
            "value",
            {"entity": "chain_1", "attribute": "cost", "value": "25", "unit": "USD"},
            session="s2",
        ),
        fact(
            "object_type", {"entity": "lights_1", "type": "bike_lights"}, session="s2"
        ),
        fact(
            "value",
            {"entity": "lights_1", "attribute": "cost", "value": "40", "unit": "USD"},
            session="s2",
        ),
        # same lights, different session, price variant -> must NOT double-count
        fact(
            "value",
            {"entity": "lights_1", "attribute": "price", "value": "40", "unit": "USD"},
            session="s3",
        ),
        fact(
            "event",
            {
                "event": "tune_up_apr20",
                "type": "tune-up",
                "date": "April 20th",
                "status": "completed",
            },
            session="s2",
        ),
        fact(
            "relation",
            {"source": "tune_up_apr20", "relation": "included", "target": "lights_1"},
            session="s2",
        ),
        fact(
            "relation",
            {"source": "tune_up_apr20", "relation": "included", "target": "chain_1"},
            session="s2",
        ),
        fact("object_type", {"entity": "rack_1", "type": "bike_rack"}, session="s1"),
        fact("status", {"entity": "rack_1", "status": "pending"}, session="s1"),
        fact(
            "action",
            {
                "subject": "user",
                "object": "rack_1",
                "verb": "order",
                "status": "planned",
            },
            session="s1",
        ),
    ]


def faith_facts() -> list[dict[str, typing.Any]]:
    """food drive (Dec 10), midnight mass (Dec 24, with a "24th" variant), bible
    study (Dec 17), upcoming study (planned). Mirrors 5a7937c8."""
    return [
        fact(
            "event",
            {
                "event": "food_drive_1",
                "type": "food_drive",
                "date": "December 10th",
                "status": "completed",
            },
            session="s1",
        ),
        fact(
            "event",
            {
                "event": "mass_1",
                "type": "midnight_mass",
                "date": "December 24",
                "status": "completed",
            },
            session="s2",
        ),
        fact("time", {"entity": "mass_1", "date": "December 24th"}, session="s2"),
        fact(
            "event",
            {
                "event": "study_dec17",
                "type": "bible_study",
                "date": "December 17th",
                "status": "completed",
            },
            session="s2",
        ),
        fact(
            "event",
            {
                "event": "study_upcoming",
                "type": "bible_study",
                "date": "next week",
                "status": "planned",
            },
            session="s3",
        ),
    ]


# --- norm_day
def test_norm_day_collapses_ordinals() -> None:
    assert env.norm_day("December 24th") == env.norm_day("December 24") == "december 24"
    assert env.norm_day("  April  10th ") == "april 10"


# --- classify_relation
def test_classify_relation_separates_values_and_times_from_real_edges() -> None:
    assert env.classify_relation("used_for", "daily commutes") is None
    assert env.classify_relation("bought_for", "$120") == "value"
    assert env.classify_relation("cost", "25") == "value"  # verb-based
    assert env.classify_relation("purchased_on", "April 10th") == "time"  # verb-based
    assert (
        env.classify_relation("happened", "December 10th") == "time"
    )  # date in target


def test_classify_value_role_from_explicit_attributes() -> None:
    assert (
        env.classify_value_role(
            "people_reached", "12000", "people", question="total people reached"
        )
        == env.VALUE_ROLE_ANSWER
    )
    assert (
        env.classify_value_role(
            "points_required", "100", "points", question="points needed to redeem"
        )
        == env.VALUE_ROLE_THRESHOLD
    )
    assert (
        env.classify_value_role(
            "redemption_cost", "100", "points", question="points needed to redeem"
        )
        == env.VALUE_ROLE_THRESHOLD
    )
    assert (
        env.classify_value_role(
            "points_balance", "300", "points", question="points needed to redeem"
        )
        == env.VALUE_ROLE_CURRENT_BALANCE
    )
    assert env.classify_value_role("subtotal", "50", "items") == env.VALUE_ROLE_SUBTOTAL
    assert (
        env.classify_value_role("previous_total", "50", "items")
        == env.VALUE_ROLE_HISTORICAL
    )
    assert (
        env.classify_value_role("clicks", "50", "clicks")
        == env.VALUE_ROLE_DISTRACTOR
    )


# --- is_dup_edge
def test_is_dup_edge_token_subset_matches_across_formatting() -> None:
    assert (
        env.is_dup_edge("the local bike shop downtown", ["bike_shop_downtown"]) is True
    )
    assert env.is_dup_edge("daily commutes", ["bike_shop_downtown"]) is False
    assert env.is_dup_edge("anything", []) is False


# --- vote_tag
def test_vote_tag_unanimous() -> None:
    assert env.vote_tag(["certain_in", "certain_in", "certain_in"]) == "certain_in"
    assert env.vote_tag(["certain_out", "certain_out", "certain_out"]) == "certain_out"


def test_vote_tag_include_exclude_split_is_contested() -> None:
    assert env.vote_tag(["certain_in", "certain_out", "certain_in"]) == "contested"


def test_vote_tag_lone_dissent_does_not_flip_stable_judgment() -> None:
    assert env.vote_tag(["certain_in", "certain_in", "contested"]) == "certain_in"


def test_vote_tag_contested_plurality_is_contested() -> None:
    assert env.vote_tag(["certain_in", "contested", "contested"]) == "contested"


def test_vote_tag_empty_is_none() -> None:
    assert env.vote_tag([]) is None
    assert env.vote_tag([None, ""]) is None  # type: ignore[list-item]


# --- resolve_entities
def test_resolve_clusters_and_inherits_event_dates() -> None:
    bundles = env.resolve_entities(gpt4_facts())
    helmet = bundles["helmet_1"]
    assert helmet.usd == 120.0
    assert (
        "April 10th" in helmet.dates
    )  # date came from a different session than the price
    assert ("purchased_at", "bike_shop_downtown") in helmet.relations
    lights = bundles["lights_1"]
    assert lights.usd == 40.0  # one value despite appearing in two sessions
    assert (
        "April 20th" in lights.dates
    )  # inherited from the tune-up event via `included`
    chain = bundles["chain_1"]
    assert chain.usd == 25.0
    assert "April 20th" in chain.dates
    rack = bundles["rack_1"]
    assert rack.usd is None
    assert "pending" in rack.statuses


def test_resolve_respects_distinct_and_coreference() -> None:
    facts = [
        *gpt4_facts(),
        fact("object_type", {"entity": "chain_old", "type": "chain"}, session="s2"),
        fact("distinct", {"a": "chain_1", "b": "chain_old"}, session="s2"),
        fact(
            "coreference", {"entity": "lights_2", "same_as": "lights_1"}, session="s2"
        ),
        fact(
            "value",
            {"entity": "lights_2", "attribute": "cost", "value": "40", "unit": "USD"},
            session="s2",
        ),
    ]
    # chain_1 and chain_old must NOT merge (distinct)
    chain_root = _root_of("chain_1", facts)
    old_root = _root_of("chain_old", facts)
    assert chain_root != old_root
    # lights_2 same_as lights_1 -> one cluster
    assert _root_of("lights_1", facts) == _root_of("lights_2", facts)


def _root_of(handle: str, facts: list[dict[str, typing.Any]]) -> str:
    bundles = env.resolve_entities(facts)
    for root, bundle in bundles.items():
        if handle in bundle.handles:
            return root
    raise AssertionError(f"{handle} not found in any bundle")


# --- select_candidates / intent
def test_detect_intent() -> None:
    assert (
        env.detect_intent("How many days did I spend on faith activities?")
        == env.AGGREGATION_DAYS
    )
    assert (
        env.detect_intent("How much money did I spend on bikes?") == env.AGGREGATION_SUM
    )


def test_select_candidates_money_includes_priced_and_purchasable() -> None:
    bundles = env.resolve_entities(gpt4_facts())
    candidates = env.select_candidates(bundles, env.AGGREGATION_SUM)
    assert set(candidates) == {"helmet_1", "chain_1", "lights_1", "rack_1"}


def test_select_candidates_days_includes_dated_events() -> None:
    bundles = env.resolve_entities(faith_facts())
    candidates = env.select_candidates(bundles, env.AGGREGATION_DAYS)
    assert set(candidates) == {
        "food_drive_1",
        "mass_1",
        "study_dec17",
        "study_upcoming",
    }


# --- aggregate_envelope
def test_envelope_money_collapses_to_185() -> None:
    bundles = env.resolve_entities(gpt4_facts())
    candidates = env.select_candidates(bundles, env.AGGREGATION_SUM)
    judged = {
        "helmet_1": "certain_in",
        "chain_1": "certain_in",
        "lights_1": "certain_in",
        "rack_1": "certain_out",
    }
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_SUM, candidates)
    assert result.certain == 185.0
    assert result.possible == 185.0
    assert result.collapsed is True
    assert result.pivot == ()
    assert result.consistent is True


def test_envelope_money_opens_when_an_item_is_contested() -> None:
    bundles = env.resolve_entities(gpt4_facts())
    candidates = env.select_candidates(bundles, env.AGGREGATION_SUM)
    judged = {
        "helmet_1": "certain_in",
        "chain_1": "certain_in",
        "lights_1": "contested",
        "rack_1": "certain_out",
    }
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_SUM, candidates)
    assert [result.certain, result.possible] == [145.0, 185.0]
    assert result.collapsed is False
    assert result.pivot == ("lights_1",)
    assert result.consistent is True


def test_envelope_days_opens_to_2_3_with_food_drive_pivot() -> None:
    bundles = env.resolve_entities(faith_facts())
    candidates = env.select_candidates(bundles, env.AGGREGATION_DAYS)
    judged = {
        "food_drive_1": "contested",
        "mass_1": "certain_in",
        "study_dec17": "certain_in",
        "study_upcoming": "certain_out",
    }
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_DAYS, candidates)
    # mass's "December 24"/"December 24th" are ONE day; upcoming excluded (planned)
    assert [result.certain, result.possible] == [2.0, 3.0]
    assert result.pivot == ("food_drive_1",)
    assert result.consistent is True


def test_envelope_days_collapses_to_3_when_food_drive_certain_in() -> None:
    bundles = env.resolve_entities(faith_facts())
    candidates = env.select_candidates(bundles, env.AGGREGATION_DAYS)
    judged = {
        "food_drive_1": "certain_in",
        "mass_1": "certain_in",
        "study_dec17": "certain_in",
        "study_upcoming": "certain_out",
    }
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_DAYS, candidates)
    assert [result.certain, result.possible] == [3.0, 3.0]
    assert result.collapsed is True


def test_envelope_days_uses_typed_time_duration_quantity() -> None:
    facts = [
        fact("object_type", {"entity": "trip_1", "type": "trip"}),
        fact(
            "quantity",
            {
                "entity": "trip_1",
                "dimension": "time.duration",
                "value": "36",
                "unit": "hours",
            },
        ),
        fact("object_type", {"entity": "trip_2", "type": "trip"}),
        fact(
            "quantity",
            {
                "entity": "trip_2",
                "dimension": "time.duration",
                "value": "2",
                "unit": "weeks",
            },
        ),
    ]
    bundles = env.resolve_entities(facts)
    candidates = env.select_candidates(bundles, env.AGGREGATION_DAYS)
    assert set(candidates) == {"trip_1", "trip_2"}
    judged = {"trip_1": "certain_in", "trip_2": "contested"}
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_DAYS, candidates)
    assert [result.certain, result.possible] == [1.5, 15.5]
    assert result.pivot == ("trip_2",)


def test_envelope_days_does_not_infer_duration_from_legacy_attribute_name() -> None:
    facts = [
        fact("object_type", {"entity": "trip_1", "type": "trip"}),
        fact(
            "value",
            {
                "entity": "trip_1",
                "attribute": "duration",
                "value": "15",
                "unit": "days",
            },
        ),
    ]
    bundles = env.resolve_entities(facts)
    candidates = env.select_candidates(bundles, env.AGGREGATION_DAYS)
    assert candidates == []


# --- fold_recovered_relations
def test_fold_recovered_relations_keeps_only_verified_new_edges() -> None:
    bundle = env.EntityBundle(
        root="helmet_1",
        handles=("helmet_1",),
        types=("helmet",),
        usd=120.0,
        values=(("price", "120", "USD"),),
        quantities=(),
        dates=(),
        statuses=(),
        relations=(("purchased_at", "bike_shop_downtown"),),
        actions=(),
        events=(),
        sessions=("s1",),
    )
    sentences = [
        "I use the Bell Zephyr helmet for my daily commutes.",
        "I got it at the local bike shop downtown for $120.",
    ]
    recovered = [
        {
            "relation": "used_for",
            "target": "my daily commutes",
            "evidence_span": "for my daily commutes",
        },
        {
            "relation": "bought_at",
            "target": "the local bike shop downtown",
            "evidence_span": "at the local bike shop downtown",
        },
        {"relation": "bought_for", "target": "$120", "evidence_span": "for $120"},
        {
            "relation": "used_for",
            "target": "hallucinated",
            "evidence_span": "not in any sentence",
        },
    ]
    updated, added = env.fold_recovered_relations(bundle, recovered, sentences)
    assert added == [
        ("used_for", "my daily commutes")
    ]  # dup (bike shop), value ($120), and unverified dropped
    assert ("used_for", "my daily commutes") in updated.relations
    assert ("purchased_at", "bike_shop_downtown") in updated.relations


# --- guards
def test_span_resolution_counts_resolved_spans() -> None:
    facts = [
        fact("value", {}, span_ok=True),
        fact("value", {}, span_ok=False),
        fact("value", {}, span_ok=True),
    ]
    assert env.span_resolution(facts) == (2, 3)


def test_zero_fact_sessions_flags_unrepresented_answer_sessions() -> None:
    facts = [fact("value", {}, session="s1"), fact("value", {}, session="s2")]
    assert env.zero_fact_sessions(facts, ["s1", "s2", "s3"]) == ["s3"]


# --- new aggregation shapes: instances + entity_select
def test_detect_intent_four_shapes() -> None:
    assert env.detect_intent("How much money did I raise?") == env.AGGREGATION_SUM
    assert (
        env.detect_intent("How many days of faith activities?") == env.AGGREGATION_DAYS
    )
    assert env.detect_intent("How many tanks do I have?") == env.AGGREGATION_INSTANCES
    assert (
        env.detect_intent("How many fitness classes do I attend?")
        == env.AGGREGATION_INSTANCES
    )
    assert (
        env.detect_intent("Which grocery store did I use most?")
        == env.AGGREGATION_ENTITY
    )
    assert env.detect_intent("What did I buy for the gift?") == env.AGGREGATION_ENTITY
    assert env.detect_intent("Where do I take yoga?") == env.AGGREGATION_ENTITY
    assert (
        env.detect_intent("What date did I volunteer?")
        == env.AGGREGATION_DATE
    )
    assert (
        env.detect_intent("When did I volunteer at the shelter?")
        == env.AGGREGATION_DATE
    )
    assert (
        env.detect_intent("How many hours of jogging and yoga did I do?")
        == env.AGGREGATION_DURATION
    )
    assert (
        env.detect_intent("What total number of people did the campaign reach?")
        == env.AGGREGATION_SUM_VALUE
    )
    assert (
        env.detect_intent("How many people reached did my posts get?")
        == env.AGGREGATION_SUM_VALUE
    )
    assert (
        env.detect_intent("How many points do I need to earn to redeem a product?")
        == env.AGGREGATION_LOOKUP
    )
    assert (
        env.detect_intent("How many rare items do I have in total?")
        == env.AGGREGATION_STATED_TOTAL
    )


def instance_facts() -> list[dict[str, typing.Any]]:
    """Three tanks (two set up, one being set up) + the speaker. Mirrors 46a3abf7."""
    return [
        fact("object_type", {"entity": "user", "type": "person"}, session="s1"),
        fact("object_type", {"entity": "tank_1", "type": "fish tank"}, session="s1"),
        fact("object_type", {"entity": "tank_2", "type": "fish tank"}, session="s2"),
        fact("object_type", {"entity": "tank_3", "type": "fish tank"}, session="s3"),
        fact("object_type", {"entity": "mileage_goal", "type": "goal"}, session="s1"),
    ]


def test_select_candidates_instances_excludes_user_and_abstract() -> None:
    bundles = env.resolve_entities(instance_facts())
    cands = env.select_candidates(bundles, env.AGGREGATION_INSTANCES)
    assert set(cands) == {"tank_1", "tank_2", "tank_3"}  # user + goal excluded


def test_envelope_instances_counts_distinct_certain_and_possible() -> None:
    bundles = env.resolve_entities(instance_facts())
    cands = env.select_candidates(bundles, env.AGGREGATION_INSTANCES)
    # two tanks set up (certain), the third still being set up (contested)
    judged = {"tank_1": "certain_in", "tank_2": "certain_in", "tank_3": "contested"}
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_INSTANCES, cands)
    assert [result.certain, result.possible] == [2.0, 3.0]
    assert result.pivot == ("tank_3",)
    assert result.consistent is True


def test_pickup_return_counts_obligations_not_wardrobe_identity() -> None:
    facts = [
        fact("object_type", {"entity": "boots", "type": "boots"}),
        fact(
            "relation",
            {"source": "boots", "relation": "exchanged_for", "target": "larger size"},
        ),
        fact("object_type", {"entity": "blazer", "type": "blazer"}),
        fact(
            "relation",
            {"source": "blazer", "relation": "dry_cleaning_for", "target": "pickup"},
        ),
    ]
    question = "How many items of clothing do I need to pick up or return from a store?"
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_INSTANCES, question=question)
    judged = {"boots": "certain_in", "blazer": "contested"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_INSTANCES, cands, question=question
    )
    assert [result.certain, result.possible] == [2.0, 3.0]
    assert result.pivot == ("blazer",)


def test_exchange_does_not_double_count_ordinary_instance_questions() -> None:
    facts = [
        fact("object_type", {"entity": "boots", "type": "boots"}),
        fact(
            "relation",
            {"source": "boots", "relation": "exchanged_for", "target": "larger size"},
        ),
    ]
    question = "How many pairs of boots have I mentioned?"
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_INSTANCES, question=question)
    judged = {"boots": "certain_in"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_INSTANCES, cands, question=question
    )
    assert [result.certain, result.possible] == [1.0, 1.0]


def test_envelope_entity_select_returns_candidate_sets() -> None:
    facts = [
        fact(
            "object_type", {"entity": "thrive", "type": "grocery store"}, session="s1"
        ),
        fact(
            "value",
            {"entity": "thrive", "attribute": "name", "value": "Thrive Market"},
            session="s1",
        ),
        fact(
            "object_type", {"entity": "walmart", "type": "grocery store"}, session="s2"
        ),
        fact(
            "value",
            {"entity": "walmart", "attribute": "name", "value": "Walmart"},
            session="s2",
        ),
    ]
    bundles = env.resolve_entities(facts)
    assert bundles["thrive"].label == "Thrive Market"  # name beats the bare type
    cands = env.select_candidates(bundles, env.AGGREGATION_ENTITY)
    assert set(cands) == {"thrive", "walmart"}
    judged = {"thrive": "certain_in", "walmart": "contested"}
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_ENTITY, cands)
    assert result.certain_labels == ("Thrive Market",)
    assert result.possible_labels == ("Thrive Market", "Walmart")
    assert result.collapsed is False  # one certain, one contested alternative -> open
    assert result.pivot == ("walmart",)


def test_envelope_sum_value_sums_people_reached() -> None:
    facts = [
        fact("object_type", {"entity": "campaign_1", "type": "campaign"}, session="s1"),
        fact(
            "value",
            {
                "entity": "campaign_1",
                "attribute": "people_reached",
                "value": "12000",
                "unit": "people",
            },
            session="s1",
        ),
        fact("object_type", {"entity": "campaign_2", "type": "campaign"}, session="s2"),
        fact(
            "value",
            {
                "entity": "campaign_2",
                "attribute": "people_reached",
                "value": "500",
                "unit": "people",
            },
            session="s2",
        ),
        fact(
            "value",
            {
                "entity": "campaign_2",
                "attribute": "clicks",
                "value": "50",
                "unit": "clicks",
            },
            session="s2",
        ),
        fact(
            "value",
            {
                "entity": "campaign_2",
                "attribute": "campaign_cost",
                "value": "25",
                "unit": "USD",
            },
            session="s2",
        ),
    ]
    bundles = env.resolve_entities(facts)
    question = "What total number of people reached did the campaigns get?"
    cands = env.select_candidates(bundles, env.AGGREGATION_SUM_VALUE, question=question)
    assert set(cands) == {"campaign_1", "campaign_2"}
    judged = {"campaign_1": "certain_in", "campaign_2": "contested"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_SUM_VALUE, cands, question=question
    )
    assert [result.certain, result.possible] == [12000.0, 12500.0]
    assert result.pivot == ("campaign_2",)


def test_60036106_sum_value_ignores_unrelated_numeric_values() -> None:
    facts = [
        fact("object_type", {"entity": "facebook_ad", "type": "ad campaign"}),
        fact(
            "value",
            {
                "entity": "facebook_ad",
                "attribute": "people_reached",
                "value": "2000",
                "unit": "people",
            },
        ),
        fact(
            "value",
            {
                "entity": "facebook_ad",
                "attribute": "clicks",
                "value": "50",
                "unit": "clicks",
            },
        ),
        fact(
            "value",
            {
                "entity": "facebook_ad",
                "attribute": "campaign_cost",
                "value": "102.50",
                "unit": "USD",
            },
        ),
        fact("object_type", {"entity": "influencer", "type": "influencer"}),
        fact(
            "value",
            {
                "entity": "influencer",
                "attribute": "followers",
                "value": "10,000",
                "unit": "people",
            },
        ),
    ]
    question = (
        "What total number of people reached did my Facebook ad campaign and "
        "Instagram influencer collaboration get?"
    )
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_SUM_VALUE, question=question)
    assert set(cands) == {"facebook_ad", "influencer"}
    judged = {"facebook_ad": "certain_in", "influencer": "certain_in"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_SUM_VALUE, cands, question=question
    )
    assert [result.certain, result.possible] == [12000.0, 12000.0]


def test_sum_value_collapses_duplicate_same_context_answer_values() -> None:
    facts = [
        fact("object_type", {"entity": "campaign_1", "type": "facebook campaign"}),
        fact(
            "value",
            {
                "entity": "campaign_1",
                "attribute": "people_reached",
                "value": "2000",
                "unit": "people",
            },
        ),
        fact("object_type", {"entity": "campaign_previous", "type": "ad campaign"}),
        fact(
            "value",
            {
                "entity": "campaign_previous",
                "attribute": "people_reached",
                "value": "2000",
                "unit": "people",
            },
        ),
        fact("object_type", {"entity": "influencer", "type": "influencer"}),
        fact(
            "value",
            {
                "entity": "influencer",
                "attribute": "followers",
                "value": "10000",
                "unit": "people",
            },
        ),
    ]
    question = "What was the total number of people reached?"
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_SUM_VALUE, question=question)
    judged = {candidate: "certain_in" for candidate in cands}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_SUM_VALUE, cands, question=question
    )
    assert [result.certain, result.possible] == [12000.0, 12000.0]


def test_envelope_duration_sum_normalizes_minutes_to_hours() -> None:
    facts = [
        fact("object_type", {"entity": "jog_1", "type": "jogging"}, session="s1"),
        fact(
            "value",
            {
                "entity": "jog_1",
                "attribute": "duration",
                "value": "30",
                "unit": "minutes",
            },
            session="s1",
        ),
        fact("object_type", {"entity": "yoga_1", "type": "yoga"}, session="s2"),
        fact(
            "value",
            {
                "entity": "yoga_1",
                "attribute": "duration",
                "value": "0.5",
                "unit": "hours",
            },
            session="s2",
        ),
    ]
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_DURATION)
    judged = {"jog_1": "certain_in", "yoga_1": "contested"}
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_DURATION, cands)
    assert [result.certain, result.possible] == [0.5, 1.0]
    assert result.pivot == ("yoga_1",)


def test_envelope_lookup_scalar_uses_candidate_value_range() -> None:
    facts = [
        fact(
            "object_type",
            {"entity": "sephora_reward", "type": "reward_threshold"},
            session="s1",
        ),
        fact(
            "value",
            {
                "entity": "sephora_reward",
                "attribute": "points_required",
                "value": "100",
                "unit": "points",
            },
            session="s1",
        ),
        fact(
            "object_type",
            {"entity": "current_points", "type": "balance"},
            session="s1",
        ),
        fact(
            "value",
            {
                "entity": "current_points",
                "attribute": "points_balance",
                "value": "40",
                "unit": "points",
            },
            session="s1",
        ),
    ]
    bundles = env.resolve_entities(facts)
    question = "How many points do I need to earn to redeem a free product?"
    cands = env.select_candidates(bundles, env.AGGREGATION_LOOKUP, question=question)
    judged = {"sephora_reward": "certain_in", "current_points": "certain_in"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_LOOKUP, cands, question=question
    )
    assert [result.certain, result.possible] == [100.0, 100.0]
    assert result.collapsed is True


def test_9ee3ecd6_lookup_prefers_threshold_over_current_balance() -> None:
    facts = [
        fact("object_type", {"entity": "skincare_reward", "type": "reward"}),
        fact(
            "value",
            {
                "entity": "skincare_reward",
                "attribute": "points_required",
                "value": "100",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "beauty_insider_account", "type": "account"}),
        fact(
            "value",
            {
                "entity": "beauty_insider_account",
                "attribute": "current_points",
                "value": "300",
                "unit": "points",
            },
        ),
    ]
    question = (
        "How many points do I need to earn to redeem a free skincare product "
        "at Sephora?"
    )
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_LOOKUP, question=question)
    assert set(cands) == {"beauty_insider_account", "skincare_reward"}
    judged = {"skincare_reward": "certain_in", "beauty_insider_account": "certain_in"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_LOOKUP, cands, question=question
    )
    assert [result.certain, result.possible] == [100.0, 100.0]


def test_9ee3ecd6_lookup_collapses_to_supported_product_threshold() -> None:
    facts = [
        fact("object_type", {"entity": "free_skincare_reward", "type": "reward"}),
        fact(
            "value",
            {
                "entity": "free_skincare_reward",
                "attribute": "points_required",
                "value": "200",
                "unit": "points",
            },
        ),
        fact(
            "value",
            {
                "entity": "free_skincare_reward",
                "attribute": "points_required",
                "value": "300",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "user", "type": "person"}),
        fact(
            "value",
            {
                "entity": "user",
                "attribute": "total_points",
                "value": "200",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "cleanser", "type": "skincare product"}),
        fact(
            "value",
            {
                "entity": "cleanser",
                "attribute": "points_cost",
                "value": "100",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "serum", "type": "skincare product"}),
        fact(
            "value",
            {
                "entity": "serum",
                "attribute": "points_cost",
                "value": "100",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "toner", "type": "skincare product"}),
        fact(
            "value",
            {
                "entity": "toner",
                "attribute": "points_cost",
                "value": "100",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "face_wash", "type": "skincare product"}),
        fact(
            "value",
            {
                "entity": "face_wash",
                "attribute": "points_cost",
                "value": "50",
                "unit": "points",
            },
        ),
    ]
    question = (
        "How many points do I need to earn to redeem a free skincare product "
        "at Sephora?"
    )
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_LOOKUP, question=question)
    judged = {
        "free_skincare_reward": "certain_in",
        "user": "certain_in",
        "cleanser": "contested",
        "serum": "contested",
        "toner": "contested",
        "face_wash": "contested",
    }
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_LOOKUP, cands, question=question
    )
    assert [result.certain, result.possible] == [100.0, 100.0]
    assert result.pivot == ()
    assert result.consistent is True


def test_lookup_scalar_opens_when_lower_threshold_is_contested() -> None:
    facts = [
        fact("object_type", {"entity": "general_reward", "type": "reward"}),
        fact(
            "value",
            {
                "entity": "general_reward",
                "attribute": "points_required",
                "value": "300",
                "unit": "points",
            },
        ),
        fact("object_type", {"entity": "skincare_reward", "type": "skincare product"}),
        fact(
            "value",
            {
                "entity": "skincare_reward",
                "attribute": "redemption_cost",
                "value": "100",
                "unit": "points",
            },
        ),
    ]
    question = "How many points do I need to earn to redeem a free skincare product?"
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_LOOKUP, question=question)
    judged = {"general_reward": "certain_in", "skincare_reward": "contested"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_LOOKUP, cands, question=question
    )
    assert [result.certain, result.possible] == [100.0, 300.0]
    assert result.pivot == ("skincare_reward",)


def test_envelope_date_answer_returns_date_sets() -> None:
    facts = [
        fact("event", {"event": "dinner_1", "type": "fundraiser"}, session="s1"),
        fact("time", {"entity": "dinner_1", "date": "February 14th"}, session="s1"),
        fact("event", {"event": "other_1", "type": "fundraiser"}, session="s2"),
        fact("time", {"entity": "other_1", "date": "February"}, session="s2"),
    ]
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_DATE)
    judged = {"dinner_1": "certain_in", "other_1": "contested"}
    result = env.aggregate_envelope(bundles, judged, env.AGGREGATION_DATE, cands)
    assert result.certain_labels == ("february 14",)
    assert result.possible_labels == ("february", "february 14")
    assert [result.certain, result.possible] == [1.0, 2.0]


def test_envelope_stated_total_uses_stated_numeric_value() -> None:
    facts = [
        fact(
            "object_type",
            {"entity": "rare_collection_total", "type": "collection_total"},
            session="s1",
        ),
        fact(
            "value",
            {
                "entity": "rare_collection_total",
                "attribute": "item_count",
                "value": "99",
                "unit": "items",
            },
            session="s1",
        ),
    ]
    bundles = env.resolve_entities(facts)
    cands = env.select_candidates(bundles, env.AGGREGATION_STATED_TOTAL)
    judged = {"rare_collection_total": "certain_in"}
    result = env.aggregate_envelope(
        bundles, judged, env.AGGREGATION_STATED_TOTAL, cands
    )
    assert [result.certain, result.possible] == [99.0, 99.0]
    assert result.collapsed is True


# --- cross-session resolution (#25)
def test_parse_resolution_response_expands_clusters_to_pairs() -> None:
    raw = '{"clusters": [["lights_1", "bike_lights_2"], ["a", "b", "c"]]}'
    pairs = env.parse_resolution_response(
        raw, valid_handles={"lights_1", "bike_lights_2", "a", "b", "c"}
    )
    assert ("lights_1", "bike_lights_2") in pairs
    assert ("a", "b") in pairs and ("a", "c") in pairs  # cluster -> pairwise unions


def test_parse_resolution_response_drops_unknown_handles() -> None:
    raw = '{"clusters": [["lights_1", "ghost"]]}'
    pairs = env.parse_resolution_response(raw, valid_handles={"lights_1"})
    assert pairs == []  # singleton after dropping the unknown handle


def test_extra_unions_merge_same_item_across_sessions() -> None:
    # lights extracted under two different handles across sessions, no coreference fact
    facts = [
        fact(
            "object_type", {"entity": "lights_1", "type": "bike lights"}, session="s2"
        ),
        fact(
            "value",
            {"entity": "lights_1", "attribute": "cost", "value": "40", "unit": "USD"},
            session="s2",
        ),
        fact(
            "object_type", {"entity": "lights_2", "type": "bike lights"}, session="s3"
        ),
        fact(
            "value",
            {"entity": "lights_2", "attribute": "cost", "value": "40", "unit": "USD"},
            session="s3",
        ),
    ]
    # without resolution: two roots, two $40 -> double-count
    assert len(env.resolve_entities(facts)) == 2
    # with the cross-session merge: one root, one $40
    merged = env.resolve_entities(facts, extra_unions=[("lights_1", "lights_2")])
    assert len(merged) == 1
    assert next(iter(merged.values())).usd == 40.0


def test_extra_unions_never_merge_a_distinct_pair() -> None:
    facts = [
        fact("object_type", {"entity": "tank_1", "type": "tank"}, session="s1"),
        fact("object_type", {"entity": "tank_2", "type": "tank"}, session="s2"),
        fact("distinct", {"a": "tank_1", "b": "tank_2"}, session="s2"),
    ]
    merged = env.resolve_entities(facts, extra_unions=[("tank_1", "tank_2")])
    assert len(merged) == 2  # distinct beats a proposed merge


# --- membership parsing
def test_parse_membership_response_parses_valid_object() -> None:
    raw = (
        '{"aggregation": "sum_amount", "entities": ['
        '{"handle": "helmet_1", "membership": "certain_in", "reason": "bike_shop"},'
        '{"handle": "rack_1", "membership": "certain_out", "reason": "planned"}]}'
    )
    result = env.parse_membership_response(raw, expected_case_id="gpt4_d84a3211")
    assert result.parse_status == "parsed"
    assert result.aggregation == "sum_amount"
    assert ("helmet_1", "certain_in", "bike_shop") in result.judgments


def test_parse_membership_response_strips_fences_and_prose() -> None:
    raw = (
        "here you go:\n```json\n"
        '{"aggregation":"count_distinct_days","entities":'
        '[{"handle":"mass_1","membership":"certain_in","reason":"r"}]}\n```'
    )
    result = env.parse_membership_response(raw)
    assert result.parse_status == "parsed"
    assert result.judgments[0][0] == "mass_1"


def test_parse_membership_response_rejects_unknown_tag() -> None:
    raw = (
        '{"aggregation":"sum_amount",'
        '"entities":[{"handle":"x","membership":"maybe","reason":"r"}]}'
    )
    result = env.parse_membership_response(raw)
    assert result.parse_status == "invalid_schema"


def test_parse_membership_response_empty_and_garbage() -> None:
    assert env.parse_membership_response("").parse_status == "empty"
    assert env.parse_membership_response("no json here").parse_status == "invalid_json"

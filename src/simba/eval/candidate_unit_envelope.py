"""Typed-fact certain/possible envelope for ambiguity disambiguation.

The thesis (docs/plans/31): gold collapses genuine question-ambiguity to one
answer, so the deliverable articulates ambiguity ("X under reading A, Y under
reading B, here's the pivot") rather than guessing the collapsed point answer.

The pipeline reuses the typed extractor and never recomputes the answer
deterministically (the recursive compiler's answer-path scored 1/18):

    EXTRACT     per-session typed facts      (candidate_unit_formalizer)
      -> COMPLETE   recover dropped relations    (entity-centric re-read)
      -> RESOLVE    cross-session clustering     (handle + coreference, minus distinct)
      -> MEMBERSHIP k independent judgments     (grounded on the entity's facts)
      -> ENVELOPE   deterministic aggregate      (certain / possible / pivot)

    certain   = aggregate(certain_in)
    possible  = aggregate(certain_in U contested)
    ambiguous = certain != possible    # collapse when equal; open (a pivot) when not

The split is load-bearing: the LLM *proposes* typed facts and *judges*
membership; this module *verifies* (spans resolve, dates, arithmetic) and
*aggregates*. Contestability = instability across independent judges (split
votes => contested), which removes single-pass false-contestability noise. What
votes cannot see -- a competing-category framing the typed facts do not carry (a
food-drive is both a church activity and a charity) -- is the ratifier's job
(layer 2), not this module's.

Everything above the provider boundary (``resolve_entities``,
``aggregate_envelope``, ``vote_tag``, the classifiers and guards) is pure and
unit-tested. The provider orchestration reuses
``interpretation_runner.run_provider`` and caches raw responses by sha1.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import re
import typing

import simba.config
from simba.eval import bench_config, candidate_unit_formalizer, interpretation_runner

ENVELOPE_PROMPT_VERSION = "candidate_unit_envelope_v1"

# A value/date emitted as a "relation" is a node property, not a new edge -- it
# must not inflate recovered relations.
_VALUE_VERBS = frozenset(["cost", "costs", "bought_for", "priced_at", "price", "paid"])
_TIME_VERBS = frozenset(
    [
        "acquired_on",
        "bought_on",
        "purchased_on",
        "lubricated_on",
        "dated_on",
        "on",
        "scheduled_to_lubricate_on",
    ]
)
# An item included in / installed during a dated event inherits that event's date.
_CONTAINMENT_RELATIONS = frozenset(
    ["included", "includes", "part_of", "contains", "installed_during", "done_during"]
)
# Entity-sorted argument roles across the formalizer's predicates.
_ENTITY_ROLES = (
    "entity",
    "source",
    "subject",
    "object",
    "target",
    "a",
    "b",
    "same_as",
    "event",
)
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "our",
        "your",
        "of",
        "at",
        "in",
        "on",
        "for",
        "to",
        "local",
    }
)
_MONEY_RE = re.compile(r"\$\s?\d")
_DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

AGGREGATION_SUM = "sum_amount"
AGGREGATION_DAYS = "count_distinct_days"
AGGREGATION_INSTANCES = "count_distinct_instances"  # "how many tanks/classes/courses"
AGGREGATION_ENTITY = (
    "entity_select"  # "which store / what / where / when" -> candidate set
)
AGGREGATION_SUM_VALUE = "sum_value"  # "total number of people reached"
AGGREGATION_DURATION = "sum_duration"  # "how many hours/minutes"
AGGREGATION_LOOKUP = "lookup_scalar"  # "points needed to redeem"
AGGREGATION_DATE = "date_answer"  # "when did I..."
AGGREGATION_STATED_TOTAL = "stated_total"  # "how many rare items do I have in total"
VALUE_ROLE_ANSWER = "answer_value"
VALUE_ROLE_CURRENT_BALANCE = "current_balance"
VALUE_ROLE_THRESHOLD = "threshold"
VALUE_ROLE_DISTRACTOR = "distractor_value"
VALUE_ROLE_SUBTOTAL = "subtotal"
VALUE_ROLE_HISTORICAL = "historical_value"
_VALUE_ROLES = frozenset(
    {
        VALUE_ROLE_ANSWER,
        VALUE_ROLE_CURRENT_BALANCE,
        VALUE_ROLE_THRESHOLD,
        VALUE_ROLE_DISTRACTOR,
        VALUE_ROLE_SUBTOTAL,
        VALUE_ROLE_HISTORICAL,
    }
)
# Abstract / speaker handles that are never an answer instance.
_NON_ANSWER_TYPE_HINTS = ("person", "user", "goal", "mileage")
# value attributes that name an instance (distinguish same-type instances).
_NAME_ATTRS = frozenset({"name", "brand_model", "model", "title", "label", "brand"})
_DURATION_UNITS = frozenset(
    {"hour", "hours", "hr", "hrs", "minute", "minutes", "min", "mins", "day", "days"}
)
_MONEY_UNITS = frozenset({"usd", "$", "dollar", "dollars"})

MEMBERSHIP_CERTAIN_IN = "certain_in"
MEMBERSHIP_CERTAIN_OUT = "certain_out"
MEMBERSHIP_CONTESTED = "contested"
_MEMBERSHIP_TAGS = frozenset(
    {MEMBERSHIP_CERTAIN_IN, MEMBERSHIP_CERTAIN_OUT, MEMBERSHIP_CONTESTED}
)


# ===========================================================================
# Pure deterministic engine
# ===========================================================================
@dataclasses.dataclass(frozen=True)
class EntityBundle:
    """One resolved entity: every typed fact about it, folded across sessions."""

    root: str
    handles: tuple[str, ...]
    types: tuple[str, ...]
    usd: float | None
    values: tuple[tuple[str, str, str], ...]  # (attribute, value, unit)
    quantities: tuple[tuple[str, str, str], ...]  # (dimension, value, unit)
    dates: tuple[str, ...]
    statuses: tuple[str, ...]
    relations: tuple[tuple[str, str], ...]  # (relation, target)
    actions: tuple[tuple[str, str], ...]  # (verb, status)
    events: tuple[dict[str, typing.Any], ...]
    sessions: tuple[str, ...]
    name: str | None = None  # a stated name/brand (distinguishes same-type instances)

    @property
    def label(self) -> str:
        # prefer a stated name (Thrive Market) over the bare type (grocery store) so
        # entity_select can distinguish same-type instances.
        longest_type = max(self.types, key=len) if self.types else ""
        return self.name or longest_type or self.root


@dataclasses.dataclass(frozen=True)
class ValueFact:
    attribute: str
    value: str
    unit: str
    numeric: float
    role: str


@dataclasses.dataclass(frozen=True)
class QuantityFact:
    dimension: str
    value: str
    unit: str
    numeric: float


@dataclasses.dataclass(frozen=True)
class EnvelopeResult:
    aggregation: str
    certain: float
    possible: float
    pivot: tuple[str, ...]  # contested roots (the disambiguation pivots)
    certain_in: tuple[str, ...]
    certain_labels: tuple[str, ...] = ()  # entity_select: the certain answer set
    possible_labels: tuple[str, ...] = ()  # entity_select: certain U contested set

    @property
    def collapsed(self) -> bool:
        return self.certain == self.possible

    @property
    def consistent(self) -> bool:
        # envelope-consistency guard: open iff there is a pivot.
        return (self.certain != self.possible) == bool(self.pivot)


def normalize_tokens(text: str) -> frozenset[str]:
    """Content tokens, underscores and spaces collapsed alike, stopwords dropped.

    Lets ``bike_shop_downtown`` match ``the local bike shop downtown`` for dedup.
    """
    return frozenset(re.findall(r"[a-z0-9]+", str(text).lower())) - _STOPWORDS


def norm_day(value: str) -> str:
    """Canonical calendar-day key: ``December 24th`` == ``December 24``."""
    stripped = _ORDINAL_RE.sub(r"\1", str(value).lower())
    return re.sub(r"\s+", " ", stripped).strip()


def classify_relation(relation: str | None, target: str | None) -> str | None:
    """``"value"``/``"time"`` if a value/date posed as a relation, else ``None``."""
    verb = (relation or "").lower()
    target_text = str(target or "")
    if verb in _VALUE_VERBS or _MONEY_RE.search(target_text):
        return "value"
    if verb in _TIME_VERBS or _DATE_RE.search(target_text):
        return "time"
    return None


def is_dup_edge(target: str, existing_targets: typing.Iterable[str]) -> bool:
    """True if ``target`` is token-subset equivalent to an existing target."""
    target_tokens = normalize_tokens(target)
    if not target_tokens:
        return False
    for existing in existing_targets:
        existing_tokens = normalize_tokens(existing)
        if existing_tokens and (
            existing_tokens <= target_tokens or target_tokens <= existing_tokens
        ):
            return True
    return False


def usd_value(
    value: typing.Any, unit: typing.Any, attribute: typing.Any
) -> float | None:
    """Extract a USD amount from a ``value`` fact, else ``None``."""
    if str(unit).upper() == "USD" or str(attribute).lower() in (
        "price",
        "cost",
        "amount",
    ):
        match = re.search(r"\d[\d,]*\.?\d*", str(value))
        if match:
            return float(match.group(0).replace(",", ""))
    return None


def numeric_value(value: typing.Any) -> float | None:
    """First numeric scalar in ``value``, preserving decimals."""
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _value_rows(bundle: EntityBundle) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    for attribute, value, unit in bundle.values:
        numeric = numeric_value(value)
        if numeric is not None:
            rows.append((attribute.lower(), value, unit.lower(), numeric))
    return rows


def _quantity_rows(bundle: EntityBundle) -> list[QuantityFact]:
    rows: list[QuantityFact] = []
    for dimension, value, unit in bundle.quantities:
        numeric = numeric_value(value)
        if numeric is not None:
            rows.append(
                QuantityFact(
                    dimension=dimension.lower(),
                    value=value,
                    unit=unit.lower(),
                    numeric=numeric,
                )
            )
    return rows


def _text_blob(*parts: typing.Any) -> str:
    return " ".join(str(part).lower().replace("_", " ") for part in parts if part)


def _bundle_value_context(bundle: EntityBundle | None) -> str:
    if bundle is None:
        return ""
    return _text_blob(
        bundle.root,
        " ".join(bundle.handles),
        " ".join(bundle.types),
        " ".join(f"{relation} {target}" for relation, target in bundle.relations),
        " ".join(f"{verb} {status}" for verb, status in bundle.actions),
    )


def classify_value_role(
    attribute: str,
    value: str,
    unit: str,
    *,
    question: str = "",
    bundle: EntityBundle | None = None,
) -> str:
    """Classify a scalar ``value`` fact by its answer role.

    This is intentionally conservative. Explicit attribute names win. Generic
    numbers are treated as distractors unless their unit/context matches the
    question's answer variable strongly enough to be answer-bearing.
    """
    attr = _text_blob(attribute)
    value_text = _text_blob(value)
    unit_text = _text_blob(unit)
    question_text = _text_blob(question)
    context = _bundle_value_context(bundle)
    local = _text_blob(attr, unit_text)
    if _is_money_value(attr, value_text, unit_text) or _is_duration_value(
        attr, unit_text
    ):
        return VALUE_ROLE_DISTRACTOR
    if re.search(r"\b(previous|prev|prior|historical|formerly|old|past)\b", attr):
        return VALUE_ROLE_HISTORICAL
    if re.search(r"\b(subtotal|partial total|category total|line total)\b", attr):
        return VALUE_ROLE_SUBTOTAL
    if re.search(
        r"\b(required|requirement|threshold|redeem|redemption|needed|need to|minimum|"
        r"points cost|redemption cost|reward cost|cost in points)\b",
        local,
    ):
        return VALUE_ROLE_THRESHOLD
    if re.search(
        r"\b(balance|current|currently|available|owned|holding|have now|on hand|"
        r"total points|points earned|earned points)\b",
        local,
    ):
        return VALUE_ROLE_CURRENT_BALANCE
    if re.search(r"\b(points?)\b", unit_text) and re.search(
        r"\b(cost|earn|need|required|redeem|redemption)\b", attr
    ):
        return VALUE_ROLE_THRESHOLD
    if re.search(r"\b(points?)\b", unit_text) and re.search(
        r"\b(balance|current|available|earned|have|total)\b", attr
    ):
        return VALUE_ROLE_CURRENT_BALANCE
    if re.search(
        r"\b(clicks?|click through|ctr|rate|percent|discount|age|years?|score|"
        r"ranking|rank)\b",
        local,
    ):
        return VALUE_ROLE_DISTRACTOR
    if re.search(
        r"\b(reach|reached|impressions?|views?|followers?|audience|customers?|"
        r"users?|people)\b",
        local,
    ) and (
        "people" in question_text
        or re.search(
            r"\b(reach|reached|audience|views?|users?|customers?)\b", question_text
        )
    ):
        return VALUE_ROLE_ANSWER
    if re.search(r"\b(count|number|total|quantity|item count)\b", attr) and not (
        "price" in attr or "cost" in attr
    ):
        return VALUE_ROLE_ANSWER
    if "prev" in context and re.search(r"\b(previous|prev)\b", context):
        return VALUE_ROLE_HISTORICAL
    return VALUE_ROLE_DISTRACTOR


def _classified_value_rows(
    bundle: EntityBundle, *, question: str = ""
) -> list[ValueFact]:
    rows: list[ValueFact] = []
    for attribute, value, unit, numeric in _value_rows(bundle):
        role = classify_value_role(
            attribute, value, unit, question=question, bundle=bundle
        )
        rows.append(
            ValueFact(
                attribute=attribute,
                value=value,
                unit=unit,
                numeric=numeric,
                role=role,
            )
        )
    return rows


def _is_duration_value(attribute: str, unit: str) -> bool:
    return unit in _DURATION_UNITS or attribute in {"duration", "elapsed_time", "time"}


def _is_money_value(attribute: str, value: str, unit: str) -> bool:
    return (
        unit in _MONEY_UNITS
        or "$" in value
        or attribute in {"price", "cost", "amount_paid", "money_spent"}
    )


def _duration_hours(attribute: str, unit: str, value: float) -> float | None:
    if not _is_duration_value(attribute, unit):
        return None
    if unit in {"minute", "minutes", "min", "mins"}:
        return value / 60.0
    if unit in {"day", "days"}:
        return value * 24.0
    return value


def _duration_quantity_hours(row: QuantityFact) -> float | None:
    if row.dimension != "time.duration":
        return None
    unit = row.unit
    if unit in {"second", "seconds", "sec", "secs", "s"}:
        return row.numeric / 3600.0
    if unit in {"minute", "minutes", "min", "mins", "m"}:
        return row.numeric / 60.0
    if unit in {"hour", "hours", "hr", "hrs", "h"}:
        return row.numeric
    if unit in {"day", "days", "d"}:
        return row.numeric * 24.0
    if unit in {"week", "weeks", "wk", "wks", "w"}:
        return row.numeric * 7.0 * 24.0
    return None


def _duration_quantity_days(row: QuantityFact) -> float | None:
    hours = _duration_quantity_hours(row)
    if hours is None:
        return None
    return hours / 24.0


def _quantity_value_rows(
    bundle: EntityBundle, *, question: str = ""
) -> list[ValueFact]:
    answer_values = [
        row
        for row in _classified_value_rows(bundle, question=question)
        if row.role == VALUE_ROLE_ANSWER
        and not _is_money_value(row.attribute, row.value, row.unit)
        and not _is_duration_value(row.attribute, row.unit)
    ]
    if answer_values:
        return answer_values
    return [
        row
        for row in _classified_value_rows(bundle, question=question)
        if row.role == VALUE_ROLE_SUBTOTAL
        and not _is_money_value(row.attribute, row.value, row.unit)
        and not _is_duration_value(row.attribute, row.unit)
    ]


def _quantity_values(bundle: EntityBundle, *, question: str = "") -> list[float]:
    return [row.numeric for row in _quantity_value_rows(bundle, question=question)]


def _duration_values(bundle: EntityBundle) -> list[float]:
    values: list[float] = []
    for row in _quantity_rows(bundle):
        duration = _duration_quantity_hours(row)
        if duration is not None:
            values.append(duration)
    if values:
        return values
    for attribute, _value, unit, numeric in _value_rows(bundle):
        duration = _duration_hours(attribute, unit, numeric)
        if duration is not None:
            values.append(duration)
    return values


def _duration_day_values(bundle: EntityBundle) -> list[float]:
    return [
        duration
        for row in _quantity_rows(bundle)
        if (duration := _duration_quantity_days(row)) is not None
    ]


def _scalar_values(
    bundle: EntityBundle,
    *,
    question: str = "",
    roles: typing.Container[str] = _VALUE_ROLES,
) -> list[float]:
    return [
        row.numeric
        for row in _classified_value_rows(bundle, question=question)
        if row.role in roles and not _is_money_value(row.attribute, row.value, row.unit)
    ]


def _lookup_values(bundle: EntityBundle, *, question: str = "") -> list[float]:
    for roles in (
        {VALUE_ROLE_THRESHOLD},
        {VALUE_ROLE_ANSWER, VALUE_ROLE_SUBTOTAL},
        {VALUE_ROLE_CURRENT_BALANCE},
    ):
        values = _scalar_values(bundle, question=question, roles=roles)
        if values:
            return values
    return []


def _lookup_values_across_roots(
    bundles: dict[str, EntityBundle], roots: typing.Iterable[str], *, question: str = ""
) -> list[float]:
    roots = list(roots)
    for roles in (
        {VALUE_ROLE_THRESHOLD},
        {VALUE_ROLE_ANSWER, VALUE_ROLE_SUBTOTAL},
        {VALUE_ROLE_CURRENT_BALANCE},
    ):
        values = sorted(
            value
            for root in roots
            for value in _scalar_values(bundles[root], question=question, roles=roles)
        )
        if values:
            return values
    return []


def _lookup_threshold_support(
    bundles: dict[str, EntityBundle], roots: typing.Iterable[str], *, question: str = ""
) -> dict[float, set[str]]:
    """Threshold values supported by distinct resolved roots.

    Lookup questions often surface several scalar values of the right role. A
    single broad reward tier should not outweigh repeated, entity-specific
    product thresholds, but tied one-off thresholds should remain ambiguous.
    """
    support: dict[float, set[str]] = collections.defaultdict(set)
    for root in roots:
        for row in _classified_value_rows(bundles[root], question=question):
            if (
                row.role == VALUE_ROLE_THRESHOLD
                and not _is_money_value(row.attribute, row.value, row.unit)
            ):
                support[row.numeric].add(root)
    return dict(support)


def _lookup_consensus_answer(
    bundles: dict[str, EntityBundle], roots: typing.Iterable[str], *, question: str = ""
) -> float | None:
    """Strict threshold consensus, else ``None``.

    This is deliberately narrow: collapse only when one threshold value has more
    distinct-root support than every alternative and at least two independent
    roots support it. Otherwise the lookup remains an interval.
    """
    support = _lookup_threshold_support(bundles, roots, question=question)
    if not support:
        return None
    ranked = sorted(
        ((len(roots_for_value), value) for value, roots_for_value in support.items()),
        reverse=True,
    )
    top_count, top_value = ranked[0]
    runner_up_count = ranked[1][0] if len(ranked) > 1 else 0
    if top_count >= 2 and top_count > runner_up_count:
        return top_value
    return None


def _stated_total_values(bundle: EntityBundle, *, question: str = "") -> list[float]:
    return _scalar_values(
        bundle,
        question=question,
        roles={VALUE_ROLE_ANSWER, VALUE_ROLE_SUBTOTAL, VALUE_ROLE_CURRENT_BALANCE},
    )


def vote_tag(votes: typing.Sequence[str]) -> str | None:
    """Aggregate k independent membership votes into one tag.

    Contestability = instability: a genuine include/exclude split, or a contested
    plurality, resolves to ``contested``. Otherwise the majority tag wins. A lone
    dissenting vote does NOT flip a stable judgment.
    """
    clean = [vote for vote in votes if vote]
    if not clean:
        return None
    counts = collections.Counter(clean)
    if counts.get(MEMBERSHIP_CERTAIN_IN, 0) and counts.get(MEMBERSHIP_CERTAIN_OUT, 0):
        return MEMBERSHIP_CONTESTED  # judges split on include vs exclude
    if counts.get(MEMBERSHIP_CONTESTED, 0) * 2 >= len(clean):
        return MEMBERSHIP_CONTESTED  # contested is at least half the judges
    return counts.most_common(1)[0][0]


def _entity_handles(arguments: dict[str, typing.Any]) -> set[str]:
    return {
        str(arguments[role])
        for role in _ENTITY_ROLES
        if isinstance(arguments.get(role), str)
    }


def _primary_handle(arguments: dict[str, typing.Any], predicate: str) -> str | None:
    if predicate == "action":
        return arguments.get("object") or arguments.get("subject")
    if predicate == "event":
        return arguments.get("event")
    return (
        arguments.get("entity") or arguments.get("source") or arguments.get("subject")
    )


def resolve_entities(
    facts: typing.Sequence[dict[str, typing.Any]],
    extra_unions: typing.Iterable[tuple[str, str]] = (),
) -> dict[str, EntityBundle]:
    """Cluster per-session facts into resolved entities.

    Merge by exact handle (implicit) and by ``coreference`` (true ``same_as``
    identity), never across a ``distinct`` pair. Then fold an event's date onto
    items it ``included`` -- the lights' "April 20th" lives on the tune-up event.

    ``extra_unions`` are cross-session same-entity merges proposed by the resolution
    pass (``resolve_cross_session``); they are applied as additional unions, still
    distinct-protected (a ``distinct`` pair is never merged even if proposed).
    """
    forbidden = {
        frozenset((str(args["a"]), str(args["b"])))
        for fact in facts
        if fact.get("predicate") == "distinct"
        for args in [fact.get("arguments", {})]
        if args.get("a") and args.get("b")
    }
    parent: dict[str, str] = {}

    def find(handle: str) -> str:
        parent.setdefault(handle, handle)
        while parent[handle] != handle:
            parent[handle] = parent[parent[handle]]
            handle = parent[handle]
        return handle

    def union(left: str, right: str) -> None:
        if frozenset((left, right)) in forbidden:
            return
        parent[find(left)] = find(right)

    all_handles: set[str] = set()
    for fact in facts:
        handles = _entity_handles(fact.get("arguments", {}))
        all_handles |= handles
        for handle in handles:
            find(handle)
    for fact in facts:
        if fact.get("predicate") == "coreference":
            args = fact.get("arguments", {})
            if args.get("entity") and args.get("same_as"):
                union(str(args["entity"]), str(args["same_as"]))
    for (
        left,
        right,
    ) in extra_unions:  # cross-session same-entity merges (distinct-protected)
        if left in parent and right in parent:
            union(str(left), str(right))

    # handles belong to a root by union-find membership -- NOT by co-mention in a fact
    handles_by_root: dict[str, set[str]] = collections.defaultdict(set)
    for handle in all_handles:
        handles_by_root[find(handle)].add(handle)

    builders: dict[str, dict[str, typing.Any]] = {}

    def builder(root: str) -> dict[str, typing.Any]:
        return builders.setdefault(
            root,
            {
                "types": set(),
                "usd": None,
                "values": set(),
                "quantities": set(),
                "name": None,
                "dates": set(),
                "statuses": set(),
                "relations": set(),
                "actions": set(),
                "events": [],
                "sessions": set(),
            },
        )

    for root in handles_by_root:  # ensure every resolved entity has a bundle
        builder(root)

    for fact in facts:
        args = fact.get("arguments", {})
        predicate = fact.get("predicate", "")
        session = fact.get("_session", "")
        primary = _primary_handle(args, predicate)
        if not isinstance(
            primary, str
        ):  # structural facts (distinct) carry no entity payload
            continue
        bundle = builder(find(primary))
        if session:
            bundle["sessions"].add(session)
        if predicate == "object_type" and args.get("type"):
            bundle["types"].add(str(args["type"]))
        elif predicate == "value":
            if args.get("value") is not None:
                bundle["values"].add(
                    (
                        str(args.get("attribute", "")),
                        str(args.get("value", "")),
                        str(args.get("unit", "")),
                    )
                )
            amount = usd_value(
                args.get("value"), args.get("unit"), args.get("attribute")
            )
            if amount is not None:
                bundle["usd"] = amount
            elif (
                str(args.get("attribute", "")).lower() in _NAME_ATTRS
                and args.get("value")
                and bundle["name"] is None
            ):
                bundle["name"] = str(args["value"])
        elif predicate == "quantity":
            if args.get("value") is not None:
                bundle["quantities"].add(
                    (
                        str(args.get("dimension", "")),
                        str(args.get("value", "")),
                        str(args.get("unit", "")),
                    )
                )
        elif predicate == "time":
            if args.get("date"):
                bundle["dates"].add(str(args["date"]))
        elif predicate == "event":
            if args.get("date"):
                bundle["dates"].add(str(args["date"]))
            bundle["events"].append(dict(args))
        elif predicate == "status" and args.get("status"):
            bundle["statuses"].add(str(args["status"]))
        elif predicate == "relation" and args.get("relation"):
            bundle["relations"].add((str(args["relation"]), str(args.get("target"))))
        elif predicate == "action" and args.get("verb"):
            bundle["actions"].add((str(args["verb"]), str(args.get("status"))))

    # date inheritance across containment relations
    event_dates = {
        str(args["event"]): str(args["date"])
        for fact in facts
        if fact.get("predicate") == "event"
        for args in [fact.get("arguments", {})]
        if args.get("event") and args.get("date")
    }
    for fact in facts:
        if fact.get("predicate") != "relation":
            continue
        args = fact.get("arguments", {})
        source = str(args.get("source"))
        target = args.get("target")
        if (
            str(args.get("relation")) in _CONTAINMENT_RELATIONS
            and source in event_dates
            and isinstance(target, str)
        ):
            builder(find(target))["dates"].add(event_dates[source])

    return {
        root: _freeze_bundle(root, data, handles_by_root.get(root, {root}))
        for root, data in builders.items()
    }


def _freeze_bundle(
    root: str, data: dict[str, typing.Any], handles: typing.Iterable[str]
) -> EntityBundle:
    return EntityBundle(
        root=root,
        handles=tuple(sorted(handles)),
        types=tuple(sorted(data["types"])),
        usd=data["usd"],
        values=tuple(sorted(data["values"])),
        quantities=tuple(sorted(data["quantities"])),
        dates=tuple(sorted(data["dates"])),
        statuses=tuple(sorted(data["statuses"])),
        relations=tuple(sorted(data["relations"])),
        actions=tuple(sorted(data["actions"])),
        events=tuple(data["events"]),
        sessions=tuple(sorted(data["sessions"])),
        name=data["name"],
    )


def fold_recovered_relations(
    bundle: EntityBundle,
    recovered: typing.Sequence[dict[str, typing.Any]],
    sentences: typing.Sequence[str],
) -> tuple[EntityBundle, list[tuple[str, str]]]:
    """Fold text-verified, relations-only, deduped recovered edges into a bundle.

    A recovered edge is kept only if its span resolves in the entity's own
    sentences, it is not a value/date masquerading as a relation, and it is not a
    token-subset duplicate of an edge the bundle already has (or one accepted this
    pass). Returns the updated bundle and the genuinely-new ``(relation, target)``.
    """
    existing_targets = {target for _, target in bundle.relations}
    accepted: list[tuple[str, str]] = list(bundle.relations)
    added: list[tuple[str, str]] = []
    for edge in recovered:
        relation = edge.get("relation")
        target = str(edge.get("target", ""))
        span = edge.get("evidence_span", "")
        if not (
            isinstance(span, str) and any(span in sentence for sentence in sentences)
        ):
            continue
        if classify_relation(relation, target):
            continue
        if is_dup_edge(target, existing_targets):
            continue
        existing_targets.add(target)
        accepted.append((str(relation), target))
        added.append((str(relation), target))
    updated = dataclasses.replace(bundle, relations=tuple(sorted(set(accepted))))
    return updated, added


def detect_intent(question: str) -> str:
    """Map the question to an aggregation shape. Authoritative (the question signal is
    reliable; the LLM mislabels e.g. 'how many classes' as a day-count)."""
    q = question.lower()
    if re.search(r"how much|how many dollars|\$|how much money", q):
        return AGGREGATION_SUM
    if re.search(r"\b(?:hours?|minutes?)\b", q) and re.search(
        r"\b(?:how many|how much|total)\b", q
    ):
        return AGGREGATION_DURATION
    if re.search(r"\bpoints?\b", q) and re.search(
        r"\b(?:redeem|redemption|need|required|earn)\b", q
    ):
        return AGGREGATION_LOOKUP
    if re.search(
        r"\btotal number of\b.*\b(?:people|users|customers|views|reach|reached)\b",
        q,
    ):
        return AGGREGATION_SUM_VALUE
    if re.search(r"\b(?:people|users|customers)\s+reached\b", q):
        return AGGREGATION_SUM_VALUE
    if "how many days" in q:
        return AGGREGATION_DAYS
    if re.search(r"\bwhen\b|\bwhat date\b|\bwhich date\b", q):
        return AGGREGATION_DATE
    if re.search(r"\bhow many\b", q) and re.search(
        r"\b(?:do i have|have i got|currently have)\b.*\bin total\b", q
    ):
        return AGGREGATION_STATED_TOTAL
    if re.search(r"how many|how often|how frequently|number of", q):
        return AGGREGATION_INSTANCES
    if re.search(r"\b(which|what|where|when|who|whose)\b", q):
        return AGGREGATION_ENTITY
    return AGGREGATION_SUM


def _is_abstract(types: typing.Iterable[str]) -> bool:
    types = list(types)
    return not types or all(
        any(hint in t.lower() for hint in _NON_ANSWER_TYPE_HINTS) for t in types
    )


def select_candidates(
    bundles: dict[str, EntityBundle], intent: str, *, question: str = ""
) -> list[str]:
    """Answer candidates by shape: priced/purchasable (sum), dated activities (days),
    or typed instances (instances / entity_select)."""
    candidates: list[str] = []
    for root, bundle in bundles.items():
        if intent == AGGREGATION_SUM:
            buyish = any(
                verb in ("buy", "order", "purchase", "get", "install")
                for verb, _ in bundle.actions
            )
            if bundle.usd is not None or buyish:
                candidates.append(root)
        elif intent == AGGREGATION_SUM_VALUE:
            if _quantity_values(bundle, question=question):
                candidates.append(root)
        elif intent == AGGREGATION_DURATION:
            if _duration_values(bundle):
                candidates.append(root)
        elif intent == AGGREGATION_LOOKUP:
            if _lookup_values(bundle, question=question):
                candidates.append(root)
        elif intent == AGGREGATION_STATED_TOTAL:
            if _stated_total_values(bundle, question=question):
                candidates.append(root)
        elif intent == AGGREGATION_DAYS:
            if bundle.dates or bundle.events or _duration_day_values(bundle):
                candidates.append(root)
        elif intent == AGGREGATION_DATE:
            if bundle.dates:
                candidates.append(root)
        elif root != "user" and not _is_abstract(bundle.types):
            # instances / entity_select: every concrete typed instance is a candidate;
            # membership filters to the question's class.
            candidates.append(root)
    return sorted(candidates)


def _distinct_days(
    bundles: dict[str, EntityBundle], roots: typing.Iterable[str]
) -> set[str]:
    return {norm_day(date) for root in roots for date in bundles[root].dates}


def _count_days(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
) -> float:
    roots = list(roots)
    duration_days = [
        value for root in roots for value in _duration_day_values(bundles[root])
    ]
    if duration_days:
        return sum(duration_days)
    return float(len(_distinct_days(bundles, roots)))


def _sum_quantity_values(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
    *,
    question: str = "",
) -> float:
    total = 0.0
    seen: list[tuple[float, str, frozenset[str]]] = []
    for root in roots:
        bundle = bundles[root]
        context_tokens = normalize_tokens(
            _text_blob(
                root,
                " ".join(bundle.handles),
                " ".join(bundle.types),
                " ".join(
                    f"{relation} {target}" for relation, target in bundle.relations
                ),
            )
        )
        for row in _quantity_value_rows(bundle, question=question):
            attr_key = _quantity_attribute_key(row.attribute, row.unit)
            duplicate = any(
                row.numeric == old_numeric
                and attr_key == old_attr_key
                and bool(context_tokens & old_context_tokens)
                for old_numeric, old_attr_key, old_context_tokens in seen
            )
            if duplicate:
                continue
            seen.append((row.numeric, attr_key, context_tokens))
            total += row.numeric
    return total


def _quantity_attribute_key(attribute: str, unit: str) -> str:
    text = _text_blob(attribute, unit)
    if re.search(r"\b(reach|reached|people)\b", text):
        return "people_reached"
    if re.search(r"\b(followers?|audience)\b", text):
        return "audience_size"
    if re.search(r"\b(views?|impressions?)\b", text):
        return "impressions"
    return text


def _sum_duration_values(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
) -> float:
    return sum(value for root in roots for value in _duration_values(bundles[root]))


def _is_pickup_return_question(question: str) -> bool:
    q = _text_blob(question)
    return bool(
        re.search(r"\b(pick up|pickup|collect)\b", q)
        and re.search(r"\b(return|exchange)\b", q)
    )


def _pickup_return_obligation_weight(bundle: EntityBundle) -> float:
    """Count actionable obligations, not wardrobe identities.

    A replacement exchange can create two answer-bearing obligations for the same
    clothing sortal: returning/exchanging the original and picking up the
    replacement. Keep the rule narrow to pickup/return questions so normal
    instance counts still count canonical entities.
    """
    evidence = _text_blob(
        " ".join(f"{relation} {target}" for relation, target in bundle.relations),
        " ".join(f"{verb} {status}" for verb, status in bundle.actions),
    )
    if re.search(r"\b(exchanged|exchange|exchanged for|replacement)\b", evidence):
        return 2.0
    return 1.0


def _count_instances(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
    *,
    question: str = "",
) -> float:
    if _is_pickup_return_question(question):
        return sum(_pickup_return_obligation_weight(bundles[root]) for root in roots)
    return float(len(list(roots)))


def _scalar_range(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
    *,
    question: str = "",
    lookup: bool = False,
    stated_total: bool = False,
) -> tuple[float, float]:
    if lookup:
        values = _lookup_values_across_roots(bundles, roots, question=question)
    elif stated_total:
        values = sorted(
            value
            for root in roots
            for value in _stated_total_values(bundles[root], question=question)
        )
    else:
        values = sorted(
            value
            for root in roots
            for value in _scalar_values(bundles[root], question=question)
        )
    if not values:
        return 0.0, 0.0
    return values[0], values[-1]


def _lookup_answer(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
    *,
    question: str = "",
) -> float:
    consensus = _lookup_consensus_answer(bundles, roots, question=question)
    if consensus is not None:
        return consensus
    values = _lookup_values_across_roots(bundles, roots, question=question)
    if not values:
        return 0.0
    # Requirement lookups ask for the qualifying threshold, not the largest
    # scalar mentioned near the account or reward.
    return min(values)


def _date_labels(
    bundles: dict[str, EntityBundle],
    roots: typing.Iterable[str],
) -> tuple[str, ...]:
    return tuple(
        sorted({norm_day(date) for root in roots for date in bundles[root].dates})
    )


def aggregate_envelope(
    bundles: dict[str, EntityBundle],
    judged: dict[str, str],
    aggregation: str,
    candidates: typing.Sequence[str],
    *,
    question: str = "",
) -> EnvelopeResult:
    """Deterministic certain/possible aggregate over LLM-tagged membership."""
    certain_roots = [
        root for root in candidates if judged.get(root) == MEMBERSHIP_CERTAIN_IN
    ]
    contested_roots = [
        root for root in candidates if judged.get(root) == MEMBERSHIP_CONTESTED
    ]
    possible_roots = certain_roots + contested_roots
    certain_labels: tuple[str, ...] = ()
    possible_labels: tuple[str, ...] = ()
    if aggregation == AGGREGATION_DAYS:
        certain = _count_days(bundles, certain_roots)
        possible = _count_days(bundles, possible_roots)
    elif aggregation == AGGREGATION_SUM_VALUE:
        certain = _sum_quantity_values(bundles, certain_roots, question=question)
        possible = _sum_quantity_values(bundles, possible_roots, question=question)
    elif aggregation == AGGREGATION_DURATION:
        certain = _sum_duration_values(bundles, certain_roots)
        possible = _sum_duration_values(bundles, possible_roots)
    elif aggregation == AGGREGATION_LOOKUP:
        possible_consensus = _lookup_consensus_answer(
            bundles, possible_roots, question=question
        )
        if possible_consensus is not None:
            certain = possible = possible_consensus
            contested_roots = []
        else:
            certain_answer = _lookup_answer(bundles, certain_roots, question=question)
            possible_answer = _lookup_answer(bundles, possible_roots, question=question)
            certain = min(certain_answer, possible_answer)
            possible = max(certain_answer, possible_answer)
    elif aggregation == AGGREGATION_STATED_TOTAL:
        _certain_low, certain_high = _scalar_range(
            bundles, certain_roots, question=question, stated_total=True
        )
        _possible_low, possible_high = _scalar_range(
            bundles, possible_roots, question=question, stated_total=True
        )
        certain = certain_high
        possible = possible_high
    elif aggregation == AGGREGATION_INSTANCES:
        certain = _count_instances(bundles, certain_roots, question=question)
        possible = _count_instances(bundles, possible_roots, question=question)
    elif aggregation == AGGREGATION_DATE:
        certain_labels = _date_labels(bundles, certain_roots)
        possible_labels = _date_labels(bundles, possible_roots)
        certain = float(len(certain_labels))
        possible = float(len(possible_labels))
    elif aggregation == AGGREGATION_ENTITY:
        certain_labels = tuple(sorted({bundles[r].label for r in certain_roots}))
        possible_labels = tuple(sorted({bundles[r].label for r in possible_roots}))
        certain = float(len(certain_labels))  # the answer SET, not an interval
        possible = float(len(possible_labels))
    else:  # AGGREGATION_SUM
        certain = sum(bundles[root].usd or 0.0 for root in certain_roots)
        possible = sum(bundles[root].usd or 0.0 for root in possible_roots)
    return EnvelopeResult(
        aggregation=aggregation,
        certain=certain,
        possible=possible,
        pivot=tuple(sorted(contested_roots)),
        certain_in=tuple(sorted(certain_roots)),
        certain_labels=certain_labels,
        possible_labels=possible_labels,
    )


# --- deterministic guards (verify what is cheap; cannot verify membership) ---
def span_resolution(facts: typing.Sequence[dict[str, typing.Any]]) -> tuple[int, int]:
    """(#facts whose span resolves in its raw session, #facts) -- anti-hallucination."""
    resolved = sum(1 for fact in facts if fact.get("_span_ok"))
    return resolved, len(facts)


def zero_fact_sessions(
    facts: typing.Sequence[dict[str, typing.Any]],
    answer_session_ids: typing.Iterable[str],
) -> list[str]:
    """Answer sessions that yielded zero facts -- a recall-miss flag."""
    seen = {fact.get("_session") for fact in facts}
    return sorted(sid for sid in answer_session_ids if sid not in seen)


# ===========================================================================
# Provider payloads / parsers (pure)
# ===========================================================================
def sentences_for(
    handles: typing.Iterable[str],
    facts: typing.Sequence[dict[str, typing.Any]],
    sessions: dict[str, str],
) -> list[str]:
    """User sentences that mention any of an entity's evidence spans (sorted)."""
    spans_by_handle: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for fact in facts:
        args = fact.get("arguments", {})
        for handle in _entity_handles(args):
            spans_by_handle[handle].add(
                (fact.get("_session", ""), fact.get("evidence_span", ""))
            )
    out: list[str] = []
    for handle in sorted(handles):
        for session, span in sorted(spans_by_handle.get(handle, set())):
            for sentence in re.split(r"(?<=[.!?])\s+", sessions.get(session, "")):
                if span and span in sentence and sentence not in out:
                    out.append(sentence)
    return out


_REREAD_PROMPT = """Sentences from a user's messages that mention "{label}":
{sentences}

List EVERY relation the text EXPLICITLY states involving "{label}"
(e.g. used_for, bought_at, part_of, for, replaces, installed_during, located_at).
Return exactly one strict JSON array, no prose:
[{{"relation": "<verb>", "target": "<other thing>", "evidence_span": "<substring>"}}]
Only relations the text states. Empty array [] if none."""


def build_reread_prompt(label: str, sentences: typing.Sequence[str]) -> str:
    body = "\n".join(f"- {sentence[:300]}" for sentence in sentences)
    return _REREAD_PROMPT.format(label=label, sentences=body)


def build_membership_payload(
    *,
    case_id: str,
    question: str,
    bundles: dict[str, EntityBundle],
    candidates: typing.Sequence[str],
) -> dict[str, typing.Any]:
    entities = []
    for root in sorted(candidates):
        bundle = bundles[root]
        entities.append(
            {
                "handle": root,
                "type": ", ".join(bundle.types) or "?",
                "usd": bundle.usd,
                "values": [
                    {
                        "attribute": attr,
                        "value": value,
                        "unit": unit,
                        "role": classify_value_role(
                            attr, value, unit, question=question, bundle=bundle
                        ),
                    }
                    for attr, value, unit in bundle.values
                ],
                "quantities": [
                    {"dimension": dimension, "value": value, "unit": unit}
                    for dimension, value, unit in bundle.quantities
                ],
                "dates": list(bundle.dates),
                "statuses": list(bundle.statuses),
                "relations": [
                    f"{relation} -> {target}" for relation, target in bundle.relations
                ],
                "actions": [
                    verb + (f" [{status}]" if status and status != "None" else "")
                    for verb, status in bundle.actions
                ],
                "events": [
                    f"type={event.get('type')} date={event.get('date')} "
                    f"status={event.get('status')}"
                    for event in bundle.events
                ],
            }
        )
    return {
        "task": (
            "Judge each candidate entity's membership in the question's answer set "
            "using ONLY the typed facts listed under it. Do not use outside priors "
            "about what 'counts'."
        ),
        "prompt_version": ENVELOPE_PROMPT_VERSION,
        "case_id": case_id,
        "question": question,
        "membership_contract": [
            "certain_in: a listed relation/fact ties the entity to the question's "
            "subject.",
            "certain_out: a listed fact excludes it (e.g. status/action shows it is "
            "planned/future for a 'money already spent' question, or its type/"
            "relations place it outside the subject).",
            "contested: the listed facts genuinely leave membership OPEN -- no listed "
            "relation resolves it either way (a careful reader could defensibly "
            "include OR exclude it).",
            "In each reason, cite the specific listed fact(s) that decide it.",
        ],
        "entities": entities,
        "output_schema": {
            "aggregation": (
                f"{AGGREGATION_SUM}|{AGGREGATION_DAYS}|"
                f"{AGGREGATION_INSTANCES}|{AGGREGATION_ENTITY}|"
                f"{AGGREGATION_SUM_VALUE}|{AGGREGATION_DURATION}|"
                f"{AGGREGATION_LOOKUP}|{AGGREGATION_DATE}|"
                f"{AGGREGATION_STATED_TOTAL}"
            ),
            "entities": [
                {
                    "handle": "<root handle from the entity block>",
                    "membership": (
                        f"{MEMBERSHIP_CERTAIN_IN}|{MEMBERSHIP_CERTAIN_OUT}|"
                        f"{MEMBERSHIP_CONTESTED}"
                    ),
                    "reason": "<cite the deciding listed fact>",
                }
            ],
        },
    }


def build_membership_prompt(
    payload: dict[str, typing.Any], *, sample_index: int, samples: int
) -> str:
    return (
        f"INDEPENDENT JUDGMENT PASS #{sample_index + 1} of {samples} - "
        "judge each entity afresh from its own facts.\n"
        "Return exactly one strict JSON object matching output_schema. "
        "No markdown, prose, comments, or trailing text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


@dataclasses.dataclass(frozen=True)
class MembershipParseResult:
    parse_status: str
    aggregation: str
    judgments: tuple[tuple[str, str, str], ...]  # (handle, membership, reason)
    parse_errors: tuple[str, ...]


def parse_membership_response(
    raw_output: str,
    *,
    expected_case_id: str | None = None,
) -> MembershipParseResult:
    text = raw_output.strip()
    if not text:
        return MembershipParseResult("empty", "", (), ("empty provider output",))
    decoded = _grab_json_object(text)
    if decoded is None:
        return MembershipParseResult("invalid_json", "", (), ("no JSON object found",))
    aggregation = str(decoded.get("aggregation", ""))
    errors: list[str] = []
    judgments: list[tuple[str, str, str]] = []
    for index, entity in enumerate(decoded.get("entities", []) or []):
        if not isinstance(entity, dict):
            errors.append(f"entities[{index}] must be an object")
            continue
        handle = str(entity.get("handle", "")).strip()
        membership = str(entity.get("membership", "")).strip()
        reason = str(entity.get("reason", "")).strip()
        if not handle:
            errors.append(f"entities[{index}] missing handle")
            continue
        if membership not in _MEMBERSHIP_TAGS:
            errors.append(f"entities[{index}] unknown membership {membership!r}")
            continue
        judgments.append((handle, membership, reason))
    if not judgments:
        errors.append("no valid membership judgments")
    status = "parsed" if not errors else "invalid_schema"
    return MembershipParseResult(status, aggregation, tuple(judgments), tuple(errors))


def _grab_json_object(text: str) -> dict[str, typing.Any] | None:
    cleaned = (
        re.sub(r"^.*?```json", "", text, flags=re.S) if "```json" in text else text
    )
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(cleaned)):
        if cleaned[index] == "{":
            depth += 1
        elif cleaned[index] == "}":
            depth -= 1
            if depth == 0:
                try:
                    decoded = json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return decoded if isinstance(decoded, dict) else None
    return None


def _grab_json_array(text: str) -> list[typing.Any]:
    cleaned = (
        re.sub(r"^.*?```json", "", text, flags=re.S) if "```json" in text else text
    )
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("[")
    if start < 0:
        return []
    depth = 0
    for index in range(start, len(cleaned)):
        if cleaned[index] == "[":
            depth += 1
        elif cleaned[index] == "]":
            depth -= 1
            if depth == 0:
                try:
                    decoded = json.loads(cleaned[start : index + 1])
                except json.JSONDecodeError:
                    return []
                return decoded if isinstance(decoded, list) else []
    return []


# ===========================================================================
# Provider orchestration (live)
# ===========================================================================
def _cached_provider(
    prompt: str,
    *,
    command: str,
    timeout_seconds: int,
    cache_dir: pathlib.Path | None,
    prefix: str,
) -> str:
    cache_file: pathlib.Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        cache_file = cache_dir / f"{prefix}{digest}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
    result = interpretation_runner.run_provider(
        command=command, prompt=prompt, timeout_seconds=timeout_seconds
    )
    raw = interpretation_runner_result_text(result)
    if cache_file is not None and result.exit_code == 0 and not result.timed_out:
        cache_file.write_text(raw, encoding="utf-8")
    return raw


def interpretation_runner_result_text(
    result: interpretation_runner.ProviderResult,
) -> str:
    """The provider's ``result`` text (``claude -p --output-format json`` wraps it)."""
    raw = result.raw_output
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(decoded, dict) and isinstance(decoded.get("result"), str):
        return decoded["result"]
    return raw


def extract_facts(
    *,
    case_id: str,
    question: str,
    answer_sessions: dict[str, tuple[str, str | None]],
    command: str,
    timeout_seconds: int,
    cache_dir: pathlib.Path | None,
) -> list[dict[str, typing.Any]]:
    """Run the formalizer per answer-session; tag each fact session/span_ok."""
    facts: list[dict[str, typing.Any]] = []
    for session_id, (text, date) in answer_sessions.items():
        payload = candidate_unit_formalizer.build_formalizer_payload(
            case_id=case_id,
            question=question,
            evidence_session={"session_id": session_id, "date": date, "text": text},
            source_prompt_version=ENVELOPE_PROMPT_VERSION,
        )
        raw = _cached_provider(
            candidate_unit_formalizer.build_provider_prompt(payload),
            command=command,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
            prefix="formalize_",
        )
        parsed = candidate_unit_formalizer.parse_formalizer_response(
            raw, expected_formalizer_id=payload["formalizer_id"]
        )
        for fact in parsed.facts:
            record = fact.to_dict()
            record["_session"] = session_id
            record["_span_ok"] = fact.evidence_span in text
            facts.append(record)
    return facts


def complete_relations(
    *,
    bundles: dict[str, EntityBundle],
    candidates: typing.Sequence[str],
    facts: typing.Sequence[dict[str, typing.Any]],
    sessions: dict[str, str],
    command: str,
    timeout_seconds: int,
    cache_dir: pathlib.Path | None,
    max_candidates: int = 8,
) -> tuple[dict[str, EntityBundle], dict[str, list[tuple[str, str]]]]:
    """Re-read each candidate's sentences; fold recovered relations into its bundle.

    Capped at ``max_candidates`` re-reads (one provider call each) so instance/entity
    questions with many typed candidates do not balloon cost; relation recovery is most
    load-bearing for the priced/entity answers anyway.
    """
    recovered: dict[str, list[tuple[str, str]]] = {}
    updated = dict(bundles)
    for root in sorted(candidates)[:max_candidates]:
        bundle = updated[root]
        sentences = sentences_for(bundle.handles, facts, sessions)
        if not sentences:
            continue
        raw = _cached_provider(
            build_reread_prompt(bundle.label, sentences),
            command=command,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
            prefix="reread_",
        )
        new_bundle, added = fold_recovered_relations(
            bundle, _grab_json_array(raw), sentences
        )
        updated[root] = new_bundle
        if added:
            recovered[root] = added
    return updated, recovered


def judge_membership(
    *,
    case_id: str,
    question: str,
    bundles: dict[str, EntityBundle],
    candidates: typing.Sequence[str],
    samples: int,
    command: str,
    timeout_seconds: int,
    cache_dir: pathlib.Path | None,
) -> tuple[dict[str, str], dict[str, list[str]], str]:
    """k independent membership passes; aggregate by ``vote_tag``."""
    payload = build_membership_payload(
        case_id=case_id, question=question, bundles=bundles, candidates=candidates
    )
    votes: dict[str, list[str]] = collections.defaultdict(list)
    aggregations: list[str] = []
    for sample_index in range(samples):
        prompt = build_membership_prompt(
            payload, sample_index=sample_index, samples=samples
        )
        raw = _cached_provider(
            prompt,
            command=command,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
            prefix=f"membership_{sample_index}_",
        )
        parsed = parse_membership_response(raw, expected_case_id=case_id)
        if parsed.parse_status != "parsed":
            for handle in candidates:
                votes[handle].append(MEMBERSHIP_CONTESTED)
            continue
        if parsed.aggregation:
            aggregations.append(parsed.aggregation)
        for handle, membership, _reason in parsed.judgments:
            votes[handle].append(membership)
    judged = {
        root: tag
        for root in candidates
        if (tag := vote_tag(votes.get(root, []))) is not None
    }
    aggregation = (
        collections.Counter(aggregations).most_common(1)[0][0] if aggregations else ""
    )
    return judged, dict(votes), aggregation


_RESOLUTION_PROMPT = """Question: {question}

Entities extracted from a user's messages across several sessions. Some MAY refer
to the SAME real-world thing mentioned in different sessions (e.g. "bike lights" in
one session and "the new lights" in another are one purchase).

Group handles that denote the SAME real-world individual. Do NOT group distinct
things that merely share a type -- two different fish tanks, two different courses,
two different stores stay SEPARATE. Group only when the evidence clearly points to
the same individual thing (same name, same described purchase, same event).

ENTITIES:
{blocks}

Return exactly one strict JSON object, no prose:
{{"clusters": [["handle_a", "handle_b"]]}}
List only multi-handle same-entity groups; omit anything that stands alone."""


def build_resolution_prompt(question: str, bundles: dict[str, EntityBundle]) -> str:
    lines = []
    for root in sorted(bundles):
        if root == "user":
            continue
        bundle = bundles[root]
        parts = [f"- {root}: type={', '.join(sorted(bundle.types)) or '?'}"]
        if bundle.name:
            parts.append(f"name={bundle.name}")
        if bundle.usd is not None:
            parts.append(f"${bundle.usd:.0f}")
        if bundle.sessions:
            parts.append(f"sessions={list(bundle.sessions)}")
        rels = [
            f"{relation}->{target}" for relation, target in sorted(bundle.relations)[:3]
        ]
        if rels:
            parts.append("rel: " + "; ".join(rels))
        lines.append(" ".join(parts))
    return _RESOLUTION_PROMPT.format(question=question, blocks="\n".join(lines))


def parse_resolution_response(
    raw_output: str, *, valid_handles: typing.Container[str]
) -> list[tuple[str, str]]:
    """Parse same-entity clusters into pairwise unions, dropping unknown handles."""
    decoded = _grab_json_object(raw_output) or {}
    pairs: list[tuple[str, str]] = []
    for cluster in decoded.get("clusters", []) or []:
        if not isinstance(cluster, list):
            continue
        members = [
            str(h) for h in cluster if isinstance(h, str) and str(h) in valid_handles
        ]
        for other in members[1:]:
            pairs.append((members[0], other))
    return pairs


def resolve_cross_session(
    *,
    question: str,
    bundles: dict[str, EntityBundle],
    command: str,
    timeout_seconds: int,
    cache_dir: pathlib.Path | None,
) -> list[tuple[str, str]]:
    """LLM proposes same-real-world-entity clusters across sessions; code merges them.

    The per-session formalizer is answer-free and cannot reason across sessions, so
    the same item recurs under different handles (the lights double-count). This
    recovers identity; the merge is applied by ``resolve_entities``, distinct-protected.
    """
    if sum(1 for root in bundles if root != "user") < 2:
        return []
    raw = _cached_provider(
        build_resolution_prompt(question, bundles),
        command=command,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
        prefix="resolve_",
    )
    return parse_resolution_response(raw, valid_handles=set(bundles))


def run_envelope_case(
    row: dict[str, typing.Any],
    *,
    samples: int,
    command: str = interpretation_runner.DEFAULT_PROVIDER_COMMAND,
    timeout_seconds: int = interpretation_runner.DEFAULT_TIMEOUT_SECONDS,
    cache_dir: pathlib.Path | None = None,
    reread_max_candidates: int = 8,
    cross_session_resolution: bool = True,
) -> dict[str, typing.Any]:
    """Run the full pipeline on one LongMemEval row; return an envelope record."""
    case_id = str(row.get("question_id", ""))
    question = str(row.get("question", ""))
    answer_ids = list(row.get("answer_session_ids", []))
    dates = row.get("haystack_dates", [None] * len(row.get("haystack_session_ids", [])))
    answer_sessions: dict[str, tuple[str, str | None]] = {}
    session_text: dict[str, str] = {}
    for session_id, turns, date in zip(
        row["haystack_session_ids"], row["haystack_sessions"], dates, strict=False
    ):
        if session_id not in set(answer_ids):
            continue
        text = " ".join(turn.get("content", "") for turn in turns)
        answer_sessions[session_id] = (text, date)
        session_text[session_id] = text

    facts = extract_facts(
        case_id=case_id,
        question=question,
        answer_sessions=answer_sessions,
        command=command,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
    )
    intent = detect_intent(question)
    bundles = resolve_entities(facts)
    merges: list[tuple[str, str]] = []
    if cross_session_resolution:
        merges = resolve_cross_session(
            question=question,
            bundles=bundles,
            command=command,
            timeout_seconds=timeout_seconds,
            cache_dir=cache_dir,
        )
        if merges:
            bundles = resolve_entities(facts, extra_unions=merges)
    candidates = select_candidates(bundles, intent, question=question)
    bundles, recovered = complete_relations(
        bundles=bundles,
        candidates=candidates,
        facts=facts,
        sessions=session_text,
        command=command,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
        max_candidates=reread_max_candidates,
    )
    judged, votes, judged_aggregation = judge_membership(
        case_id=case_id,
        question=question,
        bundles=bundles,
        candidates=candidates,
        samples=samples,
        command=command,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
    )
    # detect_intent is authoritative for the shape; judged_aggregation is audit-only.
    aggregation = intent
    envelope = aggregate_envelope(
        bundles, judged, aggregation, candidates, question=question
    )
    resolved, total = span_resolution(facts)
    return {
        "case_id": case_id,
        "question": question,
        "gold": row.get("answer"),
        "aggregation": envelope.aggregation,
        "llm_aggregation": judged_aggregation,
        "cross_session_merges": [list(pair) for pair in merges],
        "n_facts": total,
        "spans_resolved": f"{resolved}/{total}",
        "zero_fact_sessions": zero_fact_sessions(facts, answer_ids),
        "candidates": list(candidates),
        "recovered_relations": {
            bundles[root].label: edges for root, edges in recovered.items()
        },
        "envelope": [envelope.certain, envelope.possible],
        "certain_labels": list(envelope.certain_labels),
        "possible_labels": list(envelope.possible_labels),
        "collapsed": envelope.collapsed,
        "pivot": [bundles[root].label for root in envelope.pivot],
        "envelope_consistent": envelope.consistent,
        "membership": {
            bundles[root].label: {
                "tag": judged.get(root),
                "votes": votes.get(root, []),
                "usd": bundles[root].usd,
                "values": [
                    {
                        "attribute": row.attribute,
                        "value": row.value,
                        "unit": row.unit,
                        "role": row.role,
                    }
                    for row in _classified_value_rows(bundles[root], question=question)
                ],
                "quantities": [
                    {
                        "dimension": row.dimension,
                        "value": row.value,
                        "unit": row.unit,
                    }
                    for row in _quantity_rows(bundles[root])
                ],
                "dates": list(bundles[root].dates),
            }
            for root in candidates
        },
    }


def _load_corpus(path: pathlib.Path) -> dict[str, dict[str, typing.Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["question_id"]): row for row in rows}


def main(argv: list[str] | None = None) -> int:
    bench: bench_config.BenchConfig = simba.config.load("bench")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_ids", nargs="*", default=["gpt4_d84a3211", "5a7937c8"])
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        default=pathlib.Path(
            bench.longmemeval_path or ".simba/benchmarks/longmemeval_s.json"
        ),
    )
    parser.add_argument(
        "--samples", type=int, default=bench.envelope_membership_samples
    )
    parser.add_argument(
        "--provider", default=interpretation_runner.DEFAULT_PROVIDER_COMMAND
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=interpretation_runner.DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(bench.envelope_cache_path),
    )
    parser.add_argument(
        "--reread-max-candidates",
        type=int,
        default=bench.envelope_reread_max_candidates,
        help=(
            "Maximum candidate relation re-read provider calls per row. "
            "Use 0 for bounded aggregate-only measurement runs."
        ),
    )
    parser.add_argument(
        "--no-cross-session-resolution",
        action="store_true",
        help="Disable cross-session entity-resolution provider calls for bounded runs.",
    )
    parser.add_argument("--report", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    corpus = _load_corpus(args.corpus)
    records = []
    for case_id in args.case_ids:
        record = run_envelope_case(
            corpus[case_id],
            samples=args.samples,
            command=args.provider,
            timeout_seconds=args.timeout_seconds,
            cache_dir=args.cache_dir,
            reread_max_candidates=args.reread_max_candidates,
            cross_session_resolution=(
                bench.envelope_cross_session_resolution
                and not args.no_cross_session_resolution
            ),
        )
        records.append(record)
        verdict = "COLLAPSE" if record["collapsed"] else "OPEN"
        print(f"\n### {case_id}  ({record['aggregation']})  gold={record['gold']}")
        print(
            f"  facts={record['n_facts']} spans={record['spans_resolved']} "
            f"zero_fact={record['zero_fact_sessions']}"
        )
        for label, member in record["membership"].items():
            extra = (
                f"${member['usd']:.0f}"
                if member["usd"] is not None
                else ",".join(member["dates"])
            )
            votes = "/".join(vote.replace("certain_", "") for vote in member["votes"])
            print(f"    [{member['tag']!s:11}] {label:26} {extra:14} votes={votes}")
        if record["aggregation"] == AGGREGATION_ENTITY:
            print(
                f"  ANSWER SET certain={record['certain_labels']} "
                f"possible={record['possible_labels']}"
            )
        print(
            f"  ENVELOPE {record['envelope']} -> {verdict}  "
            f"pivot={record['pivot'] or '-'}  "
            f"consistent={record['envelope_consistent']}"
        )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{json.dumps(records, indent=2)}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

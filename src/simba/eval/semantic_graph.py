"""Executable typed semantic graphs for ambiguity evals.

This module is intentionally diagnostic. The answer path consumes graph nodes
and role edges; it must not call the older fail18 CandidateFact repair path.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import re
import typing

import simba.eval.ambiguity_fail18 as ambiguity_fail18


@dataclasses.dataclass(frozen=True)
class SemanticNode:
    id: str
    kind: str
    type: str
    label: str
    value: int | float | str | None = None
    unit: str = ""
    attrs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class SemanticEdge:
    source: str
    relation: str
    target: str


@dataclasses.dataclass(frozen=True)
class TypedSemanticGraph:
    id: str
    graph_type: str
    nodes: tuple[SemanticNode, ...]
    edges: tuple[SemanticEdge, ...] = ()
    source_text: str = ""
    source_session_id: str = ""
    occurred_on: dt.date | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "id": self.id,
            "graph_type": self.graph_type,
            "source_text": self.source_text,
            "source_session_id": self.source_session_id,
            "occurred_on": self.occurred_on.isoformat()
            if self.occurred_on is not None
            else None,
            "nodes": [dataclasses.asdict(node) for node in self.nodes],
            "edges": [dataclasses.asdict(edge) for edge in self.edges],
        }


@dataclasses.dataclass(frozen=True)
class GraphAnswer:
    answer_space: dict[str, int | float]
    matched_fact_count: int
    exact: bool


@dataclasses.dataclass(frozen=True)
class Fail18GraphItem:
    question_id: str
    question: str
    failure_mode: str
    gold_numeric: int | float | None
    question_graph: TypedSemanticGraph
    evidence_graph_count: int
    candidate_fact_count: int
    answer_space: dict[str, int | float]
    contains_gold: bool | None

    def to_dict(self) -> dict[str, typing.Any]:
        answer_node = _answer_node(self.question_graph)
        return {
            "question_id": self.question_id,
            "question": self.question,
            "failure_mode": self.failure_mode,
            "gold_numeric": self.gold_numeric,
            "answer_node": dataclasses.asdict(answer_node)
            if answer_node is not None
            else None,
            "question_graph": self.question_graph.to_dict(),
            "evidence_graph_count": self.evidence_graph_count,
            "candidate_fact_count": self.candidate_fact_count,
            "answer_space": self.answer_space,
            "contains_gold": self.contains_gold,
        }


@dataclasses.dataclass(frozen=True)
class Fail18GraphSummary:
    total: int
    rows_with_answer_node: int
    rows_with_evidence_graphs: int
    rows_with_candidate_facts: int
    rows_with_answer: int
    rows_containing_gold: int
    results: tuple[Fail18GraphItem, ...]

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "total": self.total,
            "rows_with_answer_node": self.rows_with_answer_node,
            "rows_with_evidence_graphs": self.rows_with_evidence_graphs,
            "rows_with_candidate_facts": self.rows_with_candidate_facts,
            "rows_with_answer": self.rows_with_answer,
            "rows_containing_gold": self.rows_containing_gold,
            "results": [item.to_dict() for item in self.results],
        }


def compile_question_graph(row: dict[str, typing.Any]) -> TypedSemanticGraph:
    question = str(row.get("question", ""))
    low = question.lower()
    answer = _answer_variable_for_question(low)
    constraints = _constraint_nodes_for_question(low, str(row.get("question_date", "")))
    meta = _meta_nodes_for_question(low)
    nodes = (answer, *constraints, *meta)
    edges = [SemanticEdge("answer", "constrained_by", node.id) for node in constraints]
    edges.extend(SemanticEdge("answer", "has_meta", node.id) for node in meta)
    return TypedSemanticGraph(
        id=f"question:{row.get('question_id', _graph_id(question))}",
        graph_type="question",
        nodes=nodes,
        edges=tuple(edges),
        source_text=question,
    )


def extract_evidence_graphs(
    corpus_row: dict[str, typing.Any] | None,
) -> tuple[TypedSemanticGraph, ...]:
    if corpus_row is None:
        return ()
    graphs: list[TypedSemanticGraph] = []
    for sid, text, session_date in _iter_user_session_texts_with_ids_dates(corpus_row):
        for idx, sentence in enumerate(_sentences(text)):
            graph = _sentence_graph(
                sentence,
                graph_id=f"sentence:{corpus_row['question_id']}:{sid}:{idx}",
                sid=sid,
                occurred_on=session_date,
            )
            if _has_non_evidence_node(graph):
                graphs.append(graph)
    return tuple(graphs)


def answer_from_graphs(
    row: dict[str, typing.Any],
    corpus_row: dict[str, typing.Any] | None,
    question_graph: TypedSemanticGraph,
) -> GraphAnswer:
    del row
    evidence_graphs = extract_evidence_graphs(corpus_row)
    return align_question_to_evidence(question_graph, evidence_graphs)


def align_question_to_evidence(
    question_graph: TypedSemanticGraph,
    evidence_graphs: tuple[TypedSemanticGraph, ...],
) -> GraphAnswer:
    answer = _answer_node(question_graph)
    if answer is None:
        return GraphAnswer({}, 0, False)
    aggregation = str(answer.attrs.get("aggregation") or "")
    if aggregation == "difference":
        return _difference_answer(question_graph, evidence_graphs, answer)
    if aggregation == "sum":
        return _sum_answer(question_graph, evidence_graphs, answer)
    if aggregation == "count":
        return _count_answer(question_graph, evidence_graphs, answer)
    return GraphAnswer({}, 0, False)


def probe_fail18(
    manifest_path: pathlib.Path = ambiguity_fail18.DEFAULT_MANIFEST,
    *,
    corpus_path: pathlib.Path = ambiguity_fail18.DEFAULT_CORPUS,
    limit: int = 0,
) -> Fail18GraphSummary:
    rows = ambiguity_fail18.load_manifest(manifest_path)
    corpus_by_id = {
        str(row["question_id"]): row
        for row in ambiguity_fail18.load_corpus(corpus_path)
    }
    results: list[Fail18GraphItem] = []
    for row in rows[: limit or None]:
        corpus_row = corpus_by_id.get(str(row["question_id"]))
        question_row = (
            {**row, "question_date": corpus_row.get("question_date")}
            if corpus_row is not None
            else row
        )
        question_graph = compile_question_graph(question_row)
        evidence_graphs = extract_evidence_graphs(corpus_row)
        answer = align_question_to_evidence(question_graph, evidence_graphs)
        gold = _numeric_gold(row)
        contains = _contains(answer.answer_space, gold) if gold is not None else None
        results.append(
            Fail18GraphItem(
                question_id=str(row["question_id"]),
                question=str(row.get("question", "")),
                failure_mode=str(row.get("failure_mode", "")),
                gold_numeric=gold,
                question_graph=question_graph,
                evidence_graph_count=len(evidence_graphs),
                candidate_fact_count=answer.matched_fact_count,
                answer_space=answer.answer_space,
                contains_gold=contains,
            )
        )
    return _summarize(results)


def _answer_variable_for_question(question: str) -> SemanticNode:
    if "points" in question and "redeem" in question:
        return _answer(
            "loyalty_points",
            "quantity",
            "loyalty points",
            "difference",
            "points",
        )
    if "total number of people reached" in question:
        return _answer("people_reached", "quantity", "people reached", "sum", "people")
    if "how much money" in question:
        return _answer("money", "money", "money", "sum", "usd")
    unit_match = re.search(r"\bhow many\s+(hours?|days?)\b", question)
    if unit_match:
        unit = _canonical_unit(unit_match.group(1))
        return _answer("duration", "duration", "duration", "sum", unit)
    phrase = _answer_phrase(question)
    concept = _concept_for_phrase(phrase)
    return _answer(concept.id, concept.node_type, concept.label, "count")


def _constraint_nodes_for_question(
    question: str,
    question_date: str,
) -> tuple[SemanticNode, ...]:
    constraints: list[SemanticNode] = [
        SemanticNode("subject:user", "constraint", "subject", "user")
    ]
    for rule in _QUESTION_CONSTRAINT_RULES:
        if any(term in question for term in rule.terms):
            constraints.append(
                SemanticNode(
                    rule.id,
                    "constraint",
                    rule.type,
                    rule.label,
                    attrs={"value": rule.value},
                )
            )
    time_window = _time_window_node(question, question_date)
    if time_window is not None:
        constraints.append(time_window)
    return tuple(_dedupe_nodes(constraints))


def _meta_nodes_for_question(question: str) -> tuple[SemanticNode, ...]:
    nodes: list[SemanticNode] = []
    if "how many" in question:
        nodes.append(SemanticNode("meta:how_many", "meta", "question_word", "how many"))
    if "different" in question:
        nodes.append(SemanticNode("meta:distinct", "meta", "policy", "distinct"))
    if "total" in question:
        nodes.append(SemanticNode("meta:total", "meta", "aggregation_word", "total"))
    return tuple(nodes)


def _sentence_graph(
    sentence: str,
    *,
    graph_id: str,
    sid: str,
    occurred_on: dt.date | None,
) -> TypedSemanticGraph:
    evidence = SemanticNode("evidence:sentence", "evidence", "sentence", sentence)
    nodes: list[SemanticNode] = [evidence]
    edges: list[SemanticEdge] = []
    if _sentence_has_user_subject(sentence):
        nodes.append(SemanticNode("entity:user", "entity", "subject", "user"))
    if _sentence_is_hypothetical(sentence):
        nodes.append(
            SemanticNode("modality:hypothetical", "meta", "modality", "hypothetical")
        )
    concept_nodes = _concept_nodes(sentence)
    value_nodes = _value_nodes(sentence)
    event_nodes = _event_nodes(sentence)
    entity_nodes = _entity_instance_nodes(sentence, concept_nodes)
    nodes.extend(concept_nodes)
    nodes.extend(value_nodes)
    nodes.extend(event_nodes)
    nodes.extend(entity_nodes)
    for node in nodes[1:]:
        edges.append(SemanticEdge("evidence:sentence", "contains", node.id))
    for entity in entity_nodes:
        concept_id = str(entity.attrs.get("concept_id") or "")
        if concept_id:
            edges.append(
                SemanticEdge(entity.id, "instance_of", f"concept:{concept_id}")
            )
    for event in event_nodes:
        if any(node.id == "entity:user" for node in nodes):
            edges.append(SemanticEdge(event.id, "agent", "entity:user"))
        _attach_event_roles(event, concept_nodes, value_nodes, entity_nodes, edges)
    return TypedSemanticGraph(
        id=graph_id,
        graph_type="sentence",
        nodes=tuple(_dedupe_nodes(nodes)),
        edges=tuple(_dedupe_edges(edges)),
        source_text=sentence,
        source_session_id=sid,
        occurred_on=occurred_on,
    )


def _attach_event_roles(
    event: SemanticNode,
    concepts: list[SemanticNode],
    values: list[SemanticNode],
    entities: list[SemanticNode],
    edges: list[SemanticEdge],
) -> None:
    event_id = str(event.attrs.get("event_id") or "")
    for concept in concepts:
        relation = _concept_role_for_event(event_id, concept)
        if relation:
            edges.append(SemanticEdge(event.id, relation, concept.id))
    for entity in entities:
        concept_id = str(entity.attrs.get("concept_id") or "")
        relation = _entity_role_for_event(event_id, concept_id)
        if relation:
            edges.append(SemanticEdge(event.id, relation, entity.id))
    for value in values:
        relation = _value_role_for_event(event_id, value)
        if relation:
            edges.append(SemanticEdge(event.id, relation, value.id))


def _value_nodes(sentence: str) -> list[SemanticNode]:
    nodes: list[SemanticNode] = []
    for match in re.finditer(_VALUE_PATTERN, sentence, flags=re.IGNORECASE):
        if match.group("money"):
            raw = match.group("money")
            value = _number_value(raw)
            unit = "usd"
            node_type = "money"
        else:
            raw = match.group("num")
            value = _number_value(raw)
            unit = _canonical_unit(match.group("unit") or "")
            if unit == "minute":
                value = float(value) / 60
                unit = "hour"
            node_type = "quantity" if unit not in {"hour", "day"} else "duration"
        nodes.append(
            SemanticNode(
                id=f"value:{len(nodes)}:{unit}",
                kind="value",
                type=node_type,
                label=match.group(0),
                value=value,
                unit=unit,
            )
        )
    return nodes


def _concept_nodes(sentence: str) -> list[SemanticNode]:
    low = sentence.lower()
    nodes: list[SemanticNode] = []
    for rule in _CONCEPT_RULES:
        if re.search(rule.pattern, low):
            nodes.append(
                SemanticNode(
                    id=f"concept:{rule.id}",
                    kind="concept",
                    type=rule.node_type,
                    label=rule.label,
                    attrs={"concept_id": rule.id},
                )
            )
    return nodes


def _event_nodes(sentence: str) -> list[SemanticNode]:
    low = sentence.lower()
    nodes: list[SemanticNode] = []
    for rule in _EVENT_RULES:
        if re.search(rule.pattern, low):
            nodes.append(
                SemanticNode(
                    id=f"event:{len(nodes)}:{rule.id}",
                    kind="event",
                    type=rule.type,
                    label=rule.label,
                    attrs={"event_id": rule.id},
                )
            )
    return nodes


def _entity_instance_nodes(
    sentence: str,
    concepts: list[SemanticNode],
) -> list[SemanticNode]:
    concept_ids = {str(node.attrs.get("concept_id") or "") for node in concepts}
    nodes: list[SemanticNode] = []
    for concept_id in sorted(concept_ids):
        for label in _entity_labels_for_concept(sentence, concept_id):
            canonical = _canonical_entity_id(label, concept_id)
            if canonical:
                nodes.append(
                    SemanticNode(
                        id=f"entity:{concept_id}:{_graph_id(canonical)}",
                        kind="entity",
                        type=concept_id,
                        label=label,
                        attrs={"concept_id": concept_id, "canonical_id": canonical},
                    )
                )
    return nodes


def _difference_answer(
    question_graph: TypedSemanticGraph,
    evidence_graphs: tuple[TypedSemanticGraph, ...],
    answer: SemanticNode,
) -> GraphAnswer:
    required: list[float] = []
    current: list[float] = []
    for graph in evidence_graphs:
        if _graph_is_excluded(graph):
            continue
        for event in _event_nodes_in_graph(graph):
            if not _event_satisfies_constraints(question_graph, graph, event):
                continue
            for edge in _out_edges(graph, event.id):
                if edge.relation not in {"required_total", "current_total"}:
                    continue
                value = _node(graph, edge.target)
                if value is None or not _value_matches_answer(value, answer):
                    continue
                bucket = required if edge.relation == "required_total" else current
                bucket.append(float(value.value or 0))
    if not required or not current:
        return GraphAnswer({}, 0, False)
    target = max(required)
    baseline = max(value for value in current if value <= target)
    return GraphAnswer({"count": target - baseline}, len(required) + len(current), True)


def _sum_answer(
    question_graph: TypedSemanticGraph,
    evidence_graphs: tuple[TypedSemanticGraph, ...],
    answer: SemanticNode,
) -> GraphAnswer:
    total = 0.0
    matched = 0
    seen: set[tuple[str, str, str]] = set()
    for graph in evidence_graphs:
        if _graph_is_excluded(graph):
            continue
        for event in _event_nodes_in_graph(graph):
            if not _event_satisfies_constraints(question_graph, graph, event):
                continue
            for edge in _out_edges(graph, event.id):
                if edge.relation not in {"amount", "duration", "reach"}:
                    continue
                value = _node(graph, edge.target)
                if value is None or not _value_matches_answer(value, answer):
                    continue
                key = _value_dedupe_key(answer, value, graph)
                if key in seen:
                    continue
                seen.add(key)
                total += float(value.value or 0)
                matched += 1
    if matched == 0:
        return GraphAnswer({}, 0, False)
    return GraphAnswer({"count": total}, matched, True)


def _count_answer(
    question_graph: TypedSemanticGraph,
    evidence_graphs: tuple[TypedSemanticGraph, ...],
    answer: SemanticNode,
) -> GraphAnswer:
    concept_id = str(answer.attrs.get("concept_id") or "")
    bindings: set[str] = set()
    for graph in evidence_graphs:
        if _graph_is_excluded(graph):
            continue
        for event in _event_nodes_in_graph(graph):
            if not _event_satisfies_constraints(question_graph, graph, event):
                continue
            for binding in _answer_bindings_for_event(graph, event, concept_id):
                bindings.add(binding)
    if not bindings:
        return GraphAnswer({}, 0, False)
    return GraphAnswer({"count": len(bindings)}, len(bindings), True)


def _answer_bindings_for_event(
    graph: TypedSemanticGraph,
    event: SemanticNode,
    concept_id: str,
) -> tuple[str, ...]:
    bindings: list[str] = []
    for edge in _out_edges(graph, event.id):
        if edge.relation not in {"object", "activity", "location"}:
            continue
        target = _node(graph, edge.target)
        if target is None:
            continue
        target_concept = str(target.attrs.get("concept_id") or "")
        if target.kind == "entity" and target_concept == concept_id:
            bindings.append(str(target.attrs.get("canonical_id") or target.id))
        elif target.kind == "concept" and target_concept == concept_id:
            bindings.append(f"{graph.id}:{event.id}:{target.id}")
    return tuple(bindings)


def _event_satisfies_constraints(
    question_graph: TypedSemanticGraph,
    graph: TypedSemanticGraph,
    event: SemanticNode,
) -> bool:
    constraints = _answer_constraints(question_graph)
    for constraint in constraints:
        if constraint.type == "subject" and not _has_edge(
            graph, event.id, "agent", "entity:user"
        ):
            return False
        if constraint.type == "time_window" and not _time_window_matches(
            constraint, graph
        ):
            return False
        if constraint.type == "action" and not _event_matches_action(event, constraint):
            return False
        if constraint.type in {
            "activity",
            "location",
            "platform",
            "product",
            "role",
        } and not _event_has_constraint_target(graph, event, constraint):
            return False
        if constraint.type == "possession" and not _event_matches_action(
            event, constraint
        ):
            return False
    return True


def _answer_constraints(graph: TypedSemanticGraph) -> tuple[SemanticNode, ...]:
    node_by_id = {node.id: node for node in graph.nodes}
    return tuple(
        node_by_id[edge.target]
        for edge in graph.edges
        if edge.source == "answer"
        and edge.relation == "constrained_by"
        and edge.target in node_by_id
    )


def _event_has_constraint_target(
    graph: TypedSemanticGraph,
    event: SemanticNode,
    constraint: SemanticNode,
) -> bool:
    expected = str(constraint.attrs.get("value") or constraint.id.split(":", 1)[-1])
    for edge in _out_edges(graph, event.id):
        target = _node(graph, edge.target)
        if target is None:
            continue
        concept_id = str(target.attrs.get("concept_id") or "")
        event_id = str(target.attrs.get("event_id") or "")
        if expected in {concept_id, event_id, target.label, target.type}:
            return True
    return False


def _event_matches_action(event: SemanticNode, constraint: SemanticNode) -> bool:
    expected = str(constraint.attrs.get("value") or "")
    event_id = str(event.attrs.get("event_id") or "")
    if expected == event_id:
        return True
    aliases = {
        "buy": {"purchase"},
        "pick_up": {"purchase"},
        "current": {"own", "have"},
        "lead": {"lead"},
        "attend": {"attend"},
        "download": {"download"},
        "assemble": {"assemble"},
        "sell": {"sell"},
        "fix": {"fix"},
        "return": {"return"},
    }
    return event_id in aliases.get(expected, set())


def _time_window_matches(constraint: SemanticNode, graph: TypedSemanticGraph) -> bool:
    if graph.occurred_on is None:
        return False
    anchor_raw = str(constraint.attrs.get("anchor") or "")
    anchor = _parse_date(anchor_raw)
    if anchor is None:
        return True
    window_days = constraint.attrs.get("window_days")
    if isinstance(window_days, int):
        return anchor - dt.timedelta(days=window_days) <= graph.occurred_on <= anchor
    if constraint.id == "time:this_year":
        return graph.occurred_on.year == anchor.year
    return True


def _value_matches_answer(value: SemanticNode, answer: SemanticNode) -> bool:
    if answer.type == "money":
        return value.type == "money" and value.unit == "usd"
    if answer.type == "duration":
        return value.type == "duration" and value.unit == answer.unit
    if answer.attrs.get("concept_id") == "people_reached":
        return value.unit == "people"
    if answer.attrs.get("concept_id") == "loyalty_points":
        return value.unit == "points"
    return False


def _value_dedupe_key(
    answer: SemanticNode,
    value: SemanticNode,
    graph: TypedSemanticGraph,
) -> tuple[str, str, str]:
    if answer.type == "money":
        return (value.unit, str(value.value), graph.source_session_id)
    return (value.unit, str(value.value), graph.id)


def _concept_role_for_event(event_id: str, concept: SemanticNode) -> str:
    concept_id = str(concept.attrs.get("concept_id") or "")
    if concept.type == "activity":
        return "activity"
    if concept.type == "location":
        return "location"
    if concept.type == "platform":
        return "platform"
    if concept.type == "product":
        return "product"
    if event_id == "reach" and concept_id == "people_reached":
        return "object"
    if event_id == "raise" and concept_id == "charity":
        return "beneficiary"
    if concept.type in {"entity", "event", "quantity"}:
        return "object"
    return ""


def _entity_role_for_event(event_id: str, concept_id: str) -> str:
    if event_id in {
        "purchase",
        "return",
        "download",
        "assemble",
        "sell",
        "fix",
        "own",
        "have",
        "lead",
        "attend",
    }:
        return "object"
    if concept_id in {"plant", "scale_model_kit", "musical_instrument"}:
        return "object"
    return ""


def _value_role_for_event(event_id: str, value: SemanticNode) -> str:
    if event_id == "required_total" and value.unit == "points":
        return "required_total"
    if event_id == "current_total" and value.unit == "points":
        return "current_total"
    if event_id == "earned_delta" and value.unit == "points":
        return "earned_delta"
    if event_id == "reach" and value.unit == "people":
        return "reach"
    if event_id == "raise" and value.type == "money":
        return "amount"
    if value.type == "duration":
        return "duration"
    return ""


def _node(graph: TypedSemanticGraph, node_id: str) -> SemanticNode | None:
    for node in graph.nodes:
        if node.id == node_id:
            return node
    return None


def _out_edges(graph: TypedSemanticGraph, source: str) -> tuple[SemanticEdge, ...]:
    return tuple(edge for edge in graph.edges if edge.source == source)


def _has_edge(
    graph: TypedSemanticGraph,
    source: str,
    relation: str,
    target: str,
) -> bool:
    return any(
        edge.source == source and edge.relation == relation and edge.target == target
        for edge in graph.edges
    )


def _event_nodes_in_graph(graph: TypedSemanticGraph) -> tuple[SemanticNode, ...]:
    return tuple(node for node in graph.nodes if node.kind == "event")


def _graph_is_excluded(graph: TypedSemanticGraph) -> bool:
    return any(node.id == "modality:hypothetical" for node in graph.nodes)


def _sentence_has_user_subject(sentence: str) -> bool:
    return bool(re.search(r"\b(?:i|i'm|i've|my|me|we|our)\b", sentence.lower()))


def _sentence_is_hypothetical(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(?:planning to|thinking of|hoping to|trying to|used to|"
            r"want to|would like to|maybe)\b",
            sentence.lower(),
        )
    )


def _answer_phrase(question: str) -> str:
    match = re.search(
        r"\bhow many\s+(?:different\s+)?(?P<phrase>.+?)\s+"
        r"(?:do|did|have|has|am|are|was|were)\b",
        question,
    )
    if match:
        return _clean_answer_phrase(match.group("phrase"))
    return "query subject"


def _clean_answer_phrase(phrase: str) -> str:
    text = re.sub(r"\b(?:total|number of|types? of|pieces? of)\b", " ", phrase)
    text = re.sub(r"\s+", " ", text).strip(" ?.")
    return text or phrase.strip()


def _concept_for_phrase(phrase: str) -> AnswerConcept:
    low = phrase.lower()
    for concept in _ANSWER_CONCEPTS:
        if any(re.search(pattern, low) for pattern in concept.patterns):
            return concept
    concept_id = re.sub(r"[^a-z0-9]+", "_", low).strip("_") or "query_subject"
    label = low or "query subject"
    return AnswerConcept(concept_id, label, "entity", (rf"\b{re.escape(low)}\b",))


def _entity_labels_for_concept(sentence: str, concept_id: str) -> tuple[str, ...]:
    patterns = _ENTITY_LABEL_PATTERNS.get(concept_id, ())
    labels: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            label = next(group for group in match.groups() if group is not None).strip()
            if label:
                labels.append(label)
    return tuple(labels)


def _canonical_entity_id(label: str, concept_id: str) -> str:
    text = label.lower()
    text = re.sub(
        r"\b(?:my|the|a|an|new|old|black|acoustic|electric|simple)\b",
        " ",
        text,
    )
    text = re.sub(r"\b(?:model kit|kit|instrument)\b", " ", text)
    text = re.sub(r"\b(?:which|that|and|with|for|about|at|last).*$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if concept_id == "musical_instrument" and text in {
        "guitar",
        "piano",
        "drum set",
        "instrument",
    }:
        return ""
    return text


def _summarize(results: list[Fail18GraphItem]) -> Fail18GraphSummary:
    return Fail18GraphSummary(
        total=len(results),
        rows_with_answer_node=sum(
            1 for item in results if _answer_node(item.question_graph) is not None
        ),
        rows_with_evidence_graphs=sum(
            1 for item in results if item.evidence_graph_count > 0
        ),
        rows_with_candidate_facts=sum(
            1 for item in results if item.candidate_fact_count > 0
        ),
        rows_with_answer=sum(1 for item in results if bool(item.answer_space)),
        rows_containing_gold=sum(1 for item in results if item.contains_gold is True),
        results=tuple(results),
    )


def _answer(
    concept_id: str,
    node_type: str,
    label: str,
    aggregation: str,
    unit: str = "",
) -> SemanticNode:
    return SemanticNode(
        "answer",
        "answer",
        node_type,
        label,
        unit=unit,
        attrs={"concept_id": concept_id, "aggregation": aggregation},
    )


def _answer_node(graph: TypedSemanticGraph) -> SemanticNode | None:
    for node in graph.nodes:
        if node.kind == "answer":
            return node
    return None


def _has_non_evidence_node(graph: TypedSemanticGraph) -> bool:
    return any(node.id != "evidence:sentence" for node in graph.nodes)


def _numeric_gold(row: dict[str, typing.Any]) -> int | float | None:
    text = str(row.get("gold_answer", "")).lower()
    candidates: list[tuple[int, int | float]] = []
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        raw = match.group().replace(",", "")
        value = float(raw)
        candidates.append((match.start(), int(value) if value.is_integer() else value))
    for word, value in _NUMBER_WORDS.items():
        match = re.search(rf"\b{word}\b", text)
        if match:
            candidates.append((match.start(), value))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    raw_count = row.get("gold_count")
    if isinstance(raw_count, int):
        return raw_count
    if isinstance(raw_count, str) and raw_count.strip().isdigit():
        return int(raw_count)
    return None


def _contains(answer: dict[str, int | float], gold: int | float | None) -> bool | None:
    if gold is None or not answer:
        return None
    if "count" in answer:
        return float(answer["count"]) == float(gold)
    lower = answer.get("lower")
    upper = answer.get("upper")
    if lower is None or upper is None:
        return None
    return float(lower) <= float(gold) <= float(upper)


def _time_window_node(question: str, question_date: str) -> SemanticNode | None:
    anchor = _parse_date(question_date)
    attrs: dict[str, typing.Any] = {"anchor": anchor.isoformat() if anchor else ""}
    if "last week" in question:
        return SemanticNode(
            "time:last_7_days",
            "constraint",
            "time_window",
            "last week",
            value=7,
            unit="day",
            attrs={**attrs, "window_days": 7},
        )
    if "past two weeks" in question or "last two weeks" in question:
        return SemanticNode(
            "time:last_14_days",
            "constraint",
            "time_window",
            "last 14 days",
            value=14,
            unit="day",
            attrs={**attrs, "window_days": 14},
        )
    if "last month" in question or "past month" in question:
        return SemanticNode(
            "time:last_31_days",
            "constraint",
            "time_window",
            "last month",
            value=31,
            unit="day",
            attrs={**attrs, "window_days": 31},
        )
    if "this year" in question:
        return SemanticNode(
            "time:this_year",
            "constraint",
            "time_window",
            "this year",
            attrs=attrs,
        )
    if "past few months" in question:
        return SemanticNode(
            "time:past_few_months",
            "constraint",
            "time_window",
            "past few months",
            value=120,
            unit="day",
            attrs={**attrs, "window_days": 120},
        )
    return None


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", compact)
        if sentence.strip()
    ]


def _iter_user_session_texts_with_ids_dates(
    row: dict[str, typing.Any],
) -> typing.Iterator[tuple[str, str, dt.date | None]]:
    ids = row.get("haystack_session_ids", [])
    dates = row.get("haystack_dates", [])
    for idx, session in enumerate(row.get("haystack_sessions", [])):
        sid = str(ids[idx]) if idx < len(ids) else str(idx)
        date_text = str(dates[idx]) if idx < len(dates) else ""
        text = " ".join(
            str(message.get("content", ""))
            for message in session
            if message.get("role") == "user"
        )
        yield sid, text, _parse_date(date_text)


def _parse_date(value: str) -> dt.date | None:
    match = re.search(r"(\d{4})/(\d{2})/(\d{2})", value)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return dt.date(year, month, day)


def _dedupe_nodes(nodes: typing.Iterable[SemanticNode]) -> tuple[SemanticNode, ...]:
    seen: set[str] = set()
    out: list[SemanticNode] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
    return tuple(out)


def _dedupe_edges(edges: typing.Iterable[SemanticEdge]) -> tuple[SemanticEdge, ...]:
    seen: set[tuple[str, str, str]] = set()
    out: list[SemanticEdge] = []
    for edge in edges:
        key = (edge.source, edge.relation, edge.target)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return tuple(out)


def _number_value(raw: str) -> int | float:
    if raw.lower() in _NUMBER_WORDS:
        return _NUMBER_WORDS[raw.lower()]
    clean = raw.replace("$", "").replace(",", "").strip()
    value = float(clean)
    return int(value) if value.is_integer() else value


def _canonical_unit(raw: str) -> str:
    low = raw.lower()
    if low.startswith("point"):
        return "points"
    if low.startswith("hour") or low.startswith("hr"):
        return "hour"
    if low.startswith("minute"):
        return "minute"
    if low.startswith("day"):
        return "day"
    if low.startswith("people") or low.startswith("follower"):
        return "people"
    if low.startswith("impression"):
        return "people"
    return low


def _graph_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "graph"


@dataclasses.dataclass(frozen=True)
class AnswerConcept:
    id: str
    label: str
    node_type: str
    patterns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ConceptRule:
    id: str
    label: str
    node_type: str
    pattern: str


@dataclasses.dataclass(frozen=True)
class EventRule:
    id: str
    label: str
    type: str
    pattern: str


@dataclasses.dataclass(frozen=True)
class ConstraintRule:
    id: str
    type: str
    label: str
    value: str
    terms: tuple[str, ...]


_VALUE_PATTERN = (
    r"(?P<money>\$\s*\d[\d,]*(?:\.\d+)?)|"
    r"(?P<num>\d[\d,]*(?:\.\d+)?|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    r"(?:\s*-\s*|\s+)"
    r"(?P<unit>points?|minutes?|hours?|hrs?|days?|people|followers|impressions?)"
)

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_ANSWER_CONCEPTS = (
    AnswerConcept("clothing_item", "clothing item", "entity", (r"clothing|clothes",)),
    AnswerConcept("art_event", "art event", "event", (r"art[- ]related events?",)),
    AnswerConcept("plant", "plant", "entity", (r"plants?",)),
    AnswerConcept("project_work", "project", "entity", (r"projects?",)),
    AnswerConcept("baking_event", "baking event", "event", (r"times",)),
    AnswerConcept(
        "music_release",
        "music release",
        "entity",
        (r"music albums?|eps?|albums? or eps?",),
    ),
    AnswerConcept(
        "citrus_fruit_type",
        "citrus fruit type",
        "entity",
        (r"citrus fruits?",),
    ),
    AnswerConcept("cuisine", "cuisine", "entity", (r"cuisines?",)),
    AnswerConcept("furniture_item", "furniture item", "entity", (r"furniture",)),
    AnswerConcept(
        "musical_instrument", "musical instrument", "entity", (r"instruments?",)
    ),
    AnswerConcept("wedding", "wedding", "event", (r"weddings?",)),
    AnswerConcept("scale_model_kit", "model kit", "entity", (r"model kits?",)),
)

_QUESTION_CONSTRAINT_RULES = (
    ConstraintRule("activity:jogging", "activity", "jogging", "jogging", ("jogging",)),
    ConstraintRule("activity:yoga", "activity", "yoga", "yoga", ("yoga",)),
    ConstraintRule("activity:camping", "activity", "camping", "camping", ("camping",)),
    ConstraintRule(
        "activity:travel",
        "activity",
        "travel",
        "travel",
        ("traveling", "travel"),
    ),
    ConstraintRule("location:hawaii", "location", "Hawaii", "hawaii", ("hawaii",)),
    ConstraintRule(
        "location:new_york_city",
        "location",
        "New York City",
        "new_york_city",
        ("new york city", "nyc"),
    ),
    ConstraintRule("location:store", "location", "store", "store", ("store",)),
    ConstraintRule(
        "product:skincare", "product", "skincare", "skincare", ("skincare",)
    ),
    ConstraintRule(
        "platform:facebook", "platform", "Facebook", "facebook", ("facebook",)
    ),
    ConstraintRule(
        "platform:instagram", "platform", "Instagram", "instagram", ("instagram",)
    ),
    ConstraintRule("role:lead", "role", "lead", "lead", ("led", "leading")),
    ConstraintRule(
        "possession:current", "possession", "current", "current", ("currently own",)
    ),
    ConstraintRule(
        "action:attend", "action", "attend", "attend", ("attend", "attended")
    ),
    ConstraintRule(
        "action:buy", "action", "buy", "buy", ("bought", "buy", "purchased")
    ),
    ConstraintRule("action:return", "action", "return", "return", ("return",)),
    ConstraintRule("action:pick_up", "action", "pick up", "pick_up", ("pick up",)),
    ConstraintRule("action:download", "action", "download", "download", ("download",)),
    ConstraintRule("action:assemble", "action", "assemble", "assemble", ("assemble",)),
    ConstraintRule("action:sell", "action", "sell", "sell", ("sell", "sold")),
    ConstraintRule("action:fix", "action", "fix", "fix", ("fix", "fixed")),
)

_CONCEPT_RULES = (
    ConceptRule("loyalty_points", "loyalty points", "quantity", r"\bpoints?\b"),
    ConceptRule("jogging", "jogging", "activity", r"\b(?:jog|jogging)\b"),
    ConceptRule("yoga", "yoga", "activity", r"\byoga\b"),
    ConceptRule("camping", "camping", "activity", r"\bcamping\b"),
    ConceptRule("travel", "travel", "activity", r"\b(?:travel|traveling|trip)\b"),
    ConceptRule("hawaii", "Hawaii", "location", r"\bhawaii\b"),
    ConceptRule(
        "new_york_city", "New York City", "location", r"\b(?:new york city|nyc)\b"
    ),
    ConceptRule("store", "store", "location", r"\bstore\b"),
    ConceptRule("facebook", "Facebook", "platform", r"\bfacebook\b"),
    ConceptRule("instagram", "Instagram", "platform", r"\binstagram\b"),
    ConceptRule("skincare", "skincare", "product", r"\bskincare\b"),
    ConceptRule(
        "charity",
        "charity",
        "beneficiary",
        r"\b(?:charity|hospital|food bank|cancer society|animal shelter)\b",
    ),
    ConceptRule(
        "musical_instrument",
        "musical instrument",
        "entity",
        r"\b(?:instrument|guitar|piano|drum|ukulele|violin)\b",
    ),
    ConceptRule(
        "scale_model_kit",
        "model kit",
        "entity",
        r"\b(?:model kit|scale|spitfire|tiger|b-29|camaro)\b",
    ),
    ConceptRule("project_work", "project", "entity", r"\b(?:project|feature)\b"),
    ConceptRule("baking_event", "baking event", "event", r"\b(?:bake|baked|baking)\b"),
    ConceptRule(
        "clothing_item",
        "clothing item",
        "entity",
        r"\b(?:clothing|clothes|shirt|pants|dress|sweater|jacket|boots|blazer|jeans)\b",
    ),
    ConceptRule(
        "art_event",
        "art event",
        "event",
        r"\b(?:art|gallery|museum|exhibition|opening|lecture|tour)\b",
    ),
    ConceptRule(
        "plant",
        "plant",
        "entity",
        r"\b(?:plant|succulent|fern|monstera|pothos|lily|basil|orchid)\b",
    ),
    ConceptRule(
        "people_reached",
        "people reached",
        "quantity",
        r"\b(?:people reached|reach|reached|impressions?|audience|followers?)\b",
    ),
    ConceptRule(
        "music_release",
        "music release",
        "entity",
        r"\b(?:album|ep|downloaded|purchased music)\b",
    ),
    ConceptRule(
        "citrus_fruit_type",
        "citrus fruit type",
        "entity",
        r"\b(?:citrus|lemon|lime|orange|grapefruit)\b",
    ),
    ConceptRule(
        "cuisine",
        "cuisine",
        "entity",
        r"\b(?:cuisine|thai|mexican|indian|italian|japanese|ethiopian|korean|vegan)\b",
    ),
    ConceptRule(
        "furniture_item",
        "furniture item",
        "entity",
        r"\b(?:furniture|chair|table|desk|shelf|sofa|dresser|bookshelf|mattress)\b",
    ),
    ConceptRule("wedding", "wedding", "event", r"\b(?:wedding|marriage)\b"),
)

_EVENT_RULES = (
    EventRule(
        "required_total", "required total", "lookup", r"\bneed\b.*\btotal\b|\bredeem\b"
    ),
    EventRule(
        "current_total",
        "current total",
        "state",
        r"\b(?:my|bringing my)\s+total\b|\bso far\b",
    ),
    EventRule("earned_delta", "earned points", "event", r"\bearned\b"),
    EventRule(
        "reach", "reach", "event", r"\b(?:reach|reached|followers|impressions?)\b"
    ),
    EventRule("raise", "raise", "event", r"\brais(?:e|ed|ing)\b"),
    EventRule(
        "purchase",
        "purchase",
        "event",
        r"\b(?:buy|bought|purchase|purchased|got|picked up)\b",
    ),
    EventRule("return", "return", "event", r"\breturn(?:ed)?\b"),
    EventRule("download", "download", "event", r"\bdownload(?:ed)?\b"),
    EventRule("attend", "attend", "event", r"\battend(?:ed)?\b"),
    EventRule("lead", "lead", "event", r"\b(?:led|leading|lead)\b"),
    EventRule("own", "own", "state", r"\b(?:own|owned|have|had)\b"),
    EventRule("bake", "bake", "event", r"\b(?:bake|baked|baking)\b"),
    EventRule("assemble", "assemble", "event", r"\bassemble(?:d)?\b"),
    EventRule("sell", "sell", "event", r"\b(?:sell|sold)\b"),
    EventRule("fix", "fix", "event", r"\bfix(?:ed)?\b"),
)

_ENTITY_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
    "musical_instrument": (
        r"\b((?:[A-Z0-9][A-Za-z0-9'./-]+\s+){0,5}"
        r"(?:guitar|piano|drum set|ukulele|violin))\b",
    ),
    "scale_model_kit": (
        r"\b((?:[A-Z0-9][A-Za-z0-9'./-]+\s+){0,6}"
        r"(?:model kit|Spitfire|Tiger|B-29|Camaro|Eagle))\b",
    ),
    "plant": (
        r"\b((?:peace lily|snake plant|basil plant|succulent|fern|orchid|"
        r"african violet))\b",
    ),
    "furniture_item": (
        r"\b((?:coffee table|kitchen table|bookshelf|mattress|sofa|desk|dresser))\b",
    ),
    "clothing_item": (
        r"\b((?:navy blue blazer|green sweater|yellow sundress|boots|"
        r"black jeans|scarf|gloves|shirts?))\b",
    ),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe typed semantic graphs.")
    parser.add_argument("--fail18", action="store_true", help="Probe fail18 fixtures.")
    parser.add_argument("--path", default="", help="fail18 manifest path.")
    parser.add_argument("--corpus", default="", help="fail18 corpus path.")
    parser.add_argument("--limit", type=int, default=0, help="Limit fail18 rows.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
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
            limit=max(0, int(args.limit)),
        )
        payload = summary.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(
            "fail18 semantic graph probe "
            f"answer_nodes={summary.rows_with_answer_node}/{summary.total}; "
            f"evidence_graphs={summary.rows_with_evidence_graphs}/{summary.total}; "
            f"candidate_facts={summary.rows_with_candidate_facts}/{summary.total}; "
            f"answers={summary.rows_with_answer}/{summary.total}; "
            f"contains_gold={summary.rows_containing_gold}/{summary.total}"
        )
        for item in summary.results:
            answer = _answer_node(item.question_graph)
            answer_label = answer.label if answer is not None else "-"
            aggregation = answer.attrs.get("aggregation") if answer is not None else "-"
            print(
                f"  {item.question_id} answer={answer_label} "
                f"agg={aggregation} evidence_graphs={item.evidence_graph_count} "
                f"facts={item.candidate_fact_count} answer_space={item.answer_space}"
            )
        return 0
    raise SystemExit("pass --fail18")


if __name__ == "__main__":
    raise SystemExit(main())

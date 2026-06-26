from __future__ import annotations

import simba.eval.semantic_graph as semantic_graph


def test_compile_question_graph_separates_answer_constraints_and_meta() -> None:
    graph = semantic_graph.compile_question_graph(
        {
            "question_id": "q",
            "question": "How many hours of jogging and yoga did I do last week?",
        }
    )

    answer = next(node for node in graph.nodes if node.kind == "answer")
    constraints = {
        (node.type, node.label) for node in graph.nodes if node.kind == "constraint"
    }
    meta = {node.label for node in graph.nodes if node.kind == "meta"}

    assert answer.type == "duration"
    assert answer.unit == "hour"
    assert answer.attrs == {"concept_id": "duration", "aggregation": "sum"}
    assert ("activity", "jogging") in constraints
    assert ("activity", "yoga") in constraints
    assert ("time_window", "last week") in constraints
    assert "how many" in meta


def test_extract_evidence_graph_lifts_sentence_values_and_events() -> None:
    graph = semantic_graph._sentence_graph(
        "I went for a 30-minute jog around the neighborhood.",
        graph_id="sentence:q:1",
        sid="s1",
        occurred_on=None,
    )

    value = next(node for node in graph.nodes if node.type == "duration")
    concepts = {
        node.attrs.get("concept_id") for node in graph.nodes if node.kind == "concept"
    }

    assert value.value == 0.5
    assert value.unit == "hour"
    assert "jogging" in concepts


def test_graph_value_alignment_sums_people_reached() -> None:
    question_graph = semantic_graph.compile_question_graph(
        {
            "question_id": "q",
            "question": ("What was the total number of people reached?"),
        }
    )
    evidence = (
        semantic_graph._sentence_graph(
            "I ran a Facebook campaign that reached around 2,000 people.",
            graph_id="sentence:q:1",
            sid="s1",
            occurred_on=None,
        ),
        semantic_graph._sentence_graph(
            (
                "I worked with an Instagram influencer who promoted it "
                "to 10,000 followers."
            ),
            graph_id="sentence:q:2",
            sid="s2",
            occurred_on=None,
        ),
    )

    answer = semantic_graph.align_question_to_evidence(
        question_graph,
        evidence,
    )

    assert answer.answer_space == {"count": 12000.0}
    assert answer.matched_fact_count == 2


def test_semantic_graph_fail18_probe_has_typed_coverage() -> None:
    summary = semantic_graph.probe_fail18()

    assert summary.total == 18
    assert summary.rows_with_answer_node == 18
    assert summary.rows_with_evidence_graphs == 18
    assert summary.rows_with_answer == 9
    assert summary.rows_containing_gold == 1

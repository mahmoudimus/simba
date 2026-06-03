"""Tests for eval report formatting + the bundled seed dataset."""

from __future__ import annotations

import simba.eval.dataset as ds
import simba.eval.report as report
import simba.eval.runner as runner


def test_default_dataset_path_exists_and_loads() -> None:
    path = report.default_dataset_path()
    assert path.exists()
    d = ds.load_dataset(path)
    assert d.name == "simba-seed"
    assert len(d.cases) >= 11
    # every gold id resolves (load_dataset validates, but assert non-trivial)
    assert d.corpus_ids() >= {"g1", "p1", "emb1"}


def test_format_report_contains_metrics() -> None:
    d = ds.Dataset(
        name="tiny",
        corpus=[ds.Memory(id="m1", content="a")],
        cases=[ds.EvalCase(id="c1", query="q", relevant_ids=["m1"])],
    )
    rep = runner.run_eval(d, lambda q: ["m1"], ks=(1, 3))
    text = report.format_report(rep)
    assert "tiny" in text
    assert "1 case" in text
    assert "recall@1" in text
    assert "mrr" in text


def test_format_report_lists_worst_cases() -> None:
    d = ds.Dataset(
        name="tiny",
        corpus=[ds.Memory(id="m1", content="a"), ds.Memory(id="m2", content="b")],
        cases=[
            ds.EvalCase(id="hit", query="q1", relevant_ids=["m1"]),
            ds.EvalCase(id="miss", query="q2", relevant_ids=["m2"]),
        ],
    )
    rep = runner.run_eval(d, lambda q: ["m1"], ks=(1,))
    text = report.format_report(rep, top_n_worst=1)
    assert "miss" in text  # the failing case is surfaced

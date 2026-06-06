"""Tests for the append-only baseline store (eval result history)."""

from __future__ import annotations


def test_append_creates_file_on_first_write(tmp_path) -> None:
    from simba.eval.benchmarks.baseline_store import append_baseline, load_baselines

    p = append_baseline(
        "locomo_qa",
        {"overall": {"accuracy": 0.72}},
        root=tmp_path,
        metadata={"k": 10},
    )
    assert p.exists()
    entries = load_baselines("locomo_qa", root=tmp_path)
    assert len(entries) == 1
    assert entries[0]["report"]["overall"]["accuracy"] == 0.72
    assert entries[0]["metadata"]["k"] == 10
    assert "ts" in entries[0]


def test_append_is_append_only(tmp_path) -> None:
    from simba.eval.benchmarks.baseline_store import append_baseline, load_baselines

    append_baseline("locomo_qa", {"overall": {"accuracy": 0.70}}, root=tmp_path)
    append_baseline("locomo_qa", {"overall": {"accuracy": 0.73}}, root=tmp_path)
    entries = load_baselines("locomo_qa", root=tmp_path)
    assert len(entries) == 2
    assert entries[0]["report"]["overall"]["accuracy"] == 0.70
    assert entries[1]["report"]["overall"]["accuracy"] == 0.73


def test_different_names_go_to_different_files(tmp_path) -> None:
    from simba.eval.benchmarks.baseline_store import append_baseline

    p1 = append_baseline("locomo_qa", {}, root=tmp_path)
    p2 = append_baseline("locomo_recall", {}, root=tmp_path)
    assert p1 != p2
    assert p1.name == "locomo_qa.jsonl"
    assert p2.name == "locomo_recall.jsonl"


def test_baseline_dir_config_respected(tmp_path) -> None:
    import simba.config
    import simba.eval.config

    simba.config.set_value(
        "eval", "baseline_dir", ".custom/baselines", scope="local", root=tmp_path
    )
    from simba.eval.benchmarks.baseline_store import append_baseline

    p = append_baseline("test", {}, root=tmp_path)
    assert ".custom/baselines" in str(p)

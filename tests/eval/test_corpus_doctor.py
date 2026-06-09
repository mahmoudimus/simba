"""Tests for the corpus_doctor inject->detect->score harness (Phase-7)."""

from __future__ import annotations

import simba.config
import simba.eval.corpus_doctor as cd


def _edge(eid: int, subj: str, pred: str, obj: str) -> dict:
    return {
        "id": eid,
        "subject": subj,
        "predicate": pred,
        "object": obj,
        "valid_from": "",
        "valid_to": None,
    }


def _base_corpus() -> list[dict]:
    return [
        _edge(1, "alice", "uses", "vim"),
        _edge(2, "bob", "prefers", "emacs"),
        _edge(3, "carol", "fixes", "build"),
        _edge(4, "dave", "uses", "rust"),
    ]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_corpus_doctor_section_registered() -> None:
    assert "corpus_doctor" in simba.config.list_sections()


def test_config_defaults_disabled() -> None:
    cfg = cd.CorpusDoctorConfig()
    assert cfg.enabled is False
    assert cfg.seed == 42
    assert cfg.num_corruptions_per_corpus == 5
    assert cfg.detection_threshold == 0.5
    assert "antonym" in cfg.contradiction_types


def test_config_kinds_tuple_parses() -> None:
    cfg = cd.CorpusDoctorConfig(contradiction_types="antonym, duplicate")
    assert cfg.kinds_tuple() == ("antonym", "duplicate")


# --------------------------------------------------------------------------
# inject
# --------------------------------------------------------------------------


def test_inject_adds_edges_and_ground_truth() -> None:
    corpus = _base_corpus()
    corrupted, truth = cd.inject(corpus, kinds=["antonym"], count=1, seed=42)
    assert len(corrupted) == len(corpus) + 1
    assert len(truth) == 1
    assert truth[0].contradiction_type == "antonym"
    # the injected edge id must be present in the corrupted corpus
    injected_ids = {e["id"] for e in corrupted}
    assert truth[0].edge_id_b in injected_ids


def test_inject_does_not_mutate_input() -> None:
    corpus = _base_corpus()
    before = [dict(e) for e in corpus]
    cd.inject(corpus, kinds=["antonym"], count=1, seed=1)
    assert corpus == before


def test_inject_is_deterministic_for_fixed_seed() -> None:
    a_corrupted, a_truth = cd.inject(_base_corpus(), kinds=["antonym"], count=2, seed=7)
    b_corrupted, b_truth = cd.inject(_base_corpus(), kinds=["antonym"], count=2, seed=7)
    assert a_corrupted == b_corrupted
    assert [t.edge_id_a for t in a_truth] == [t.edge_id_a for t in b_truth]
    assert [t.edge_id_b for t in a_truth] == [t.edge_id_b for t in b_truth]


def test_inject_antonym_pair_shares_endpoints_with_opposite_predicate() -> None:
    corpus = _base_corpus()
    corrupted, truth = cd.inject(corpus, kinds=["antonym"], count=1, seed=42)
    by_id = {e["id"]: e for e in corrupted}
    orig = by_id[truth[0].edge_id_a]
    inj = by_id[truth[0].edge_id_b]
    assert inj["subject"] == orig["subject"]
    assert inj["object"] == orig["object"]
    assert inj["predicate"] != orig["predicate"]


def test_inject_duplicate_pair_is_identical_triple() -> None:
    corpus = _base_corpus()
    corrupted, truth = cd.inject(corpus, kinds=["duplicate"], count=1, seed=3)
    by_id = {e["id"]: e for e in corrupted}
    orig = by_id[truth[0].edge_id_a]
    inj = by_id[truth[0].edge_id_b]
    assert (inj["subject"], inj["predicate"], inj["object"]) == (
        orig["subject"],
        orig["predicate"],
        orig["object"],
    )
    assert truth[0].contradiction_type == "duplicate"


def test_inject_temporal_overlap_same_predicate_overlapping_window() -> None:
    corpus = _base_corpus()
    corrupted, truth = cd.inject(corpus, kinds=["temporal_overlap"], count=1, seed=5)
    by_id = {e["id"]: e for e in corrupted}
    orig = by_id[truth[0].edge_id_a]
    inj = by_id[truth[0].edge_id_b]
    assert inj["subject"] == orig["subject"]
    assert inj["object"] == orig["object"]
    assert truth[0].contradiction_type == "temporal_overlap"


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def _truth(eid_a: int, eid_b: int) -> cd.InjectionResult:
    return cd.InjectionResult(
        corpus_id="t",
        edge_id_a=eid_a,
        edge_id_b=eid_b,
        contradiction_type="antonym",
        injected_edge={},
        original_edge=None,
    )


def test_score_tp_when_one_edge_of_pair_flagged() -> None:
    m = cd.score({10}, [_truth(10, 11)])
    assert m.true_positives == 1
    assert m.false_negatives == 0
    assert m.recall == 1.0


def test_score_fn_when_pair_missed() -> None:
    m = cd.score({99}, [_truth(10, 11)])
    assert m.true_positives == 0
    assert m.false_negatives == 1
    assert m.recall == 0.0


def test_score_fp_counts_unexpected_flags() -> None:
    m = cd.score({10, 99}, [_truth(10, 11)])
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.precision == 0.5


def test_score_perfect_detection_f1_one() -> None:
    m = cd.score({10, 20}, [_truth(10, 11), _truth(20, 21)])
    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_score_empty_truth_and_detections_is_zeroed() -> None:
    m = cd.score(set(), [])
    assert m.true_positives == 0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


# --------------------------------------------------------------------------
# run_corpus_doctor_eval (end-to-end)
# --------------------------------------------------------------------------


def test_run_disabled_returns_noop() -> None:
    cfg = cd.CorpusDoctorConfig(enabled=False)
    m = cd.run_corpus_doctor_eval([], detect_fn=lambda _e: set(), cfg=cfg)
    assert m.true_positives == 0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_run_roundtrip_perfect_detector_recall_one() -> None:
    cfg = cd.CorpusDoctorConfig(
        enabled=True, num_corruptions_per_corpus=2, contradiction_types="antonym"
    )

    captured: dict = {}

    def perfect_detect(edges: list[dict]) -> set[int]:
        # flag the highest ids — the injected edges are appended last
        captured["edges"] = edges
        ids = sorted(e["id"] for e in edges)
        return set(ids[-cfg.num_corruptions_per_corpus :])

    m = cd.run_corpus_doctor_eval(_base_corpus(), detect_fn=perfect_detect, cfg=cfg)
    assert m.recall == 1.0
    assert m.false_negatives == 0
    # detector saw the corrupted (larger) corpus
    assert len(captured["edges"]) > len(_base_corpus())


def test_run_roundtrip_blind_detector_recall_zero() -> None:
    cfg = cd.CorpusDoctorConfig(
        enabled=True, num_corruptions_per_corpus=2, contradiction_types="antonym"
    )
    m = cd.run_corpus_doctor_eval(_base_corpus(), detect_fn=lambda _e: set(), cfg=cfg)
    assert m.true_positives == 0
    assert m.recall == 0.0

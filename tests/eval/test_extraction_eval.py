"""Tests for the extraction-quality evaluator scaffold (gold + metrics)."""

from __future__ import annotations

import simba.eval.extraction_eval as ee


def test_keep_drop_prf_perfect():
    r = ee.keep_drop_prf(
        ["uses jwt", "peewee isolation bug"], ["uses jwt", "peewee isolation bug"]
    )
    assert r["precision"] == 1.0
    assert r["recall"] == 1.0
    assert r["f1"] == 1.0


def test_keep_drop_prf_partial():
    r = ee.keep_drop_prf(
        ["uses jwt", "irrelevant"], ["uses jwt", "peewee isolation bug"]
    )
    assert r["precision"] == 0.5
    assert r["recall"] == 0.5


def test_keep_drop_prf_normalizes_case_and_whitespace():
    r = ee.keep_drop_prf(["  Uses   JWT "], ["uses jwt"])
    assert r["recall"] == 1.0


def test_keep_drop_prf_empty_prediction_is_zero_recall():
    r = ee.keep_drop_prf([], ["a", "b"])
    assert r["recall"] == 0.0


def test_recurrence_hit_rate_half():
    cached = ["auth uses jwt"]
    later = ["how does auth work", "what is the db name"]
    hit = ee.recurrence_hit_rate(
        cached, later, match=lambda q, c: "auth" in q and "auth" in c
    )
    assert hit == 0.5


def test_recurrence_hit_rate_empty_is_zero():
    assert ee.recurrence_hit_rate([], [], match=lambda q, c: True) == 0.0


def test_evaluate_goldset_micro_average():
    gold = [
        ee.GoldWindow(window="w1", expected_keep=["A", "B"]),
        ee.GoldWindow(window="w2", expected_keep=["C"]),
    ]
    # predictor keeps A and C, misses B → recall 2/3
    res = ee.evaluate(gold, predict=lambda w: ["A"] if w == "w1" else ["C"])
    assert res["recall"] == round(2 / 3, 3)

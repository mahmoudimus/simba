"""Tests for the eval config section."""

from __future__ import annotations

import simba.config
import simba.eval.config as ec


def test_eval_section_registered() -> None:
    assert "eval" in simba.config.list_sections()


def test_defaults() -> None:
    cfg = ec.EvalConfig()
    assert cfg.ks == "1,3,5,10"
    assert cfg.dataset == ""


def test_ks_tuple_parses() -> None:
    assert ec.EvalConfig(ks="1,3,5").ks_tuple() == (1, 3, 5)


def test_ks_tuple_ignores_blanks_and_junk() -> None:
    assert ec.EvalConfig(ks="1, ,3,x,5").ks_tuple() == (1, 3, 5)


def test_ks_tuple_fallback_when_empty() -> None:
    assert ec.EvalConfig(ks="").ks_tuple() == (1, 3, 5, 10)

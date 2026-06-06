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


def test_ircot_defaults() -> None:
    cfg = ec.EvalConfig()
    assert cfg.ircot_enabled is False
    assert cfg.ircot_max_steps == 4
    assert cfg.ircot_k_per_step == 3
    assert cfg.ircot_k_final == 10


def test_load_config_applies_overrides() -> None:
    cfg = ec.load_config(ircot_enabled=True, ircot_max_steps=2)
    assert cfg.ircot_enabled is True
    assert cfg.ircot_max_steps == 2


def test_load_config_ignores_none_overrides() -> None:
    cfg = ec.load_config(ircot_enabled=None)
    assert cfg.ircot_enabled is False

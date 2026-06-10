"""Tests for the workflow @configurable section."""

from __future__ import annotations

import dataclasses

import simba.config
import simba.workflow.config as wcfg


def test_section_registered():
    assert simba.config.list_sections().get("workflow") is wcfg.WorkflowConfig


def test_defaults():
    cfg = wcfg.WorkflowConfig()
    assert cfg.default_max_attempts == 3
    assert cfg.retry_backoff_base_seconds == 2.0
    assert cfg.retry_backoff_max_seconds == 300.0
    assert cfg.stale_after_seconds == 3600
    assert cfg.fan_out_max_workers == 8
    assert cfg.worker_mode == "detached"


def test_loadable_via_config():
    cfg = simba.config.load("workflow")
    assert cfg.default_max_attempts == 3


def test_all_fields_are_dataclass_fields():
    names = {f.name for f in dataclasses.fields(wcfg.WorkflowConfig)}
    assert names == {
        "default_max_attempts",
        "retry_backoff_base_seconds",
        "retry_backoff_max_seconds",
        "stale_after_seconds",
        "fan_out_max_workers",
        "worker_mode",
    }

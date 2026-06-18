"""Tests for cheap UserPromptSubmit retrieval triage."""

from __future__ import annotations

import simba.hooks.recall_triage as triage


def test_acknowledgement_skips_retrieval() -> None:
    result = triage.classify("thanks!")

    assert result.decision == "skip"
    assert result.reason == "acknowledgement"
    assert result.should_retrieve is False


def test_current_time_skips_retrieval() -> None:
    result = triage.classify("what is the current time?")

    assert result.decision == "skip"
    assert result.reason == "current_time_or_date"


def test_repo_or_memory_cue_retrieves() -> None:
    result = triage.classify("what is next from the borrow roadmap?")

    assert result.decision == "recall"
    assert result.reason == "memory_or_repo_cue"
    assert result.should_retrieve is True


def test_unknown_prompt_is_uncertain_and_retrieves() -> None:
    result = triage.classify("please think carefully about this")

    assert result.decision == "uncertain"
    assert result.should_retrieve is True


def test_render_outputs_diagnostics_block() -> None:
    out = triage.render(triage.RecallTriage("skip", "acknowledgement"))

    assert "<recall-triage>" in out
    assert "decision: skip" in out
    assert "reason: acknowledgement" in out

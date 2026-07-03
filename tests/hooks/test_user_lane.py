"""Cross-project user lane (spec 33 Phase 3).

The audit's "smarter about ME" gap: PREFERENCE is 4% of the corpus, almost
all project directives, recallable only inside the project that learned them
— the harness has no portable user model. When ``memory.user_lane_enabled``
is on, UserPromptSubmit adds ONE cross-project PREFERENCE slot at a high
similarity floor, rendered as ``source="user-model"``. Default off →
byte-identical.
"""

from __future__ import annotations

import json
import unittest.mock

import pytest

import simba.hooks.config as hooks_config
import simba.hooks.user_prompt_submit as ups


class _MemCfg:
    def __init__(self, enabled: bool) -> None:
        self.user_lane_enabled = enabled
        self.user_lane_min_similarity = 0.55
        self.user_lane_max_results = 1


@pytest.fixture
def _hook_cfg(monkeypatch):
    cfg = hooks_config.HooksConfig(prompt_min_length=1)
    monkeypatch.setattr(ups, "_cfg", lambda: cfg)
    return cfg


def _run(monkeypatch, *, lane_enabled: bool, recall_side_effect) -> str:
    monkeypatch.setattr(
        "simba.hooks._memory_client._memory_cfg", lambda: _MemCfg(lane_enabled)
    )
    with unittest.mock.patch(
        "simba.hooks._memory_client.recall_memories",
        side_effect=recall_side_effect,
    ) as mock_recall:
        out = ups.main(
            {
                "prompt": "how should I structure this migration",
                "cwd": None,  # skip CLAUDE.md/RAG side effects
                "session_id": "sess-lane",
            }
        )
    ctx = json.loads(out)["hookSpecificOutput"].get("additionalContext", "")
    return ctx, mock_recall


def test_user_lane_appends_cross_project_preference(monkeypatch, _hook_cfg) -> None:
    main_mem = [
        {"id": "m1", "type": "GOTCHA", "content": "project fact", "similarity": 0.8}
    ]
    lane_mem = [
        {
            "id": "u1",
            "type": "PREFERENCE",
            "content": "User requires a triage table before batch-acting",
            "similarity": 0.7,
        }
    ]
    ctx, mock_recall = _run(
        monkeypatch,
        lane_enabled=True,
        recall_side_effect=[main_mem, lane_mem],
    )
    assert 'source="user-model"' in ctx
    assert "triage table" in ctx
    # Second call is the lane: cross-project (no project_path), PREFERENCE
    # only, its own floor and cap.
    lane_call = mock_recall.call_args_list[1]
    assert lane_call.kwargs["project_path"] is None
    assert lane_call.kwargs["filters"] == {"types": ["PREFERENCE"]}
    assert lane_call.kwargs["min_similarity"] == 0.55
    assert lane_call.kwargs["max_results"] == 1


def test_user_lane_dedupes_against_main_recall(monkeypatch, _hook_cfg) -> None:
    shared = {
        "id": "m1",
        "type": "PREFERENCE",
        "content": "User wants baselines committed first",
        "similarity": 0.8,
    }
    ctx, _ = _run(
        monkeypatch,
        lane_enabled=True,
        recall_side_effect=[[shared], [shared]],
    )
    assert ctx.count("baselines committed first") == 1


def test_user_lane_off_by_default_single_recall(monkeypatch, _hook_cfg) -> None:
    main_mem = [
        {"id": "m1", "type": "GOTCHA", "content": "project fact", "similarity": 0.8}
    ]
    ctx, mock_recall = _run(
        monkeypatch,
        lane_enabled=False,
        recall_side_effect=[main_mem],
    )
    assert mock_recall.call_count == 1
    assert 'source="user-model"' not in ctx


def test_user_lane_fires_even_when_main_recall_empty(monkeypatch, _hook_cfg) -> None:
    lane_mem = [
        {
            "id": "u2",
            "type": "PREFERENCE",
            "content": "User prefers explicit dry-run flags",
            "similarity": 0.66,
        }
    ]
    ctx, _ = _run(
        monkeypatch,
        lane_enabled=True,
        recall_side_effect=[[], lane_mem],
    )
    assert 'source="user-model"' in ctx
    assert "dry-run flags" in ctx

"""Tests for the per-turn preflight flag (spec 28).

Reuses the spec-25 signal-flag tempfile plumbing: a tiny session-scoped JSON flag
recording whether a ``simba preflight`` fired this turn, so the PreToolUse gate
can require it before a mutating tool. Set by preflight, reset on UserPromptSubmit
(turn boundary).
"""

from __future__ import annotations

import simba.guardian.preflight_flag as pf


class TestPreflightFlag:
    def test_path_is_session_scoped(self):
        assert pf.flag_path("s1") != pf.flag_path("s2")
        assert "s1" in pf.flag_path("s1").name

    def test_set_then_present_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.set_preflight("sess", task="review PR")
        assert pf.preflight_ran("sess") is True

    def test_unset_session_is_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        assert pf.preflight_ran("never") is False

    def test_reset_clears(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.set_preflight("sess")
        assert pf.preflight_ran("sess") is True
        pf.reset_preflight("sess")
        assert pf.preflight_ran("sess") is False
        pf.reset_preflight("sess")  # idempotent

    def test_corrupt_flag_is_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.flag_path("sess").write_text("not json{")
        assert pf.preflight_ran("sess") is False

    def test_empty_session_id_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.set_preflight("")  # no-op, never raises
        assert pf.preflight_ran("") is False
        pf.reset_preflight("")


class TestMandateArming:
    def test_arm_then_armed_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.arm_mandate("sess")
        assert pf.mandate_armed("sess") is True

    def test_unarmed_is_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        assert pf.mandate_armed("sess") is False

    def test_arm_and_preflight_are_independent_flags(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.arm_mandate("sess")
        assert pf.mandate_armed("sess") is True
        assert pf.preflight_ran("sess") is False  # distinct flag

    def test_reset_mandate_clears(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.arm_mandate("sess")
        pf.reset_mandate("sess")
        assert pf.mandate_armed("sess") is False
        pf.reset_mandate("sess")  # idempotent

    def test_empty_session_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pf, "_TMP_DIR", tmp_path)
        pf.arm_mandate("")
        assert pf.mandate_armed("") is False
        pf.reset_mandate("")

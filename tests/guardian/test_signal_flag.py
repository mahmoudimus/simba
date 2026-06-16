"""Tests for the guardian per-session signal-flag plumbing.

The flag records whether the model's *previous* response carried the
``[✓ rules]`` signal, so UserPromptSubmit can skip re-injecting the CORE block
when the rules are still present (Proposal A, spec 25).
"""

from __future__ import annotations

import simba.guardian.signal_flag as sf


class TestFlagPath:
    def test_path_is_session_scoped(self):
        p1 = sf.flag_path("sess-1")
        p2 = sf.flag_path("sess-2")
        assert p1 != p2
        assert "sess-1" in p1.name
        assert "sess-2" in p2.name

    def test_empty_session_id_is_handled(self):
        # An empty session id still yields a stable, distinct path.
        p = sf.flag_path("")
        assert p.name


class TestRecordAndRead:
    def test_record_present_then_read_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        sf.record_signal("sess-x", present=True)
        assert sf.signal_present("sess-x") is True

    def test_record_absent_then_read_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        sf.record_signal("sess-x", present=False)
        assert sf.signal_present("sess-x") is False

    def test_no_flag_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        # Never recorded → treat as "not present" (caller fail-opens to inject).
        assert sf.signal_present("never-seen") is False

    def test_corrupt_flag_returns_false(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        sf.flag_path("sess-x").write_text("not json{")
        assert sf.signal_present("sess-x") is False

    def test_reset_removes_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        sf.record_signal("sess-x", present=True)
        assert sf.flag_path("sess-x").exists()
        sf.reset_signal("sess-x")
        assert not sf.flag_path("sess-x").exists()
        # Reset is idempotent — removing a missing flag is a no-op.
        sf.reset_signal("sess-x")


class TestShouldInject:
    """should_inject() encodes the Proposal A decision (fail-open)."""

    def test_first_prompt_no_flag_injects(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        assert sf.should_inject("fresh-session") is True

    def test_prior_signal_present_skips(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        sf.record_signal("s", present=True)
        assert sf.should_inject("s") is False

    def test_prior_signal_missing_injects(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        sf.record_signal("s", present=False)
        assert sf.should_inject("s") is True

    def test_empty_session_id_injects(self, monkeypatch, tmp_path):
        # No session id → can't track decay → fail-open to inject.
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)
        assert sf.should_inject("") is True

    def test_read_error_fails_open(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sf, "_TMP_DIR", tmp_path)

        def boom(_session_id):
            raise OSError("disk gone")

        monkeypatch.setattr(sf, "signal_present", boom)
        assert sf.should_inject("s") is True

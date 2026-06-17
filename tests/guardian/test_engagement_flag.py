"""Per-turn engagement record (spec 27): did simba surface activity this turn?"""

from __future__ import annotations

import simba.guardian.engagement_flag as ef


class TestEngagementFlag:
    def test_records_and_reads_activity(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        ef.record_engagement("sess-A", ledger="🦁☑ recalled 2 (top 0.5)")
        assert ef.engaged("sess-A") is True
        assert ef.last_ledger("sess-A") == "🦁☑ recalled 2 (top 0.5)"

    def test_unrecorded_session_not_engaged(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        assert ef.engaged("never-seen") is False
        assert ef.last_ledger("never-seen") == ""

    def test_no_session_id_is_noop(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        ef.record_engagement("", ledger="🦁☑ idle (nothing matched)")
        assert ef.engaged("") is False

    def test_reset_clears(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        ef.record_engagement("sess-R", ledger="🦁☑ recalled 1 (top 0.9)")
        ef.reset_engagement("sess-R")
        assert ef.engaged("sess-R") is False

    def test_corrupt_flag_fails_soft(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        ef.flag_path("sess-C").write_text("{not json")
        assert ef.engaged("sess-C") is False
        assert ef.last_ledger("sess-C") == ""

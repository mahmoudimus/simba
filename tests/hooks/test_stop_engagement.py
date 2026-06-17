"""Stop echo-verify of the 🦁☑ engagement marker (spec 27, Phase B)."""

from __future__ import annotations

import json

import simba.hooks.stop


class _MarkerCfg:
    engagement_marker_enabled = True
    reasoning_verify_enabled = False


def _patch_cfg(monkeypatch) -> None:
    monkeypatch.setattr(
        simba.hooks.stop, "_hooks_cfg", lambda: _MarkerCfg(), raising=False
    )


class TestStopEchoVerify:
    def test_off_by_default_no_echo_check(self, tmp_path, monkeypatch) -> None:
        # Characterization: lever OFF (default) → byte-identical to today. Even with
        # a recorded engagement, no echo nudge is emitted.
        import simba.guardian.engagement_flag as ef
        import simba.hooks.config

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        # Pin the cfg to the real default (both spec-27 levers off) so an ambient
        # dogfood .simba/config.toml can't flip on the echo-verify / block path.
        monkeypatch.setattr(
            simba.hooks.stop,
            "_hooks_cfg",
            lambda: simba.hooks.config.HooksConfig(),
            raising=False,
        )
        ef.record_engagement("sess-off", ledger="🦁☑ recalled 2 (top 0.5)")
        out = simba.hooks.stop.main(
            {
                "response": "answer with no marker",
                "cwd": str(tmp_path),
                "session_id": "sess-off",
            }
        )
        assert json.loads(out) == {}

    def test_on_flags_missing_echo_when_simba_acted(
        self, tmp_path, monkeypatch
    ) -> None:
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        _patch_cfg(monkeypatch)
        ef.record_engagement("sess-miss", ledger="🦁☑ recalled 2 (top 0.5)")
        out = simba.hooks.stop.main(
            {
                "response": "Here is my answer [✓ rules]",
                "cwd": str(tmp_path),
                "session_id": "sess-miss",
            }
        )
        reason = json.loads(out).get("stopReason", "")
        assert "🦁☑" in reason

    def test_on_no_flag_when_echo_present(self, tmp_path, monkeypatch) -> None:
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        _patch_cfg(monkeypatch)
        ef.record_engagement("sess-ok", ledger="🦁☑ recalled 2 (top 0.5)")
        out = simba.hooks.stop.main(
            {
                "response": "Done. 🦁☑ recalled 2 (top 0.5) [✓ rules]",
                "cwd": str(tmp_path),
                "session_id": "sess-ok",
            }
        )
        # Echo present → no engagement nudge in the stopReason.
        reason = json.loads(out).get("stopReason", "")
        assert "echo" not in reason.lower()

    def test_on_no_flag_when_simba_idle(self, tmp_path, monkeypatch) -> None:
        # simba never recorded engagement this turn → nothing to verify.
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        _patch_cfg(monkeypatch)
        out = simba.hooks.stop.main(
            {
                "response": "answer with no marker",
                "cwd": str(tmp_path),
                "session_id": "sess-idle",
            }
        )
        reason = json.loads(out).get("stopReason", "")
        assert "🦁☑" not in reason

    def test_on_clears_record_after_check(self, tmp_path, monkeypatch) -> None:
        # The per-turn record ages out after Stop reads it.
        import simba.guardian.engagement_flag as ef

        monkeypatch.setattr(ef, "_TMP_DIR", tmp_path)
        _patch_cfg(monkeypatch)
        ef.record_engagement("sess-clr", ledger="🦁☑ recalled 1 (top 0.9)")
        simba.hooks.stop.main(
            {
                "response": "done",
                "cwd": str(tmp_path),
                "session_id": "sess-clr",
            }
        )
        assert ef.engaged("sess-clr") is False

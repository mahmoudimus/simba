"""hooks/recall_query: strip harness + simba boilerplate before embedding a query."""

from __future__ import annotations

import types

import simba.hooks.reasoning_verify
import simba.hooks.recall_query as rq
import simba.hooks.user_prompt_submit

_NOTIFICATION = """<task-notification>
<task-id>a63540c21b6003d6f</task-id>
<tool-use-id>toolu_01JNDeq6</tool-use-id>
<output-file>/private/tmp/x/tasks/a6.output</output-file>
<status>completed</status>
<summary>Agent "Review memory repos group A" finished</summary>
<note>A task-notification fires each time this agent stops.</note>
<result>billion-context moved blocks to egress_block_state.</result>
<usage><subagent_tokens>83825</subagent_tokens></usage>
</task-notification>"""


class TestClean:
    def test_plain_text_unchanged(self) -> None:
        text = "why does the daemon RSS climb after a reembed?"
        assert rq.clean(text) == text

    def test_empty(self) -> None:
        assert rq.clean("") == ""

    def test_drops_engagement_line_and_signal(self) -> None:
        reply = (
            "🦁☑ recalled 3 (top 0.47)\nThe agent working on #352 is done.\n\n[✓ rules]"
        )
        assert rq.clean(reply) == "The agent working on #352 is done."

    def test_engagement_line_inline_with_text_is_dropped_whole_line(self) -> None:
        # The ledger is always its own line; a line starting with it is all marker.
        assert rq.clean("🦁☑ idle (nothing matched) [✓ rules]") == ""

    def test_drops_system_reminder_block(self) -> None:
        text = "<system-reminder>\nSYSTEM NOTIFICATION\n</system-reminder>\nfix the bug"
        assert rq.clean(text) == "fix the bug"

    def test_task_notification_keeps_summary_and_result_only(self) -> None:
        out = rq.clean(_NOTIFICATION)
        assert out == (
            'Agent "Review memory repos group A" finished\n'
            "billion-context moved blocks to egress_block_state."
        )
        for noise in ("task-id", "toolu_", "/private/tmp", "subagent_tokens", "<"):
            assert noise not in out

    def test_unterminated_notification_still_cleaned(self) -> None:
        text = "<task-notification><status>completed</status><result>partial report"
        assert rq.clean(text) == "partial report"

    def test_drops_token_footer(self) -> None:
        assert rq.clean("hello world\n[simba: ~320 tokens | ✓ rules]") == "hello world"

    def test_idempotent(self) -> None:
        once = rq.clean("🦁☑ x\n" + _NOTIFICATION + "\n[✓ rules]")
        assert rq.clean(once) == once


class TestLever:
    def test_off_returns_text_verbatim(self) -> None:
        cfg = types.SimpleNamespace(recall_query_clean_enabled=False)
        text = "🦁☑ idle\n[✓ rules]"
        assert rq.recall_query(text, cfg) is text

    def test_missing_attr_is_off(self) -> None:
        assert rq.recall_query("🦁☑ idle", object()) == "🦁☑ idle"

    def test_on_cleans(self) -> None:
        cfg = types.SimpleNamespace(recall_query_clean_enabled=True)
        assert rq.recall_query("🦁☑ idle\nreal words", cfg) == "real words"

    def test_default_off(self) -> None:
        import simba.hooks.config

        assert simba.hooks.config.HooksConfig().recall_query_clean_enabled is False


class TestReasoningVerifyQuery:
    def _cfg(self, on: bool):
        return types.SimpleNamespace(
            recall_query_clean_enabled=on,
            pitfall_gate_types="FAILURE",
            pitfall_gate_max_results=5,
        )

    def _run(self, monkeypatch, on: bool, response: str) -> list[str]:
        seen: list[str] = []

        def fake_recall(query, **_):
            seen.append(query)
            return []

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", fake_recall)
        monkeypatch.setattr("simba.memory.pitfall.pitfall_note", lambda *a, **k: "")
        simba.hooks.reasoning_verify.verify(response, None, self._cfg(on))
        return seen

    def test_on_embeds_reply_without_markers(self, monkeypatch) -> None:
        seen = self._run(monkeypatch, True, "🦁☑ recalled 1\nskip the test\n[✓ rules]")
        assert seen == ["skip the test"]

    def test_on_marker_only_reply_skips_recall(self, monkeypatch) -> None:
        assert self._run(monkeypatch, True, "🦁☑ idle\n\n[✓ rules]") == []

    def test_off_embeds_reply_verbatim(self, monkeypatch) -> None:
        reply = "🦁☑ recalled 1\nskip the test\n[✓ rules]"
        assert self._run(monkeypatch, False, reply) == [reply]


class TestPromptRecallQuery:
    def _run(self, monkeypatch, on: bool, prompt: str) -> list[str]:
        seen: list[str] = []
        import simba.hooks.config

        base = simba.hooks.config.HooksConfig()
        base.recall_query_clean_enabled = on
        monkeypatch.setattr(simba.hooks.user_prompt_submit, "_cfg", lambda: base)
        monkeypatch.setattr(
            simba.hooks.user_prompt_submit,
            "_recall_prompt",
            lambda query, project_path, cfg: seen.append(query) or [],
        )
        simba.hooks.user_prompt_submit.run({"prompt": prompt, "session_id": ""})
        return seen

    def test_on_notification_prompt_recalls_on_its_content(self, monkeypatch) -> None:
        seen = self._run(monkeypatch, True, _NOTIFICATION)
        assert len(seen) == 1
        assert seen[0].startswith('Agent "Review memory repos group A" finished')

    def test_on_boilerplate_only_prompt_skips_recall(self, monkeypatch) -> None:
        prompt = "<system-reminder>x</system-reminder>"
        assert self._run(monkeypatch, True, prompt) == []

    def test_off_recalls_on_raw_prompt(self, monkeypatch) -> None:
        assert self._run(monkeypatch, False, _NOTIFICATION) == [_NOTIFICATION]


class TestTracksEmitters:
    """The cleaner reads the emitters' constants, so renaming a marker at its
    source changes what is stripped. These feed real emitter output through."""

    def test_strips_every_prompt_ledger_form(self) -> None:
        import simba.hooks.engagement as eng

        for line in (
            eng.prompt_ledger(memory_count=3, top_similarity=0.47),
            eng.prompt_ledger(memory_count=0, top_similarity=0.0),
            eng.append_gate_action(
                eng.prompt_ledger(memory_count=1, top_similarity=0.5),
                eng.gate_action_label("warn", "rm -rf build"),
            ),
        ):
            assert rq.clean(f"{line}\nreal words") == "real words"

    def test_strips_the_footer_the_hooks_emit(self) -> None:
        import simba.hooks.engagement as eng

        for tags in ("~320 tokens", "~320 tokens | ✓ rules", "~9 tokens injected"):
            assert rq.clean(f"real words\n{eng.footer(tags)}") == "real words"

    def test_strips_the_signal_the_guardian_checks(self) -> None:
        import simba.guardian.check_signal as cs

        reply = f"done\n\n{cs.SIGNAL}"
        assert cs.check_signal(reply)
        assert rq.clean(reply) == "done"

    def test_patterns_follow_a_renamed_marker(self, monkeypatch) -> None:
        import importlib

        import simba.hooks.engagement as eng

        monkeypatch.setattr(eng, "MARKER", "🐯✔")
        reloaded = importlib.reload(rq)
        try:
            assert (
                reloaded.clean("🐯✔ recalled 2 (top 0.40)\nreal words") == "real words"
            )
        finally:
            monkeypatch.undo()
            importlib.reload(rq)

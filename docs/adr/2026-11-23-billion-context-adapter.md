# Billion Context adapter

Status: accepted, 2026-11-23. Code: `src/simba/integrations/billion_context.py`,
called from the Stop hook (`hooks/stop.py`) and PreToolUse
(`hooks/pre_tool_use.py`). Tickets: sim-i8kc, sim-oorr, sim-mudk, sim-kq35;
proposal for sim-5f4u below. Source studied: billion-context 0.1.216
(`docs/plugin-protocol.md`, `docs/configuration.md`, `src/codex-compact.ts`,
`src/rules-feature.ts`, `src/agent/native-bootstrap.ts`).

## Problem

Billion Context (bili) compresses the live context itself. While it runs, the
host's own compaction rarely or never happens: pi does not compact, `bili
claude` sets `DISABLE_AUTO_COMPACT=1`, and Codex's compaction requests are
answered with bili's own summaries. simba hangs two behaviours on compaction:
exporting the transcript for extraction, and re-injecting CORE rules after the
model loses context. Both stopped happening. simba's "context is low" notice
also misfires, because under bili the transcript keeps growing while the
model's context does not.

## Decisions

1. **Find bili through what it already publishes.** bili's launchers and its
   pi-native bootstrap set `BILLION_CONTEXT_PROXY` in the host process once the
   proxy is up. Hooks inherit the host's environment, but the daemon does not,
   so `simba hook` and the pi bridge copy it into the payload as `bili_origin`.
   The daemon's own environment and bili's `proxy-origin` state file are
   fallbacks. The pi-native proxy listens on a random port, so no fixed port is
   assumed.

2. **Detect folds from the documented plugin protocol, with a store fallback.**
   At each turn end, `GET /__bili/plugin/status?conversationId=<session id>`
   lists the conversation's blocks; the sorted active block ids are the
   conversation's signature. When no proxy answers, the same ids come from
   bili's SQLite store, opened read-only (`conversations` maps a host
   conversation id to bili's session id). Both paths produce identical
   signatures (verified live: 64 active blocks either way), so a turn that
   falls back does not look like a fold. For Codex, bili's
   `[bili] context summary after compaction:` marker in the rollout's last
   256 KiB also counts. The first sighting of a conversation records a
   baseline, except that a Codex marker always counts, so the first Codex fold
   is not missed.

3. **React to a fold the way simba reacts to a compaction.** Reset the
   rules-signal flag (CORE is re-injected on the next prompt) and hand the
   transcript to the PreCompact export in a detached process (`trigger:
   bili_fold`), which also queues background extraction. The pi bridge
   already exports after bili's `compress` tool; a fold can produce a second,
   cheap export, which the export's unchanged-size check and the extraction
   cursor absorb.

4. **Keep CORE rules as bili rules.** bili's `acp_rule` tool records short
   reminders that are protected from folds. When the proxy's manifest
   advertises it (`compress.rules: true`, off by default, proxy mode only),
   simba records each CORE rule once per conversation, prefixed `[simba] `,
   at most 40 rules of 300 characters (bili allows 50). Rules are re-synced
   only when the CORE text changes; while `acp_rule` is off, simba re-checks
   at most hourly.

   simba turns the feature on itself (`ensure_rules_enabled`, run once from
   SessionStart and by `simba bili setup`): when an installed agent routes
   through bili (pi's `packages`, Claude's `ANTHROPIC_BASE_URL` or plugin, or
   Codex's `[mcp_servers.bili]`), it sets `compress.rules: true` in bili's
   config, after backing it up. It records that it did so and never flips the
   setting back if the user turns it off. A JSONC config (bili accepts
   comments) is left alone so no comment is lost. Note bili's own caveat:
   with rules on, the model may also delete or clear rules; simba's are
   re-added when CORE changes.

5. **Skip the context-low notice while bili manages the session** (seen in
   the last 6 hours).

6. **Stay cheap and fail-soft.** One reused HTTP client with a 2 s timeout and
   no environment proxies (a fresh client cost ~0.6 s of setup per call;
   status answers in about 0.1 s). A read-only query on two small indexed
   tables. A small JSON state file per conversation in `~/.cache/simba/bili`.
   Any failure leaves the turn unchanged.

7. **simba does not compress.** bili warns about other compressors and
   running two would compress everything twice. simba only observes and
   reacts.

8. **Automatic setup, recorded and reversible** (sim-r899). SessionStart runs
   `billion_context_setup.apply(auto=True)` once: when an installed agent uses
   bili, it changes bili's config and returns a user-facing notice ("Detected
   Billion Context ...; run `simba bili restore` to restore settings"), shown
   as the hook's `systemMessage` (pi: a UI notification). Each changed key is
   recorded with its previous value in `~/.cache/simba/bili/setup.json`;
   `simba bili restore` reverts exactly those keys, leaving the user's other
   edits, and SessionStart never re-applies after setup or restore.

9. **Learnings from bili's summaries** (sim-ezyi). On the first sighting of a
   session and on every fold, simba queues a `bili:<session>` job for the
   background extraction worker. It reads the session's active block
   summaries read-only, one row at a time and each cut to one chunk in SQL,
   skips blocks already extracted (recorded per session), and sends them to
   the host's extraction CLI. The transcript export a fold triggers is not
   queued for extraction too (triggers `bili_fold`, `context_compress`).

## Contributing to bili's summaries (sim-5f4u)

Decided 2026-11-23: no relay, no upstream change. Setup can point bili's
`compress.externalSummary` at deepseek-flash through a `simba-deepseek`
provider recipe, so a cheap model writes fold summaries, and simba extracts
from those summaries (decision 9). The catch is the key: pi keeps its DeepSeek
key in `auth.json`, and bili's pi lane never reports stored credentials
(`src/agent/pi.ts:226`). `simba bili setup --summary-key-from-pi` copies it
into bili's own private store (`<config>.summary-credentials.json`, mode 600,
referenced as `secret:simba-deepseek`). Copying a secret between tools is an
explicit opt-in, never automatic; restore removes it.

## Superseded proposal

bili has no hook for adding text to its summaries. Two options:

- **Upstream change**: a plugin-protocol callback that returns text to append.
  This is a public pull request to ranxianglei/billion-context and needs the
  user's approval.
- **Be bili's summarizer (preferred, no upstream change)**: bili's documented
  `compress.externalSummary` sends every fold's summary request to a
  configured provider recipe, and loopback HTTP is allowed. The simba daemon
  could serve an OpenAI-compatible endpoint that relays each request to a
  real model (e.g. deepseek-flash), adds CORE rules and relevant memories to
  the summary instructions, and records the folded range for extraction. Only
  summary requests pass through simba. This changes how every session's folds
  are summarized and needs model credentials in the daemon, so it waits for
  the user's decision.

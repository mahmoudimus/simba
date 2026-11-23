# Headless transcript extraction

Status: accepted, 2026-11-23. Code: `src/simba/transcripts/llm_extract.py`,
`src/simba/transcripts/config.py`, `src/simba/hooks/pre_compact.py`,
`simba transcript extract`. Tickets: sim-hrka, sim-7xdd (epic sim-wh54).

## Problem

PreCompact exports a transcript and marks it `pending_extraction`. Until now
only a Claude Code session start turned that into memories: it told the model
to dispatch an extraction subagent. Pi has no such step and Codex rarely took
it, so on 2026-11-23 there were 147 pending exports (18 from pi, the rest from
Claude Code and Codex). Billion Context and Jev also keep pi from compacting,
so pi exports were rare to begin with (fixed separately: the pi bridge now
exports at context boundaries, see the README's pi section).

## Decisions

### 1. The model returns JSON and has no tools

Each chunk goes to a model run that cannot call tools, MCP servers, extensions
or skills, and loads no context files. The model returns a JSON list of
learnings; simba validates each entry (known type, `content` under
`memory.max_content_length`, confidence in [0, 1]) and stores the survivors
through the daemon's `/store`, so dedup and supersession apply.

Why:
- Transcripts contain pasted text, logs and web pages. With no tools, an
  instruction hidden in a transcript cannot make the extraction run execute
  commands, write files or call MCP servers. The worst it can do is return bad
  JSON, which validation rejects. The prompt also tells the model to treat the
  transcript as data.
- simba, not the model, decides what is stored, and storage goes through one
  path (`/store`) instead of the model running `simba memory store` itself.

Costs, accepted:
- The model sees only the chunk. It cannot check a path or command against
  the repository, so a learning can be stale or wrong in ways it cannot see.
- Without a schema, a model sometimes returns no valid JSON: 1 of 30 chunks
  in the first pi run (deepseek-flash). That export stays pending and is
  retried. Claude and Codex runs use schema-enforced output
  (`--json-schema`, `--output-schema`), which removes this failure for them.
- Codex always has a shell tool. The run uses a read-only sandbox in an empty
  temp directory, without the user's config (no MCP servers or hooks) or exec
  rules. That is the closest Codex gets to no tools.

### 2. Each host's transcripts go to that host's own CLI and provider

| Export from | Runner | Model (config key) |
|---|---|---|
| pi | `pi -p` | `deepseek/deepseek-flash` (`transcripts.model_pi`) |
| Claude Code | `claude -p` | `claude-haiku-5-5` (`transcripts.model_claude`) |
| Codex | `codex exec` | `gpt-5.6-luna`, reasoning `max` (`transcripts.model_codex`, `transcripts.codex_reasoning_effort`) |

Why: a work transcript (e.g. vgs, kirafin) should not be sent to a provider
its session never used. The smallest model of each host keeps the cost low.
Using each host's CLI also reuses its existing login. The host is recorded in
the export metadata (`source`) from the hook's client, or sniffed from the
exported JSONL's first line for older exports. An export whose host is
unknown is never sent anywhere automatically.

pi exports are the exception to "same provider": a pi session may have used
several providers (xai, cursor, deepseek, openrouter), and its extraction goes
to deepseek-flash. The user chose this on 2026-11-23.

### 3. simba's own hooks are off inside extraction runs

`claude -p` and `codex exec` would otherwise run simba's installed hooks:
recall memories into the extraction prompt, export the extraction session,
and queue another extraction. Every child gets `SIMBA_HOOKS_DISABLED=1`, which
makes `simba hook` and `simba hook-canonical` no-ops. (`claude --bare` also
skips hooks, but it only accepts `ANTHROPIC_API_KEY`, not the user's OAuth
login, so it is not used.)

### 4. Memory is bounded; one extraction at a time

The user's requirement: no runaway memory use, which has taken the whole
machine down before (see the 2026-07-17 RSS incident in
`hooks/pre_compact.py`).

- The transcript is streamed line by line from a byte offset. At most one
  chunk (`transcripts.chunk_chars`, 240,000 characters) is held per model call;
  a longer line is read in 1 MiB pieces. `<thinking>` blocks are dropped while
  streaming.
- The background worker holds a machine-wide file lock
  (`~/.cache/simba/extract/worker.lock`), so only one extraction runs at a
  time. It processes one export and one model call at a time, at `nice 10`.
  Export hooks only add a queue file and start the worker if the lock is free.
- Node CLIs (pi, claude) get `--max-old-space-size=1024`
  (`transcripts.node_max_old_space_mb`). Each model call has a timeout
  (`transcripts.call_timeout`, 900 s).
- Measured peak RSS on 2026-11-23 (macOS, `/usr/bin/time -l`): `pi -p`
  146.6 MB, `claude -p` 330.0 MB, `codex exec` about 204 MB, the simba worker
  42.5 MB. The largest export's markdown was 22.7 MB, which is never loaded
  whole.
- A manual run may use `--jobs N` (default `transcripts.manual_jobs`, 2); it
  takes the same lock, so it never overlaps the worker.

### 5. Automatic for new exports only

After a real export, PreCompact marks the export `queued_extraction`, adds it
to the queue and starts the worker. Only the triggering export is queued; the
backlog is extracted only by an explicit `simba transcript extract --all`.
If extraction fails, the worker sets the export back to `pending_extraction`,
so Claude Code's session-start directive (or a later manual run) can still
pick it up. A queued export is not offered to that directive, which prevents
extracting the same transcript twice.

### 6. Incremental extraction

`extract_cursor.json` next to the export records how many bytes of
`transcript.md` were extracted (the export rewrites `metadata.json`, so the
cursor lives in its own file). A re-export of a longer session sends only the
new turns. An export is marked extracted, and the cursor advanced, only when
every chunk succeeded. A learning held for supersession review
(`pending_confirmation`) counts as stored.

## Turning it off

`simba config set transcripts.auto_extract false` stops the background path;
manual `simba transcript extract` still works.

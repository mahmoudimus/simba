"""The Pi bridge exports transcripts at context boundaries, not only compaction."""

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

HARNESS = r"""
import assert from "node:assert/strict";
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import bridge from "./src/simba/pi/extension/simba.ts";
const sessionFile = process.env.SESSION_FILE;
const cliMarker = process.env.CLI_MARKER;
const handlers = new Map();
const on = (event, handler) => handlers.set(event, handler);
bridge({on, getActiveTools: () => []});
const exports = [];
const deadlines = [];
const timeout = AbortSignal.timeout;
AbortSignal.timeout = (ms) => {
  deadlines.push(ms);
  return timeout(ms);
};
let daemon = "up";
let release;
let hold = false;
globalThis.fetch = async (url, init) => {
  if (String(url).endsWith("/hook/pre_compact")) {
    if (daemon === "down") throw new TypeError("fetch failed");
    exports.push(JSON.parse(init.body));
    if (daemon === "slow") throw new DOMException("timed out", "TimeoutError");
    if (hold) await new Promise((resolve) => (release = resolve));
  }
  return {ok: true, json: async () => ({})};
};
const ctx = {cwd: "/test", hasUI: false, sessionManager: {
  getSessionFile: () => sessionFile,
  getSessionId: () => "pi-session",
  getBranch: () => [],
}};
const settle = (ms = 10) => new Promise((resolve) => setTimeout(resolve, ms));
const toolResult = (toolName, isError = false) =>
  handlers.get("tool_result")({toolName, isError, input: {}, content: []}, ctx);
"""

SCENARIO = r"""
// Billion Context's compress tool is a context boundary.
await toolResult("compress");
await settle();
assert.equal(exports.length, 1);
assert.equal(exports[0].trigger, "context_compress");
assert.equal(exports[0].transcript_path, sessionFile);
assert.equal(exports[0].session_id, "pi-session");
assert.equal(deadlines.at(-1), 2_000, "wait only for the handoff");

// A failed compress and other tools do not export.
await toolResult("compress", true);
await toolResult("bash");
await settle();
assert.equal(exports.length, 1);

// Small growth does not export; 4 MB of growth does, once.
await handlers.get("agent_end")({messages: []}, ctx);
await settle();
assert.equal(exports.length, 1);
appendFileSync(sessionFile, "x".repeat(4_000_000));
await handlers.get("agent_end")({messages: []}, ctx);
await settle();
assert.equal(exports.length, 2);
assert.equal(exports[1].trigger, "checkpoint");
await handlers.get("agent_end")({messages: []}, ctx);
await settle();
assert.equal(exports.length, 2);

// One handoff at a time: a compress during a handoff does not start another.
hold = true;
appendFileSync(sessionFile, "x".repeat(4_000_000));
await handlers.get("agent_end")({messages: []}, ctx);
await settle();
await toolResult("compress");
await settle();
assert.equal(exports.length, 3);

// Compaction does not wait for the export, even while one is held.
assert.equal(handlers.get("session_before_compact")({}, ctx), undefined);
await settle();
assert.equal(exports.length, 4);
assert.equal(exports[3].trigger, "compaction");
hold = false;
release();
await settle();

// A slow daemon already has the request: no second export through the CLI.
daemon = "slow";
await handlers.get("session_shutdown")({reason: "quit"}, ctx);
assert.equal(exports.at(-1).trigger, "shutdown");
await settle(300);
assert.equal(existsSync(cliMarker), false, "timeout must not fall back");

// An unreachable daemon falls back to a detached CLI export.
daemon = "down";
await handlers.get("session_shutdown")({reason: "quit"}, ctx);
for (let i = 0; i < 100 && !existsSync(cliMarker); i++) await settle(20);
const cli = JSON.parse(readFileSync(cliMarker, "utf8"));
assert.equal(cli.trigger, "shutdown");
assert.equal(cli.transcript_path, sessionFile);
"""


def test_exports_at_context_boundaries(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the Pi bridge")
    session_file = tmp_path / "session.jsonl"
    session_file.write_text("{}\n")
    marker = tmp_path / "cli-export.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "simba"
    fake.write_text(
        f'#!/bin/sh\n[ "$1 $2" = "hook-canonical pre_compact" ] && cat > "{marker}"\n'
    )
    fake.chmod(0o755)
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-"],
        input=HARNESS + SCENARIO,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "SESSION_FILE": str(session_file),
            "CLI_MARKER": str(marker),
        },
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

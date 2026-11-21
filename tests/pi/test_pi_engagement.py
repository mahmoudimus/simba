"""Execute the bundled bridge against a minimal Pi event surface."""

import pathlib
import shutil
import subprocess

import pytest


def test_engagement_is_ui_only():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute the Pi bridge")
    root = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, "--experimental-strip-types", "--input-type=module", "-"],
        input=r"""
import assert from "node:assert/strict";
import bridge from "./src/simba/pi/extension/simba.ts";
const handlers = new Map();
bridge({
  on: (event, handler) => handlers.set(event, handler),
  getActiveTools: () => ["mcp__simba_rlm__rlm_grep", "mcp__simba_rlm__rlm_peek"]
});
const ledger = "🦁☑ recalled 2 (top 0.80)";
const marker = `<engagement-marker>\n${ledger}\n` +
  "Echo the 🦁☑ line above verbatim.\n</engagement-marker>";
const guidance = "<recalled-memories>Use scoped updates.</recalled-memories>";
let response;
const calls = [];
const deadlines = [];
const timeout = AbortSignal.timeout;
AbortSignal.timeout = (ms) => {
  deadlines.push(ms);
  return timeout(ms);
};
globalThis.fetch = async (url) => {
  calls.push(url);
  return {ok: true, json: async () => response};
};
const statuses = [];
const ctx = {cwd: "/test", hasUI: true,
  ui: {setStatus: (key, value) => statuses.push([key, value])}};
response = {additional_context: `${marker}\n${guidance}`, memory_count: 2};
const output = await handlers.get("before_agent_start")({prompt: "do work"}, ctx);
assert.equal(output.message.content, guidance, "receipt must not enter model context");
assert.deepEqual(statuses.at(-1), ["simba", ledger]);
assert.equal(handlers.has("context"), false, "no per-call injection");
assert.equal(calls.length, 1);
assert.equal(deadlines[0], 15_000, "allow prompt retrieval to finish");
response = {};
await handlers.get("session_start")({}, ctx);
assert.equal(deadlines.at(-1), 3_000, "other hooks keep their short deadline");
response = {additional_context: marker};
assert.equal(
  await handlers.get("before_agent_start")({prompt: "work"}, ctx), undefined);
response = {additional_context: guidance};
const plain = await handlers.get("before_agent_start")({prompt: "work"}, ctx);
assert.equal(plain.message.content, guidance);
assert.deepEqual(statuses.at(-1), ["simba", undefined], "clear stale receipt");
response = {additional_context: ""};
assert.equal(
  await handlers.get("before_agent_start")({prompt: "work"}, ctx), undefined);
response = {additional_context: `${marker}\n${guidance}`};
const headless = await handlers.get("before_agent_start")(
  {prompt: "work"}, {cwd: "/test", hasUI: false});
assert.equal(headless.message.content, guidance);

const pointer = "synthetic-session";
response = {additional_context: guidance +
  "\n<rlm-pointers>\nUse rlm_grep/rlm_peek\n" +
  `  - ${pointer} :: Prior decision\n</rlm-pointers>`};
const offered = await handlers.get("before_agent_start")({prompt: "work"}, ctx);
assert.ok(offered.message.content.includes(pointer));
assert.match(offered.message.content,
  /mcp__simba_rlm__rlm_grep\/mcp__simba_rlm__rlm_peek/);
assert.equal(statuses.findLast(([key]) => key === "simba-rlm")[1],
  "RLM: 1 pointer available, 0 followed");

const result = handlers.get("tool_result");
const receipt = () => statuses.findLast(([key]) => key === "simba-rlm")[1];
const event = (toolName, payload, extra = {}) => ({
  toolName, input: {transcript_id: pointer}, isError: false,
  content: [{type: "text", text: JSON.stringify(payload)}], ...extra
});
await result(event("rlm_grep", {matches: []}), ctx);
assert.match(receipt(), /0 matches/);
await result(event("rlm_peek", {error: "transcript not available"}), ctx);
assert.match(receipt(), /lookup failed/);
await result(event("mcp__simba-rlm__rlm_grep", {matches: [{line: 1}]}), ctx);
assert.match(receipt(), /searched.*1 match/);
await result(event("rlm_peek", {text: "Prior decision"}), ctx);
assert.match(receipt(), /read/);
await result(event("rlm_peek", {text: "Other transcript"},
  {input: {transcript_id: "unrelated"}}), ctx);
assert.match(receipt(), /read.*synthetic-session/);

response = {additional_context: guidance};
await handlers.get("before_agent_start")({prompt: "new turn"}, ctx);
assert.equal(receipt(), undefined);
""",
        text=True,
        cwd=root,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

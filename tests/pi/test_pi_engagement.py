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
bridge({on: (event, handler) => handlers.set(event, handler)});
const ledger = "🦁☑ recalled 2 (top 0.80)";
const marker = `<engagement-marker>\n${ledger}\n` +
  "Echo the 🦁☑ line above verbatim.\n</engagement-marker>";
const guidance = "<recalled-memories>Use scoped updates.</recalled-memories>";
let response;
const calls = [];
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
""",
        text=True,
        cwd=root,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

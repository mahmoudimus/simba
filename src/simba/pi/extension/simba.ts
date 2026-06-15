/**
 * simba — memory loop bridge for the pi coding agent.
 *
 * Pure marshalling: each pi lifecycle event is forwarded to simba (daemon HTTP,
 * CLI fallback) and the canonical result is applied to pi's event result. No
 * recall/ranking/guardian logic lives here — it all runs in Python.
 */
import { spawn } from "node:child_process";
import type {
  ExtensionAPI,
  ExtensionContext,
  BeforeAgentStartEvent,
  AgentEndEvent,
} from "@earendil-works/pi-coding-agent";

const DAEMON = process.env.SIMBA_DAEMON_URL || "http://localhost:8741";

interface Canonical {
  additional_context?: string;
  suppress_output?: boolean;
  block_reason?: string | null;
}

function lastAssistantText(messages: Array<{ role?: string; content?: unknown }>): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "assistant") continue;
    const c = m.content;
    if (typeof c === "string") return c;
    if (Array.isArray(c)) {
      return c
        .filter((p): p is { type: "text"; text: string } =>
          Boolean(p && typeof p === "object" && (p as { type?: string }).type === "text"))
        .map((p) => p.text)
        .join("\n");
    }
    return "";
  }
  return "";
}

function viaCli(event: string, payload: Record<string, unknown>): Promise<Canonical> {
  return new Promise((resolve) => {
    const child = spawn("simba", ["hook-canonical", event], { stdio: ["pipe", "pipe", "ignore"] });
    let out = "";
    child.stdout.on("data", (d: Buffer) => (out += d.toString()));
    child.on("close", () => {
      try {
        resolve(JSON.parse(out));
      } catch {
        resolve({});
      }
    });
    child.on("error", () => resolve({}));
    child.stdin.end(JSON.stringify(payload));
  });
}

async function callSimba(event: string, payload: Record<string, unknown>): Promise<Canonical> {
  try {
    const resp = await fetch(`${DAEMON}/hook/${event}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) return (await resp.json()) as Canonical;
  } catch {
    /* daemon down — fall back to the CLI */
  }
  return viaCli(event, payload);
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (_e, ctx: ExtensionContext) => {
    const r = await callSimba("session_start", { cwd: ctx.cwd });
    if (r.additional_context && ctx.hasUI) ctx.ui.notify(r.additional_context, "info");
  });

  pi.on("before_agent_start", async (e: BeforeAgentStartEvent, ctx: ExtensionContext) => {
    const r = await callSimba("prompt_submit", { prompt: e.prompt, cwd: ctx.cwd });
    if (r.additional_context) {
      return {
        message: { customType: "simba-memory", content: r.additional_context, display: true },
      };
    }
  });

  pi.on("agent_end", async (e: AgentEndEvent, ctx: ExtensionContext) => {
    await callSimba("stop", {
      response: lastAssistantText(e.messages),
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
    });
  });

  pi.on("session_before_compact", async (_e, ctx: ExtensionContext) => {
    await callSimba("pre_compact", {
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
      session_id: ctx.sessionManager.getSessionId(),
    });
  });
}

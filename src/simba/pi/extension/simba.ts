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
  ToolCallEvent,
  ToolCallEventResult,
} from "@earendil-works/pi-coding-agent";

const DAEMON = process.env.SIMBA_DAEMON_URL || "http://localhost:8741";

/** Surface what simba did to stderr — visible even in `pi -p`. No magic. */
function note(msg: string): void {
  process.stderr.write(`[simba: ${msg}]\n`);
}

interface Canonical {
  additional_context?: string;
  suppress_output?: boolean;
  memory_count?: number;
  block_reason?: string | null;
  // A silent rewrite of the tool arguments (redirect rewrite). For bash we apply
  // transform.command by mutating event.input.command in place.
  transform?: { command?: string; reason?: string } | null;
  // A directive that block-only harnesses (pi tool_call) enforce as a hard block
  // — a strong TOOL_RULE match. Claude/Codex inject it as context instead.
  escalated_block?: string | null;
}

// pi lowercases tool names ("bash", "edit", …); simba's PreToolUse speaks the
// Claude convention ("Bash", "Edit", …). Unknown tools fall back to
// capitalize-first so a custom tool still reaches the gate with a sane name.
const PI_TOOL_TO_CLAUDE: Record<string, string> = {
  bash: "Bash",
  edit: "Edit",
  write: "Write",
  read: "Read",
  grep: "Grep",
  find: "Find",
};

function claudeToolName(piName: string): string {
  return PI_TOOL_TO_CLAUDE[piName] ?? piName.charAt(0).toUpperCase() + piName.slice(1);
}

// Flatten an assistant message's content to text. Handles a plain string and the
// structured `{type:"text",text}[]` shape; ignores tool-call / non-text blocks.
function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p): p is { type: "text"; text: string } =>
        Boolean(p && typeof p === "object" && (p as { type?: string }).type === "text"))
      .map((p) => p.text)
      .join("\n");
  }
  return "";
}

function lastAssistantText(messages: Array<{ role?: string; content?: unknown }>): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role !== "assistant") continue;
    return messageText(m.content);
  }
  return "";
}

// The agent's most recent reasoning, pulled from the live session branch (pi has
// no transcript file at tool_call time). The pitfall/doctrine gate keys on this.
// Defensive: getBranch() returns a SessionEntry union; only message entries carry
// `.message`, so narrow via `as any` and skip non-message entries silently.
function lastAssistantReasoning(sessionManager: { getBranch(): unknown[] }): string {
  let branch: unknown[];
  try {
    branch = sessionManager.getBranch();
  } catch {
    return "";
  }
  for (let i = branch.length - 1; i >= 0; i--) {
    const msg = (branch[i] as { message?: { role?: string; content?: unknown } })?.message;
    if (!msg || msg.role !== "assistant") continue;
    return messageText(msg.content);
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
    note(r.additional_context ? "session ready — memory active" : "daemon unavailable");
    if (r.additional_context && ctx.hasUI) ctx.ui.notify(r.additional_context, "info");
  });

  pi.on("before_agent_start", async (e: BeforeAgentStartEvent, ctx: ExtensionContext) => {
    const r = await callSimba("prompt_submit", { prompt: e.prompt, cwd: ctx.cwd });
    if (r.additional_context) {
      const n = r.memory_count ?? 0;
      note(n > 0 ? `${n} memories injected` : "project rules injected");
      return {
        message: { customType: "simba-memory", content: r.additional_context, display: true },
      };
    }
    note("nothing to inject");
  });

  // Tool gate: before a tool runs, ask simba's PreToolUse path whether to block
  // or silently rewrite it. Same redirect/TOOL_RULE rules as Claude/Codex; the
  // daemon decides (gated by redirect_enabled / rule_check_enabled), the bridge
  // only enforces. pi tool_call has no context channel, so context-only
  // injections (weak TOOL_RULE, recall, context-low warnings) are dropped.
  pi.on("tool_call", async (e: ToolCallEvent, ctx: ExtensionContext): Promise<ToolCallEventResult | void> => {
    const tool_name = claudeToolName(e.toolName);
    const tool_input =
      e.toolName === "bash"
        ? { command: (e.input as { command?: string }).command }
        : (e.input as Record<string, unknown>);

    // Pass the agent's last reasoning so the daemon's pitfall/doctrine gate can fire
    // (it only reads this when pitfall_gate_enabled). Cheap, in-memory — always send.
    const thinking = lastAssistantReasoning(ctx.sessionManager);
    const r = await callSimba("pre_tool", { tool_name, tool_input, cwd: ctx.cwd, thinking });

    if (r.transform && r.transform.command && e.toolName === "bash") {
      // Silent rewrite — mutate the (mutable) bash input in place and allow.
      (e.input as { command?: string }).command = r.transform.command;
      note(`rewrote → ${r.transform.command}`);
      return;
    }
    if (r.block_reason || r.escalated_block) {
      const reason = r.block_reason || r.escalated_block || "blocked by simba";
      note(`blocked tool — ${reason}`);
      return { block: true, reason };
    }
    // No gate fired — allow with the original input (context injections dropped).
  });

  pi.on("agent_end", async (e: AgentEndEvent, ctx: ExtensionContext) => {
    await callSimba("stop", {
      response: lastAssistantText(e.messages),
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
    });
    note("session captured");
  });

  pi.on("session_before_compact", async (_e, ctx: ExtensionContext) => {
    await callSimba("pre_compact", {
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
      session_id: ctx.sessionManager.getSessionId(),
    });
    note("transcript exported");
  });
}

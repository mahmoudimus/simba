/**
 * simba — memory loop bridge for the pi coding agent.
 *
 * Pure marshalling: each pi lifecycle event is forwarded to simba (daemon HTTP,
 * CLI fallback) and the canonical result is applied to pi's event result. No
 * recall/ranking/guardian logic lives here — it all runs in Python.
 */
import { spawn } from "node:child_process";
import { statSync } from "node:fs";
import type {
  ExtensionAPI,
  ExtensionContext,
  BeforeAgentStartEvent,
  AgentEndEvent,
  ToolCallEvent,
  ToolCallEventResult,
} from "@earendil-works/pi-coding-agent";
const DAEMON = process.env.SIMBA_DAEMON_URL || "http://localhost:8741";

// Context managers such as Billion Context and Jev shrink the live context
// without Pi compaction, so session_before_compact rarely fires. Export the
// transcript at those boundaries too: after Billion Context's `compress` tool,
// whenever the session file has grown by this many bytes, and at shutdown.
const CONTEXT_COMPRESS_TOOL = "compress";
const CHECKPOINT_GROWTH_BYTES = 4_000_000;
// An export streams the whole session file, which can take seconds. The daemon
// runs it on its own worker thread and keeps going after the caller stops
// waiting, so the bridge only waits long enough to hand the request over.
const EXPORT_HANDOFF_MS = 2_000;

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
  system_message?: string;
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

function rlmOperation(name: string): string | undefined {
  return /(?:^|_)rlm_(grep|peek|window)$/.exec(name)?.[1];
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

// Engagement receipts are telemetry, not user instructions. Split the tagged
// prompt receipt here so this also works with already-running Simba daemons.
// Actual recall and CORE guidance remain in the model-facing context.
function promptPresentation(context: string): { guidance: string; ledger?: string } {
  let ledger: string | undefined;
  const guidance = context.replace(
    /<engagement-marker>\s*([\s\S]*?)<\/engagement-marker>/g,
    (_block, body: string) => {
      ledger = body.split("\n").map((line) => line.trim()).find(Boolean);
      return "";
    },
  ).trim();
  return { guidance, ledger };
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

function sessionBytes(file: string): number {
  try {
    return statSync(file).size;
  } catch {
    return 0;
  }
}

// Hand an export to the daemon without waiting for it to finish. A handoff
// timeout means the daemon has the request and is still exporting, so it must
// not fall back to the CLI, which would export the same file a second time.
// Only an unreachable or failing daemon falls back, to a detached CLI process.
async function sendExport(payload: Record<string, unknown>): Promise<void> {
  try {
    const resp = await fetch(`${DAEMON}/hook/pre_compact`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-simba-client": "pi" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(EXPORT_HANDOFF_MS),
    });
    if (resp.ok) return;
  } catch (err) {
    if ((err as { name?: string } | undefined)?.name === "TimeoutError") return;
  }
  try {
    const child = spawn("simba", ["hook-canonical", "pre_compact"], {
      detached: true,
      stdio: ["pipe", "ignore", "ignore"],
      env: { ...process.env, SIMBA_CLIENT: "pi" },
    });
    child.on("error", () => {});
    child.stdin?.end(JSON.stringify(payload));
    child.unref();
  } catch {
    /* no simba on PATH: nothing to export with */
  }
}

function viaCli(event: string, payload: Record<string, unknown>): Promise<Canonical> {
  return new Promise((resolve) => {
    // SIMBA_CLIENT=pi so the CLI fallback (and any daemon call it makes)
    // attributes this traffic to pi, matching the direct-fetch header.
    const child = spawn("simba", ["hook-canonical", event], {
      stdio: ["pipe", "pipe", "ignore"],
      env: { ...process.env, SIMBA_CLIENT: "pi" },
    });
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
  // Prompt submission can run recall, conflict checking, and project search
  // before returning context. Keep the shorter deadline for other hooks.
  const timeoutMs = event === "prompt_submit" ? 15_000 : 3_000;
  try {
    const resp = await fetch(`${DAEMON}/hook/${event}`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-simba-client": "pi" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (resp.ok) return (await resp.json()) as Canonical;
  } catch {
    /* daemon down — fall back to the CLI */
  }
  return viaCli(event, payload);
}

export default function (pi: ExtensionAPI) {
  let offeredTranscripts = new Set<string>();
  let exportedBytes = 0;
  let exporting: Promise<unknown> | undefined;

  // Hand the transcript to simba's PreCompact export. The daemon skips a file
  // that has not changed since its last export, and runs one export per session
  // at a time. `trigger` tells it whether the model's context was reduced.
  // Pi's session file is append-only, so nothing has to wait for the export.
  function exportTranscript(ctx: ExtensionContext, trigger: string): Promise<unknown> {
    const transcript_path = ctx.sessionManager.getSessionFile() ?? "";
    if (!transcript_path) return Promise.resolve();
    exportedBytes = sessionBytes(transcript_path);
    const run = sendExport({
      cwd: ctx.cwd,
      transcript_path,
      session_id: ctx.sessionManager.getSessionId(),
      trigger,
    }).then(() => note(`transcript export queued (${trigger})`));
    const tracked: Promise<unknown> = run.finally(() => {
      if (exporting === tracked) exporting = undefined;
    });
    exporting = tracked;
    return tracked;
  }

  pi.on("session_start", async (_e, ctx: ExtensionContext) => {
    const r = await callSimba("session_start", { cwd: ctx.cwd });
    note(r.additional_context ? "session ready — memory active" : "daemon unavailable");
    if (r.additional_context && ctx.hasUI) ctx.ui.notify(r.additional_context, "info");
    // Human-facing notes (e.g. "ran simba bili setup automatically").
    if (r.system_message) {
      if (ctx.hasUI) ctx.ui.notify(r.system_message, "info");
      else note(r.system_message);
    }
  });

  pi.on("before_agent_start", async (e: BeforeAgentStartEvent, ctx: ExtensionContext) => {
    const r = await callSimba("prompt_submit", { prompt: e.prompt, cwd: ctx.cwd });
    const presentation = promptPresentation(r.additional_context ?? "");
    const { ledger } = presentation;
    let guidance = presentation.guidance;
    if (ctx.hasUI) ctx.ui.setStatus("simba", ledger);
    else if (ledger) note(ledger);
    const previousHadPointers = offeredTranscripts.size > 0;
    const pointerBlock = /<rlm-pointers>([\s\S]*?)<\/rlm-pointers>/.exec(guidance)?.[1] ?? "";
    offeredTranscripts = new Set(
      Array.from(pointerBlock.matchAll(/^\s*- ([\w.-]+) ::/gm), (match) => match[1]),
    );
    const count = offeredTranscripts.size;
    if (count) {
      const tools = pi.getActiveTools?.() ?? [];
      const grep = tools.find((name) => rlmOperation(name) === "grep");
      const peek = tools.find((name) => rlmOperation(name) === "peek");
      if (grep && peek) {
        guidance = guidance.replace(
          pointerBlock, () => pointerBlock.replace("rlm_grep/rlm_peek", `${grep}/${peek}`),
        );
      }
      const status = `RLM: ${count} pointer${count === 1 ? "" : "s"} available, 0 followed`;
      if (ctx.hasUI) ctx.ui.setStatus("simba-rlm", status);
      else note(status);
    } else if (ctx.hasUI && previousHadPointers) {
      ctx.ui.setStatus("simba-rlm", undefined);
    }
    if (guidance) {
      const n = r.memory_count ?? 0;
      note(n > 0 ? `${n} memories injected` : "project rules injected");
      return {
        message: { customType: "simba-memory", content: guidance, display: true },
      };
    }
    note("nothing to inject");
  });

  pi.on("tool_result", (e, ctx) => {
    if (e.toolName === CONTEXT_COMPRESS_TOOL && !e.isError && !exporting) {
      void exportTranscript(ctx, "context_compress");
    }
    const operation = rlmOperation(e.toolName);
    const transcriptId = (e.input as { transcript_id?: unknown }).transcript_id;
    if (!operation || typeof transcriptId !== "string" || !offeredTranscripts.has(transcriptId)) return;

    let status = `RLM: lookup failed for ${transcriptId}`;
    if (!e.isError) {
      try {
        const result = JSON.parse(messageText(e.content)) as { matches?: unknown; text?: unknown };
        if (operation === "grep" && Array.isArray(result.matches)) {
          const count = result.matches.length;
          status = `RLM: searched ${transcriptId} (${count} match${count === 1 ? "" : "es"})`;
        } else if (typeof result.text === "string") {
          status = result.text ?
            `RLM: read ${transcriptId} (${result.text.length} chars)` :
            `RLM: empty read from ${transcriptId}`;
        }
      } catch {
        // An unrecognized result is not evidence that the pointer was read.
        status = `RLM: lookup result unverified for ${transcriptId}`;
      }
    }
    if (ctx.hasUI) ctx.ui.setStatus("simba-rlm", status);
    else note(status);
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

  // No per-LLM-call context handler: Pi converts custom messages to user
  // messages. Repeating a receipt there masquerades as fresh user input.
  // Relevant memories and rules arrive once through before_agent_start above.

  // Tier-2 (pi-only): doctrine-verify the FINALIZED assistant message — the
  // tools-free-output catch (a wrong conclusion stated in prose has no tool to
  // gate). The daemon returns a correction (block_reason) on a violation; we
  // annotate the message IN PLACE (keeping the role, as pi requires) so the
  // correction rides with it. Off (default) → daemon returns no block, no-op.
  pi.on("message_end", async (e, ctx: ExtensionContext) => {
    const msg = e.message;
    if (!msg || msg.role !== "assistant") return; // only verify assistant output
    const r = await callSimba("message_end", {
      message_text: messageText(msg.content),
      cwd: ctx.cwd,
    });
    if (!r.block_reason) return; // no violation — leave the message as-is
    // Defensive: today's pi types guarantee AssistantMessage.content is an array,
    // but a future/non-Anthropic provider could surface a string. Spreading a
    // string would shatter it into chars — bail (no-op) rather than corrupt it.
    if (!Array.isArray(msg.content)) return;
    note("doctrine violation — annotated finalized message");
    const correction = { type: "text" as const, text: `\n\n${r.block_reason}` };
    // Keep the original role; append the correction to the assistant content.
    const content = [...msg.content, correction];
    return { message: { ...msg, content } };
  });

  pi.on("agent_end", async (e: AgentEndEvent, ctx: ExtensionContext) => {
    await callSimba("stop", {
      response: lastAssistantText(e.messages),
      cwd: ctx.cwd,
      transcript_path: ctx.sessionManager.getSessionFile() ?? "",
      session_id: ctx.sessionManager.getSessionId(),
      // Billion Context's pi-native bootstrap exports its in-process proxy here.
      bili_origin: process.env.BILLION_CONTEXT_PROXY ?? "",
    });
    note("session captured");
    const file = ctx.sessionManager.getSessionFile();
    if (file && !exporting && sessionBytes(file) - exportedBytes >= CHECKPOINT_GROWTH_BYTES) {
      void exportTranscript(ctx, "checkpoint");
    }
  });

  // Compaction appends to the session file, so it does not wait for the export.
  pi.on("session_before_compact", (_e, ctx: ExtensionContext) => {
    void exportTranscript(ctx, "compaction");
  });

  // Shutdown waits only for the handoff, so the request leaves before Pi exits.
  pi.on("session_shutdown", async (_e, ctx: ExtensionContext) => {
    await exportTranscript(ctx, "shutdown");
  });
}

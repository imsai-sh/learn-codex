// =============================================================================
// step08 / sandbox.ts — 长寿命 agent + 一族 agent-team 工具
// -----------------------------------------------------------------------------
// 在 step07 的工具集（run_bash / read_file / write_file / edit_file / update_plan
// + spawn_sub_agent）之上，step08 把 agent 变成了 runtime 里的一等公民：
//
//   spawn_agent  — 起一条后台 agent“线程”，附带初始 instruction；立刻返回。
//   send_input   — 把后续指令排进已有 agent 的队列；如果它当前 idle，就把
//                  worker 重新拉起来。
//   wait_agent   — 阻塞等待某个 agent 走到 final 状态（completed/failed/closed），
//                  带超时。
//   close_agent  — 把 agent 标记为关闭，拒绝新输入并立刻置为 closed 终态。
//   list_agents  — 返回当前所有 agent 的快照，给父 agent 做调度参考。
//
// 同时，原本在 step04~step07 里同步等待的 spawn_sub_agent 也被改写：它现在
// 走的是“spawn 一条后台 agent → 用 wait_for_agent_status 等它结束 → 把
// last_result 当成工具返回值”这套统一管线。
//
// 实现要点：
//   * Rust 用 tokio::spawn 起 worker；TS 这里用一个普通 async 函数 + `void` 触发
//     “fire and forget”——它会跑在事件循环里，不阻塞调用方。
//   * Rust 用 watch::channel 等 final 状态；TS 用 agent_team.ts 里的
//     StatusBroadcaster + Promise.race(setTimeout) 做超时等待。
//   * 同一个 agent 同一时刻只允许一条 worker 在跑（tryStartWorker 守门）。
//     当 worker 拉空了队列、刚把 workerActive 置 false 时，如果发现 enqueue
//     与之竞争塞了新输入进来，就再 CAS 拉起来一轮，避免“塞了输入但没人跑”。
// =============================================================================

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { fetch } from "undici";

import {
  AgentSnapshot,
  AgentStatus,
  AgentTeamManager,
  AgentThread,
  isFinalStatus,
} from "./agent_team.js";

// ----------------------------- 通用结果结构 -----------------------------

export interface ToolResult {
  ok: boolean;
  tool: string;
  message: string;
  error_code?: string;
  data: Record<string, unknown>;
}

export interface TruncatedText {
  content: string;
  truncated: boolean;
  omitted_chars: number;
}

function serializeToolResult(r: ToolResult): string {
  try {
    return JSON.stringify(r, null, 2);
  } catch (e) {
    return JSON.stringify({
      ok: false,
      tool: "internal",
      message: `failed to serialize tool result: ${(e as Error).message}`,
      error_code: "serialization_failed",
      data: {},
    });
  }
}

export function toolSuccess(tool: string, message: string, data: Record<string, unknown>): string {
  return serializeToolResult({ ok: true, tool, message, data });
}

export function toolError(
  tool: string,
  errorCode: string,
  message: string,
  data: Record<string, unknown>,
): string {
  return serializeToolResult({ ok: false, tool, message, error_code: errorCode, data });
}

function toolNotFoundOutput(tool: string, args: string): string {
  return toolError(tool, "tool_not_found", `tool ${tool} not found`, { arguments: args });
}

function parseArgs<T>(
  tool: string,
  argumentsJson: string,
): { ok: true; value: T } | { ok: false; output: string } {
  try {
    return { ok: true, value: JSON.parse(argumentsJson) as T };
  } catch (e) {
    return {
      ok: false,
      output: toolError(
        tool,
        "invalid_arguments",
        `failed to parse arguments for ${tool}: ${(e as Error).message}`,
        { arguments: argumentsJson },
      ),
    };
  }
}

// ----------------------------- 一个迷你“读写锁” -----------------------------

/**
 * ReadWriteLock 的语义：
 *   - acquireRead():   多读并行。当且仅当“当前没有写锁、也没有写者在排队”时立刻拿到。
 *   - acquireWrite():  独占。会等到所有读者退出，并优先于后续读者。
 *
 * 写者排队期间，新读者必须等（避免写饥饿）；释放后按 FIFO 唤醒。
 */
class ReadWriteLock {
  private readers = 0;
  private writer = false;
  private waiters: Array<{ kind: "reader" | "writer"; resolve: () => void }> = [];

  async acquireRead(): Promise<() => void> {
    if (!this.writer && !this.waiters.some((w) => w.kind === "writer")) {
      this.readers += 1;
      return () => this.releaseRead();
    }
    await new Promise<void>((resolve) => {
      this.waiters.push({ kind: "reader", resolve });
    });
    this.readers += 1;
    return () => this.releaseRead();
  }

  async acquireWrite(): Promise<() => void> {
    if (!this.writer && this.readers === 0) {
      this.writer = true;
      return () => this.releaseWrite();
    }
    await new Promise<void>((resolve) => {
      this.waiters.push({ kind: "writer", resolve });
    });
    this.writer = true;
    return () => this.releaseWrite();
  }

  private releaseRead(): void {
    this.readers -= 1;
    if (this.readers === 0) this.tryWake();
  }

  private releaseWrite(): void {
    this.writer = false;
    this.tryWake();
  }

  private tryWake(): void {
    if (this.writer || this.readers > 0) return;
    if (this.waiters.length === 0) return;

    const next = this.waiters[0];
    if (next.kind === "writer") {
      this.waiters.shift();
      next.resolve();
      return;
    }
    while (this.waiters.length > 0 && this.waiters[0].kind === "reader") {
      const r = this.waiters.shift()!;
      r.resolve();
    }
  }
}

// ----------------------------- ToolHandler / ToolRegistry -----------------------------

export interface ToolHandler {
  name(): string;
  spec(): unknown;
  /** 默认 false：与同批工具不并发执行（写锁）。 */
  supportsParallelToolCalls?(): boolean;
  /** 默认 true：dispatch 时通过 registry 的读写锁协调。 */
  requiresDispatchLock?(): boolean;
  handle(registry: ToolRegistry, argumentsJson: string): Promise<string>;
}

/** 一次工具调用的请求体。 */
export interface ToolInvocation {
  call_id: string;
  tool_name: string;
  arguments: string;
}

/** 一次工具调用的结果体（保留 call_id 用来与请求 1:1 配对）。 */
export interface ToolInvocationResult {
  call_id: string;
  tool_name: string;
  output: string;
}

export interface PlanItem {
  step: string;
  status: "pending" | "in_progress" | "completed";
}

function isValidStatus(s: string): s is PlanItem["status"] {
  return s === "pending" || s === "in_progress" || s === "completed";
}

/**
 * ToolRegistry —— 工具表 + 共享会话状态。
 *
 * 在 step08 里它额外背着一个 AgentTeamManager：所有 spawn_agent / send_input /
 * wait_agent 共享同一份 agent 表，子 agent worker 在跑 tool calls 时也是
 * 通过这里的 dispatchMany 进入 —— 也就是说工具集对父 agent 和子 agent 完全
 * 是同一份。
 */
export class ToolRegistry {
  private readonly handlers = new Map<string, ToolHandler>();
  private planState: PlanItem[] = [];
  private readonly parallelLock = new ReadWriteLock();
  private readonly agentTeam = new AgentTeamManager();

  register(handler: ToolHandler): void {
    console.log(`[registry] registering tool: ${handler.name()}`);
    this.handlers.set(handler.name(), handler);
  }

  async dispatch(name: string, argumentsJson: string): Promise<string | null> {
    const h = this.handlers.get(name);
    if (!h) return null;
    console.log(`[registry] dispatching tool: ${name}`);

    const requiresLock = h.requiresDispatchLock?.() ?? true;
    if (!requiresLock) {
      return h.handle(this, argumentsJson);
    }
    const supportsParallel = h.supportsParallelToolCalls?.() ?? false;
    const release = supportsParallel
      ? await this.parallelLock.acquireRead()
      : await this.parallelLock.acquireWrite();
    try {
      return await h.handle(this, argumentsJson);
    } finally {
      release();
    }
  }

  async dispatchMany(invocations: ToolInvocation[]): Promise<ToolInvocationResult[]> {
    return Promise.all(
      invocations.map(async (inv) => {
        try {
          const out = await this.dispatch(inv.tool_name, inv.arguments);
          return {
            call_id: inv.call_id,
            tool_name: inv.tool_name,
            output: out ?? toolNotFoundOutput(inv.tool_name, inv.arguments),
          };
        } catch (e) {
          // 任意未捕获错误转换为结构化失败 —— 不让一个工具的崩溃拖垮整批。
          return {
            call_id: inv.call_id,
            tool_name: inv.tool_name,
            output: toolError(
              inv.tool_name,
              "tool_task_failed",
              `tool task failed: ${(e as Error).message}`,
              { arguments: inv.arguments },
            ),
          };
        }
      }),
    );
  }

  getSpecs(): unknown[] {
    return Array.from(this.handlers.values()).map((h) => h.spec());
  }

  // ----- agent team 暴露给 handler 的内部 API -----

  spawnAgentThread(role: string, systemPrompt: string, instruction: string): AgentThread {
    return this.agentTeam.spawnAgent(role, systemPrompt, instruction);
  }

  getAgentThread(id: string): AgentThread | undefined {
    return this.agentTeam.get(id);
  }

  agentSnapshots(): AgentSnapshot[] {
    return this.agentTeam.listSnapshots();
  }

  updatePlanState(plan: PlanItem[]): { ok: true; plan: PlanItem[] } | { ok: false; reason: string } {
    const invalid = plan.find((it) => !isValidStatus(it.status));
    if (invalid) {
      return { ok: false, reason: `invalid plan status '${invalid.status}' for step '${invalid.step}'` };
    }
    if (plan.filter((it) => it.status === "in_progress").length > 1) {
      return { ok: false, reason: "plan can contain at most one in_progress step" };
    }
    this.planState = plan.map((it) => ({ ...it }));
    return { ok: true, plan: this.planState.map((it) => ({ ...it })) };
  }
}

// ----------------------------- 共享辅助 -----------------------------

const MAX_TOOL_CONTENT_CHARS = 10_000;
const BASH_OUTPUT_CHARS_PER_STREAM = 4_000;
const COMMAND_TIMEOUT_MS = 10_000;
const DEMO_SUPPORTS_PARALLEL_TOOL_CALLS = true;

const SUB_AGENT_SYSTEM_PROMPT =
  "你是一个辅助主 agent 的子 agent。你可以使用多种工具。请高效地完成分配给你的任务。";
const DEFAULT_WAIT_AGENT_TIMEOUT_MS = 30_000;

interface AgentExecutionConfig {
  apiKey: string;
  baseUrl: string;
  modelName: string;
}

function checkSafeCommand(cmd: string): { ok: true } | { ok: false; reason: string } {
  const normalized = cmd.trim().toLowerCase();
  for (const k of [
    "rm -rf",
    "mkfs",
    "dd if=",
    "halt",
    "reboot",
    "shutdown",
    "> /dev/sda",
    "sudo ",
    "chmod -r 777 /",
  ]) {
    if (normalized.includes(k)) return { ok: false, reason: `command rejected by policy: contains '${k}'` };
  }
  return { ok: true };
}

function truncateText(content: string, maxChars: number): TruncatedText {
  if (content.length <= maxChars) return { content, truncated: false, omitted_chars: 0 };
  const half = Math.floor(maxChars / 2);
  const prefix = content.slice(0, half);
  const suffix = content.slice(content.length - half);
  const omitted_chars = content.length - (prefix.length + suffix.length);
  return {
    content: `${prefix}\n\n... [TRUNCATED ${omitted_chars} CHARACTERS] ...\n\n${suffix}`,
    truncated: true,
    omitted_chars,
  };
}

async function executeBash(cmd: string): Promise<string> {
  const safe = checkSafeCommand(cmd);
  if (!safe.ok) return toolError("run_bash", "policy_denied", safe.reason, { cmd });
  console.log(`[sandbox] executing command: ${cmd}`);
  const cwd = process.cwd();
  return new Promise<string>((resolve) => {
    const child = spawn("bash", ["-lc", cmd], { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let so = "";
    let se = "";
    let timedOut = false;
    let spawnError: Error | null = null;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, COMMAND_TIMEOUT_MS);
    child.stdout?.on("data", (c: Buffer) => (so += c.toString("utf8")));
    child.stderr?.on("data", (c: Buffer) => (se += c.toString("utf8")));
    child.on("error", (e) => (spawnError = e));
    child.on("close", (code) => {
      clearTimeout(timer);
      if (spawnError) {
        resolve(
          toolError("run_bash", "spawn_failed", `terminal execution failed: ${spawnError.message}`, {
            cmd,
            cwd,
          }),
        );
        return;
      }
      if (timedOut) {
        resolve(
          toolError(
            "run_bash",
            "timeout",
            `command exceeded ${COMMAND_TIMEOUT_MS / 1000} seconds and was terminated`,
            { cmd, cwd, timeout_secs: COMMAND_TIMEOUT_MS / 1000 },
          ),
        );
        return;
      }
      const stdout = truncateText(so, BASH_OUTPUT_CHARS_PER_STREAM);
      const stderr = truncateText(se, BASH_OUTPUT_CHARS_PER_STREAM);
      const data = { cmd, cwd, exit_code: code, stdout, stderr, timed_out: false };
      resolve(
        code === 0
          ? toolSuccess("run_bash", "command executed successfully", data)
          : toolError("run_bash", "non_zero_exit", "command exited with a non-zero status", data),
      );
    });
  });
}

async function readFileImpl(path: string): Promise<string> {
  console.log(`[sandbox] reading file: ${path}`);
  try {
    const content = await fs.readFile(path, "utf8");
    return toolSuccess("read_file", `read file successfully: ${path}`, {
      path,
      content: truncateText(content, MAX_TOOL_CONTENT_CHARS),
    });
  } catch (e) {
    return toolError("read_file", "read_failed", `read file failed: ${(e as Error).message}`, { path });
  }
}

async function writeFileImpl(path: string, content: string): Promise<string> {
  console.log(`[sandbox] writing file: ${path}`);
  try {
    await fs.writeFile(path, content);
    return toolSuccess("write_file", `file written successfully: ${path}`, {
      path,
      bytes_written: content.length,
    });
  } catch (e) {
    return toolError("write_file", "write_failed", `write file failed: ${(e as Error).message}`, {
      path,
      bytes_attempted: content.length,
    });
  }
}

async function editFileImpl(path: string, target: string, replacement: string): Promise<string> {
  console.log(`[sandbox] editing file: ${path}`);
  let content: string;
  try {
    content = await fs.readFile(path, "utf8");
  } catch (e) {
    return toolError("edit_file", "read_failed", `edit file failed: ${(e as Error).message}`, { path });
  }
  if (!content.includes(target)) {
    return toolError("edit_file", "target_not_found", "edit file failed: target string not found", {
      path,
      target,
    });
  }
  const newContent = content.split(target).join(replacement);
  try {
    await fs.writeFile(path, newContent);
    return toolSuccess("edit_file", `file edited successfully: ${path}`, {
      path,
      target: truncateText(target, Math.floor(MAX_TOOL_CONTENT_CHARS / 4)),
      replacement: truncateText(replacement, Math.floor(MAX_TOOL_CONTENT_CHARS / 4)),
    });
  } catch (e) {
    return toolError("edit_file", "write_failed", `edit file failed: ${(e as Error).message}`, { path });
  }
}

// ----------------------------- 后台 agent worker -----------------------------

/**
 * 跑一个 agent 的 worker 循环：尽量把 pending_inputs 抽干。每条 instruction
 * 都会被推进 history（role:"user"），然后调用 runAgentTurn 去和模型反复做
 * tool-call 直到拿到一段最终文本。
 *
 * 退出条件：
 *   - agent 被 close()；
 *   - 队列空 + markWorkerStopped 后没人塞新输入。
 *
 * 这里有一个易踩的边界：把 workerActive 置回 false 之后，必须再检查一次队列，
 * 因为 enqueueInput 是 fire-and-forget，可能正好在“我们决定退出”和“真正退出”
 * 之间塞了新输入进来 —— 这种情况下重新 CAS 拉一轮。
 */
function startAgentWorker(
  registry: ToolRegistry,
  agent: AgentThread,
  config: AgentExecutionConfig,
): void {
  if (!agent.tryStartWorker()) {
    return;
  }

  // 故意不 await：等价于 Rust 的 tokio::spawn —— 让 worker 在事件循环里独立跑，
  // 调用方（spawn_agent / send_input 工具）立刻拿到“已派发”的同步反馈。
  void (async () => {
    while (true) {
      if (agent.isClosed()) {
        agent.markWorkerStopped();
        break;
      }

      const instruction = agent.takeNextInput();
      if (instruction === null) {
        // 队列暂时空：先把 worker 标记为停下来，再做一次“是否还有输入”的
        // double check。如果在我们置 false 的瞬间被塞了新输入，但当前
        // workerActive 还是 false（说明没人接），就 CAS 重新接管。
        agent.markWorkerStopped();
        if (agent.hasPendingInputs() && agent.tryStartWorker()) {
          continue;
        }
        break;
      }

      agent.setStatus("running");
      agent.pushHistoryItem({ role: "user", content: instruction });

      try {
        const finalContent = await runAgentTurn(registry, agent, config);
        if (!agent.isClosed()) {
          agent.setLastResult(finalContent);
          agent.setStatus("completed");
        }
      } catch (e) {
        if (!agent.isClosed()) {
          agent.setLastError((e as Error).message);
          agent.setStatus("failed");
        }
      }
    }
  })();
}

/**
 * 跑“一轮”agent：从当前 history 出发反复请求模型，直到模型给出非 tool_calls
 * 的最终回答；中途模型每次返回 tool_calls 都通过 registry.dispatchMany 跑，
 * 然后把工具结果 push 回 agent.history。
 *
 * 这个函数会抛错（外层 startAgentWorker 转成 set_last_error）。
 */
async function runAgentTurn(
  registry: ToolRegistry,
  agent: AgentThread,
  config: AgentExecutionConfig,
): Promise<string> {
  const toolSpecs = registry.getSpecs();

  while (true) {
    if (agent.isClosed()) {
      throw new Error("agent closed before completion");
    }

    const history = agent.historySnapshot();
    const res = await fetch(config.baseUrl, {
      method: "POST",
      headers: { Authorization: `Bearer ${config.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: config.modelName,
        messages: history,
        tools: toolSpecs,
        parallel_tool_calls: DEMO_SUPPORTS_PARALLEL_TOOL_CALLS,
        temperature: 0.2,
      }),
    });

    if (!res.ok) {
      throw new Error(`sub-agent API call failed: HTTP ${res.status}`);
    }

    let body: any;
    try {
      body = await res.json();
    } catch (e) {
      throw new Error(`failed to parse sub-agent response: ${(e as Error).message}`);
    }

    const choice = body.choices?.[0];
    const message = choice?.message;
    if (!message) {
      throw new Error("sub-agent response missing choices[0].message");
    }
    agent.pushHistoryItem(message);

    const wantsTool = choice.finish_reason === "tool_calls" || Array.isArray(message.tool_calls);
    if (!wantsTool) {
      return typeof message.content === "string" && message.content.length > 0
        ? message.content
        : "Done";
    }

    const toolCalls: any[] = message.tool_calls ?? [];
    console.log(`[agent-team] ${agent.id()} requested ${toolCalls.length} tool call(s)`);

    const invocations: ToolInvocation[] = [];
    for (const tc of toolCalls) {
      const fname = tc.function?.name as string | undefined;
      const fargs = tc.function?.arguments as string | undefined;
      const callId = tc.id as string | undefined;
      if (!fname || fargs === undefined || !callId) continue;
      console.log(`[agent-team] ${agent.id()} tool request: ${fname}(${fargs})`);
      invocations.push({ call_id: callId, tool_name: fname, arguments: fargs });
    }

    const outputs = await registry.dispatchMany(invocations);
    for (const out of outputs) {
      agent.pushHistoryItem({ role: "tool", content: out.output, tool_call_id: out.call_id });
    }
  }
}

/**
 * 等待某个 agent 进入 final 状态（completed/failed/closed），带超时。
 *
 * 返回 [status, timedOut]：
 *   - timedOut=true 时 status 是“当前最新”状态而不是 final 状态（用于回报“还没好”）。
 *
 * 实现是 Promise.race：广播器的 once + setTimeout，谁先到就用谁。注意要在
 * winner 确定后清掉对方（clearTimeout / 让 once 的 listener 自然失效）以
 * 避免 Node 进程因为悬挂的 timer 退不出去。
 */
async function waitForAgentStatus(
  agent: AgentThread,
  timeoutMs: number,
): Promise<{ status: AgentStatus; timedOut: boolean }> {
  const current = agent.getStatus();
  if (isFinalStatus(current)) {
    return { status: current, timedOut: false };
  }

  let timer: NodeJS.Timeout | null = null;
  const finalStatus: Promise<{ status: AgentStatus; timedOut: false }> = agent
    .waitForStatus(isFinalStatus)
    .then((s) => ({ status: s, timedOut: false }));
  const timeout: Promise<{ status: AgentStatus; timedOut: true }> = new Promise((resolve) => {
    timer = setTimeout(() => {
      resolve({ status: agent.getStatus(), timedOut: true });
    }, timeoutMs);
  });

  try {
    return await Promise.race([finalStatus, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// ----------------------------- 普通工具 Handler -----------------------------

export class RunBashHandler implements ToolHandler {
  name(): string {
    return "run_bash";
  }
  supportsParallelToolCalls(): boolean {
    return true;
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description:
          "Execute a bash command on the user's machine to list files, read code, or make changes.",
        parameters: {
          type: "object",
          properties: {
            cmd: {
              type: "string",
              description:
                "The bash command string to execute (e.g., 'pwd', 'ls -la', 'cat file.rs')",
            },
          },
          required: ["cmd"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(_r: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ cmd: string }>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    return executeBash(p.value.cmd);
  }
}

export class ReadFileHandler implements ToolHandler {
  name(): string {
    return "read_file";
  }
  supportsParallelToolCalls(): boolean {
    return true;
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Read the content of a file from the disk.",
        parameters: {
          type: "object",
          properties: { path: { type: "string", description: "Absolute or relative path to the file." } },
          required: ["path"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(_r: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ path: string }>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    return readFileImpl(p.value.path);
  }
}

export class WriteFileHandler implements ToolHandler {
  name(): string {
    return "write_file";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Create or overwrite a file with new content.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Path where the file should be written." },
            content: { type: "string", description: "The full content to write into the file." },
          },
          required: ["path", "content"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(_r: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ path: string; content: string }>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    return writeFileImpl(p.value.path, p.value.content);
  }
}

export class EditFileHandler implements ToolHandler {
  name(): string {
    return "edit_file";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Edit an existing file by replacing a target string with a new string.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "Path to the file to edit." },
            target: { type: "string", description: "The exact string within the file to be replaced." },
            replacement: {
              type: "string",
              description: "The new string to insert instead of the target.",
            },
          },
          required: ["path", "target", "replacement"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(_r: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ path: string; target: string; replacement: string }>(
      this.name(),
      argumentsJson,
    );
    if (!p.ok) return p.output;
    return editFileImpl(p.value.path, p.value.target, p.value.replacement);
  }
}

// ----------------------------- update_plan -----------------------------

interface UpdatePlanArgs {
  explanation?: string;
  plan: PlanItem[];
}

export class PlanHandler implements ToolHandler {
  name(): string {
    return "update_plan";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description:
          "Updates the task plan. Use this at the start of complex tasks to decompose them into steps, and keep it updated as you progress.",
        parameters: {
          type: "object",
          properties: {
            explanation: { type: "string", description: "An optional explanation for the plan change." },
            plan: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  step: { type: "string" },
                  status: { type: "string", enum: ["pending", "in_progress", "completed"] },
                },
                required: ["step", "status"],
                additionalProperties: false,
              },
            },
          },
          required: ["plan"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<UpdatePlanArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;
    const result = registry.updatePlanState(args.plan);
    if (!result.ok) {
      return toolError(this.name(), "invalid_plan", `plan update rejected: ${result.reason}`, {
        explanation: args.explanation,
        plan: args.plan,
      });
    }
    console.log("\n[plan] update received");
    if (args.explanation) console.log(`Explanation: ${args.explanation}`);
    for (const item of result.plan) console.log(`  - ${item.step} [${item.status}]`);
    console.log();
    return toolSuccess(this.name(), "plan updated successfully", {
      explanation: args.explanation,
      plan: result.plan,
    });
  }
}

// ----------------------------- agent team 工具族 -----------------------------

interface SpawnAgentArgs {
  instruction: string;
  role?: string;
}

/**
 * spawn_agent —— 起一条后台 agent，立刻返回 agent_id；不等它跑完。
 *
 * 与 spawn_sub_agent 的本质区别：spawn_sub_agent 同步阻塞等结果，spawn_agent
 * 是 fire-and-forget，配合 send_input / wait_agent / list_agents 让模型自己
 * 在多个 agent 之间编排。
 */
export class SpawnAgentHandler implements ToolHandler {
  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    private readonly modelName: string,
  ) {}

  name(): string {
    return "spawn_agent";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Spawn a background agent thread and give it an initial instruction.",
        parameters: {
          type: "object",
          properties: {
            instruction: { type: "string", description: "The initial task for the agent." },
            role: { type: "string", description: "Optional agent role label." },
          },
          required: ["instruction"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<SpawnAgentArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const role = args.role ?? "default";
    const agent = registry.spawnAgentThread(role, SUB_AGENT_SYSTEM_PROMPT, args.instruction);
    console.log(`[agent-team] spawned agent: ${agent.id()} [${role}]`);

    startAgentWorker(registry, agent, {
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
      modelName: this.modelName,
    });

    return toolSuccess(this.name(), "agent spawned successfully", {
      agent_id: agent.id(),
      role,
      status: agent.getStatus(),
      agent_snapshots: registry.agentSnapshots(),
    });
  }
}

interface SendAgentInputArgs {
  agent_id: string;
  instruction: string;
}

/**
 * send_input —— 把一条新指令排进已有 agent 的队列。如果该 agent 当前 idle
 * （worker 没在跑），需要重新拉起 worker；不然 worker 自己会下一轮拉到。
 */
export class SendAgentInputHandler implements ToolHandler {
  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    private readonly modelName: string,
  ) {}

  name(): string {
    return "send_input";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Send a follow-up instruction to an existing agent thread.",
        parameters: {
          type: "object",
          properties: {
            agent_id: { type: "string", description: "The target agent identifier." },
            instruction: {
              type: "string",
              description: "The instruction to queue for the target agent.",
            },
          },
          required: ["agent_id", "instruction"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<SendAgentInputArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const agent = registry.getAgentThread(args.agent_id);
    if (!agent) {
      return toolError(this.name(), "agent_not_found", `agent ${args.agent_id} not found`, {
        agent_id: args.agent_id,
      });
    }
    if (agent.isClosed()) {
      return toolError(this.name(), "agent_closed", `agent ${args.agent_id} is closed`, {
        agent_id: args.agent_id,
      });
    }

    console.log(`[agent-team] queued input for ${args.agent_id}: ${args.instruction}`);
    agent.enqueueInput(args.instruction);
    // 始终调一次 startAgentWorker：如果当前正有 worker 跑着，tryStartWorker
    // 会返回 false 直接跳过；否则就把它重新拉起来。
    startAgentWorker(registry, agent, {
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
      modelName: this.modelName,
    });

    return toolSuccess(this.name(), "input queued successfully", {
      agent_id: args.agent_id,
      status: agent.getStatus(),
      agent_snapshots: registry.agentSnapshots(),
    });
  }
}

interface WaitAgentArgs {
  agent_id: string;
  timeout_ms?: number;
}

/**
 * wait_agent —— 等某个 agent 走到 final 状态。带 timeout_ms（默认 30s）。
 * 即便超时也能返回 —— data.timed_out 为 true，data.status 是“此刻最新”。
 */
export class WaitAgentHandler implements ToolHandler {
  name(): string {
    return "wait_agent";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Wait for an agent to reach a final state or until the timeout expires.",
        parameters: {
          type: "object",
          properties: {
            agent_id: { type: "string", description: "The target agent identifier." },
            timeout_ms: { type: "integer", description: "Optional timeout in milliseconds." },
          },
          required: ["agent_id"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<WaitAgentArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const agent = registry.getAgentThread(args.agent_id);
    if (!agent) {
      return toolError(this.name(), "agent_not_found", `agent ${args.agent_id} not found`, {
        agent_id: args.agent_id,
      });
    }

    const timeoutMs = args.timeout_ms ?? DEFAULT_WAIT_AGENT_TIMEOUT_MS;
    const { status, timedOut } = await waitForAgentStatus(agent, timeoutMs);

    return toolSuccess(this.name(), timedOut ? "wait timed out" : "agent reached a final state", {
      agent_id: args.agent_id,
      status,
      timed_out: timedOut,
      final_output: agent.lastResult(),
      error: agent.lastError(),
    });
  }
}

interface CloseAgentArgs {
  agent_id: string;
}

/**
 * close_agent —— 把 agent 标记成关闭。worker 会在下一次循环顶端看到 closed
 * 后自己退出；setStatus("closed") 也会唤醒所有挂在 wait_agent 上的等待者。
 */
export class CloseAgentHandler implements ToolHandler {
  name(): string {
    return "close_agent";
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "Mark an agent as closed so it no longer accepts new work.",
        parameters: {
          type: "object",
          properties: {
            agent_id: { type: "string", description: "The target agent identifier." },
          },
          required: ["agent_id"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<CloseAgentArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const agent = registry.getAgentThread(args.agent_id);
    if (!agent) {
      return toolError(this.name(), "agent_not_found", `agent ${args.agent_id} not found`, {
        agent_id: args.agent_id,
      });
    }

    agent.close();
    console.log(`[agent-team] closed agent: ${args.agent_id}`);

    return toolSuccess(this.name(), "agent closed successfully", {
      agent_id: args.agent_id,
      status: agent.getStatus(),
      agent_snapshots: registry.agentSnapshots(),
    });
  }
}

/**
 * list_agents —— 只读地把所有 agent 的快照吐出来给父 agent 看。可与其它
 * 只读工具并行（supportsParallelToolCalls=true）。
 */
export class ListAgentsHandler implements ToolHandler {
  name(): string {
    return "list_agents";
  }
  supportsParallelToolCalls(): boolean {
    return true;
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description: "List the currently known agent threads and their statuses.",
        parameters: {
          type: "object",
          properties: {},
          additionalProperties: false,
        },
      },
    };
  }
  async handle(registry: ToolRegistry, _argumentsJson: string): Promise<string> {
    return toolSuccess(this.name(), "listed agents successfully", {
      agents: registry.agentSnapshots(),
    });
  }
}

// ----------------------------- spawn_sub_agent（同步等待版） -----------------------------

interface SubAgentArgs {
  instruction: string;
}

/**
 * step08 的 spawn_sub_agent 实际上变成了“spawn_agent + wait_agent”的语法糖：
 * 起一条后台 agent，同步等它进 final 状态，然后把 last_result 直接当成
 * 工具结果返回给父 agent。这样父 agent 拿到的体验仍然是“一行 tool call 拿到
 * 一段总结”，但底层走的是同一套 agent runtime —— 有助于未来在 step09 里
 * 引入“父子关系 / 强制取消”等概念。
 */
export class SubAgentHandler implements ToolHandler {
  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    private readonly modelName: string,
  ) {}

  name(): string {
    return "spawn_sub_agent";
  }
  // 关键：sub-agent 自己内部要再调工具，不能被父 dispatch 的写锁卡住。
  requiresDispatchLock(): boolean {
    return false;
  }
  spec(): unknown {
    return {
      type: "function",
      function: {
        name: this.name(),
        description:
          "Spawn a sub-agent to perform a specific sub-task. The sub-agent has access to all your tools.",
        parameters: {
          type: "object",
          properties: {
            instruction: {
              type: "string",
              description:
                "The specific instruction for the sub-agent. Be clear and provide necessary context.",
            },
          },
          required: ["instruction"],
          additionalProperties: false,
        },
      },
    };
  }

  async handle(registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<SubAgentArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const agent = registry.spawnAgentThread("default", SUB_AGENT_SYSTEM_PROMPT, args.instruction);
    const snapshot = agent.snapshot();
    console.log(`[agent-team] spawned agent: ${snapshot.id} [${snapshot.role}]`);
    console.log(`[sub-agent] task assigned: ${args.instruction}`);

    startAgentWorker(registry, agent, {
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
      modelName: this.modelName,
    });

    const { status, timedOut } = await waitForAgentStatus(agent, DEFAULT_WAIT_AGENT_TIMEOUT_MS);
    if (timedOut) {
      return toolError(
        this.name(),
        "wait_timed_out",
        "子 agent 在超时前未完成",
        {
          agent_id: agent.id(),
          instruction: args.instruction,
          status,
        },
      );
    }

    if (status === "completed") {
      console.log("[sub-agent] task completed");
      return toolSuccess(this.name(), "子 agent 任务完成", {
        agent_id: agent.id(),
        instruction: args.instruction,
        final_content: agent.lastResult(),
        agent_snapshots: registry.agentSnapshots(),
      });
    }
    if (status === "failed") {
      return toolError(
        this.name(),
        "sub_agent_failed",
        agent.lastError() ?? "子 agent 执行失败",
        {
          agent_id: agent.id(),
          instruction: args.instruction,
          status: "failed",
        },
      );
    }
    if (status === "closed") {
      return toolError(
        this.name(),
        "sub_agent_closed",
        "子 agent 在完成之前已被关闭",
        {
          agent_id: agent.id(),
          instruction: args.instruction,
        },
      );
    }
    // pending / running：理论上 waitForAgentStatus 在没超时的情况下不会返回这种值，
    // 不过为完整起见仍然给一个分支（与 Rust 保持一致）。
    return toolError(
      this.name(),
      "sub_agent_incomplete",
      "子 agent 没有进入最终状态",
      {
        agent_id: agent.id(),
        instruction: args.instruction,
        status,
      },
    );
  }
}

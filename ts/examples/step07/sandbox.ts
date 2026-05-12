// =============================================================================
// step07 / sandbox.ts — 与 step06 完全一致的工具实现
// -----------------------------------------------------------------------------
// step07 的真正演进发生在 agent_team.ts（AgentThread / AgentTeamManager 雏形）。
// 工具层本身保持和 step06 一致：parallel_tool_calls + 读写锁 + dispatchMany。
// 这里独立放一份是为了 step07 可以单独运行而不需要跨目录 import。
// =============================================================================

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { fetch } from "undici";

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
 * 这个实现满足 step06 的需求：
 *   - 写者排队期间，新读者必须等（避免写饥饿）；
 *   - 释放后按 FIFO 唤醒。
 */
class ReadWriteLock {
  private readers = 0;
  private writer = false;
  // 等待队列。一个 "writer" 节点对应一个 acquireWrite，"reader" 节点对应 acquireRead。
  private waiters: Array<{ kind: "reader" | "writer"; resolve: () => void }> = [];

  async acquireRead(): Promise<() => void> {
    // 没有写者占用，且队伍前方没有写者在排队 —— 立刻进。
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
    // 队头是读者：把所有连续的读者一起唤醒（不能跨过写者）。
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

export class ToolRegistry {
  private readonly handlers = new Map<string, ToolHandler>();
  private planState: PlanItem[] = [];
  private readonly parallelLock = new ReadWriteLock();

  register(handler: ToolHandler): void {
    console.log(`[registry] registering tool: ${handler.name()}`);
    this.handlers.set(handler.name(), handler);
  }

  /**
   * dispatch 一次：
   *  - 找不到 handler 返回 null；
   *  - 不需要锁则直接执行；
   *  - 支持并行的工具拿读锁；其它拿写锁。
   */
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

  /**
   * 把一批 tool calls 同时发起，但每个调用内部都还会去抢 registry 的读写锁，
   * 所以“并发”其实是“可并发的工具之间并发，不可并发的工具之间排队”。
   */
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

// ----------------------------- 普通工具 Handler -----------------------------

export class RunBashHandler implements ToolHandler {
  name(): string {
    return "run_bash";
  }
  // 跑命令本身可能写文件，但教学上把它视为可与其它 read/bash 并行 —— 与 Rust 一致。
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

// ----------------------------- spawn_sub_agent（用 dispatchMany） -----------------------------

interface SubAgentArgs {
  instruction: string;
}

export class SubAgentHandler implements ToolHandler {
  constructor(
    private readonly apiKey: string,
    private readonly baseUrl: string,
    private readonly modelName: string,
  ) {}

  name(): string {
    return "spawn_sub_agent";
  }
  // 关键：sub-agent 不要被父 dispatch 的写锁卡住，否则它内部的工具调用会与自己死锁。
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

    console.log(`[sub-agent] spawning with instruction: ${args.instruction}`);
    const history: any[] = [
      {
        role: "system",
        content:
          "你是一个辅助主 agent 的子 agent。你可以使用多种工具。请高效地完成分配给你的任务。",
      },
      { role: "user", content: args.instruction },
    ];
    const toolSpecs = registry.getSpecs();

    while (true) {
      const res = await fetch(this.baseUrl, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: this.modelName,
          messages: history,
          tools: toolSpecs,
          parallel_tool_calls: DEMO_SUPPORTS_PARALLEL_TOOL_CALLS,
          temperature: 0.2,
        }),
      });

      if (!res.ok) {
        return toolError(
          this.name(),
          "sub_agent_request_failed",
          `sub-agent API call failed: HTTP ${res.status}`,
          { instruction: args.instruction },
        );
      }

      let body: any;
      try {
        body = await res.json();
      } catch (e) {
        return toolError(
          this.name(),
          "sub_agent_response_invalid",
          `failed to parse sub-agent response: ${(e as Error).message}`,
          { instruction: args.instruction },
        );
      }

      const choice = body.choices?.[0];
      const message = choice?.message;
      if (!message) {
        return toolError(
          this.name(),
          "sub_agent_response_invalid",
          "sub-agent response missing choices[0].message",
          { instruction: args.instruction },
        );
      }
      history.push(message);

      const wantsTool =
        choice.finish_reason === "tool_calls" || Array.isArray(message.tool_calls);
      if (!wantsTool) {
        const finalContent =
          typeof message.content === "string" && message.content.length > 0
            ? message.content
            : "Done";
        console.log("[sub-agent] task completed");
        return toolSuccess(this.name(), "子 agent 任务完成", {
          instruction: args.instruction,
          final_content: finalContent,
        });
      }

      const toolCalls: any[] = message.tool_calls ?? [];
      console.log(`[sub-agent] tool batch received: ${toolCalls.length} call(s)`);

      // step06 关键：批量并发派发，而不是串行 for-of。
      const invocations: ToolInvocation[] = [];
      for (const tc of toolCalls) {
        const fname = tc.function?.name as string | undefined;
        const fargs = tc.function?.arguments as string | undefined;
        const callId = tc.id as string | undefined;
        if (!fname || fargs === undefined || !callId) continue;
        console.log(`[sub-agent] requested tool: ${fname}(${fargs})`);
        invocations.push({ call_id: callId, tool_name: fname, arguments: fargs });
      }

      const outputs = await registry.dispatchMany(invocations);
      for (const out of outputs) {
        history.push({ role: "tool", content: out.output, tool_call_id: out.call_id });
      }
    }
  }
}

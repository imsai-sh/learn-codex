// =============================================================================
// step05 / sandbox.ts — 与 step04 几乎一致的工具实现
// -----------------------------------------------------------------------------
// step05 真正的“演进”发生在主循环：上下文压缩 + skill 注入。
// 工具层（run_bash / read_file / write_file / edit_file / update_plan /
// spawn_sub_agent）保持与 step04 完全一致 —— 这里独立放一份是为了让 step05
// 可以单独运行，不用跨目录 import。
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

// ----------------------------- ToolRegistry（带状态） -----------------------------

/**
 * step04 中的 ToolHandler 多了一个 registry 参数：
 * 这样工具内部可以反过来调用 registry，从而实现“工具调用工具”、“代理调用代理”。
 */
export interface ToolHandler {
  name(): string;
  spec(): unknown;
  handle(registry: ToolRegistry, argumentsJson: string): Promise<string>;
}

export interface PlanItem {
  step: string;
  status: "pending" | "in_progress" | "completed";
}

function isValidStatus(s: string): s is PlanItem["status"] {
  return s === "pending" || s === "in_progress" || s === "completed";
}

/**
 * ToolRegistry 现在不仅是“工具表”，还承载 session 范围内的共享状态（plan_state）。
 * 把状态放在 registry 上的好处：sub-agent 也能看到同一份 plan，不会失忆。
 */
export class ToolRegistry {
  private readonly handlers = new Map<string, ToolHandler>();
  private planState: PlanItem[] = [];

  register(handler: ToolHandler): void {
    console.log(`[registry] registering tool: ${handler.name()}`);
    this.handlers.set(handler.name(), handler);
  }

  async dispatch(name: string, argumentsJson: string): Promise<string | null> {
    const h = this.handlers.get(name);
    if (!h) return null;
    console.log(`[registry] dispatching tool: ${name}`);
    return h.handle(this, argumentsJson);
  }

  getSpecs(): unknown[] {
    return Array.from(this.handlers.values()).map((h) => h.spec());
  }

  /**
   * 校验并写入 plan，返回写入后的最新 plan 副本（避免外部直接拿到内部数组的引用）。
   * 校验语义和 Rust 版完全一致。
   */
  updatePlanState(plan: PlanItem[]): { ok: true; plan: PlanItem[] } | { ok: false; reason: string } {
    const invalid = plan.find((it) => !isValidStatus(it.status));
    if (invalid) {
      return {
        ok: false,
        reason: `invalid plan status '${invalid.status}' for step '${invalid.step}'`,
      };
    }
    const inProgressCount = plan.filter((it) => it.status === "in_progress").length;
    if (inProgressCount > 1) {
      return { ok: false, reason: "plan can contain at most one in_progress step" };
    }
    this.planState = plan.map((it) => ({ ...it }));
    return { ok: true, plan: this.planState.map((it) => ({ ...it })) };
  }
}

// ----------------------------- 公共辅助 -----------------------------

const MAX_TOOL_CONTENT_CHARS = 100_000;
const BASH_OUTPUT_CHARS_PER_STREAM = 40_000;
const COMMAND_TIMEOUT_MS = 10_000;

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

// ----------------------------- 普通工具的 Handler -----------------------------

export class RunBashHandler implements ToolHandler {
  name(): string {
    return "run_bash";
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
  async handle(_registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ cmd: string }>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    return executeBash(p.value.cmd);
  }
}

export class ReadFileHandler implements ToolHandler {
  name(): string {
    return "read_file";
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
  async handle(_registry: ToolRegistry, argumentsJson: string): Promise<string> {
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
  async handle(_registry: ToolRegistry, argumentsJson: string): Promise<string> {
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
  async handle(_registry: ToolRegistry, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ path: string; target: string; replacement: string }>(
      this.name(),
      argumentsJson,
    );
    if (!p.ok) return p.output;
    return editFileImpl(p.value.path, p.value.target, p.value.replacement);
  }
}

// ----------------------------- PlanHandler（写入 registry） -----------------------------

interface UpdatePlanArgs {
  explanation?: string;
  plan: PlanItem[];
}

/**
 * step04 的 PlanHandler 不再仅“校验+回显”，而是把 plan 写入 registry.planState。
 * 这样后续即便发生 sub-agent 嵌套调用，也能共享同一份 plan。
 */
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

// ----------------------------- step04 新增：spawn_sub_agent -----------------------------

interface SubAgentArgs {
  instruction: String;
}

/**
 * SubAgentHandler 的核心思想：
 *   - 它是一个普通工具，但内部会**新开一个独立的 agent loop**；
 *   - 它的子 agent 拥有完全相同的工具集（registry.getSpecs），
 *     所以子 agent 可以读文件、跑命令，甚至再 spawn_sub_agent；
 *   - 子 agent 的所有工具调用都通过 registry.dispatch 走回主调度器，
 *     这意味着 plan_state 等共享状态自然可见。
 *
 * 限制：这一版子 agent 的 history 完全独立，不继承父 agent 的对话；
 * 这种简化模型在后面 step09 会被 fork_context 取代。
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

    // 子 agent 的初始 history：仅包含一个 system 提示和用户 instruction。
    const history: unknown[] = [
      {
        role: "system",
        content:
          "你是一个辅助主 agent 的子 agent。你可以使用多种工具。请高效地完成分配给你的任务。",
      },
      { role: "user", content: args.instruction },
    ];
    const toolSpecs = registry.getSpecs();

    // 嵌套 agent loop（结构与外层 agent loop 几乎一样）
    while (true) {
      const res = await fetch(this.baseUrl, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          model: this.modelName,
          messages: history,
          tools: toolSpecs,
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

      let responseJson: any;
      try {
        responseJson = await res.json();
      } catch (e) {
        return toolError(
          this.name(),
          "sub_agent_response_invalid",
          `failed to parse sub-agent response: ${(e as Error).message}`,
          { instruction: args.instruction },
        );
      }

      const choice = responseJson.choices?.[0];
      const message = choice?.message;
      if (!message) {
        return toolError(
          this.name(),
          "sub_agent_response_invalid",
          "sub-agent response missing choices[0].message",
          { instruction: args.instruction },
        );
      }
      // 把整条 assistant 消息原样压入子 agent 的历史。
      history.push(message);

      const wantsToolCall =
        choice.finish_reason === "tool_calls" || Array.isArray(message.tool_calls);

      if (!wantsToolCall) {
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
      for (const tc of toolCalls) {
        const fname = tc.function?.name as string | undefined;
        const fargs = tc.function?.arguments as string | undefined;
        if (!fname || fargs === undefined) continue;
        console.log(`[sub-agent] requested tool: ${fname}(${fargs})`);
        // 关键：用同一份 registry 派发 —— 共享所有能力 + 共享 plan 状态
        const out = await registry.dispatch(fname, fargs);
        if (out !== null) {
          history.push({ role: "tool", content: out, tool_call_id: tc.id });
        } else {
          history.push({
            role: "tool",
            content: toolError(fname, "tool_not_found", `tool ${fname} not found`, {
              arguments: fargs,
            }),
            tool_call_id: tc.id,
          });
        }
      }
      // 让子 agent 继续基于工具结果思考。
    }
  }
}

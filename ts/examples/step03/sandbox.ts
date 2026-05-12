// =============================================================================
// step03 / sandbox.ts — 在 step02 基础上新增 update_plan 工具
// -----------------------------------------------------------------------------
// 这一阶段开始让 Agent 不只是“会调工具”，而是“会组织任务”：
//   - 引入 PlanHandler，用 update_plan 工具维护一个步骤列表；
//   - 每个步骤有 step（描述）+ status（pending / in_progress / completed）；
//   - 在 system prompt 里要求模型对复杂任务必须先 update_plan。
// =============================================================================

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";

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

// ----------------------------- 工具基础设施 -----------------------------

export interface ToolHandler {
  name(): string;
  spec(): unknown;
  handle(argumentsJson: string): Promise<string>;
}

export class ToolRegistry {
  private readonly handlers = new Map<string, ToolHandler>();

  register(handler: ToolHandler): void {
    console.log(`[registry] registering tool: ${handler.name()}`);
    this.handlers.set(handler.name(), handler);
  }

  async dispatch(name: string, argumentsJson: string): Promise<string | null> {
    const handler = this.handlers.get(name);
    if (!handler) return null;
    console.log(`[registry] dispatching tool: ${name}`);
    return handler.handle(argumentsJson);
  }

  getSpecs(): unknown[] {
    return Array.from(this.handlers.values()).map((h) => h.spec());
  }
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

// ----------------------------- 通用辅助 -----------------------------

const MAX_TOOL_CONTENT_CHARS = 10_000;
const BASH_OUTPUT_CHARS_PER_STREAM = 4_000;
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
    if (normalized.includes(k)) {
      return { ok: false, reason: `command rejected by policy: contains '${k}'` };
    }
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

// ----------------------------- 底层实现 -----------------------------

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

// ----------------------------- ToolHandler 实现 -----------------------------

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
              description: "The bash command string to execute (e.g., 'pwd', 'ls -la', 'cat file.rs')",
            },
          },
          required: ["cmd"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(argumentsJson: string): Promise<string> {
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
  async handle(argumentsJson: string): Promise<string> {
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
  async handle(argumentsJson: string): Promise<string> {
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
  async handle(argumentsJson: string): Promise<string> {
    const p = parseArgs<{ path: string; target: string; replacement: string }>(
      this.name(),
      argumentsJson,
    );
    if (!p.ok) return p.output;
    return editFileImpl(p.value.path, p.value.target, p.value.replacement);
  }
}

// ----------------------------- step03 新增：update_plan -----------------------------

/**
 * 计划中的一项任务。
 *  - step:   人类可读的步骤描述。
 *  - status: 必须是 pending / in_progress / completed 三选一。
 */
export interface PlanItem {
  step: string;
  status: "pending" | "in_progress" | "completed";
}

export interface UpdatePlanArgs {
  explanation?: string;
  plan: PlanItem[];
}

function isValidStatus(s: string): s is PlanItem["status"] {
  return s === "pending" || s === "in_progress" || s === "completed";
}

/**
 * PlanHandler：协调“任务分解”这件事。
 * 行为约束：
 *   1. 同一时刻只能有一个步骤是 in_progress；
 *   2. status 字段必须是允许的三个枚举值之一；
 *   3. 验证通过后，把当前 plan 完整地回显出来，让模型继续推理。
 *
 * 注意 step03 没有把 plan 状态持久化到 registry（那是 step04 才会做的事），
 * 这里只是“校验 + 打印 + 回执”。
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
            explanation: {
              type: "string",
              description: "An optional explanation for the plan change.",
            },
            plan: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  step: { type: "string" },
                  status: {
                    type: "string",
                    enum: ["pending", "in_progress", "completed"],
                  },
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
  async handle(argumentsJson: string): Promise<string> {
    const p = parseArgs<UpdatePlanArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    // 校验 1：每个 step 的 status 必须合法。
    const invalid = args.plan.find((it) => !isValidStatus(it.status));
    if (invalid) {
      return toolError(
        this.name(),
        "invalid_plan",
        `plan update rejected: invalid status '${invalid.status}' for step '${invalid.step}'`,
        { explanation: args.explanation, plan: args.plan },
      );
    }

    // 校验 2：最多一个步骤同时 in_progress。
    const inProgressCount = args.plan.filter((it) => it.status === "in_progress").length;
    if (inProgressCount > 1) {
      return toolError(
        this.name(),
        "invalid_plan",
        "plan update rejected: plan can contain at most one in_progress step",
        { explanation: args.explanation, plan: args.plan },
      );
    }

    console.log("\n[plan] update received");
    if (args.explanation) console.log(`Explanation: ${args.explanation}`);
    for (const item of args.plan) console.log(`  - ${item.step} [${item.status}]`);
    console.log();

    return toolSuccess(this.name(), "plan updated successfully", {
      explanation: args.explanation,
      plan: args.plan,
    });
  }
}

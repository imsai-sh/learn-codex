// =============================================================================
// step02 / sandbox.ts — 引入 ToolRegistry
// -----------------------------------------------------------------------------
// step01 中“工具”是写死在主循环里的一个 if 分支。在 step02 中我们把它抽出来：
//
//   - 定义 ToolHandler 接口（name / spec / handle）；
//   - 用一个 ToolRegistry 来注册和派发工具；
//   - 在已有 run_bash 之外，再加 read_file / write_file / edit_file。
//
// 这一阶段开始体现 Harness 的一个关键特征：
//
//   工具不是 if/else 写死，而是一个 runtime 管理的“能力集合”。
// =============================================================================

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";

// ---------------------------------------------------------------------------
// 工具结果的统一结构（与 step01 完全一致，保留以方便单文件对照）
// ---------------------------------------------------------------------------

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

function serializeToolResult(result: ToolResult): string {
  try {
    return JSON.stringify(result, null, 2);
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

export function toolSuccess(
  tool: string,
  message: string,
  data: Record<string, unknown>,
): string {
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

// ---------------------------------------------------------------------------
// ToolHandler：一个工具至少要回答三个问题：
//   - 我叫什么（name）
//   - 我的入参 schema 长什么样（spec，会发给模型）
//   - 给我一段“字符串化的 JSON 入参”，我要怎么处理（handle）
// ---------------------------------------------------------------------------

export interface ToolHandler {
  name(): string;
  /** OpenAI function-calling 协议规定的 spec，会作为 tools 字段发给模型。 */
  spec(): unknown;
  /** 真正的执行入口。返回值是“工具的结构化输出（已序列化为字符串）”。 */
  handle(argumentsJson: string): Promise<string>;
}

/**
 * ToolRegistry —— 工具的中央调度器。
 * - register: 把一个 ToolHandler 注册进来；
 * - dispatch: 按名字执行某个工具；
 * - getSpecs: 把所有工具的 schema 收集起来送给模型。
 */
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

  /** 返回所有工具的 spec 列表，供 chat completions 的 tools 字段使用。 */
  getSpecs(): unknown[] {
    return Array.from(this.handlers.values()).map((h) => h.spec());
  }
}

// ---------------------------------------------------------------------------
// 通用辅助：截断、命令安全检查
// ---------------------------------------------------------------------------

const MAX_TOOL_CONTENT_CHARS = 10_000;
const BASH_OUTPUT_CHARS_PER_STREAM = 4_000;
const COMMAND_TIMEOUT_MS = 10_000;

function checkSafeCommand(cmd: string): { ok: true } | { ok: false; reason: string } {
  const normalized = cmd.trim().toLowerCase();
  const dangerousKeywords = [
    "rm -rf",
    "mkfs",
    "dd if=",
    "halt",
    "reboot",
    "shutdown",
    "> /dev/sda",
    "sudo ",
    "chmod -r 777 /",
  ];
  for (const keyword of dangerousKeywords) {
    if (normalized.includes(keyword)) {
      return { ok: false, reason: `command rejected by policy: contains '${keyword}'` };
    }
  }
  return { ok: true };
}

function truncateText(content: string, maxChars: number): TruncatedText {
  if (content.length <= maxChars) {
    return { content, truncated: false, omitted_chars: 0 };
  }
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

// ---------------------------------------------------------------------------
// 底层工具实现：bash / read / write / edit
// ---------------------------------------------------------------------------

/** 在当前目录用 bash -lc 跑一条命令，返回结构化结果（永不抛异常）。 */
async function executeBash(cmd: string): Promise<string> {
  const safe = checkSafeCommand(cmd);
  if (!safe.ok) {
    return toolError("run_bash", "policy_denied", safe.reason, { cmd });
  }

  console.log(`[sandbox] executing command: ${cmd}`);
  const cwd = process.cwd();

  return new Promise<string>((resolve) => {
    const child = spawn("bash", ["-lc", cmd], {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdoutBuf = "";
    let stderrBuf = "";
    let timedOut = false;
    let spawnError: Error | null = null;

    const killTimer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, COMMAND_TIMEOUT_MS);

    child.stdout?.on("data", (chunk: Buffer) => {
      stdoutBuf += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderrBuf += chunk.toString("utf8");
    });
    child.on("error", (err) => {
      spawnError = err;
    });

    child.on("close", (code: number | null) => {
      clearTimeout(killTimer);
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
      const stdout = truncateText(stdoutBuf, BASH_OUTPUT_CHARS_PER_STREAM);
      const stderr = truncateText(stderrBuf, BASH_OUTPUT_CHARS_PER_STREAM);
      const data = { cmd, cwd, exit_code: code, stdout, stderr, timed_out: false };
      if (code === 0) {
        resolve(toolSuccess("run_bash", "command executed successfully", data));
      } else {
        resolve(
          toolError("run_bash", "non_zero_exit", "command exited with a non-zero status", data),
        );
      }
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
  // 注意：String.replace 默认只替换第一次出现，与 Rust 中的 String::replace
  // 行为不同。Rust 是“全部替换”，所以这里改用 split+join 实现“替换全部出现”。
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

// ---------------------------------------------------------------------------
// 各个工具的 ToolHandler 实现
// ---------------------------------------------------------------------------

/**
 * 一个公共辅助：把“字符串化 JSON”解析为期望的 args 类型，失败就返回结构化错误。
 * 用 unknown + cast 而非 generic 是为了便于在错误分支 return ToolResult 字符串。
 */
function parseArgs<T>(tool: string, argumentsJson: string): { ok: true; value: T } | { ok: false; output: string } {
  try {
    return { ok: true, value: JSON.parse(argumentsJson) as T };
  } catch (e) {
    return {
      ok: false,
      output: toolError(tool, "invalid_arguments", `failed to parse arguments for ${tool}: ${(e as Error).message}`, {
        arguments: argumentsJson,
      }),
    };
  }
}

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
  async handle(argumentsJson: string): Promise<string> {
    const parsed = parseArgs<{ cmd: string }>(this.name(), argumentsJson);
    if (!parsed.ok) return parsed.output;
    return executeBash(parsed.value.cmd);
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
          properties: {
            path: { type: "string", description: "Absolute or relative path to the file." },
          },
          required: ["path"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(argumentsJson: string): Promise<string> {
    const parsed = parseArgs<{ path: string }>(this.name(), argumentsJson);
    if (!parsed.ok) return parsed.output;
    return readFileImpl(parsed.value.path);
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
    const parsed = parseArgs<{ path: string; content: string }>(this.name(), argumentsJson);
    if (!parsed.ok) return parsed.output;
    return writeFileImpl(parsed.value.path, parsed.value.content);
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
    const parsed = parseArgs<{ path: string; target: string; replacement: string }>(
      this.name(),
      argumentsJson,
    );
    if (!parsed.ok) return parsed.output;
    return editFileImpl(parsed.value.path, parsed.value.target, parsed.value.replacement);
  }
}

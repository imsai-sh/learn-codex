// =============================================================================
// step01 / sandbox.ts
// -----------------------------------------------------------------------------
// 这是最简版本的“沙箱层”。它只暴露一个能力：执行 bash 命令并以结构化的 JSON
// 形式返回结果（包括 stdout、stderr、exit_code 等字段），供模型在下一轮
// 推理时直接观察。
//
// 设计原则（和 Rust 版本完全对齐）：
//   1. 危险命令在执行前先被字符串黑名单拒绝；
//   2. 真正的执行用 `bash -lc <cmd>`，等同于 Rust 中 Command::new("bash")
//      .arg("-lc").arg(cmd)，保留 login shell 的环境；
//   3. 命令最长 10 秒；
//   4. stdout / stderr 各自截断到 4000 字符以避免污染上下文；
//   5. 不论成功或失败，都返回一段 JSON 字符串，模型只需读 ok / error_code 字段
//      就能决定下一步动作。
// =============================================================================

import { spawn } from "node:child_process";

// ---------------------------------------------------------------------------
// 常量：截断阈值 / 超时阈值
// ---------------------------------------------------------------------------

/** 单条流（stdout 或 stderr）允许的最大字符数，超出会做“前后留尾”截断。 */
const BASH_OUTPUT_CHARS_PER_STREAM = 4_000;

/** 命令的最大执行时间（毫秒）。超时会被强制终止。 */
const COMMAND_TIMEOUT_MS = 10_000;

// ---------------------------------------------------------------------------
// 工具结果的数据形状
// ---------------------------------------------------------------------------

/**
 * 所有工具调用都使用统一的结果形状。
 *  - ok:        true 表示业务成功；false 表示失败。
 *  - tool:      产生该结果的工具名，便于模型在多工具混跑时定位。
 *  - message:   人类可读的概述。
 *  - error_code 仅在 ok=false 时出现，用于模型的可预测分支判断。
 *  - data:      具体业务字段（命令、退出码、截断后的输出等）。
 */
export interface ToolResult {
  ok: boolean;
  tool: string;
  message: string;
  error_code?: string;
  data: Record<string, unknown>;
}

/** 截断后的文本附加元信息：是否被截、丢了多少字符。 */
export interface TruncatedText {
  content: string;
  truncated: boolean;
  omitted_chars: number;
}

/** 把 ToolResult 序列化为字符串（带缩进，便于人类调试时阅读）。 */
function serializeToolResult(result: ToolResult): string {
  try {
    return JSON.stringify(result, null, 2);
  } catch (e) {
    // 在 TS 中 JSON.stringify 极少失败（除了循环引用），但还是把错误回报模型。
    return JSON.stringify({
      ok: false,
      tool: "internal",
      message: `failed to serialize tool result: ${(e as Error).message}`,
      error_code: "serialization_failed",
      data: {},
    });
  }
}

/** 构造成功结果。 */
export function toolSuccess(
  tool: string,
  message: string,
  data: Record<string, unknown>,
): string {
  return serializeToolResult({ ok: true, tool, message, data });
}

/** 构造失败结果。error_code 是机读的失败类型标签。 */
export function toolError(
  tool: string,
  errorCode: string,
  message: string,
  data: Record<string, unknown>,
): string {
  return serializeToolResult({ ok: false, tool, message, error_code: errorCode, data });
}

// ---------------------------------------------------------------------------
// 安全策略（教学版）
// ---------------------------------------------------------------------------

/**
 * 对命令做简单的黑名单匹配。
 * 注意：这只是教学用途的“象征性”保护，真实生产环境必须用更强的策略
 * （沙箱进程、白名单、二次审批等）。
 */
export function checkSafeCommand(cmd: string): { ok: true } | { ok: false; reason: string } {
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

// ---------------------------------------------------------------------------
// 文本截断：超过 maxChars 时只保留首尾各一半，中间塞一个明显的占位符。
// ---------------------------------------------------------------------------

export function truncateText(content: string, maxChars: number): TruncatedText {
  // JavaScript 中 String 是 UTF-16 序列；为了与 Rust 行为对齐，这里按
  // 字符数（code unit）计长度，并直接 slice，足以满足教学场景。
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
// 真正的命令执行
// ---------------------------------------------------------------------------

/**
 * 执行一条 bash 命令。永远返回一个序列化好的 ToolResult 字符串。
 * - 即使执行失败（spawn 失败、退出码非零、超时…）也不抛异常，
 *   而是把错误编码进结构化结果里——这样上层 agent loop 不需要区分
 *   异常路径和正常路径。
 */
export async function executeBash(cmd: string): Promise<string> {
  const safe = checkSafeCommand(cmd);
  if (!safe.ok) {
    return toolError("run_bash", "policy_denied", safe.reason, { cmd });
  }

  console.log(`[sandbox] executing command: ${cmd}`);
  const cwd = process.cwd();

  return new Promise<string>((resolve) => {
    // 用 -lc 是为了让命令运行在 login shell 上下文里，能继承用户环境变量。
    const child = spawn("bash", ["-lc", cmd], {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdoutBuf = "";
    let stderrBuf = "";
    let timedOut = false;
    let spawnError: Error | null = null;

    // 用 setTimeout 实现 10s 超时；超时后强杀子进程。
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
      // spawn 阶段失败（例如 bash 不存在）。会随后触发 close。
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
      const data = {
        cmd,
        cwd,
        exit_code: code,
        stdout,
        stderr,
        timed_out: false,
      };

      // 与 Rust 版完全对齐：退出码 0 视为成功，否则报 non_zero_exit 错。
      if (code === 0) {
        resolve(toolSuccess("run_bash", "command executed successfully", data));
      } else {
        resolve(toolError("run_bash", "non_zero_exit", "command exited with a non-zero status", data));
      }
    });
  });
}

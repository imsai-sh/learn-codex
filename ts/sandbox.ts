// =============================================================================
// sandbox.ts — 主版本工具沙箱（与 examples/step09/sandbox.ts 等价）
// -----------------------------------------------------------------------------
// 这一份对应 src/sandbox.rs，能力相当于 step09：在 step08 的 agent 后台 worker
// 之上引入“执行上下文”和“父子关系”。每一次工具调用都带着“是谁在调用”的元数据：
//
//   ToolExecutionContext { caller_agent_id?, caller_role?, caller_depth }
//
// 它沿着 dispatch / handle 一路传下去，于是工具可以做出“依赖调用方身份”的
// 决策。在 step09 里有四件具体的事用到它：
//
//   1) explorer 角色不允许写文件 / 改文件系统：
//      - write_file / edit_file 直接拒；
//      - run_bash 命中 mutating 关键字时也拒。
//      这套权限边界就是“agent 角色”的实际意义。
//
//   2) spawn_delegated_agent 把“父 agent 是谁、深度多少”从 context 推断出来，
//      再把 fork_context 选项交给 build_agent_history —— 决定要不要把父 agent
//      的非 system 历史塞进新 agent 的初始上下文。
//
//   3) 深度上限 MAX_AGENT_DEPTH 防止 agent 链路无限套娃。
//
//   4) notify_parent_of_agent_update 在子 agent 走到 Completed/Failed/Closed
//      时，往父 agent 的 history 里塞一条 system 消息，让父 agent 在下一轮
//      采样时自然地“感知”子任务的结果。
//
// 这些机制组合起来，就让“agent 团队”从 step08 的“一群孤岛”变成了 step09 的
// “有结构的小型组织”。
// =============================================================================

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { fetch } from "undici";

import {
  AgentRole,
  AgentRoleHelpers,
  AgentSnapshot,
  AgentSpawnRequest,
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

// ----------------------------- 一个迷你“读写锁”（与 step06 一致） -----------------------------

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

// ----------------------------- ToolExecutionContext -----------------------------

/**
 * 每次工具调用都会带着这个上下文。它的存在让“工具可以根据是谁在调自己”
 * 来做不同决策（最典型：explorer 不能写）。
 *
 * - root() 表示调用来自顶层 agent loop（step09.ts 的主循环）；这种情况下
 *   caller_role 是 undefined，所有权限检查都默认放行；
 * - for_agent() 表示调用来自某个 AgentThread 的 run_agent_turn；这时
 *   role / depth / id 都填进来，下游工具就能感知“这是 explorer 在调”。
 */
export interface ToolExecutionContext {
  caller_agent_id?: string;
  caller_role?: AgentRole;
  caller_depth: number;
}

export const ToolExecutionContext = {
  /** 顶层调用：没有调用方 agent，深度从 0 起算。 */
  root(): ToolExecutionContext {
    return { caller_depth: 0 };
  },

  /** 由某个 AgentThread 发起的调用：把 id / role / depth 拎出来。 */
  forAgent(agent: AgentThread): ToolExecutionContext {
    return {
      caller_agent_id: agent.id(),
      caller_role: agent.role(),
      caller_depth: agent.depth(),
    };
  },
};

// ----------------------------- ToolHandler / ToolRegistry -----------------------------

export interface ToolHandler {
  name(): string;
  spec(): unknown;
  /** 默认 false：与同批工具不并发执行（写锁）。 */
  supportsParallelToolCalls?(): boolean;
  /** 默认 true：dispatch 时通过 registry 的读写锁协调。 */
  requiresDispatchLock?(): boolean;
  /**
   * step09 的关键变化：handle 多了 context 参数。所有需要做权限判断或父子推
   * 断的工具（write_file / edit_file / run_bash / spawn_agent / spawn_sub_agent）
   * 都直接读它。
   */
  handle(registry: ToolRegistry, context: ToolExecutionContext, argumentsJson: string): Promise<string>;
}

/** 一次工具调用的请求体；和 step08 比多了 context。 */
export interface ToolInvocation {
  call_id: string;
  tool_name: string;
  arguments: string;
  context: ToolExecutionContext;
}

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
  private readonly agentTeam = new AgentTeamManager();

  register(handler: ToolHandler): void {
    console.log(`[registry] registering tool: ${handler.name()}`);
    this.handlers.set(handler.name(), handler);
  }

  /**
   * 单次 dispatch：找不到 handler 返回 null；不需要锁则直接执行；支持并行的
   * 工具拿读锁；其它拿写锁。step09 多了 context 参数，会一路转发到 handler。
   */
  async dispatch(
    name: string,
    context: ToolExecutionContext,
    argumentsJson: string,
  ): Promise<string | null> {
    const h = this.handlers.get(name);
    if (!h) return null;
    console.log(`[registry] dispatching tool: ${name}`);

    const requiresLock = h.requiresDispatchLock?.() ?? true;
    if (!requiresLock) {
      return h.handle(this, context, argumentsJson);
    }
    const supportsParallel = h.supportsParallelToolCalls?.() ?? false;
    const release = supportsParallel
      ? await this.parallelLock.acquireRead()
      : await this.parallelLock.acquireWrite();
    try {
      return await h.handle(this, context, argumentsJson);
    } finally {
      release();
    }
  }

  async dispatchMany(invocations: ToolInvocation[]): Promise<ToolInvocationResult[]> {
    return Promise.all(
      invocations.map(async (inv) => {
        try {
          const out = await this.dispatch(inv.tool_name, inv.context, inv.arguments);
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

  spawnAgentThread(request: AgentSpawnRequest): AgentThread {
    return this.agentTeam.spawnAgent(request);
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

// ----------------------------- 共享常量 -----------------------------

const MAX_TOOL_CONTENT_CHARS = 100_000;
const BASH_OUTPUT_CHARS_PER_STREAM = 40_000;
const COMMAND_TIMEOUT_MS = 10_000;
const DEMO_SUPPORTS_PARALLEL_TOOL_CALLS = true;
/**
 * agent 树的最大深度。根 agent 是 0，根直接 spawn 出来的是 1，孙子 agent 是 2。
 * 超过 2 直接拒掉，避免递归 spawn 把模型 / API 调用爆掉。
 */
const MAX_AGENT_DEPTH = 2;

const DEFAULT_WAIT_AGENT_TIMEOUT_MS = 30_000;

interface AgentExecutionConfig {
  apiKey: string;
  baseUrl: string;
  modelName: string;
}

// ----------------------------- 角色 / 历史构造 / 通知父亲 -----------------------------

/**
 * 构造一个新 agent 的初始 history。
 *
 *  - 永远以 role 对应的 system prompt 起头；
 *  - 如果 fork_context=true 且有父 agent，把父 agent 的非 system 历史
 *    （history.slice(1)）追加进来，再补一条 system 消息说明继承关系；
 *  - 否则就只留 role 自己的 system 消息。
 *
 * 返回 (history, fork_applied) —— fork_applied 用来回传给工具结果，让父 agent
 * 看到“fork_context 真的被应用了吗”。
 */
function buildAgentHistory(
  role: AgentRole,
  parent: AgentThread | undefined,
  forkContext: boolean,
): { history: unknown[]; forkApplied: boolean } {
  const history: unknown[] = [
    { role: "system", content: AgentRoleHelpers.systemPrompt(role) },
  ];

  if (!parent) return { history, forkApplied: false };
  if (!forkContext) return { history, forkApplied: false };

  // 父亲的历史第一条是它自己的 system prompt，对孩子不适用 —— 跳掉。
  const parentHistory = parent.historySnapshot();
  for (let i = 1; i < parentHistory.length; i++) history.push(parentHistory[i]);

  history.push({
    role: "system",
    content: `你是由父 agent ${parent.id()} 以 ${AgentRoleHelpers.label(
      role,
    )} 角色 spawn 出来的。上方是从父 agent 继承下来的对话上下文。`,
  });
  return { history, forkApplied: true };
}

/**
 * 如果 caller 是 explorer（不允许 mutation），返回一个结构化的 tool_error；
 * 其它情况返回 null（放行）。reason 用来告诉模型“为什么这次被拒”。
 */
function mutationNotAllowed(
  tool: string,
  context: ToolExecutionContext,
  reason: string,
): string | null {
  const role = context.caller_role;
  if (role === undefined) return null;
  if (AgentRoleHelpers.allowsFileMutation(role)) return null;
  return toolError(tool, "role_violation", reason, {
    caller_agent_id: context.caller_agent_id,
    caller_role: role,
  });
}

/**
 * “这条 bash 命令看起来像在改东西吗？” 用粗糙的子串匹配实现 —— 不是为了卡死
 * 真正的恶意命令（那是 check_safe_command 的活），而是为了让 explorer 在最常
 * 见的写路径上立刻被拦下。误判风险存在但可接受：模型收到 role_violation 就
 * 知道换 worker 角色或父 agent 自己来跑。
 */
function commandLooksMutating(cmd: string): boolean {
  const normalized = cmd.trim().toLowerCase();
  const patterns = [
    "rm ",
    "mv ",
    "cp ",
    "mkdir ",
    "touch ",
    "chmod ",
    "chown ",
    "tee ",
    "sed -i",
    "perl -i",
    "patch ",
    "git apply",
    "git commit",
    "cargo add",
    "npm install",
    "pip install",
    " >",
    " >>",
  ];
  return patterns.some((p) => normalized.includes(p));
}

/**
 * 子 agent 走到终态时，往父 agent 的 history 里压入一条 system 消息。
 * 这样父 agent 在下一次 run_agent_turn 取 history snapshot 时就能自然看到
 * “儿子已经完成 / 失败 / 被关闭了”，不需要单独的事件机制。
 *
 * Pending / Running 不通知 —— 它们不是“变化”，没必要污染父亲历史。
 */
function notifyParentOfAgentUpdate(
  registry: ToolRegistry,
  agent: AgentThread,
  status: AgentStatus,
): void {
  const parentId = agent.parentAgentId();
  if (!parentId) return;
  const parent = registry.getAgentThread(parentId);
  if (!parent) return;

  let content: string;
  switch (status) {
    case "completed":
      content = `子 agent ${agent.id()} [${AgentRoleHelpers.label(
        agent.role(),
      )}] 已完成任务。\n结果：\n${agent.getLastResult() ?? "（无结果）"}`;
      break;
    case "failed":
      content = `子 agent ${agent.id()} [${AgentRoleHelpers.label(
        agent.role(),
      )}] 失败了。\n错误：\n${agent.getLastError() ?? "（未知错误）"}`;
      break;
    case "closed":
      content = `子 agent ${agent.id()} [${AgentRoleHelpers.label(
        agent.role(),
      )}] 在完成之前已被关闭。`;
      break;
    case "pending":
    case "running":
      return;
  }

  parent.pushHistoryItem({ role: "system", content });
  console.log(`[agent-team] notified parent ${parentId} about child ${agent.id()}`);
}

// ----------------------------- agent 后台 worker -----------------------------

/**
 * 启动一个 agent 的后台执行循环（如果还没在跑）。语义和 step08 完全一致，
 * 只是终态时多了一步 notify_parent_of_agent_update。
 */
function startAgentWorker(
  registry: ToolRegistry,
  agent: AgentThread,
  config: AgentExecutionConfig,
): void {
  if (!agent.tryStartWorker()) return;

  // 用 IIFE 启一个独立的 Promise 链；它的生命周期长于调用者，因此我们不 await。
  void (async () => {
    while (true) {
      if (agent.isClosed()) {
        // 被外部 close 掉时也要给父亲一封“已关闭”的便条。
        notifyParentOfAgentUpdate(registry, agent, "closed");
        agent.markWorkerStopped();
        break;
      }

      const instruction = agent.takeNextInput();
      if (instruction === null) {
        // 没活了就放下 worker 标记。中间窗口里如果又有新输入进来，重新抢一次。
        agent.markWorkerStopped();
        if (agent.hasPendingInputs() && agent.tryStartWorker()) continue;
        break;
      }

      agent.setStatus("running");
      // initial_input / 后续 send_input 都是在这个时刻才以 user 角色压入 history。
      agent.pushHistoryItem({ role: "user", content: instruction });

      try {
        const finalContent = await runAgentTurn(registry, agent, config);
        if (!agent.isClosed()) {
          agent.setLastResult(finalContent);
          agent.setStatus("completed");
          notifyParentOfAgentUpdate(registry, agent, "completed");
        }
      } catch (err) {
        if (!agent.isClosed()) {
          agent.setLastError((err as Error).message);
          agent.setStatus("failed");
          notifyParentOfAgentUpdate(registry, agent, "failed");
        }
      }
    }
  })();
}

/**
 * 一次 agent turn：和 step08 一样不断采样 → 派发 tool calls → 把工具结果回填，
 * 直到模型给出一段不再要工具的最终文本。
 *
 * 关键点：每次构造 ToolInvocation 时都用 ToolExecutionContext.forAgent(agent)，
 * 这样下游的 sandbox 工具能把这个 agent 的 role / depth 当作权限输入。
 */
async function runAgentTurn(
  registry: ToolRegistry,
  agent: AgentThread,
  config: AgentExecutionConfig,
): Promise<string> {
  const toolSpecs = registry.getSpecs();

  while (true) {
    if (agent.isClosed()) throw new Error("agent closed before completion");

    const history = agent.historySnapshot();
    const res = await fetch(config.baseUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` },
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
    if (!message) throw new Error("sub-agent response missing choices[0].message");

    agent.pushHistoryItem(message);

    const wantsTool = choice.finish_reason === "tool_calls" || Array.isArray(message.tool_calls);
    if (!wantsTool) {
      const finalContent = typeof message.content === "string" ? message.content : "Done";
      return finalContent.length > 0 ? finalContent : "Done";
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
      invocations.push({
        call_id: callId,
        tool_name: fname,
        arguments: fargs,
        // 这里就是 step09 的“身份注入”入口。
        context: ToolExecutionContext.forAgent(agent),
      });
    }

    const outputs = await registry.dispatchMany(invocations);
    for (const out of outputs) {
      agent.pushHistoryItem({ role: "tool", content: out.output, tool_call_id: out.call_id });
    }
  }
}

/**
 * 等到 agent 走到 final 状态或者超时。返回 (status, timed_out)。
 *
 * - 如果当前已经是 final，直接返回（不算 timed_out）。
 * - 否则订阅 statusBroadcaster；任意一次状态变化都触发检查；
 * - 设一个一次性定时器作为兜底。
 *
 * 注意：单次 await 后立即 unsubscribe / clearTimeout，避免泄露监听器。
 */
async function waitForAgentStatus(
  agent: AgentThread,
  timeoutMs: number,
): Promise<{ status: AgentStatus; timedOut: boolean }> {
  const current = agent.status();
  if (isFinalStatus(current)) return { status: current, timedOut: false };

  return new Promise((resolve) => {
    let settled = false;
    const unsubscribe = agent.subscribeStatus((s) => {
      if (settled) return;
      if (isFinalStatus(s)) {
        settled = true;
        clearTimeout(timer);
        unsubscribe();
        resolve({ status: s, timedOut: false });
      }
    });
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      unsubscribe();
      resolve({ status: agent.status(), timedOut: true });
    }, timeoutMs);
  });
}

// ----------------------------- 普通工具底层实现 -----------------------------

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
              description: "The bash command string to execute (e.g., 'pwd', 'ls -la', 'cat file.rs')",
            },
          },
          required: ["cmd"],
          additionalProperties: false,
        },
      },
    };
  }
  async handle(_r: ToolRegistry, context: ToolExecutionContext, argumentsJson: string): Promise<string> {
    const p = parseArgs<{ cmd: string }>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    // explorer 仍然可以跑只读命令，所以这里只在“看起来像 mutation”的命令上才检查角色。
    if (commandLooksMutating(p.value.cmd)) {
      const denied = mutationNotAllowed(
        this.name(),
        context,
        "explorer agents may only run read-only shell commands",
      );
      if (denied) return denied;
    }
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
  async handle(_r: ToolRegistry, _context: ToolExecutionContext, argumentsJson: string): Promise<string> {
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
  async handle(_r: ToolRegistry, context: ToolExecutionContext, argumentsJson: string): Promise<string> {
    // write_file 永远是 mutation，先做角色检查再解析参数。
    const denied = mutationNotAllowed(
      this.name(),
      context,
      "explorer agents are not allowed to write files",
    );
    if (denied) return denied;
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
  async handle(_r: ToolRegistry, context: ToolExecutionContext, argumentsJson: string): Promise<string> {
    const denied = mutationNotAllowed(
      this.name(),
      context,
      "explorer agents are not allowed to edit files",
    );
    if (denied) return denied;
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
  async handle(registry: ToolRegistry, _context: ToolExecutionContext, argumentsJson: string): Promise<string> {
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

// ----------------------------- 多 agent 协作工具 -----------------------------

/**
 * 解析模型传来的 role 字符串。缺省按 default 处理；不识别就返回 Err 让上层
 * 转成 invalid_role 工具错误。
 */
function resolveAgentRole(label: string | undefined): { ok: true; role: AgentRole } | { ok: false; reason: string } {
  const effective = label ?? "default";
  const role = AgentRoleHelpers.parse(effective);
  if (!role) {
    return {
      ok: false,
      reason: `unsupported role '${effective}'; expected one of: default, explorer, worker`,
    };
  }
  return { ok: true, role };
}

/**
 * spawn_agent / spawn_sub_agent 的共用代码：
 *   1) 从 context.caller_agent_id 推断父 agent；
 *   2) 推断 depth：父存在 ⇒ caller_depth+1；否则 ⇒ 1（顶层 agent 主动 spawn 的孩子算第 1 层）；
 *   3) 检查深度上限；
 *   4) 调 buildAgentHistory 决定要不要 fork 父历史；
 *   5) 把 AgentSpawnRequest 交给 manager 注册。
 *
 * 不在这里启动 worker —— 调用方决定是不是要立刻启动（spawn_agent 是；
 * spawn_sub_agent 也是，但还会同步 wait）。
 */
function spawnDelegatedAgent(
  registry: ToolRegistry,
  context: ToolExecutionContext,
  role: AgentRole,
  instruction: string,
  forkContext: boolean,
): { ok: true; agent: AgentThread; forkApplied: boolean } | { ok: false; reason: string } {
  const parent = context.caller_agent_id ? registry.getAgentThread(context.caller_agent_id) : undefined;
  const depth = parent ? context.caller_depth + 1 : 1;
  if (depth > MAX_AGENT_DEPTH) {
    return {
      ok: false,
      reason: `agent depth limit exceeded: requested depth ${depth}, maximum is ${MAX_AGENT_DEPTH}`,
    };
  }

  const { history, forkApplied } = buildAgentHistory(role, parent, forkContext);
  const request: AgentSpawnRequest = {
    role,
    depth,
    initial_history: history,
    initial_input: instruction,
  };
  if (parent) request.parent_agent_id = parent.id();
  const agent = registry.spawnAgentThread(request);
  return { ok: true, agent, forkApplied };
}

interface SpawnAgentArgs {
  instruction: string;
  role?: string;
  fork_context?: boolean;
}

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
            role: {
              type: "string",
              enum: ["default", "explorer", "worker"],
              description: "Optional agent role label.",
            },
            fork_context: {
              type: "boolean",
              description: "When true, inherit the parent agent's non-system conversation history.",
            },
          },
          required: ["instruction"],
          additionalProperties: false,
        },
      },
    };
  }

  async handle(registry: ToolRegistry, context: ToolExecutionContext, argumentsJson: string): Promise<string> {
    const p = parseArgs<SpawnAgentArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const resolved = resolveAgentRole(args.role);
    if (!resolved.ok) {
      return toolError(this.name(), "invalid_role", resolved.reason, { role: args.role });
    }
    const role = resolved.role;
    const forkContext = args.fork_context ?? false;

    const spawned = spawnDelegatedAgent(registry, context, role, args.instruction, forkContext);
    if (!spawned.ok) {
      return toolError(this.name(), "spawn_rejected", spawned.reason, {
        instruction: args.instruction,
        role,
        fork_context: forkContext,
      });
    }
    const { agent, forkApplied } = spawned;
    console.log(`[agent-team] spawned agent: ${agent.id()} [${AgentRoleHelpers.label(role)}]`);

    startAgentWorker(registry, agent, {
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
      modelName: this.modelName,
    });

    return toolSuccess(this.name(), "agent spawned successfully", {
      agent_id: agent.id(),
      role,
      parent_agent_id: agent.parentAgentId(),
      depth: agent.depth(),
      fork_context_requested: forkContext,
      fork_context_applied: forkApplied,
      status: agent.status(),
      agent_snapshots: registry.agentSnapshots(),
    });
  }
}

interface SendAgentInputArgs {
  agent_id: string;
  instruction: string;
}

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

  async handle(registry: ToolRegistry, _context: ToolExecutionContext, argumentsJson: string): Promise<string> {
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
    // 队列里有新指令但 worker 可能已经退出了，所以再尝试启动一次。
    startAgentWorker(registry, agent, {
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
      modelName: this.modelName,
    });

    return toolSuccess(this.name(), "input queued successfully", {
      agent_id: args.agent_id,
      status: agent.status(),
      agent_snapshots: registry.agentSnapshots(),
    });
  }
}

interface WaitAgentArgs {
  agent_id: string;
  timeout_ms?: number;
}

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

  async handle(registry: ToolRegistry, _context: ToolExecutionContext, argumentsJson: string): Promise<string> {
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
      final_output: agent.getLastResult(),
      error: agent.getLastError(),
    });
  }
}

interface CloseAgentArgs {
  agent_id: string;
}

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

  async handle(registry: ToolRegistry, _context: ToolExecutionContext, argumentsJson: string): Promise<string> {
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
      status: agent.status(),
      agent_snapshots: registry.agentSnapshots(),
    });
  }
}

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
        parameters: { type: "object", properties: {}, additionalProperties: false },
      },
    };
  }
  async handle(registry: ToolRegistry, _context: ToolExecutionContext, _arguments: string): Promise<string> {
    return toolSuccess(this.name(), "listed agents successfully", {
      agents: registry.agentSnapshots(),
    });
  }
}

// ----------------------------- spawn_sub_agent（同步等待版本） -----------------------------

interface SubAgentArgs {
  instruction: string;
}

/**
 * 和 spawn_agent 的区别：spawn_sub_agent 会“同步等到子 agent 跑完”再返回。
 * 因此对父 agent 来说它更像一个普通工具调用，行为上接近 step04 的 sub_agent，
 * 但内部走的是 step08+ 的后台 worker + waitForAgentStatus。
 *
 * step09 的关键改动：父子关系和 fork_context 都通过 spawnDelegatedAgent
 * 正确建立 —— role 固定为 default，fork_context 默认为 true（孩子要继承
 * 父的对话上下文，否则它什么都不知道）。
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
  // 关键：sub-agent 内部还会发工具调用，如果它自己被父 dispatch 的写锁卡住，会死锁。
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

  async handle(registry: ToolRegistry, context: ToolExecutionContext, argumentsJson: string): Promise<string> {
    const p = parseArgs<SubAgentArgs>(this.name(), argumentsJson);
    if (!p.ok) return p.output;
    const args = p.value;

    const spawned = spawnDelegatedAgent(registry, context, "default", args.instruction, true);
    if (!spawned.ok) {
      return toolError(this.name(), "spawn_rejected", spawned.reason, {
        instruction: args.instruction,
      });
    }
    const { agent, forkApplied } = spawned;
    const snap = agent.snapshot();
    console.log(`[agent-team] spawned agent: ${snap.id} [${AgentRoleHelpers.label(snap.role)}]`);
    console.log(`[sub-agent] task assigned: ${args.instruction}`);

    startAgentWorker(registry, agent, {
      apiKey: this.apiKey,
      baseUrl: this.baseUrl,
      modelName: this.modelName,
    });

    const { status, timedOut } = await waitForAgentStatus(agent, DEFAULT_WAIT_AGENT_TIMEOUT_MS);
    if (timedOut) {
      return toolError(this.name(), "wait_timed_out", "子 agent 在超时前未完成", {
        agent_id: agent.id(),
        instruction: args.instruction,
        status,
      });
    }

    switch (status) {
      case "completed":
        console.log("[sub-agent] task completed");
        return toolSuccess(this.name(), "子 agent 任务完成", {
          agent_id: agent.id(),
          instruction: args.instruction,
          fork_context_applied: forkApplied,
          final_content: agent.getLastResult(),
          agent_snapshots: registry.agentSnapshots(),
        });
      case "failed":
        return toolError(
          this.name(),
          "sub_agent_failed",
          agent.getLastError() ?? "子 agent 执行失败",
          {
            agent_id: agent.id(),
            instruction: args.instruction,
            status: "failed",
          },
        );
      case "closed":
        return toolError(this.name(), "sub_agent_closed", "子 agent 在完成之前已被关闭", {
          agent_id: agent.id(),
          instruction: args.instruction,
        });
      case "pending":
      case "running":
        return toolError(
          this.name(),
          "sub_agent_incomplete",
          "子 agent 没有进入最终状态",
          { agent_id: agent.id(), instruction: args.instruction, status },
        );
    }
  }
}

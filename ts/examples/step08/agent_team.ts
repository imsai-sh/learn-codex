// =============================================================================
// step08 / agent_team.ts — 让 agent 成为一个“可对话的常驻实体”
// -----------------------------------------------------------------------------
// step07 已经把 AgentThread / AgentTeamManager 立起来了，但当时一个 agent 还是
// “一次性的子任务”：spawn_sub_agent 同步等它跑完就结束。step08 的关键升级是把
// agent 变成一条“后台运行、可以被反复 send_input 的常驻线程”，于是这里的数据
// 结构需要承载几件新东西：
//
//   1. pending_inputs 队列 —— 父 agent 可以随时塞下一条任务进来。
//   2. last_result / last_error —— worker 跑完一轮后把成果留在这里，等
//      wait_agent 之类的查询来取。
//   3. worker_active 标志 —— 用来做“同一时刻只允许一条 worker 在跑这个 agent”
//      的 compare-and-set 守卫；如果队列里又来新输入，会再触发一次 worker 启动。
//   4. closed 标志 + Closed 状态 —— 显式关闭后拒绝新的 send_input。
//   5. 状态广播 (StatusBroadcaster) —— wait_agent 需要 await 一个“agent 状态
//      达到 final”的事件，对应 Rust 里的 watch::channel。Node 没有原生的 watch
//      channel，所以我们用一个最小 publish/once 的 pub/sub 来实现。
//
// 这里的所有“锁”都不是真的 mutex —— Node 是单线程事件循环，对 Map / 数组等
// 同步访问本身就是原子的。worker_active 也只是一个普通布尔，靠“函数体内不
// await 就完成 CAS”这一点保证 try_start_worker 是无竞争的。
// =============================================================================

/**
 * Agent 的运行状态。
 *
 *   pending   — 还没被 worker 拉起；初始状态。
 *   running   — worker 正在跑这一轮。
 *   completed — 上一轮成功，并且队列里没有新任务。
 *   failed    — 上一轮抛错。
 *   closed    — 显式调用 close() 关闭；不再接受新输入。
 *
 * 注意 completed/failed 仍然可以从队列里再拿到任务变回 running —— 这是
 * "long-running agent" 的核心：状态不是单调的，而是反复进入又离开 final 状态。
 * 不过 closed 是终态，不会再被翻回去。
 */
export type AgentStatus = "pending" | "running" | "completed" | "failed" | "closed";

/** wait_agent 用：判断一个状态是不是“可以停下来回报给父 agent”的状态。 */
export function isFinalStatus(status: AgentStatus): boolean {
  return status === "completed" || status === "failed" || status === "closed";
}

/** 给上层（list_agents 工具）回报时的快照结构。字段名要和 Rust 序列化一致。 */
export interface AgentSnapshot {
  id: string;
  role: string;
  status: AgentStatus;
  history_items: number;
  pending_inputs: number;
}

/**
 * 极简的状态广播器：替代 Rust 的 tokio::sync::watch。
 *
 * 我们只暴露两件事：
 *   - publish(v):       告诉所有订阅者“状态变成了 v”；订阅是“一次性”的，
 *                       触发后会自己摘掉自己（once 语义）。
 *   - once(predicate):  返回一个 Promise，会在 publish 出来的值第一次满足
 *                       predicate 时 resolve。配合 Promise.race + setTimeout
 *                       就能实现 wait_for_agent_status 的超时等待。
 *
 * 之所以不做“持续订阅”是因为在我们的场景里，wait_agent 拿到 final 状态就
 * 直接返回了，不需要长时间的 stream 语义。
 */
class StatusBroadcaster<T> {
  private listeners: Array<(value: T) => void> = [];

  publish(value: T): void {
    // 复制一份再遍历：listener 内部可能会反过来再注册新 listener，
    // 直接遍历原数组会导致行为不确定。
    for (const listener of [...this.listeners]) {
      listener(value);
    }
  }

  /** 订阅一次：第一次满足 predicate 的值就 resolve，并把自己从订阅列表里摘掉。 */
  once(predicate: (value: T) => boolean): Promise<T> {
    return new Promise<T>((resolve) => {
      const listener = (value: T) => {
        if (!predicate(value)) return;
        const idx = this.listeners.indexOf(listener);
        if (idx >= 0) this.listeners.splice(idx, 1);
        resolve(value);
      };
      this.listeners.push(listener);
    });
  }
}

/**
 * 一个独立的 agent“线程”。这里的“线程”是逻辑上的 —— Node 是单线程事件循环，
 * 物理上是异步任务，但每个 AgentThread 都拥有独立的 history / 状态 / 队列。
 *
 * 与 step07 相比新增字段：
 *   - pendingInputs: 父 agent 排队的任务列表。
 *   - statusBroadcaster: 状态变更通知（用于 wait_agent 的超时等待）。
 *   - lastResult / lastError: worker 一轮结束后留下的成果。
 *   - workerActive: 互斥标志，保证同一时刻只有一个 worker 在跑这个 agent。
 *   - closedFlag: 一旦置 true，agent 永久拒绝新输入并进入 closed 终态。
 */
export class AgentThread {
  private status: AgentStatus = "pending";
  private readonly history: unknown[];
  private readonly pendingInputs: string[] = [];
  private readonly statusBroadcaster = new StatusBroadcaster<AgentStatus>();
  private lastResultValue: string | null = null;
  private lastErrorValue: string | null = null;
  private workerActive = false;
  private closedFlag = false;

  constructor(
    private readonly _id: string,
    private readonly _role: string,
    systemPrompt: string,
  ) {
    // 与 Rust 一致：初始 history 只放 system prompt，user 那一条由 worker
    // 在拉到 instruction 时再 push 进去（因为同一 agent 可能被 send_input
    // 多次，每条 user 消息都要按时间顺序穿插进 history）。
    this.history = [{ role: "system", content: systemPrompt }];
  }

  id(): string {
    return this._id;
  }

  // ------------------- pending_inputs 队列 -------------------

  /** 把一条新指令塞进队尾。Node 单线程下 push 本身就是原子的。 */
  enqueueInput(input: string): void {
    this.pendingInputs.push(input);
  }

  /** 取队首；空队列返回 null。 */
  takeNextInput(): string | null {
    if (this.pendingInputs.length === 0) return null;
    return this.pendingInputs.shift()!;
  }

  hasPendingInputs(): boolean {
    return this.pendingInputs.length > 0;
  }

  // ------------------- worker 互斥 -------------------

  /**
   * 比较并置位：如果当前没有 worker 在跑，就置 true 并返回 true；否则返回 false。
   * Node 单线程让我们可以在这一段“读 → 写”之间不被打断 —— 不需要真原子指令。
   */
  tryStartWorker(): boolean {
    if (this.workerActive) return false;
    this.workerActive = true;
    return true;
  }

  markWorkerStopped(): void {
    this.workerActive = false;
  }

  // ------------------- status 与广播 -------------------

  setStatus(status: AgentStatus): void {
    this.status = status;
    this.statusBroadcaster.publish(status);
  }

  getStatus(): AgentStatus {
    return this.status;
  }

  /** 等到状态满足 predicate 时 resolve；调用方负责套 timeout。 */
  waitForStatus(predicate: (s: AgentStatus) => boolean): Promise<AgentStatus> {
    return this.statusBroadcaster.once(predicate);
  }

  // ------------------- history -------------------

  pushHistoryItem(item: unknown): void {
    this.history.push(item);
  }

  /** 返回 history 的浅拷贝，避免 worker 在外部代码迭代时被 push 影响。 */
  historySnapshot(): unknown[] {
    return [...this.history];
  }

  // ------------------- last_result / last_error -------------------

  setLastResult(result: string): void {
    this.lastResultValue = result;
    // 与 Rust 一致：写入 result 时把 error 清空。
    this.lastErrorValue = null;
  }

  setLastError(error: string): void {
    this.lastErrorValue = error;
  }

  lastResult(): string | null {
    return this.lastResultValue;
  }

  lastError(): string | null {
    return this.lastErrorValue;
  }

  // ------------------- 关闭 -------------------

  close(): void {
    this.closedFlag = true;
    this.setStatus("closed");
  }

  isClosed(): boolean {
    return this.closedFlag;
  }

  // ------------------- snapshot -------------------

  snapshot(): AgentSnapshot {
    return {
      id: this._id,
      role: this._role,
      status: this.status,
      history_items: this.history.length,
      pending_inputs: this.pendingInputs.length,
    };
  }
}

/**
 * 整个 session 唯一的 AgentTeamManager。它负责：
 *   - 给每个新 agent 分配递增 id；
 *   - 把所有 agent 集中保存，按 id 查询；
 *   - 提供按 id 排序后的快照列表给 list_agents 工具。
 *
 * step08 相对 step07 的增量是 get(id) —— send_input / wait_agent / close_agent
 * 都需要按 id 反查到具体的 AgentThread。
 */
export class AgentTeamManager {
  private nextId = 1;
  private readonly agents = new Map<string, AgentThread>();

  spawnAgent(role: string, systemPrompt: string, initialInput: string): AgentThread {
    const id = `agent-${this.nextId++}`;
    const agent = new AgentThread(id, role, systemPrompt);
    // 与 Rust 一致：spawn 时直接把 initial_input 排进队尾，让 worker
    // 拉起来后第一件事就是处理它。
    agent.enqueueInput(initialInput);
    this.agents.set(id, agent);
    return agent;
  }

  get(id: string): AgentThread | undefined {
    return this.agents.get(id);
  }

  /** 按 id 升序返回所有 agent 的快照（与 Rust 版本的 sort_by 行为对齐）。 */
  listSnapshots(): AgentSnapshot[] {
    return Array.from(this.agents.values())
      .map((a) => a.snapshot())
      .sort((a, b) => a.id.localeCompare(b.id));
  }
}

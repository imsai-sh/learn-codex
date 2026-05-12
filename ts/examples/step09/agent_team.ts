// =============================================================================
// step09 / agent_team.ts — 给 AgentThread 引入“角色 / 父子关系 / 深度”三件套
// -----------------------------------------------------------------------------
// step08 把 agent 做成 runtime 里可以并发跑的“后台 worker”，但所有 agent 都是
// 同质的。step09 在不破坏 step08 协议的前提下，加上了三层结构性元数据：
//
//   1) AgentRole —— default / explorer / worker。
//      explorer 角色在 sandbox 层会被禁掉一切“写文件 / 改文件系统 / 改仓库”
//      的工具，只允许检视；worker 是干活的；default 仍然是通用代理。
//      这是“给子 agent 配权限边界”的最小可教学版本。
//
//   2) parent_agent_id + depth —— 每个被 spawn 出来的 agent 都记得自己是谁的
//      孩子，以及离根 agent 多深。sandbox 用 depth 实现深度上限；用
//      parent_agent_id 实现“孩子完成 / 失败 / 关闭时给父亲的 history 注入一条
//      系统消息”，这样父 agent 自然能感知子任务的结果。
//
//   3) AgentSpawnRequest —— 把上面这些字段统一成一个构造请求，避免
//      AgentThread 构造函数变得越来越长。spawn 方时一次性塞过来。
//
// 教学重点：所有结构变化的目的都不是“多功能”，而是“让权限和因果关系可表达”。
// =============================================================================

/**
 * 三种内置角色。snake_case 是为了和 Rust 的 #[serde(rename_all = "snake_case")]
 * 对齐 —— 这些 label 会出现在工具返回的 JSON、agent_snapshots 等字段里，被模型
 * 直接读到，所以保持稳定的对外名字很重要。
 */
export type AgentRole = "default" | "explorer" | "worker";

/**
 * AgentRoleHelpers 是 Rust 里 `impl AgentRole` 的 TS 等价物。把和角色相关的纯
 * 函数集中到一个对象上，调用方写起来像 `AgentRoleHelpers.systemPrompt(role)`，
 * 比满屏的 switch 干净。
 */
export const AgentRoleHelpers = {
  /** 解析模型/上层传来的角色字符串。空白容忍，大小写不敏感；不识别就返回 null。 */
  parse(label: string): AgentRole | null {
    switch (label.trim().toLowerCase()) {
      case "default":
        return "default";
      case "explorer":
        return "explorer";
      case "worker":
        return "worker";
      default:
        return null;
    }
  },

  /** 序列化用的 label。当前等价于 role 本身，但保留这个函数让调用方可以无脑用。 */
  label(role: AgentRole): string {
    return role;
  },

  /**
   * 每个角色对应一段 system prompt。注意这些字符串决定了子 agent 的实际行为，
   * 改动时请慎重权衡（角色边界由 sandbox 层强制实现，prompt 主要是给模型的语义引导）。
   */
  systemPrompt(role: AgentRole): string {
    switch (role) {
      case "default":
        return "你是一个被委派的 agent，正在协助主 agent。请负责任地使用工具，并高效完成分配给你的任务。";
      case "explorer":
        return "你是一个 explorer（探查）角色的 agent。请专注于检视、读取文件和收集线索，不要修改文件，也不要执行任何会带来副作用的 shell 操作。";
      case "worker":
        return "你是一个 worker（执行）角色的 agent。请完成一项边界清晰的实现任务，谨慎使用工具，并给父 agent 留下一份清晰的结果。";
    }
  },

  /**
   * 这个角色是否被允许做“可能改动文件系统 / 仓库状态”的事。
   * 只有 explorer 是 false —— 它是被设计来做只读勘察的；其它角色（default /
   * worker）继承父 agent 的全部权限。sandbox 在 write_file / edit_file 以及
   * 看起来像 mutating 的 bash 命令上都会调用这个判断。
   */
  allowsFileMutation(role: AgentRole): boolean {
    return role !== "explorer";
  },
};

/** Agent 的状态机和 step08 一致；is_final 同样用来判断是否到达终态。 */
export type AgentStatus = "pending" | "running" | "completed" | "failed" | "closed";

export function isFinalStatus(status: AgentStatus): boolean {
  return status === "completed" || status === "failed" || status === "closed";
}

/**
 * 给 list_agents 之类工具回报时的快照结构。新增了 role / parent_agent_id /
 * depth，模型可以直接看出整棵 agent 树的形状。字段命名严格保持 snake_case，
 * 因为它们会被序列化到工具返回值里。
 */
export interface AgentSnapshot {
  id: string;
  role: AgentRole;
  parent_agent_id?: string;
  depth: number;
  status: AgentStatus;
  history_items: number;
  pending_inputs: number;
}

/**
 * spawnAgent 的入参。把所有“一次性初始化”的字段放在一起：
 *   - role / parent_agent_id / depth：上面三件套；
 *   - initial_history：调用方（sandbox.build_agent_history）已经构造好的初始
 *     历史，至少包含一条 system 消息。这里不会再追加任何东西；
 *   - initial_input：第一条用户指令。注意它不会立刻被压入 history，而是先放
 *     进 pending_inputs 队列，由 worker 在每个 turn 开始前以 user 角色压入
 *     history。这样能保证“一次 spawn 等于一次 user turn”的语义。
 */
export interface AgentSpawnRequest {
  role: AgentRole;
  parent_agent_id?: string;
  depth: number;
  initial_history: unknown[];
  initial_input: string;
}

/**
 * 一个最小的状态广播器，对应 Rust 中的 tokio::sync::watch。每次 setStatus 时
 * 把当前状态推给所有订阅者，等待方（wait_agent / SubAgent 同步等待）就能在
 * status 变成终态时立刻醒来。我们没有跨真线程的语义，所以一个简单的
 * Promise + listener 队列就够用。
 */
class StatusBroadcaster {
  private listeners: Array<(status: AgentStatus) => void> = [];

  publish(status: AgentStatus): void {
    // 拷贝一份再遍历，listener 自身在收到时可能取消订阅。
    for (const l of [...this.listeners]) l(status);
  }

  /** 注册一次性监听。返回的 unsubscribe 用于在外部主动取消（例如超时）。 */
  subscribe(listener: (status: AgentStatus) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const i = this.listeners.indexOf(listener);
      if (i >= 0) this.listeners.splice(i, 1);
    };
  }
}

/**
 * 一个独立的 agent“线程”。Node 是单线程事件循环，所以这里的“线程”是逻辑上
 * 的：每个 AgentThread 有自己独立的 history / pending_inputs / status，互不
 * 污染。step09 相对 step08 多了 role / parent_agent_id / depth 三个字段。
 */
export class AgentThread {
  private readonly _id: string;
  private readonly _role: AgentRole;
  private readonly _parentAgentId: string | undefined;
  private readonly _depth: number;
  private readonly history: unknown[];
  private readonly pendingInputs: string[];
  private currentStatus: AgentStatus = "pending";
  private readonly statusBroadcaster = new StatusBroadcaster();
  private lastResult: string | null = null;
  private lastError: string | null = null;
  private workerActive = false;
  private closedFlag = false;

  constructor(id: string, request: AgentSpawnRequest) {
    this._id = id;
    this._role = request.role;
    this._parentAgentId = request.parent_agent_id;
    this._depth = request.depth;
    // 直接接管调用方已经构造好的 initial_history；不做拷贝，因为构造方之后
    // 不会再持有引用。pending_inputs 用 initial_input 起头 —— 它代表“第一条
    // 待处理的 user instruction”。
    this.history = [...request.initial_history];
    this.pendingInputs = [request.initial_input];
  }

  id(): string {
    return this._id;
  }

  role(): AgentRole {
    return this._role;
  }

  parentAgentId(): string | undefined {
    return this._parentAgentId;
  }

  depth(): number {
    return this._depth;
  }

  enqueueInput(input: string): void {
    this.pendingInputs.push(input);
  }

  /** 取出最早排队的一条指令；空队列返回 null。和 Rust 的 take_next_input 等价。 */
  takeNextInput(): string | null {
    if (this.pendingInputs.length === 0) return null;
    return this.pendingInputs.shift()!;
  }

  hasPendingInputs(): boolean {
    return this.pendingInputs.length > 0;
  }

  /**
   * 把“当前 worker 是否在跑”这个标志原子地从 false 翻成 true。
   * Rust 里用 AtomicBool::compare_exchange 来防止两个 spawn 出来的 worker
   * 同时进入；JS 单线程其实不需要原子操作，但保留同样的 API 形状让上层逻辑
   * 不用改。
   */
  tryStartWorker(): boolean {
    if (this.workerActive) return false;
    this.workerActive = true;
    return true;
  }

  markWorkerStopped(): void {
    this.workerActive = false;
  }

  setStatus(status: AgentStatus): void {
    this.currentStatus = status;
    this.statusBroadcaster.publish(status);
  }

  status(): AgentStatus {
    return this.currentStatus;
  }

  /** 让外部（wait_for_agent_status）订阅状态流。 */
  subscribeStatus(listener: (status: AgentStatus) => void): () => void {
    return this.statusBroadcaster.subscribe(listener);
  }

  pushHistoryItem(item: unknown): void {
    this.history.push(item);
  }

  /** 浅拷贝 history，避免外部直接污染内部数组。run_agent_turn 每轮都要这个。 */
  historySnapshot(): unknown[] {
    return [...this.history];
  }

  setLastResult(result: string): void {
    this.lastResult = result;
    // 一次成功的产出会顺带清掉残留的 error 文本，避免父 agent 看到“成功+错误”混在一起。
    this.lastError = null;
  }

  setLastError(error: string): void {
    this.lastError = error;
  }

  getLastResult(): string | null {
    return this.lastResult;
  }

  getLastError(): string | null {
    return this.lastError;
  }

  close(): void {
    this.closedFlag = true;
    this.setStatus("closed");
  }

  isClosed(): boolean {
    return this.closedFlag;
  }

  snapshot(): AgentSnapshot {
    const snap: AgentSnapshot = {
      id: this._id,
      role: this._role,
      depth: this._depth,
      status: this.currentStatus,
      history_items: this.history.length,
      pending_inputs: this.pendingInputs.length,
    };
    if (this._parentAgentId !== undefined) snap.parent_agent_id = this._parentAgentId;
    return snap;
  }
}

/**
 * 整个 session 唯一的 AgentTeamManager。它只负责存储和查询，不负责调度
 * （调度逻辑在 sandbox.start_agent_worker 里）。spawn_agent 直接接受一个
 * AgentSpawnRequest，让 sandbox 那边的 build_agent_history / depth 检查可以
 * 一次构造完，再交给 manager 注册。
 */
export class AgentTeamManager {
  private nextAgentId = 1;
  private readonly agents = new Map<string, AgentThread>();

  spawnAgent(request: AgentSpawnRequest): AgentThread {
    const id = `agent-${this.nextAgentId++}`;
    const agent = new AgentThread(id, request);
    this.agents.set(id, agent);
    return agent;
  }

  get(id: string): AgentThread | undefined {
    return this.agents.get(id);
  }

  /** 按 id 升序返回所有 agent 的快照（与 Rust sort_by 行为对齐）。 */
  listSnapshots(): AgentSnapshot[] {
    return Array.from(this.agents.values())
      .map((a) => a.snapshot())
      .sort((a, b) => a.id.localeCompare(b.id));
  }
}

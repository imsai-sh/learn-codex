// =============================================================================
// step07 / agent_team.ts — AgentThread / AgentTeamManager 雏形
// -----------------------------------------------------------------------------
// step07 引入了“agent 是 runtime 里的一个一等公民”这件事。它还没有把所有 team
// 工具都接进去（spawn_agent / wait_agent 这些是 step08 才有），但定下了基本数据
// 结构：
//
//   AgentThread {
//     id, role, history[], status
//   }
//
//   AgentTeamManager {
//     spawn_agent(role, system_prompt, instruction) -> AgentThread
//     list_snapshots() -> AgentSnapshot[]
//   }
//
// 这一阶段对应 README 里 Step 07 的核心问题：
//
//   一个 agent 在 runtime 里到底是什么？
// =============================================================================

/** Agent 的运行状态。step07 暂时只用 pending；后续 step 会扩展。 */
export type AgentStatus = "pending" | "running" | "completed" | "failed";

/** 给上层（list_agents 之类的工具）回报时的快照结构。 */
export interface AgentSnapshot {
  id: string;
  role: string;
  status: AgentStatus;
  history_items: number;
}

/**
 * 一个独立的 agent“线程”。注意这里的“线程”是逻辑上的 —— Node.js 是单线程
 * 事件循环，物理上仍然是异步任务，但每个 AgentThread 都有自己的 history、
 * status 等独立状态，互不污染。
 */
export class AgentThread {
  private status: AgentStatus = "pending";

  constructor(
    private readonly _id: string,
    private readonly _role: string,
    private readonly history: unknown[],
  ) {}

  id(): string {
    return this._id;
  }

  setStatus(status: AgentStatus): void {
    this.status = status;
  }

  pushHistoryItem(item: unknown): void {
    this.history.push(item);
  }

  /** 返回 history 的浅拷贝，避免外部直接修改内部数组。 */
  historySnapshot(): unknown[] {
    return [...this.history];
  }

  snapshot(): AgentSnapshot {
    return {
      id: this._id,
      role: this._role,
      status: this.status,
      history_items: this.history.length,
    };
  }
}

/**
 * 整个 session 唯一的 AgentTeamManager。它负责：
 *   - 给每个新 agent 分配一个递增 id；
 *   - 把所有 agent 集中保存，按 id 查询；
 *   - 提供按 id 排序后的快照列表，便于 list_agents 工具回报。
 */
export class AgentTeamManager {
  private nextId = 1;
  private readonly agents = new Map<string, AgentThread>();

  spawnAgent(role: string, systemPrompt: string, instruction: string): AgentThread {
    const id = `agent-${this.nextId++}`;
    // 初始 history 至少包含 system + 用户的第一条 instruction。
    const history: unknown[] = [
      { role: "system", content: systemPrompt },
      { role: "user", content: instruction },
    ];
    const agent = new AgentThread(id, role, history);
    this.agents.set(id, agent);
    return agent;
  }

  /** 按 id 升序返回所有 agent 的快照（与 Rust 版本的 sort_by 行为对齐）。 */
  listSnapshots(): AgentSnapshot[] {
    return Array.from(this.agents.values())
      .map((a) => a.snapshot())
      .sort((a, b) => a.id.localeCompare(b.id));
  }
}

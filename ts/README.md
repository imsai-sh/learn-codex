# learn-codex (TypeScript 版)

本目录是 `src/examples/` 下 Rust 教学示例的 TypeScript 等价实现，每一步逐渐演进出一个类似 Codex 的 Agent Harness。

## 结构

```text
ts/
├── package.json
├── tsconfig.json
├── examples/
│   ├── step01/  最小 function-calling 循环（仅 run_bash）
│   ├── step02/  + ToolRegistry（read_file/write_file/edit_file）
│   ├── step03/  + update_plan（任务分解）
│   ├── step04/  + spawn_sub_agent（递归子 Agent）
│   ├── step05/  + skills 加载 + 历史压缩
│   ├── step06/  + parallel_tool_calls（并发工具调用）
│   ├── step07/  + AgentThread / AgentTeamManager 雏形
│   ├── step08/  + spawn_agent / send_input / wait_agent / close_agent / list_agents
│   └── step09/  + AgentRole / fork_context / depth / 权限边界
├── main.ts        与 step09 等价的合并版主实现
├── sandbox.ts     主版本沙箱
├── skills.ts      主版本 skill 加载器
└── agent_team.ts  主版本 Agent 团队管理器
```

## 安装与运行

```bash
cd ts
npm install

# 推荐：用 .env 配置（已经接入 dotenv，启动时自动加载 ts/.env）
cp .env.example .env
# 编辑 .env 填入实际值

# 然后直接运行某一步示例
npm run step01
npm run step09
npm run main
```

环境变量也可以用传统方式提供（`export OPENAI_API_KEY=...` 或 `OPENAI_API_KEY=... npm run step01`），dotenv 不会覆盖已经存在的环境变量，所以两种方式可以并存。

## 与 Rust 版本的关键差异

| 主题 | Rust | TypeScript |
| --- | --- | --- |
| 异步运行时 | tokio | Node.js 内建事件循环 |
| HTTP 客户端 | reqwest | undici 的 fetch |
| 并发原语 | `Arc<Mutex<...>>` / `tokio::sync::watch` | 闭包 + 自定义事件订阅类（基于 Promise） |
| 错误传播 | `Result<T, E>` + `?` | 抛出/捕获 + 自定义结构化错误对象 |
| 子线程 | `tokio::spawn` | `setImmediate` / 异步函数（Node 的事件循环里全是单线程） |
| 文件系统 | `std::fs` | `node:fs/promises` |

> 注意：TypeScript 实现保留了原 Rust 的语义和接口形状（包括函数命名、字段名等），方便与 Rust 版本做并排阅读。

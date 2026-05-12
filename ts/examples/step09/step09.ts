// =============================================================================
// step09 / step09.ts — 顶层主循环
// -----------------------------------------------------------------------------
// 主循环本身和 step08 几乎完全一样：
//   - 注册全套工具（普通工具 + agent team 协作工具）；
//   - 每轮都重新加载 skills 并按用户输入注入；
//   - 必要时跑一次 compact_history 把上下文压短；
//   - 让模型采样一次，处理它返回的 tool_calls，否则打印最终回答。
//
// step09 唯一新增的事是：把 ToolExecutionContext.root() 塞进每个顶层
// ToolInvocation 里。这样下游的 sandbox 工具就能用同一套权限/父子关系机制
// 处理“顶层人类用户的工具调用”和“子 agent 的工具调用”—— 区别仅在于 context
// 是不是 root：
//   - root 没有 caller_role ⇒ 所有 mutation 检查都直接放行；
//   - root 没有 caller_agent_id ⇒ spawnDelegatedAgent 推断 depth=1，没有父亲。
//
// 这就是 step09 在主循环侧的全部改动。
// =============================================================================

// 自动从当前工作目录的 .env 加载 OPENAI_* 等环境变量；必须在任何读取
// process.env 的代码之前执行（dotenv/config 这个 side-effect import 自带）。
import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { promises as fs } from "node:fs";
import { fetch } from "undici";

import {
  CloseAgentHandler,
  EditFileHandler,
  ListAgentsHandler,
  PlanHandler,
  ReadFileHandler,
  RunBashHandler,
  SendAgentInputHandler,
  SpawnAgentHandler,
  SubAgentHandler,
  ToolExecutionContext,
  ToolInvocation,
  ToolRegistry,
  WaitAgentHandler,
  WriteFileHandler,
} from "./sandbox.js";
import {
  buildSkillInjectionMessages,
  collectExplicitSkillMentions,
  loadSkills,
  renderSkillsSection,
} from "./skills.js";

// ----------------------------- 协议结构 -----------------------------

interface Message {
  role: "system" | "user" | "assistant" | "tool";
  // assistant 在请求工具时 content 可以为 null。
  content?: string | null;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
}

interface ChatResponse {
  choices: Array<{ message: Message; finish_reason: string }>;
}

const DEMO_SUPPORTS_PARALLEL_TOOL_CALLS = true;

// ----------------------------- 上下文压缩（同 step05/06） -----------------------------

const COMPACTION_CHAR_THRESHOLD = 20_000;
const COMPACTION_KEEP_RECENT_USER_TURNS = 2;
const COMPACTION_TOOL_TEXT_CHARS = 1_200;
const COMPACTION_SUMMARY_PREFIX = "对话前文（在历史压缩时生成的摘要）：";

function truncateForCompaction(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const each = Math.floor(maxChars / 2);
  const prefix = text.slice(0, each);
  const suffix = text.slice(text.length - each);
  const omitted = text.length - (prefix.length + suffix.length);
  return `${prefix}\n\n... [TRUNCATED ${omitted} CHARACTERS] ...\n\n${suffix}`;
}

function findRecentHistoryStart(history: Message[], keepRecentUserTurns: number): number {
  let seen = 0;
  for (let i = history.length - 1; i >= 1; i--) {
    if (history[i].role === "user") {
      seen += 1;
      if (seen === keepRecentUserTurns) return i;
    }
  }
  return 1;
}

function sanitizeMessageForCompaction(message: Message): Message | null {
  switch (message.role) {
    case "user": {
      const c = (message.content ?? "").trim();
      return c.length > 0 ? { role: "user", content: message.content! } : null;
    }
    case "assistant": {
      let content: string | undefined;
      if (typeof message.content === "string" && message.content.trim().length > 0) {
        content = message.content;
      } else if (message.tool_calls && message.tool_calls.length > 0) {
        content = `助手请求调用了以下工具：${message.tool_calls
          .map((t) => t.function.name)
          .join("、")}`;
      }
      return content ? { role: "assistant", content } : null;
    }
    case "tool": {
      const c = (message.content ?? "").trim();
      if (c.length === 0) return null;
      return {
        role: "assistant",
        content: `观察到的工具结果：\n${truncateForCompaction(message.content!, COMPACTION_TOOL_TEXT_CHARS)}`,
      };
    }
    default:
      return null;
  }
}

async function compactHistory(
  apiKey: string,
  baseUrl: string,
  modelName: string,
  history: Message[],
): Promise<Message[]> {
  const totalChars = history.reduce((acc, m) => acc + (m.content?.length ?? 0), 0);
  if (totalChars < COMPACTION_CHAR_THRESHOLD || history.length <= 6) return history;

  console.log(`\n[context] history too long (${totalChars} chars); starting compaction`);

  const systemMessage = history[0];
  const recentStart = findRecentHistoryStart(history, COMPACTION_KEEP_RECENT_USER_TURNS);
  const recentMessages = history.slice(recentStart);
  const toSummarize = history.slice(1, recentStart);

  const sanitized = toSummarize
    .map(sanitizeMessageForCompaction)
    .filter((m): m is Message => m !== null);
  if (sanitized.length === 0) {
    console.log(
      "[context] history is long but no compactable history exists before the retained recent turns",
    );
    return history;
  }

  const summarizationPrompt =
    "请简明地总结以下对话历史，同时保留关键事实、用户意图和重要的工具输出。目标是为后续推理保留必要的上下文。";
  const summaryRequest: Message[] = [
    { role: "system", content: summarizationPrompt },
    ...sanitized,
  ];

  const res = await fetch(baseUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model: modelName, messages: summaryRequest, temperature: 0.3 }),
  });

  if (!res.ok) {
    const status = res.status;
    let body = "<empty response body>";
    try {
      body = await res.text();
    } catch {}
    console.log(`[context] compaction failed (status: ${status})`);
    console.log(`[context] response body: ${body}`);
    console.log("[context] continuing with full history");
    return history;
  }

  const responseBody = (await res.json()) as ChatResponse;
  const summaryText = responseBody.choices[0]?.message.content ?? "";
  console.log("[context] compaction completed");

  const summaryMessage: Message = {
    role: "system",
    content: `${COMPACTION_SUMMARY_PREFIX}\n${summaryText}`,
  };
  return [systemMessage, summaryMessage, ...recentMessages];
}

function composeSystemPrompt(basePrompt: string, skillsSection: string | null): string {
  if (!skillsSection) return basePrompt;
  return `${basePrompt}\n${skillsSection}\n`;
}

function buildRequestMessages(history: Message[], turnSkillMessages: Message[]): Message[] {
  if (history.length === 0 || turnSkillMessages.length === 0) return [...history];
  return [history[0], ...turnSkillMessages, ...history.slice(1)];
}

async function skillsRoot(): Promise<string> {
  try {
    const stat = await fs.stat("demo/skills");
    if (stat.isDirectory()) return "demo/skills";
  } catch {}
  return "skills";
}

// ----------------------------- 把一批 tool_calls 跑掉 -----------------------------

/**
 * 主循环侧的 tool 派发。所有工具调用都用 ToolExecutionContext.root()：
 * 表示这是“最外层人类用户驱动的调用”，没有 caller_agent_id / caller_role —— 所
 * 以 sandbox 里的 mutation 检查都直接放行，spawnDelegatedAgent 也会把新 agent
 * 算作 depth=1 的根级孩子（没有父亲）。
 */
async function executeToolCalls(
  registry: ToolRegistry,
  conversationHistory: Message[],
  toolCalls: NonNullable<Message["tool_calls"]>,
): Promise<void> {
  console.log(`\n[registry] tool batch received: ${toolCalls.length} call(s)`);

  const invocations: ToolInvocation[] = toolCalls.map((tc) => {
    console.log(`[registry] tool request: ${tc.function.name}(${tc.function.arguments})`);
    return {
      call_id: tc.id,
      tool_name: tc.function.name,
      arguments: tc.function.arguments,
      context: ToolExecutionContext.root(),
    };
  });

  const results = await registry.dispatchMany(invocations);
  for (const r of results) {
    console.log(`[registry] tool completed: ${r.tool_name}`);
    conversationHistory.push({ role: "tool", content: r.output, tool_call_id: r.call_id });
  }
}

// ----------------------------- 主循环 -----------------------------

async function main(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY ?? "your-api_key";
  const baseUrl = process.env.OPENAI_BASE_URL ?? "your-base_url";
  const modelName = process.env.OPENAI_MODEL_NAME ?? "your-model";

  const registry = new ToolRegistry();
  // 普通工具
  registry.register(new RunBashHandler());
  registry.register(new ReadFileHandler());
  registry.register(new WriteFileHandler());
  registry.register(new EditFileHandler());
  registry.register(new PlanHandler());
  registry.register(new ListAgentsHandler());
  // agent team 协作工具
  registry.register(new SpawnAgentHandler(apiKey, baseUrl, modelName));
  registry.register(new SendAgentInputHandler(apiKey, baseUrl, modelName));
  registry.register(new WaitAgentHandler());
  registry.register(new CloseAgentHandler());
  registry.register(new SubAgentHandler(apiKey, baseUrl, modelName));

  // system prompt 完全照搬 Rust step09.rs 的字符串。注意它显式提到 role /
  // fork_context / depth limit —— 这些都对应 sandbox 里实现的能力。
  const baseSystemPrompt = `
你是一个命令行助手。你可以在用户的 macOS 文件系统上执行 bash 命令并管理文件。
请始终通过提供的工具显式地展开你的思考过程，用它们来探索系统或执行用户要求的任务。
工具的返回结果是结构化 JSON，请先查看字段再决定下一步动作。
当多个互相独立的只读任务或 shell 任务可以安全并发推进时，优先在一次响应里同时发出多条工具调用。
不要把文件修改或计划更新与其它工具调用并发执行，除非它们之间确实彼此独立且安全。

对于复杂任务，你**必须**使用 \`update_plan\` 工具来：
1. 在一开始把任务拆成若干个可管理的步骤；
2. 开始执行某一步时，把它标记为 \`in_progress\`；
3. 完成后，把它标记为 \`completed\`。

你还可以使用 \`spawn_sub_agent\` 工具，把具体的子任务委派给一个独立的 agent 实例去完成。
对于更长时间的协作，优先使用：
- \`spawn_agent\` 创建一个后台 agent；
- 通过 \`role\` 在 \`default\` / \`explorer\` / \`worker\` 之间选择子 agent 的角色；
- 当子 agent 需要继承父 agent 的近期上下文时，使用 \`fork_context\`；
- \`send_input\` 给该 agent 排入后续任务；
- \`wait_agent\` 等待其完成；
- \`close_agent\` 让它停止接收新任务；
- \`list_agents\` 查看当前 agent 团队的状态。
请遵守 agent 的深度限制，避免不必要地嵌套 spawn agent。
`;

  let conversationHistory: Message[] = [{ role: "system", content: baseSystemPrompt }];
  const toolsDefinitions = registry.getSpecs();

  const rl = createInterface({ input, output });

  while (true) {
    const userInput = (await rl.question("> ")).trim();
    if (!userInput) continue;
    if (userInput === "exit" || userInput === "quit") {
      console.log("Bye!");
      rl.close();
      break;
    }

    // 每轮都重新扫一遍 skills 目录，让用户可以热添加 skill 而不必重启。
    const root = await skillsRoot();
    const loaded = await loadSkills(root);
    for (const w of loaded.warnings) console.log(`[skills] ${w}`);

    conversationHistory[0] = {
      role: "system",
      content: composeSystemPrompt(baseSystemPrompt, renderSkillsSection(loaded.skills)),
    };

    const mentioned = collectExplicitSkillMentions(userInput, loaded.skills);
    const { messages: skillMessageContents, warnings: skillWarnings } =
      await buildSkillInjectionMessages(mentioned);
    for (const w of skillWarnings) console.log(`[skills] ${w}`);
    for (const s of mentioned) console.log(`[skills] injecting skill for turn: ${s.name}`);

    // 注入消息只对“当前这一次模型采样”生效，不会污染长期 history。
    const turnSkillMessages: Message[] = skillMessageContents.map((content) => ({
      role: "system",
      content,
    }));

    conversationHistory.push({ role: "user", content: userInput });

    while (true) {
      try {
        conversationHistory = await compactHistory(apiKey, baseUrl, modelName, conversationHistory);
      } catch (e) {
        console.log(`[context] compaction error: ${(e as Error).message}`);
      }

      const requestMessages = buildRequestMessages(conversationHistory, turnSkillMessages);

      const res = await fetch(baseUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: modelName,
          messages: requestMessages,
          tools: toolsDefinitions,
          parallel_tool_calls: DEMO_SUPPORTS_PARALLEL_TOOL_CALLS,
          temperature: 0.2,
        }),
      });

      if (!res.ok) {
        console.log(`\nAPI Error: ${await res.text()}`);
        break;
      }

      const body = (await res.json()) as ChatResponse;
      const choice = body.choices[0];
      const assistantMessage = choice.message;
      conversationHistory.push(assistantMessage);

      const wantsToolCall =
        choice.finish_reason === "tool_calls" || (assistantMessage.tool_calls?.length ?? 0) > 0;

      if (!wantsToolCall) {
        if (assistantMessage.content) {
          console.log(`\n[agent] final response\n${assistantMessage.content}\n`);
        }
        break;
      }

      await executeToolCalls(registry, conversationHistory, assistantMessage.tool_calls!);
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

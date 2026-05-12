// =============================================================================
// step05 / step05.ts — 加入“按需 skill 注入”和“历史压缩”
// -----------------------------------------------------------------------------
// 这是第一个“跑长了真的不会爆炸”的版本。新增的两件事：
//
// 1) Skill 注入
//    - 每轮开始前重新扫描 skills/ 目录，列在 system prompt 里；
//    - 当本轮用户输入显式带 $skill-name 时，把对应 SKILL.md 读出来作为
//      额外的 system 消息塞到本轮请求里（只对“当前这一次模型采样”生效）；
//    - 这样既不污染长期历史，也能让模型按需吸收外部知识。
//
// 2) 上下文压缩 (compact_history)
//    - 当对话历史的字符总数超过阈值，系统会自动调用一次模型来生成摘要；
//    - 摘要替换掉中段内容，但保留 system 消息和最近 N 个用户回合。
// =============================================================================

// 自动从当前工作目录的 .env 加载 OPENAI_* 等环境变量；必须在任何读取
// process.env 的代码之前执行（dotenv/config 这个 side-effect import 自带）。
import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { promises as fs } from "node:fs";
import { fetch } from "undici";

import {
  EditFileHandler,
  PlanHandler,
  ReadFileHandler,
  RunBashHandler,
  SubAgentHandler,
  ToolRegistry,
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
  // assistant 在请求工具时 content 可以是 null，所以这里允许 null。
  content?: string | null;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
}

interface ChatResponse {
  choices: Array<{ message: Message; finish_reason: string }>;
}

// ----------------------------- 上下文压缩 -----------------------------

/** 触发压缩的字符数阈值。短于这个长度时不做任何动作。 */
const COMPACTION_CHAR_THRESHOLD = 20_000;
/** 压缩时保留最近多少个 user 回合（不会被摘要吃掉）。 */
const COMPACTION_KEEP_RECENT_USER_TURNS = 2;
/** 压缩时单条 tool 输出文本的最大字符数。 */
const COMPACTION_TOOL_TEXT_CHARS = 1_200;
/** 摘要消息的固定前缀，方便日后区分。 */
const COMPACTION_SUMMARY_PREFIX = "对话前文（在历史压缩时生成的摘要）：";

/** 把过长的 tool 输出截短：保留首尾两半，中间塞一个占位说明。 */
function truncateForCompaction(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const each = Math.floor(maxChars / 2);
  const prefix = text.slice(0, each);
  const suffix = text.slice(text.length - each);
  const omitted = text.length - (prefix.length + suffix.length);
  return `${prefix}\n\n... [TRUNCATED ${omitted} CHARACTERS] ...\n\n${suffix}`;
}

/**
 * 找到“最近 N 个 user 回合”的起点下标。
 * 历史结构通常是 [system, user, assistant, tool, user, assistant, ...]，
 * 这里从尾部往前扫，倒数第 N 个 user 消息出现的位置就是“保留区”的起点。
 */
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

/**
 * 把一条普通历史消息“清洗”成一份可以送进总结模型的纯文本消息。
 *  - tool 消息会被改写成 "观察到的工具结果：\n<截断后的文本>"，
 *    role 也改成 "assistant"，因为很多 chat 模型会拒绝非配对的 role="tool"；
 *  - 空 content 的 assistant 会被压成“助手请求调用了以下工具：...”；
 *  - 空字符串/全空白会被丢弃。
 */
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
        const names = message.tool_calls.map((tc) => tc.function.name);
        content = `助手请求调用了以下工具：${names.join("、")}`;
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

/**
 * 触发条件：
 *   - 历史字符总数超过阈值；
 *   - 历史长度大于 6（避免“一开始就压缩，没东西可压”）。
 *
 * 真实改动：在 [system, ..., recent_start) 这一段插入一条 system 摘要消息，
 * 然后保留 [recent_start, end) 区间不动。
 */
async function compactHistory(
  apiKey: string,
  baseUrl: string,
  modelName: string,
  history: Message[],
): Promise<Message[]> {
  const totalChars = history.reduce((acc, m) => acc + (m.content?.length ?? 0), 0);
  if (totalChars < COMPACTION_CHAR_THRESHOLD || history.length <= 6) {
    return history;
  }

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

// ----------------------------- 系统消息组合 -----------------------------

/** 在 base prompt 之后追加（可选的）skills 段落。 */
function composeSystemPrompt(basePrompt: string, skillsSection: string | null): string {
  if (!skillsSection) return basePrompt;
  return `${basePrompt}\n${skillsSection}\n`;
}

/**
 * 给本次请求构造完整 messages：
 *   [history[0]=system, 本轮临时 skill messages..., history[1..]]
 *
 * 注意 turn_skill_messages 不会被写回 conversationHistory —— 这是“按需注入”
 * 的关键：它们只在“这一次和模型交互”时存在，不会污染长期历史。
 */
function buildRequestMessages(history: Message[], turnSkillMessages: Message[]): Message[] {
  if (history.length === 0 || turnSkillMessages.length === 0) return [...history];
  return [history[0], ...turnSkillMessages, ...history.slice(1)];
}

/** skills 文件夹的位置：优先 demo/skills，其次 ./skills。 */
async function skillsRoot(): Promise<string> {
  try {
    const stat = await fs.stat("demo/skills");
    if (stat.isDirectory()) return "demo/skills";
  } catch {}
  return "skills";
}

// ----------------------------- 主循环 -----------------------------

async function main(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY ?? "your-api_key";
  const baseUrl = process.env.OPENAI_BASE_URL ?? "your-base_url";
  const modelName = process.env.OPENAI_MODEL_NAME ?? "your-model";

  const registry = new ToolRegistry();
  registry.register(new RunBashHandler());
  registry.register(new ReadFileHandler());
  registry.register(new WriteFileHandler());
  registry.register(new EditFileHandler());
  registry.register(new PlanHandler());
  registry.register(new SubAgentHandler(apiKey, baseUrl, modelName));

  const baseSystemPrompt = `
你是一个命令行助手。你可以在用户的 macOS 文件系统上执行 bash 命令并管理文件。
请始终通过提供的工具显式地展开你的思考过程，用它们来探索系统或执行用户要求的任务。
工具的返回结果是结构化 JSON，请先查看字段再决定下一步动作。

对于复杂任务，你**必须**使用 \`update_plan\` 工具来：
1. 在一开始把任务拆成若干个可管理的步骤；
2. 开始执行某一步时，把它标记为 \`in_progress\`；
3. 完成后，把它标记为 \`completed\`。

你还可以使用 \`spawn_sub_agent\` 工具，把具体的子任务委派给一个独立的 agent 实例去完成。
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

    // 每轮重新扫描一次 skills，让目录里 SKILL.md 的增删能被立即感知。
    const root = await skillsRoot();
    const loaded = await loadSkills(root);
    for (const w of loaded.warnings) console.log(`[skills] ${w}`);

    // 把新的 skills 列表合并进 system prompt（覆盖 history[0] 的 content）。
    conversationHistory[0] = {
      role: "system",
      content: composeSystemPrompt(baseSystemPrompt, renderSkillsSection(loaded.skills)),
    };

    // 找出本轮被显式提及的 skill，并把它们的 SKILL.md 读出来。
    const mentioned = collectExplicitSkillMentions(userInput, loaded.skills);
    const { messages: skillMessageContents, warnings: skillWarnings } =
      await buildSkillInjectionMessages(mentioned);
    for (const w of skillWarnings) console.log(`[skills] ${w}`);
    for (const s of mentioned) console.log(`[skills] injecting skill for turn: ${s.name}`);

    const turnSkillMessages: Message[] = skillMessageContents.map((content) => ({
      role: "system",
      content,
    }));

    conversationHistory.push({ role: "user", content: userInput });

    while (true) {
      // 在采样前先尝试压缩 —— 即使没到阈值，函数内部会自己短路返回。
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

      for (const toolCall of assistantMessage.tool_calls!) {
        const { name, arguments: argumentsJson } = toolCall.function;
        console.log(`\n[registry] tool request: ${name}(${argumentsJson})`);
        const out = await registry.dispatch(name, argumentsJson);
        if (out !== null) {
          conversationHistory.push({ role: "tool", content: out, tool_call_id: toolCall.id });
        } else {
          console.log(`[registry] tool not found: ${name}`);
          conversationHistory.push({
            role: "tool",
            content: JSON.stringify({
              ok: false,
              tool: name,
              message: `tool ${name} not found`,
              error_code: "tool_not_found",
              data: { arguments: argumentsJson },
            }),
            tool_call_id: toolCall.id,
          });
        }
      }
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

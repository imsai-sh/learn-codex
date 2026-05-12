// =============================================================================
// step02 / step02.ts — Tool Registry 化的 Agent Loop
// -----------------------------------------------------------------------------
// 与 step01 的差异：
//   1. 工具不再写死在主循环里，而是 register 到一个 ToolRegistry；
//   2. 一轮 assistant 消息里如果有多条 tool_calls，会按顺序处理而不是只取第一条；
//   3. 找不到工具时也回填一个结构化错误，让模型自己发现并自我修复。
// =============================================================================

// 自动从当前工作目录的 .env 加载 OPENAI_* 等环境变量；必须在任何读取
// process.env 的代码之前执行（dotenv/config 这个 side-effect import 自带）。
import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { fetch } from "undici";

import {
  EditFileHandler,
  ReadFileHandler,
  RunBashHandler,
  ToolRegistry,
  WriteFileHandler,
} from "./sandbox.js";

interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface ChatResponse {
  choices: Array<{ message: Message; finish_reason: string }>;
}

async function main(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY ?? "your-api_key";
  const baseUrl = process.env.OPENAI_BASE_URL ?? "your-base_url";
  const modelName = process.env.OPENAI_MODEL_NAME ?? "your-model";

  // 这是 step02 的核心改动：注册一个工具集合而不是写死。
  const registry = new ToolRegistry();
  registry.register(new RunBashHandler());
  registry.register(new ReadFileHandler());
  registry.register(new WriteFileHandler());
  registry.register(new EditFileHandler());

  const systemPrompt = `
你是一个命令行助手。你可以在用户的 macOS 文件系统上执行 bash 命令并管理文件。
请始终通过提供的工具显式地展开你的思考过程，用它们来探索系统或执行用户要求的任务。
工具的返回结果是结构化 JSON，请先查看字段再决定下一步动作。
`;

  const conversationHistory: Message[] = [{ role: "system", content: systemPrompt }];
  // 把所有工具的 spec 一次性导出，整个 session 内都不需要重新生成。
  const toolsDefinitions = registry.getSpecs();

  const rl = createInterface({ input, output });

  while (true) {
    const userInput = (await rl.question("> ")).trim();
    if (userInput.length === 0) continue;
    if (userInput === "exit" || userInput === "quit") {
      console.log("Bye!");
      rl.close();
      break;
    }

    conversationHistory.push({ role: "user", content: userInput });

    while (true) {
      const payload = {
        model: modelName,
        messages: conversationHistory,
        tools: toolsDefinitions,
        temperature: 0.2,
      };

      const res = await fetch(baseUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        console.log(`\nAPI Error: ${await res.text()}`);
        break;
      }

      const responseBody = (await res.json()) as ChatResponse;
      const choice = responseBody.choices[0];
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

      // 顺序处理这一批 tool_calls：每条都通过 registry.dispatch 查找并执行，
      // 然后以 role="tool" 回填到历史里。
      for (const toolCall of assistantMessage.tool_calls!) {
        const functionName = toolCall.function.name;
        const argumentsJson = toolCall.function.arguments;
        console.log(`\n[registry] tool request: ${functionName}(${argumentsJson})`);

        const toolOutput = await registry.dispatch(functionName, argumentsJson);
        if (toolOutput !== null) {
          conversationHistory.push({
            role: "tool",
            content: toolOutput,
            tool_call_id: toolCall.id,
          });
        } else {
          // 找不到工具：给模型一个清晰的“tool_not_found”，让它自己改用别的工具。
          console.log(`[registry] tool not found: ${functionName}`);
          conversationHistory.push({
            role: "tool",
            content: JSON.stringify({
              ok: false,
              tool: functionName,
              message: `tool ${functionName} not found`,
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

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

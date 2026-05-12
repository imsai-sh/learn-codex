// =============================================================================
// step03 / step03.ts — 加上 update_plan 后的 Agent Loop
// -----------------------------------------------------------------------------
// 与 step02 的差异：
//   - 在 ToolRegistry 中额外注册 PlanHandler；
//   - system prompt 里强制要求“复杂任务先 update_plan 再动手”。
// 主循环本身没有变化。
// =============================================================================

// 自动从当前工作目录的 .env 加载 OPENAI_* 等环境变量；必须在任何读取
// process.env 的代码之前执行（dotenv/config 这个 side-effect import 自带）。
import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { fetch } from "undici";

import {
  EditFileHandler,
  PlanHandler,
  ReadFileHandler,
  RunBashHandler,
  ToolRegistry,
  WriteFileHandler,
} from "./sandbox.js";

interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string;
}

interface ChatResponse {
  choices: Array<{ message: Message; finish_reason: string }>;
}

async function main(): Promise<void> {
  const apiKey = process.env.OPENAI_API_KEY ?? "";
  const baseUrl = process.env.OPENAI_BASE_URL ?? "";
  const modelName = process.env.OPENAI_MODEL_NAME ?? "";

  const registry = new ToolRegistry();
  registry.register(new RunBashHandler());
  registry.register(new ReadFileHandler());
  registry.register(new WriteFileHandler());
  registry.register(new EditFileHandler());
  // step03 新增：让模型可以维护一个明确的执行计划。
  registry.register(new PlanHandler());

  const systemPrompt = `
你是一个命令行助手。你可以在用户的 macOS 文件系统上执行 bash 命令并管理文件。
请始终通过提供的工具显式地展开你的思考过程，用它们来探索系统或执行用户要求的任务。
工具的返回结果是结构化 JSON，请先查看字段再决定下一步动作。

对于复杂任务，你**必须**使用 \`update_plan\` 工具来：
1. 在一开始把任务拆成若干个可管理的步骤；
2. 开始执行某一步时，把它标记为 \`in_progress\`；
3. 完成后，把它标记为 \`completed\`。
这能让你保持思路清晰，也能让用户透明地看到进展。
`;

  const conversationHistory: Message[] = [{ role: "system", content: systemPrompt }];
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

    conversationHistory.push({ role: "user", content: userInput });

    while (true) {
      const res = await fetch(baseUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: modelName,
          messages: conversationHistory,
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

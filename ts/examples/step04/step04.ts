// =============================================================================
// step04 / step04.ts — 引入 sub-agent 后的主循环
// -----------------------------------------------------------------------------
// 与 step03 相比的差异：
//   1. ToolRegistry 现在通过 dispatch 把自己作为参数注入 handler；
//   2. 多注册了一个 SubAgentHandler，后者会用 apiKey/baseUrl/modelName 去
//      启动嵌套 agent loop；
//   3. system prompt 增加一句话告诉模型：你可以委派任务。
// 主循环本身没变。
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
  SubAgentHandler,
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
  const apiKey = process.env.OPENAI_API_KEY ?? "your-api_key";
  const baseUrl = process.env.OPENAI_BASE_URL ?? "your-base_url";
  const modelName = process.env.OPENAI_MODEL_NAME ?? "your-model";

  const registry = new ToolRegistry();
  registry.register(new RunBashHandler());
  registry.register(new ReadFileHandler());
  registry.register(new WriteFileHandler());
  registry.register(new EditFileHandler());
  registry.register(new PlanHandler());
  // step04 新增：把“产生子 agent”的能力作为一个普通工具注册进来。
  registry.register(new SubAgentHandler(apiKey, baseUrl, modelName));

  const systemPrompt = `
你是一个命令行助手。你可以在用户的 macOS 文件系统上执行 bash 命令并管理文件。
请始终通过提供的工具显式地展开你的思考过程，用它们来探索系统或执行用户要求的任务。
工具的返回结果是结构化 JSON，请先查看字段再决定下一步动作。

对于复杂任务，你**必须**使用 \`update_plan\` 工具来：
1. 在一开始把任务拆成若干个可管理的步骤；
2. 开始执行某一步时，把它标记为 \`in_progress\`；
3. 完成后，把它标记为 \`completed\`。

你还可以使用 \`spawn_sub_agent\` 工具，把具体的子任务委派给一个独立的 agent 实例去完成。
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

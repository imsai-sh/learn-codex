// =============================================================================
// step01 / step01.ts — 最小可运行的 Agent Loop
// -----------------------------------------------------------------------------
// 这是整个教学序列的起点。它只做一件事：
//
//   while True:
//     1. 把对话历史 + 工具定义发给模型；
//     2. 看模型返回的 finish_reason；
//        - 是 "tool_calls" 就执行第一条工具调用，把结果回填到历史里，再继续；
//        - 否则就把模型的最终回答打印出来，结束这一轮 user 输入。
//
// 这一阶段对应 README 里的 “Prompt + Tools + Loop”。
//
// 设计上特意保留了几个“朴素感”的细节：
//   - 一次只处理一条 tool_call（即使模型返回多条），方便阅读循环结构；
//   - 没有抽象 ToolRegistry，直接 if-else 派发；
//   - 没有上下文压缩、没有 sub-agent、没有 plan。
// 这些能力会在后续 step 里逐步加进来。
// =============================================================================

// 自动从当前工作目录的 .env 加载 OPENAI_* 等环境变量；必须在任何读取
// process.env 的代码之前执行（dotenv/config 这个 side-effect import 自带）。
import "dotenv/config";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { fetch } from "undici";

import { executeBash } from "./sandbox.js";

// ---------------------------------------------------------------------------
// 与 OpenAI Chat Completions 协议保持一致的消息形状
// ---------------------------------------------------------------------------

/**
 * 一条对话消息。和 Rust 版的 Message 结构对齐：
 *  - role:           system / user / assistant / tool
 *  - content:        文本内容；assistant 在请求工具时可以为 null。
 *  - tool_calls:     模型“想要调用哪些工具”的清单，只出现在 assistant 消息上。
 *  - tool_call_id:   当前消息是某次工具调用的回执时，必须指明对应的 id，
 *                    模型才能把结果和请求关联起来。
 */
interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content?: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

/** 模型请求的一次工具调用。 */
interface ToolCall {
  id: string;
  type: "function";
  function: FunctionCall;
}

/**
 * 模型实际发起的“函数调用”指令。
 * 注意：arguments 是一个**字符串化的 JSON**，需要我们自己解一次。
 * 这是 OpenAI 协议的规定，不是 bug。
 */
interface FunctionCall {
  name: string;
  arguments: string;
}

/** Chat Completions 响应中我们关心的字段。 */
interface ChatResponse {
  choices: Array<{
    message: Message;
    finish_reason: string;
  }>;
}

/** run_bash 工具的入参形状。 */
interface RunBashArgs {
  cmd: string;
}

// ---------------------------------------------------------------------------
// 主循环
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  // 这三个环境变量保持和 Rust 版完全一样：
  //   OPENAI_API_KEY      鉴权用
  //   OPENAI_BASE_URL     兼容 OpenAI 协议的 Chat Completions endpoint
  //   OPENAI_MODEL_NAME   具体模型名
  const apiKey = process.env.OPENAI_API_KEY ?? "your-api_key";
  const baseUrl = process.env.OPENAI_BASE_URL ?? "your-base_url";
  const modelName = process.env.OPENAI_MODEL_NAME ?? "your-model";

  const systemPrompt = `
你是一个命令行助手。你可以在用户的 macOS 文件系统上执行 bash 命令。
请始终通过 \`run_bash\` 工具显式地展开你的思考过程，用它来探索系统或执行用户要求的任务。
工具的返回结果是结构化 JSON，请先查看字段再决定下一步动作。
`;

  // 用一个数组来保存“对话史”。它会在循环里不断被 append。
  const conversationHistory: Message[] = [
    { role: "system", content: systemPrompt },
  ];

  // 把工具能力以 OpenAI function-calling 的 schema 暴露给模型。
  // 字段命名来自 https://platform.openai.com/docs/guides/function-calling
  const toolsDefinitions = [
    {
      type: "function",
      function: {
        name: "run_bash",
        description:
          "Execute a bash command on the user's machine to list files, read code, or make changes.",
        parameters: {
          type: "object",
          properties: {
            cmd: {
              type: "string",
              description:
                "The bash command string to execute (e.g., 'pwd', 'ls -la', 'cat file.rs')",
            },
          },
          required: ["cmd"],
          additionalProperties: false,
        },
      },
    },
  ];

  // Node 的逐行交互输入。Rust 用的是 std::io::stdin().read_line。
  const rl = createInterface({ input, output });

  // 外层循环：每一次迭代是一次完整的“用户提问 -> 模型响应（可能多轮 tool）”。
  // 内层循环：在没有 tool call 之前持续与模型交互。
  while (true) {
    const userInput = (await rl.question("> ")).trim();
    if (userInput.length === 0) continue;
    if (userInput === "exit" || userInput === "quit") {
      console.log("Bye!");
      rl.close();
      break;
    }

    conversationHistory.push({ role: "user", content: userInput });

    // 内层循环：处理 tool_call -> 工具结果 -> 再次采样 这条链路。
    while (true) {
      const payload = {
        model: modelName,
        messages: conversationHistory,
        tools: toolsDefinitions,
        // 用低温保证模型的工具调用更稳定（不要瞎发挥）。
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
        const errText = await res.text();
        console.log(`\nAPI Error: ${errText}`);
        break;
      }

      const responseBody = (await res.json()) as ChatResponse;
      const choice = responseBody.choices[0];
      const assistantMessage = choice.message;

      // 关键：assistant 整条消息必须原样压回历史，否则后续工具结果回填时
      // 模型会找不到对应的 tool_call_id。
      conversationHistory.push(assistantMessage);

      // 判断模型是请求工具，还是给出了最终答复。
      const wantsToolCall =
        choice.finish_reason === "tool_calls" ||
        (assistantMessage.tool_calls?.length ?? 0) > 0;

      if (wantsToolCall) {
        const toolCalls = assistantMessage.tool_calls!;
        // step01 故意只处理“第一条” tool call，把循环结构压到最简。
        const toolCall = toolCalls[0];
        if (!toolCall) break;

        const functionName = toolCall.function.name;
        const argumentsJson = toolCall.function.arguments;
        console.log(`\n[agent] requested tool: ${functionName}(${argumentsJson})`);

        if (functionName === "run_bash") {
          let args: RunBashArgs | null = null;
          try {
            args = JSON.parse(argumentsJson) as RunBashArgs;
          } catch {
            console.log(`Failed to parse bash arguments: ${argumentsJson}`);
            break;
          }

          // 执行命令拿到结构化结果，然后以 role="tool" 的形式回填。
          const toolOutput = await executeBash(args.cmd);
          conversationHistory.push({
            role: "tool",
            content: toolOutput,
            tool_call_id: toolCall.id,
          });
          // continue：让模型“看到”工具结果，再决定下一步。
          continue;
        }
      } else {
        // 模型给出最终答复 —— 当前 turn 结束。
        if (assistantMessage.content) {
          console.log(`\n[agent] final response\n${assistantMessage.content}\n`);
        }
        break;
      }
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

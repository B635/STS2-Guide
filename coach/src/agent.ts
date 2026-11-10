// One worker loop per flagged decision (orchestrator–worker): the model may
// call the engine's tools up to MAX_TOOL_CALLS times, then must hand in a
// Finding through submit_finding. The verifier checks it; on failure the
// errors go back to the model once (REPAIRS), after that the flag is dropped.

import type OpenAI from "openai";
import { z } from "zod";
import { Engine, EngineToolError } from "./engine.js";
import type { ToolSpec } from "./engine.js";
import { Finding } from "./finding.js";
import { costUsd } from "./llm.js";
import type { DecisionDetail, RunSummary } from "./schemas.js";
import type { Trace } from "./trace.js";
import { verifyFinding } from "./verifier.js";
import type { Evidence } from "./verifier.js";

export const MAX_TOOL_CALLS = 6;
export const REPAIRS = 1;
const MAX_TURNS = MAX_TOOL_CALLS + REPAIRS + 4; // a few turns of slack for nudges
const SUBMIT = "submit_finding";

type Message = OpenAI.Chat.Completions.ChatCompletionMessageParam;
type Tool = OpenAI.Chat.Completions.ChatCompletionTool;

// The agent's own tools: only the calls that need judgement. What every flag
// needs anyway (the comparison, the deck, the allowed what-if changes) is
// fetched by the program before the loop starts: across 55 traced flags the
// model called those three every time, in nearly the same order.
export const AGENT_TOOLS = new Set(["what_if", "get_card", "get_relic"]);

export const SYSTEM_PROMPT = `你是《杀戮尖塔 2》的赛后复盘助手。确定性的牌组评分引擎已经算出：在某个决定上，引擎的首选和玩家的选择不同，而且差距明显。需要的资料已经查好给你（T0 决策细节、T1 两个选项逐项比较、T2 当时的牌组、T3 可以检验的改动）。你的工作是检验假设，再交出一条结构化结论。

规则：
1. 只能依据工具结果说话。每个理由都要注明依据的结果编号（T0–T3 是已给的资料，之后每次工具调用的结果会接着编号）。
2. 理由要对应引擎的因子（factor 代码，例如 deck_fitness:defense），并选 T1 里差距最大的因子，最多 3 条。
3. 用 what_if 检验假设：找出“改了什么，引擎就会同意玩家的选择”（结果里 agrees_with_player_after 为 true）。先根据 T1 的差距和 T2 的牌组想好最可能的改动再试，例如补上缺口最大的那一项、或拿了之前出现过的某张牌；一次没翻转就换一个思路再试，最多试 4 种。找到就写进 what_if；都没翻转，就如实写试了哪些改动、首选都没变。T3 为空时不用 what_if。
4. 语气平和，用简体中文，每句话简短。不要说“失误”“错误”，这是引擎的看法，不是定论；引擎和玩家的一致率只有约一半。
5. 文字里出现的数字和卡牌、遗物名称，必须出现在你引用的工具结果里。不要自己计算新数字。
6. 写给玩家看：用每回合的实际数值说明（例如“防御每回合 6.9 → 8.9，需求 8.5”），不要写因子代码、“因子 +4.11”这类引擎内部分数，也不要写 T0、T1 这些编号。
7. 工具调用最多 ${MAX_TOOL_CALLS} 次。查完就调用 ${SUBMIT} 提交，不要输出其他文字。`;

function toOpenAiTool(spec: ToolSpec): Tool {
  // run_id is filled in by the orchestrator; the model never sees it.
  const schema = structuredClone(spec.inputSchema) as { properties?: Record<string, unknown>; required?: string[] };
  delete schema.properties?.run_id;
  if (schema.required) schema.required = schema.required.filter((r) => r !== "run_id");
  return { type: "function", function: { name: spec.name, description: spec.description, parameters: schema } };
}

const SUBMIT_TOOL: Tool = {
  type: "function",
  function: {
    name: SUBMIT,
    description: "Hand in the finding for this decision. It is checked against the tool results it cites.",
    parameters: z.toJSONSchema(Finding) as Record<string, unknown>,
  },
};

export interface FlagResult {
  flag: string;
  status: "verified" | "dropped";
  finding?: Finding;
  errors: string[];
  evidence: Evidence[];
  toolCalls: number;
  repairs: number;
  usage: { prompt_tokens: number; completion_tokens: number; cost_usd: number; llm_calls: number };
}

export interface AgentContext {
  engine: Engine;
  llm: OpenAI;
  model: string;
  tools: ToolSpec[];
  names: ReadonlySet<string>;
  trace: Trace;
  /** DeepSeek thinking mode: no temperature, no forced tool_choice, and its reasoning must be sent back. */
  thinking: boolean;
}

function compactDetail(detail: DecisionDetail): unknown {
  // The full detail is T0; options beyond the top five rarely matter and cost tokens.
  const keep = new Set([detail.engine_top, detail.chosen]);
  const options = detail.options.filter((o, i) => i < 5 || keep.has(o.id));
  return { ...detail, options: options.map(({ dimensions: _d, ...o }) => o) };
}

export async function explainFlag(ctx: AgentContext, summary: RunSummary, detail: DecisionDetail): Promise<FlagResult> {
  const flag = detail.id;
  const t0 = compactDetail(detail);
  const evidence: Evidence[] = [{ id: "T0", tool: "get_decision", result: t0 }];
  // Fixed context, fetched by the program (traced as tool calls, not counted against the agent's budget).
  const fixed: Array<[string, Record<string, unknown>]> = [
    ["compare_options", { decision_id: flag, option_a: detail.engine_top, option_b: detail.chosen }],
    ["get_deck", { decision_id: flag }],
  ];
  const canWhatIf = detail.kind === "card_reward";
  if (canWhatIf) fixed.push(["what_if_choices", { decision_id: flag }]);
  for (const [tool, args] of fixed) {
    const id = `T${evidence.length}`;
    const output = await ctx.trace.span(
      "tool.prefetch", flag,
      () => ctx.engine.call(tool, { ...args, run_id: summary.run_id }),
      () => ({ tool, args, result_id: id }),
    );
    evidence.push({ id, tool, result: output });
  }
  if (!canWhatIf) evidence.push({ id: "T3", tool: "what_if_choices", result: {} });
  const result: FlagResult = {
    flag, status: "dropped", errors: [], evidence, toolCalls: 0, repairs: 0,
    usage: { prompt_tokens: 0, completion_tokens: 0, cost_usd: 0, llm_calls: 0 },
  };
  const tools = [
    ...ctx.tools.filter((t) => AGENT_TOOLS.has(t.name) && (canWhatIf || t.name !== "what_if")).map(toOpenAiTool),
    SUBMIT_TOOL,
  ];
  const messages: Message[] = [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content:
        `角色 ${summary.character}，进阶 ${summary.ascension ?? 0}，游戏版本 ${summary.build_id ?? "?"}。\n` +
        `要解释的决定：${flag}（第 ${detail.act} 幕第 ${detail.floor} 层）。玩家选了 ${detail.chosen}，引擎首选 ${detail.engine_top}，` +
        `牌组分数差 ${detail.gap}。\n` + evidence.map((e) => `${e.id}（${e.tool}）= ${JSON.stringify(e.result)}`).join("\n"),
    },
  ];

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    const mustSubmit = result.toolCalls >= MAX_TOOL_CALLS;
    const request = ctx.thinking
      ? { model: ctx.model, messages, tools, thinking: { type: "enabled" } }
      : {
          model: ctx.model, messages, tools, temperature: 0, thinking: { type: "disabled" },
          tool_choice: mustSubmit ? { type: "function" as const, function: { name: SUBMIT } } : ("auto" as const),
        };
    const response = await ctx.trace.span(
      "llm.call", flag,
      // `thinking` is a DeepSeek extension the OpenAI types do not know; the SDK sends it as is.
      () => ctx.llm.chat.completions.create(request as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming),
      (r) => ({ model: ctx.model, turn, usage: r.usage, finish_reason: r.choices[0]?.finish_reason }),
    );
    if (response.usage) {
      result.usage.prompt_tokens += response.usage.prompt_tokens;
      result.usage.completion_tokens += response.usage.completion_tokens;
      result.usage.cost_usd += costUsd(ctx.model, response.usage as never);
    }
    result.usage.llm_calls++;
    const message = response.choices[0]?.message;
    if (!message) break;
    messages.push(message);
    const calls = (message.tool_calls ?? []).filter((c) => c.type === "function");
    if (calls.length === 0) {
      messages.push({ role: "user", content: `请调用 ${SUBMIT} 提交结论。` });
      continue;
    }
    for (const call of calls) {
      const reply = (content: unknown) =>
        messages.push({ role: "tool", tool_call_id: call.id, content: typeof content === "string" ? content : JSON.stringify(content) });
      let args: unknown;
      try {
        args = JSON.parse(call.function.arguments || "{}");
      } catch {
        reply({ error: "arguments are not valid JSON" });
        continue;
      }
      if (call.function.name === SUBMIT) {
        const parsed = Finding.safeParse(args);
        const errors = parsed.success
          ? verifyFinding(parsed.data, detail, evidence, ctx.names).errors
          : parsed.error.issues.map((i) => `${i.path.join(".")}: ${i.message}`);
        ctx.trace.event("verify", flag, { ok: errors.length === 0, errors, repairs: result.repairs, finding: args });
        if (errors.length === 0 && parsed.success) {
          result.status = "verified";
          result.finding = parsed.data;
          return result;
        }
        result.errors = errors;
        if (result.repairs >= REPAIRS) return result;
        result.repairs++;
        reply({ rejected: true, errors, instruction: "fix these and call submit_finding again" });
        continue;
      }
      if (result.toolCalls >= MAX_TOOL_CALLS) {
        reply({ error: `tool budget of ${MAX_TOOL_CALLS} calls used; call ${SUBMIT} now` });
        continue;
      }
      result.toolCalls++;
      const id = `T${evidence.length}`;
      try {
        const output = await ctx.trace.span(
          "tool.call", flag,
          () => ctx.engine.call(call.function.name, { ...(args as object), run_id: summary.run_id }),
          () => ({ tool: call.function.name, args, result_id: id }),
        );
        evidence.push({ id, tool: call.function.name, result: output });
        reply({ result_id: id, result: output });
      } catch (error) {
        const text = error instanceof EngineToolError ? error.message : String(error);
        ctx.trace.event("tool.error", flag, { tool: call.function.name, args, error: text });
        reply({ error: text });
      }
    }
  }
  result.errors.push("no verified finding within the turn limit");
  return result;
}

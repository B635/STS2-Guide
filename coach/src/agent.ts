// One worker loop per decision (orchestrator–worker): the model may call the
// engine's tools up to MAX_TOOL_CALLS times, then must hand in a Finding
// through submit_finding. The verifier checks it; on failure the errors go
// back to the model once (REPAIRS), after that the decision is dropped.
//
// The same loop serves the review (part A: one tier-1 flag, engine and player
// disagree clearly) and the follow-up questions (part B: an investigator for
// any scored decision). What it compares depends on the decision:
// - "flag": tier 1, the engine's pick against the player's choice;
// - "disagree": they differ but the engine is less sure (tier 2); same pair,
//   and the prompt asks for a more guarded finding;
// - "agree": they agree; the pick against the runner-up, so the reasons say
//   why the engine likes the choice, and the what-if looks for what would
//   have moved the pick.
// A decision with no such pair (no engine pick, a choice the engine could not
// score, or two options no factor separates) gets no model call at all.

import type OpenAI from "openai";
import { z } from "zod";
import { Engine, EngineToolError } from "./engine.js";
import type { ToolSpec } from "./engine.js";
import { Finding } from "./finding.js";
import { costUsd } from "./llm.js";
import type { DecisionDetail, RunSummary } from "./schemas.js";
import type { Trace } from "./trace.js";
import { separatingFactors, verifyFinding } from "./verifier.js";
import type { Evidence, Pair } from "./verifier.js";

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

// Part B variants: the flag prompt with the first paragraph and the what-if rule swapped.
const DISAGREE_INTRO = `你是《杀戮尖塔 2》的赛后复盘助手。确定性的牌组评分引擎在某个决定上的首选和玩家的选择不同，但引擎对这个分歧把握不大或差距不大（原因见下面的说明），所以它没有被列为“值得想一想”。需要的资料已经查好给你（T0 决策细节、T1 两个选项逐项比较、T2 当时的牌组、T3 可以检验的改动）。你的工作是检验假设，再交出一条结构化结论；语气要比平常更保留，可以在建议里说明这只是引擎不太有把握的看法。`;
const AGREE_INTRO = `你是《杀戮尖塔 2》的赛后复盘助手。确定性的牌组评分引擎在某个决定上的首选和玩家的选择相同。需要的资料已经查好给你（T0 决策细节、T1 这个选择和第二名选项的逐项比较、T2 当时的牌组、T3 可以检验的改动）。你的工作是说明引擎为什么认为这个选择比第二名好，再交出一条结构化结论。`;
const AGREE_WHAT_IF = `3. 可以用 what_if 检验这个选择有多稳：找出“改了什么，引擎就会改选别的”（结果里 pick_changed 为 true）。先根据 T1 的差距和 T2 的牌组想好最可能的改动再试，最多试 3 种。找到就写进 what_if；都没改变，就如实写首选没变。T3 为空时不用 what_if。`;
const RULE_2_AGREE = `2. 理由要对应引擎的因子（factor 代码，例如 deck_fitness:defense），并选 T1 里这个选择和第二名差距最大的因子，最多 3 条。`;

export type Relation = "flag" | "disagree" | "agree";

/** What one decision's finding compares; a = the engine's pick, b = the player's choice or the runner-up. */
export interface Subject {
  relation: Relation;
  pair: { a: string; b: string };
}

/** Why a decision cannot be investigated, in the engine's own terms. */
export type NoSubject = { relation: "none"; reason: "engine_no_pick" | "choice_unknown" | "choice_not_scored" | "no_factor_difference" };

export function subjectOf(detail: DecisionDetail): Subject | NoSubject {
  const top = detail.engine_top, chosen = detail.chosen;
  if (!top) return { relation: "none", reason: "engine_no_pick" };
  if (!chosen) return { relation: "none", reason: "choice_unknown" };
  const scored = (id: string) => detail.options.some((o) => o.id === id && o.deck_score !== null);
  let subject: Subject;
  if (top !== chosen) {
    if (!scored(top) || !scored(chosen)) return { relation: "none", reason: "choice_not_scored" };
    subject = { relation: detail.tier === 1 ? "flag" : "disagree", pair: { a: top, b: chosen } };
  } else {
    const runnerUp = [...detail.options]
      .filter((o) => o.id !== top && o.deck_score !== null)
      .sort((x, y) => (x.rank ?? 99) - (y.rank ?? 99))[0];
    if (!runnerUp) return { relation: "none", reason: "no_factor_difference" };
    subject = { relation: "agree", pair: { a: top, b: runnerUp.id } };
  }
  // A finding needs at least one factor between the two (a rest site scored by its rule alone has none).
  if (separatingFactors(detail, subject.pair).length === 0) return { relation: "none", reason: "no_factor_difference" };
  return subject;
}

export function systemPrompt(relation: Relation): string {
  if (relation === "flag") return SYSTEM_PROMPT;
  const [, rules] = SYSTEM_PROMPT.split("\n\n");
  const intro = relation === "agree" ? AGREE_INTRO : DISAGREE_INTRO;
  let body = rules!;
  if (relation === "agree") {
    body = body.replace(/^2\. .*$/m, RULE_2_AGREE).replace(/^3\. .*$/m, AGREE_WHAT_IF);
  }
  return `${intro}\n\n${body}`;
}

export function toOpenAiTool(spec: ToolSpec): Tool {
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

/** Part B: what the lead asks of one investigator. */
export interface InvestigateOptions {
  /** The player's question, so the finding helps answer it. */
  focus?: string;
  /** Why the engine is less sure here (tier 2), in Chinese, written by the program. */
  note?: string;
}

function compactDetail(detail: DecisionDetail): unknown {
  // The full detail is T0; options beyond the top five rarely matter and cost tokens.
  const keep = new Set([detail.engine_top, detail.chosen]);
  const options = detail.options.filter((o, i) => i < 5 || keep.has(o.id));
  return { ...detail, options: options.map(({ dimensions: _d, ...o }) => o) };
}

/** Part A: explain one tier-1 flag. */
export async function explainFlag(ctx: AgentContext, summary: RunSummary, detail: DecisionDetail): Promise<FlagResult> {
  // A tier-1 flag always has a pair; the fallback keeps part A's behaviour if the engine ever disagrees.
  const subject = subjectOf(detail);
  const flag: Subject = subject.relation === "none" ? { relation: "flag", pair: { a: detail.engine_top ?? "", b: detail.chosen ?? "" } } : subject;
  return investigate(ctx, summary, detail, flag);
}

function deckScore(detail: DecisionDetail, id: string): number {
  return detail.options.find((o) => o.id === id)?.deck_score ?? 0;
}

function brief(summary: RunSummary, detail: DecisionDetail, subject: Subject, options: InvestigateOptions): string {
  const flag = detail.id;
  const head = `角色 ${summary.character}，进阶 ${summary.ascension ?? 0}，游戏版本 ${summary.build_id ?? "?"}。\n`;
  let what: string;
  if (subject.relation === "agree") {
    const gap = Math.round((deckScore(detail, subject.pair.a) - deckScore(detail, subject.pair.b)) * 100) / 100;
    what = `要解释的决定：${flag}（第 ${detail.act} 幕第 ${detail.floor} 层）。玩家和引擎都选了 ${subject.pair.a}，第二名是 ${subject.pair.b}，` +
      `牌组分数差 ${gap}。\n`;
  } else {
    what = `要解释的决定：${flag}（第 ${detail.act} 幕第 ${detail.floor} 层）。玩家选了 ${detail.chosen}，引擎首选 ${detail.engine_top}，` +
      `牌组分数差 ${detail.gap}。\n`;
    if (subject.relation === "disagree" && options.note) what += `引擎对这个分歧没那么有把握：${options.note}。\n`;
  }
  return head + what;
}

/**
 * The loop itself: prefetch the fixed evidence (T0–T3), let the model test
 * hypotheses with what_if and look up facts, check what it hands in.
 */
export async function investigate(
  ctx: AgentContext,
  summary: RunSummary,
  detail: DecisionDetail,
  subject: Subject,
  options: InvestigateOptions = {},
): Promise<FlagResult> {
  const flag = detail.id;
  const pair: Pair = subject.pair;
  const t0 = compactDetail(detail);
  const evidence: Evidence[] = [{ id: "T0", tool: "get_decision", result: t0 }];
  // Fixed context, fetched by the program (traced as tool calls, not counted against the agent's budget).
  const fixed: Array<[string, Record<string, unknown>]> = [
    ["compare_options", { decision_id: flag, option_a: subject.pair.a, option_b: subject.pair.b }],
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
  const focus = options.focus ? `\n玩家的问题：${options.focus}\n结论要能帮助回答这个问题。` : "";
  const messages: Message[] = [
    { role: "system", content: systemPrompt(subject.relation) },
    {
      role: "user",
      content: brief(summary, detail, subject, options) + evidence.map((e) => `${e.id}（${e.tool}）= ${JSON.stringify(e.result)}`).join("\n") + focus,
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
          ? verifyFinding(parsed.data, detail, evidence, ctx.names, pair).errors
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

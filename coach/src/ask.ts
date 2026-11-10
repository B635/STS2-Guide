// Follow-up questions about one reviewed run (part B): a lead agent, a pool
// of investigators and a critic.
//
// Per question:
// 1. the lead states a short plan (a forced `plan` call: which decisions, why);
// 2. it gathers evidence: cheap read tools itself, and `investigate` for the
//    "why" of a decision, which runs one investigator loop (agent.ts) per
//    decision, in parallel; an investigator's report holds the engine's own
//    comparison and, if it passed the verifier, its finding;
// 3. it writes a draft (streamed to the caller as it is produced);
// 4. the program checks every sentence of the draft: numbers and names must
//    appear in the evidence of this conversation (checkText); then the
//    critic, a separate model call, lists claims the evidence does not
//    support;
// 5. if either found something, the lead rewrites once; in the rewrite,
//    sentences that still fail the program check or that the critic still
//    quotes are cut, and the answer says how many could not be confirmed.
//
// Budgets are per question and named below. The run's overview (E0: the run,
// counts, every decision with the engine's verdict, what was not scored, the
// player notes) is given up front: like the fixed evidence of part A, every
// question needs it, so the program fetches it once per session.

import type OpenAI from "openai";
import { investigate, subjectOf } from "./agent.js";
import type { AgentContext, FlagResult, Subject } from "./agent.js";
import { critique } from "./critic.js";
import type { CriticIssue } from "./critic.js";
import type { Engine } from "./engine.js";
import { EngineToolError } from "./engine.js";
import { KIND_LABELS, factorLabel, notStrictLabel, skippedLabel } from "./labels.js";
import { costUsd, streamChat } from "./llm.js";
import type { Usage } from "./llm.js";
import { characterName, lookupId } from "./names.js";
import { noteLine } from "./notes.js";
import type { PlayerNote } from "./notes.js";
import { headline } from "./report.js";
import { DecisionDetail, DecisionList, RunSummary } from "./schemas.js";
import type { DecisionRow } from "./schemas.js";
import { checkText } from "./verifier.js";
import type { Evidence } from "./verifier.js";

/** Model calls of the lead per question (plan, gathering, draft); the last one must be the answer. */
export const MAX_LEAD_TURNS = 8;
/** Read-tool calls of the lead per question (an `investigate` call counts once here). */
export const MAX_LEAD_TOOL_CALLS = 8;
/** Investigator loops per question, over all `investigate` calls. */
export const MAX_INVESTIGATORS = 4;
/** Rewrites after the program check and the critic. */
export const REWRITES = 1;
/**
 * Conversation memory: the last MAX_HISTORY questions and their final
 * answers go back to the lead verbatim (not their tool calls), and only their
 * evidence still counts for the program check and the critic. Older turns are
 * dropped whole, so the context stays bounded however long the chat runs.
 */
export const MAX_HISTORY = 4;

export const UNCONFIRMED = "这个问题我没能用引擎的结果确认。可以换个问法，或者问某一层的具体决定。";

type Message = OpenAI.Chat.Completions.ChatCompletionMessageParam;
type Tool = OpenAI.Chat.Completions.ChatCompletionTool;

const DIMENSIONS = ["frontload", "aoe", "defense", "scaling", "cycle", "energy"];

const LEAD_TOOLS: Tool[] = [
  {
    type: "function",
    function: {
      name: "plan",
      description: "State the plan first: one or two sentences in simplified Chinese on which decisions you will look at and why.",
      parameters: {
        type: "object",
        properties: {
          text: { type: "string", description: "the plan, one or two short sentences, simplified Chinese" },
          decisions: { type: "array", items: { type: "string" }, description: "decision IDs from E0 you intend to look at (may be empty)" },
        },
        required: ["text", "decisions"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_decision",
      description: "One scored decision in full: the options with their scores and the engine's factor messages (per-turn values against the need).",
      parameters: { type: "object", properties: { decision_id: { type: "string" } }, required: ["decision_id"] },
    },
  },
  {
    type: "function",
    function: {
      name: "get_deck",
      description: "The deck, relics, potions, HP and gold at the moment of a decision.",
      parameters: { type: "object", properties: { decision_id: { type: "string" } }, required: ["decision_id"] },
    },
  },
  {
    type: "function",
    function: {
      name: "deck_trend",
      description: "At every card reward of the run: the deck's per-turn value (have) against what the coming fights need (need), per dimension, with below_need marked. For patterns over the run.",
      parameters: { type: "object", properties: { dimension: { type: "string", enum: DIMENSIONS, description: "one dimension; omit for all" } } },
    },
  },
  {
    type: "function",
    function: {
      name: "get_card",
      description: "A card's facts from the game (cost, type, description, upgrade).",
      parameters: { type: "object", properties: { card: { type: "string", description: "card ID or its Chinese name" } }, required: ["card"] },
    },
  },
  {
    type: "function",
    function: {
      name: "get_relic",
      description: "A relic's facts from the game.",
      parameters: { type: "object", properties: { relic: { type: "string", description: "relic ID or its Chinese name" } }, required: ["relic"] },
    },
  },
  {
    type: "function",
    function: {
      name: "investigate",
      description:
        `Hand decisions to investigators, one each, run in parallel (at most ${MAX_INVESTIGATORS} per question). Each compares the options factor by factor, ` +
        "tests what-if changes, and returns the engine's comparison plus a checked finding. Use it for why / should questions about a decision.",
      parameters: {
        type: "object",
        properties: {
          decision_ids: { type: "array", items: { type: "string" }, minItems: 1, maxItems: MAX_INVESTIGATORS },
          focus: { type: "string", description: "what the player wants to know about these decisions, simplified Chinese" },
        },
        required: ["decision_ids", "focus"],
      },
    },
  },
];

/** What the engine can and cannot answer; the lead and the critic both get it. */
export const SCOPE = `引擎能回答的：这局每个选牌、商店（只比较买或离开）、休息点（升级价值只算得到下限，把握很低）决定上，引擎的首选、玩家的选择、分差和原因；每次选牌时牌组每回合的输出、防御等数值和这一幕的需求（deck_trend）；卡牌和遗物的说明。
引擎不能回答的：战斗过程（打出多少伤害、为什么输掉某场战斗、出牌顺序）、地图路线、事件、药水的使用、引擎没法评分的决定（E0 的 not_scored），以及别的对局的细节。`;

export const LEAD_PROMPT = `你是《杀戮尖塔 2》的赛后复盘助手，正在和玩家聊他打完的一局。你根据一个确定性评分引擎的结果回答他的问题。

${SCOPE}
遇到引擎不能回答的问题，直接说引擎回答不了，再说它能看什么。

工作方式：
1. 先调用 plan，用一两句话说明你打算看哪些决定、为什么。
2. 需要事实就用查询工具（get_decision、get_deck、deck_trend、get_card、get_relic），不要猜。
3. 问到某个决定“为什么”“该不该”，用 investigate 交给调查员（只是确认引擎和玩家一致、不需要解释原因的决定，不用调查）：每个调查员深入查一个决定，交回引擎的逐项比较（comparison）和核对过的结论（finding；其中 what_if 是引擎把当时的牌组改一改再重新计分的结果）。几个决定放在同一次 investigate 里一起交出去（每个问题最多 ${MAX_INVESTIGATORS} 个）。
4. 只查回答这个问题需要的决定。调查报告只比较两个选项；问到别的选项（例如跳过）的分数，用 get_decision 查全部选项。
5. 查完就直接写回答，不再调用工具。回答会先由程序核对数字和名称，再由审查员核对每个说法。

回答规则：
- 简体中文，简短（一般不超过 200 字），先给结论再给理由。纯文字，不用 Markdown 的粗体或标题；要列点就用“- ”开头。
- 只说工具结果里有的事。数字、卡牌和遗物名称照抄工具结果；不要自己相加、相减或数个数（E0 的 stats 里已有的计数可以用）。
- 引擎的看法只是一种意见：在这位玩家以往的对局里，它和玩家的选择大约一半一致（这一局的一致数见 E0 的 run.engine_agrees）。语气平和，不要说“失误”“错误”。
- 不要写决定编号（例如 F15-card）、卡牌或遗物的英文 ID、因子代码、工具名（例如 what_if，要说“把当时的牌组改一改再重新计分”）、E1 这类编号；说“第 15 层的选牌”，卡牌用中文名。
- 玩家笔记来自以往对局，只在相关时提一句。
- 查不到依据的，就说引擎的结果里看不出来。
- 不要提调查员、核对、审查这些内部步骤，直接说引擎的看法。`;

export interface Session {
  summary: RunSummary;
  decisions: DecisionList;
  notes: PlayerNote[];
  context: Evidence; // E0
  turns: Turn[]; // at most MAX_HISTORY
  nextEvidence: number;
  reports: Map<string, Evidence>; // investigator reports of the session, by decision ID
  asked: number;
}

interface Turn {
  question: string;
  answer: string;
  evidence: Evidence[];
}

export type AskEvent =
  | { type: "plan"; text: string }
  | { type: "tool"; tool: string; detail: string }
  | { type: "investigate"; decisions: string[] }
  | { type: "investigated"; decision: string; status: string }
  | { type: "draft"; delta: string }
  | { type: "draft_discarded" }
  | { type: "check"; stage: Stage; failed: string[] }
  | { type: "critic"; stage: Stage; issues: CriticIssue[] }
  | { type: "rewrite"; delta: string };

type Stage = "draft" | "rewrite";

export interface Answer {
  index: number;
  question: string;
  plan: string;
  plan_source: "model" | "program";
  draft: string;
  rewrite: string | null;
  /** The model-written part of the answer the player reads (after any cuts). */
  final: string;
  /** A sentence the program adds when something was cut. */
  notice: string;
  status: "passed" | "rewritten" | "cut" | "unconfirmed";
  checks: {
    draft_program: string[];
    draft_critic: CriticIssue[];
    rewrite_program: string[];
    rewrite_critic: CriticIssue[];
    critic_unparsed: number;
    cut: string[];
  };
  /** This question's evidence; with earlier turns' evidence and E0 it is what the answer was checked against. */
  evidence: Evidence[];
  window: string[]; // IDs of every result the answer was checked against
  tool_calls: number;
  investigators: Array<{ decision: string; status: string; relation: string; cached: boolean }>;
  usage: {
    cost_usd: number; llm_calls: number; prompt_tokens: number; completion_tokens: number;
    by_role: Record<"lead" | "investigator" | "critic", number>;
  };
  seconds: number;
}

export type AskContext = AgentContext;

// Said so that the lead cannot read "worth a look" as the engine doubting itself.
const TIER_LABELS: Record<string, string> = { "1": "不同，引擎有把握且差距明显（复盘报告里的“值得想一想”）", "0": "一致" };

function verdictLabel(row: DecisionRow): string {
  if (row.tier === 1 || row.tier === 0) return TIER_LABELS[String(row.tier)]!;
  if (row.tier === 2) return `不同，引擎把握不大（${notStrictLabel(row.not_strict_because)}）`;
  return "未判断";
}

function stats(rows: DecisionRow[]) {
  const card = rows.filter((r) => r.kind === "card_reward");
  const count = (xs: DecisionRow[]) => ({
    decisions: xs.length,
    agree: xs.filter((r) => r.tier === 0).length,
    worth_a_look: xs.filter((r) => r.tier === 1).length,
    differ_less_sure: xs.filter((r) => r.tier === 2).length,
  });
  const acts = [...new Set(rows.map((r) => r.act))].sort();
  return {
    all: count(rows),
    card_rewards: { ...count(card), player_skipped: card.filter((r) => r.chosen === "skip").length, engine_would_skip: card.filter((r) => r.engine_top === "skip").length },
    shops: count(rows.filter((r) => r.kind === "shop")),
    rest_sites: count(rows.filter((r) => r.kind === "rest")),
    by_act: Object.fromEntries(acts.map((a) => [`act_${a}`, count(rows.filter((r) => r.act === a))])),
  };
}

/** E0: everything about the run the lead needs for any question, built by the program. */
export function overview(summary: RunSummary, decisions: DecisionList, notes: PlayerNote[]): unknown {
  return {
    // The engine's own words for what it measures: answers may use them freely.
    dimensions: ["前期输出", "群伤", "防御", "成长", "循环", "能量", "每回合", "需求"],
    run: {
      character: characterName(summary.character), ascension: summary.ascension ?? 0, version: summary.build_id,
      result: summary.win ? "胜利" : `败于 ${summary.killed_by ?? "?"}`, acts_reached: summary.acts_reached,
      final_deck_size: summary.final_deck_size,
      engine_agrees: `${summary.agreement.agree}/${summary.agreement.judged}`,
      engine_agrees_in_earlier_runs: "约一半",
    },
    stats: stats(decisions.decisions),
    decisions: decisions.decisions.map((d) => ({
      id: d.id, act: d.act, floor: d.floor, kind: KIND_LABELS[d.kind] ?? d.kind,
      player: d.chosen_name ?? d.chosen, engine: d.engine_top_name ?? d.engine_top, gap: d.gap, verdict: verdictLabel(d),
    })),
    not_scored: decisions.skipped.map((s) => ({ id: s.id, floor: s.floor, kind: KIND_LABELS[s.kind] ?? s.kind, reason: skippedLabel(s.reason) })),
    // Only other runs: this run's own flags are in `decisions` already, and mixing them confused the critic.
    player_notes_from_earlier_runs: notes.filter((n) => n.run_id !== summary.run_id).map(noteLine),
  };
}

export async function openSession(engine: Engine, runPath: string, notes: PlayerNote[]): Promise<Session> {
  const summary = await engine.callParsed("review_run", { path: runPath }, RunSummary);
  const decisions = await engine.callParsed("list_decisions", { run_id: summary.run_id }, DecisionList);
  return newSession(summary, decisions, notes);
}

export function newSession(summary: RunSummary, decisions: DecisionList, notes: PlayerNote[]): Session {
  return {
    summary, decisions, notes,
    context: { id: "E0", tool: "overview", result: overview(summary, decisions, notes) },
    turns: [], nextEvidence: 1, reports: new Map(), asked: 0,
  };
}

/** Sentences of an answer, delimiters kept, so cutting some keeps the rest as written. */
export function sentences(text: string): string[] {
  const out: string[] = [];
  for (const piece of text.split(/(?<=[。！？!?；\n])/)) {
    // Line breaks stay with the sentence before them, so cutting keeps the layout.
    if (piece.trim().length === 0 && out.length > 0) out[out.length - 1] += piece;
    else if (piece.length > 0) out.push(piece);
  }
  return out.filter((s) => s.trim().length > 0);
}

// Engine IDs (DEFEND_NECROBINDER, F15-card, E3) and tool names are for tools, not for the player.
const INTERNAL_ID = /\b(?:[A-Z][A-Z0-9]*_[A-Z0-9_]+|F\d{2}-(?:card|rest|shop)\d*|[ET]\d+|what_if|deck_trend|get_decision|get_deck|get_card|get_relic|investigate)\b/;

/** The program check: every sentence's numbers and names must appear in the evidence, and no internal IDs. */
export function checkAnswer(text: string, evidence: Evidence[], names: ReadonlySet<string>): Array<{ sentence: string; errors: string[] }> {
  const results = evidence.map((e) => e.result);
  return sentences(text)
    .map((sentence) => {
      const errors = checkText("answer", sentence.trim(), results, names, "the evidence of this conversation");
      const id = INTERNAL_ID.exec(sentence);
      if (id) errors.push(`answer: write names, not the internal ID ${id[0]}`);
      return { sentence, errors };
    })
    .filter((x) => x.errors.length > 0);
}

const BULLET = /^\s*(?:[-•·*]|\d+[.、）)])\s*/;

/**
 * The sentences left after a cut. A list item goes whole when one of its
 * sentences is cut: its other sentences rarely read right without the first.
 */
export function withoutCut(list: string[], cut: ReadonlySet<string>): string[] {
  const lines: string[][] = [[]];
  for (const s of list) {
    lines[lines.length - 1]!.push(s);
    if (s.endsWith("\n")) lines.push([]);
  }
  return lines.flatMap((line) => {
    const bullet = line.length > 0 && BULLET.test(line[0]!);
    if (bullet && line.some((x) => cut.has(x))) return [];
    return line.filter((x) => !cut.has(x));
  });
}

/** Does the critic's quote point at this sentence? (It may quote part of one, or a sentence without its punctuation.) */
function quotes(issue: CriticIssue, sentence: string): boolean {
  const strip = (s: string) => s.replace(/[\s。！？!?；，,：:“”"「」]/g, "");
  const a = strip(issue.sentence), b = strip(sentence);
  return a.length > 0 && b.length > 0 && (b.includes(a) || a.includes(b));
}

function compactDetail(detail: DecisionDetail, row: DecisionRow | undefined): unknown {
  return {
    id: detail.id, kind: KIND_LABELS[detail.kind] ?? detail.kind, act: detail.act, floor: detail.floor, hp: detail.hp, max_hp: detail.max_hp, gold: detail.gold,
    player: detail.chosen, engine: detail.engine_top, confidence: detail.confidence, gap: detail.gap,
    verdict: row ? verdictLabel(row) : undefined,
    options: detail.options.slice(0, 5).map((o) => ({
      id: o.id, name: o.name, deck_score: o.deck_score, rank: o.rank, fully_scored: o.fully_scored,
      factors: o.factors.filter((f) => f.message).map((f) => f.message),
    })),
  };
}

const NO_SUBJECT: Record<string, string> = {
  engine_no_pick: "引擎在这个决定上没有给出首选",
  choice_unknown: "记录里看不出玩家的选择",
  choice_not_scored: "玩家的选择引擎没法评分",
  no_factor_difference: "引擎在这两个选项之间没有可以比较的因子（例如休息点只按规则判断）",
};

/** What the lead gets back from one investigator: the engine's comparison (program) and the checked finding (model). */
function report(detail: DecisionDetail, subject: Subject, result: FlagResult, row: DecisionRow | undefined): unknown {
  const name = (id: string) => detail.options.find((o) => o.id === id)?.name ?? id;
  const score = (id: string) => detail.options.find((o) => o.id === id)?.deck_score ?? 0;
  let head: string;
  if (subject.relation === "agree") {
    const gap = Math.round((score(subject.pair.a) - score(subject.pair.b)) * 100) / 100;
    head = `你和引擎都选了「${name(subject.pair.a)}」，牌组分数比第二名「${name(subject.pair.b)}」高 ${gap} 分`;
  } else if ((detail.gap ?? 0) > 0) {
    head = headline(detail);
  } else {
    head = `引擎首选「${name(subject.pair.a)}」，你选了「${name(subject.pair.b)}」；只看牌组分数两者相差 ${detail.gap}，引擎的首选来自它的取舍倾向（例如跳过门槛）`;
  }
  const compared = result.evidence.find((e) => e.tool === "compare_options")?.result as
    | { factors?: Array<{ factor: string; a_message: string | null; b_message: string | null }> }
    | undefined;
  const f = result.status === "verified" ? result.finding : undefined;
  return {
    decision_id: detail.id,
    where: `第 ${detail.act} 幕第 ${detail.floor} 层${KIND_LABELS[detail.kind] ?? detail.kind}`,
    verdict: row ? verdictLabel(row) : undefined,
    headline: head,
    comparison: (compared?.factors ?? []).slice(0, 3).map((x) => ({
      aspect: factorLabel(x.factor), [name(subject.pair.a)]: x.a_message, [name(subject.pair.b)]: x.b_message,
    })),
    ...(f
      ? {
          status: "verified",
          finding: {
            reasons: f.reasons.map((r) => ({ aspect: factorLabel(r.factor), text: r.text })),
            ...(f.what_if_text ? { what_if: f.what_if_text } : {}),
            advice: f.advice,
          },
        }
      : { status: "not_confirmed", note: "这个决定只有引擎的逐项比较，没有更多说明" }),
  };
}

function unavailable(decisionId: string, why: string): unknown {
  return { decision_id: decisionId, status: "unavailable", note: why };
}

interface Tally {
  cost: number;
  calls: number;
  prompt: number;
  completion: number;
  byRole: Record<"lead" | "investigator" | "critic", number>;
}

function addUsage(t: Tally, role: keyof Tally["byRole"], model: string, usage: Usage | undefined): void {
  t.calls++;
  if (!usage) return;
  const cost = costUsd(model, usage);
  t.cost += cost;
  t.byRole[role] += cost;
  t.prompt += usage.prompt_tokens;
  t.completion += usage.completion_tokens;
}

export async function ask(ctx: AskContext, session: Session, question: string, onEvent: (e: AskEvent) => void = () => {}): Promise<Answer> {
  const started = Date.now();
  const index = ++session.asked;
  const label = `Q${index}`;
  const { trace } = ctx;
  const tally: Tally = { cost: 0, calls: 0, prompt: 0, completion: 0, byRole: { lead: 0, investigator: 0, critic: 0 } };
  const evidence: Evidence[] = [];
  const window = () => [session.context, ...session.turns.flatMap((t) => t.evidence), ...evidence];
  const answer: Answer = {
    index, question, plan: "", plan_source: "model", draft: "", rewrite: null, final: "", notice: "", status: "passed",
    checks: { draft_program: [], draft_critic: [], rewrite_program: [], rewrite_critic: [], critic_unparsed: 0, cut: [] },
    evidence, window: [], tool_calls: 0, investigators: [],
    usage: { cost_usd: 0, llm_calls: 0, prompt_tokens: 0, completion_tokens: 0, by_role: tally.byRole },
    seconds: 0,
  };
  const rows = new Map(session.decisions.decisions.map((d) => [d.id, d]));
  const keep = (tool: string, result: unknown): Evidence => {
    const e = { id: `E${session.nextEvidence++}`, tool, result };
    evidence.push(e);
    return e;
  };

  const messages: Message[] = [
    { role: "system", content: `${LEAD_PROMPT}\n\nE0（这局的概况，程序给出）= ${JSON.stringify(session.context.result)}` },
    ...session.turns.flatMap((t): Message[] => [{ role: "user", content: t.question }, { role: "assistant", content: t.answer }]),
    { role: "user", content: question },
  ];

  const leadCall = async (stage: string, request: Record<string, unknown>, onDelta?: (d: string) => void) => {
    const streamed = await trace.span(
      "llm.lead", label,
      () => streamChat(ctx.llm, { model: ctx.model, temperature: 0, thinking: { type: "disabled" }, ...request }, onDelta),
      (r) => ({ model: ctx.model, stage, usage: r.usage, finish_reason: r.finish_reason, tool_calls: r.message.tool_calls?.map((c) => c.function.name) ?? [] }),
    );
    addUsage(tally, "lead", ctx.model, streamed.usage);
    return streamed;
  };

  // --- tools -------------------------------------------------------------
  let investigatorsUsed = 0;
  const runInvestigations = async (ids: string[], focus: string): Promise<unknown[]> => {
    onEvent({ type: "investigate", decisions: ids });
    // Everything up to the first await runs in order for each ID, so the budget is counted before any loop starts.
    return Promise.all(ids.map(async (id): Promise<unknown> => {
      const cached = session.reports.get(id);
      if (cached) {
        answer.investigators.push({ decision: id, status: "cached", relation: "", cached: true });
        onEvent({ type: "investigated", decision: id, status: "cached" });
        trace.event("investigate.cached", id, { question: label, result_id: cached.id });
        return { ...(cached.result as object), note_cached: `已在 ${cached.id} 查过` };
      }
      const skipped = session.decisions.skipped.find((s) => s.id === id);
      if (skipped) return unavailable(id, `这个决定引擎没法评分：${skippedLabel(skipped.reason)}`);
      if (investigatorsUsed >= MAX_INVESTIGATORS) {
        trace.event("investigate.over_budget", id, { question: label });
        return unavailable(id, `investigator budget of ${MAX_INVESTIGATORS} per question used`);
      }
      investigatorsUsed++;
      let detail: DecisionDetail;
      try {
        detail = await ctx.engine.callParsed("get_decision", { run_id: session.summary.run_id, decision_id: id }, DecisionDetail);
      } catch (error) {
        return unavailable(id, error instanceof EngineToolError ? error.message : String(error));
      }
      const subject = subjectOf(detail);
      if (subject.relation === "none") {
        investigatorsUsed--; // no model call
        answer.investigators.push({ decision: id, status: "unavailable", relation: "none", cached: false });
        trace.event("investigate.skipped", id, { question: label, reason: subject.reason });
        return { ...unavailable(id, NO_SUBJECT[subject.reason] ?? subject.reason) as object, headline: detail.engine_top === detail.chosen ? undefined : headline(detail) };
      }
      const slot = { decision: id, status: "running", relation: subject.relation, cached: false };
      answer.investigators.push(slot);
      const row = rows.get(id);
      const result = await trace.span(
        "investigate", id,
        () => investigate(ctx, session.summary, detail, subject, {
          focus,
          ...(subject.relation === "disagree" && row ? { note: notStrictLabel(row.not_strict_because) } : {}),
        }),
        (r) => ({ question: label, relation: subject.relation, status: r.status, tool_calls: r.toolCalls, repairs: r.repairs, cost_usd: r.usage.cost_usd, errors: r.errors }),
      );
      tally.cost += result.usage.cost_usd;
      tally.byRole.investigator += result.usage.cost_usd;
      tally.calls += result.usage.llm_calls;
      tally.prompt += result.usage.prompt_tokens;
      tally.completion += result.usage.completion_tokens;
      slot.status = result.status;
      onEvent({ type: "investigated", decision: id, status: result.status });
      return report(detail, subject, result, row);
    }));
  };

  const runTool = async (name: string, args: Record<string, unknown>): Promise<unknown> => {
    const run_id = session.summary.run_id;
    switch (name) {
      case "get_decision": {
        const id = String(args.decision_id ?? "");
        const detail = await ctx.engine.callParsed("get_decision", { run_id, decision_id: id }, DecisionDetail);
        return compactDetail(detail, rows.get(id));
      }
      case "get_deck":
        return ctx.engine.call("get_deck", { run_id, decision_id: String(args.decision_id ?? "") });
      case "deck_trend":
        return ctx.engine.call("deck_trend", { run_id, ...(args.dimension ? { dimension: String(args.dimension) } : {}) });
      case "get_card": {
        const id = lookupId("cards", String(args.card ?? ""));
        if (!id) throw new EngineToolError(`unknown card ${String(args.card)}`);
        return ctx.engine.call("get_card", { run_id, card_id: id });
      }
      case "get_relic": {
        const id = lookupId("relics", String(args.relic ?? ""));
        if (!id) throw new EngineToolError(`unknown relic ${String(args.relic)}`);
        return ctx.engine.call("get_relic", { run_id, relic_id: id });
      }
      default:
        throw new EngineToolError(`no tool ${name}`);
    }
  };

  const setProgramPlan = (ids: string[], why: string) => {
    answer.plan_source = "program";
    const where = [...new Set(ids)].map((d) => {
      const r = rows.get(d);
      return r ? `第 ${r.floor} 层的${KIND_LABELS[r.kind] ?? r.kind}` : null;
    }).filter((x): x is string => x !== null);
    answer.plan = where.length ? `先看${where.join("、")}。` : "先看这局的概况。";
    trace.event("plan.program", label, { why, plan: answer.plan });
    onEvent({ type: "plan", text: answer.plan });
  };

  // --- 1. plan, 2. evidence, 3. draft --------------------------------------
  let draft = "";
  for (let turn = 0; turn < MAX_LEAD_TURNS; turn++) {
    const planTurn = turn === 0;
    const lastTurn = turn === MAX_LEAD_TURNS - 1;
    const choice = planTurn
      ? { type: "function", function: { name: "plan" } }
      : lastTurn || answer.tool_calls >= MAX_LEAD_TOOL_CALLS ? "none" : "auto";
    let streamedSome = false;
    const response = await leadCall(planTurn ? "plan" : "gather", { messages, tools: LEAD_TOOLS, tool_choice: choice }, (delta) => {
      streamedSome = true;
      onEvent({ type: "draft", delta });
    });
    const message = response.message;
    messages.push(message);
    const calls = message.tool_calls ?? [];
    if (calls.length === 0) {
      draft = (message.content ?? "").trim();
      if (draft) break;
      messages.push({ role: "user", content: "请直接写回答。" });
      continue;
    }
    if (streamedSome) onEvent({ type: "draft_discarded" }); // text before tool calls was not the answer
    const reply = (id: string, content: unknown) => messages.push({ role: "tool", tool_call_id: id, content: JSON.stringify(content) });
    if (planTurn && !calls.some((c) => c.function.name === "plan")) {
      // The forced plan call is not always honoured; the program then states what the first tools will open.
      const touched = calls.flatMap((c) => {
        try {
          const a = JSON.parse(c.function.arguments || "{}") as Record<string, unknown>;
          return [a.decision_id, ...(Array.isArray(a.decision_ids) ? a.decision_ids : [])].filter((x): x is string => typeof x === "string");
        } catch {
          return [];
        }
      });
      setProgramPlan(touched, "model did not call plan");
    }
    // Investigations of one turn run together, after the other tools of that turn.
    for (const call of calls) {
      let args: Record<string, unknown>;
      try {
        args = JSON.parse(call.function.arguments || "{}") as Record<string, unknown>;
      } catch {
        reply(call.id, { error: "arguments are not valid JSON" });
        continue;
      }
      const name = call.function.name;
      if (name === "plan") {
        const text = String(args.text ?? "").trim();
        const known = (Array.isArray(args.decisions) ? args.decisions.map(String) : []).filter((d) => rows.has(d) || session.decisions.skipped.some((s) => s.id === d));
        const errors = checkText("plan", text, [session.context.result], ctx.names);
        trace.event("plan", label, { text, decisions: known, ok: errors.length === 0 && text.length > 0, errors });
        if (text && errors.length === 0) {
          answer.plan = text;
          onEvent({ type: "plan", text: answer.plan });
        } else {
          // The plan the player sees must hold up too; otherwise the program says which decisions it will open.
          setProgramPlan(known, "plan failed the check");
        }
        reply(call.id, { ok: true });
        continue;
      }
      if (answer.tool_calls >= MAX_LEAD_TOOL_CALLS) {
        reply(call.id, { error: `tool budget of ${MAX_LEAD_TOOL_CALLS} calls used; write the answer now` });
        continue;
      }
      answer.tool_calls++;
      if (name === "investigate") {
        const ids = [...new Set((Array.isArray(args.decision_ids) ? args.decision_ids : []).map(String))];
        if (ids.length === 0) {
          reply(call.id, { error: "give decision_ids, e.g. [\"F14-card\"]" });
          continue;
        }
        const reports = await runInvestigations(ids, String(args.focus ?? question));
        // Every report is evidence (the lead reads all of them); finished ones are also kept for follow-ups.
        const kept = reports.map((r, i) => {
          const status = (r as { status?: string }).status;
          if ((r as { note_cached?: string }).note_cached) return r;
          const e = keep("investigate", r);
          if (status === "verified" || status === "not_confirmed") session.reports.set(ids[i]!, e);
          return { result_id: e.id, ...(r as object) };
        });
        reply(call.id, { reports: kept });
        continue;
      }
      const detail = name === "get_card" ? String(args.card) : name === "get_relic" ? String(args.relic) : String(args.decision_id ?? args.dimension ?? "");
      onEvent({ type: "tool", tool: name, detail });
      try {
        const output = await trace.span("tool.lead", label, () => runTool(name, args), () => ({ tool: name, args }));
        const e = keep(name, output);
        reply(call.id, { result_id: e.id, result: output });
      } catch (error) {
        const text = error instanceof EngineToolError ? error.message : String(error);
        trace.event("tool.error", label, { tool: name, args, error: text });
        reply(call.id, { error: text });
      }
    }
  }
  answer.draft = draft;
  if (!draft) {
    answer.final = UNCONFIRMED;
    answer.status = "unconfirmed";
    return finish();
  }

  // --- 4. program check, then the critic ------------------------------------
  const checkStage = async (stage: Stage, text: string) => {
    const failed = checkAnswer(text, window(), ctx.names);
    trace.event("check.program", label, { stage, ok: failed.length === 0, failed });
    onEvent({ type: "check", stage, failed: failed.map((f) => f.sentence.trim()) });
    const verdict = await critique(ctx.llm, ctx.model, question, text, window(), trace, label, SCOPE);
    tally.cost += verdict.cost_usd;
    tally.byRole.critic += verdict.cost_usd;
    tally.calls++;
    tally.prompt += verdict.prompt_tokens;
    tally.completion += verdict.completion_tokens;
    if (!verdict.parsed) answer.checks.critic_unparsed++;
    onEvent({ type: "critic", stage, issues: verdict.issues });
    return { failed, issues: verdict.issues };
  };

  const first = await checkStage("draft", draft);
  answer.checks.draft_program = first.failed.map((f) => `${f.sentence.trim()} — ${f.errors.join("; ")}`);
  answer.checks.draft_critic = first.issues;
  if (first.failed.length === 0 && first.issues.length === 0) {
    answer.final = draft;
    return finish();
  }

  // --- 5. one rewrite; what still fails is cut ------------------------------
  const problems = [
    ...first.failed.map((f) => `- 「${f.sentence.trim()}」：${f.errors.map((e) => e.replace(/^answer: /, "")).join("；")}`),
    ...first.issues.map((i) => `- 「${i.sentence}」：${i.reason}`),
  ].join("\n");
  messages.push({
    role: "user",
    content: `（核对结果，不是玩家的话）你的回答里有这些说法没有依据：\n${problems}\n请重写整段回答：改正或删掉这些说法。不要声称引擎没有某项数据（可能只是这次没查），也不要提核对、审查这些步骤。不要再调用工具，只输出新的回答。`,
  });
  const rewritten = await leadCall("rewrite", { messages, tools: LEAD_TOOLS, tool_choice: "none" }, (delta) => onEvent({ type: "rewrite", delta }));
  const rewrite = (rewritten.message.content ?? "").trim();
  answer.rewrite = rewrite;
  answer.status = "rewritten";
  let kept = sentences(rewrite);
  // The program check first: those sentences go before the critic reads the text.
  const second = checkAnswer(rewrite, window(), ctx.names);
  answer.checks.rewrite_program = second.map((f) => `${f.sentence.trim()} — ${f.errors.join("; ")}`);
  trace.event("check.program", label, { stage: "rewrite", ok: second.length === 0, failed: second });
  onEvent({ type: "check", stage: "rewrite", failed: second.map((f) => f.sentence.trim()) });
  const cut = new Set(second.map((f) => f.sentence));
  kept = withoutCut(kept, cut);
  let unmatched = 0;
  if (kept.join("").trim()) {
    const verdict = await critique(ctx.llm, ctx.model, question, kept.join(""), window(), trace, label, SCOPE);
    tally.cost += verdict.cost_usd;
    tally.byRole.critic += verdict.cost_usd;
    tally.calls++;
    tally.prompt += verdict.prompt_tokens;
    tally.completion += verdict.completion_tokens;
    if (!verdict.parsed) answer.checks.critic_unparsed++;
    answer.checks.rewrite_critic = verdict.issues;
    onEvent({ type: "critic", stage: "rewrite", issues: verdict.issues });
    for (const issue of verdict.issues) {
      const hit = kept.filter((s) => quotes(issue, s));
      if (hit.length === 0) unmatched++;
      for (const s of hit) cut.add(s);
    }
    kept = withoutCut(kept, cut);
  }
  answer.checks.cut = [...cut].map((s) => s.trim());
  answer.final = kept.join("").trim();
  if (!answer.final) {
    answer.final = UNCONFIRMED;
    answer.status = "unconfirmed";
  } else if (cut.size > 0 || unmatched > 0) {
    answer.status = "cut";
    const parts = [];
    if (cut.size > 0) parts.push(`有 ${cut.size} 句说法没能和引擎的结果核对上，已经删去`);
    if (unmatched > 0) parts.push(`审查员还对 ${unmatched} 处说法有疑问，请把它当作没能确认`);
    answer.notice = `（${parts.join("；")}。）`;
  }
  trace.event("cut", label, { cut: answer.checks.cut, unmatched });
  return finish();

  function finish(): Answer {
    answer.window = window().map((e) => e.id);
    answer.usage = {
      cost_usd: tally.cost, llm_calls: tally.calls, prompt_tokens: tally.prompt, completion_tokens: tally.completion, by_role: tally.byRole,
    };
    answer.seconds = (Date.now() - started) / 1000;
    const shown = answer.notice ? `${answer.final}\n${answer.notice}` : answer.final;
    session.turns.push({ question, answer: shown, evidence });
    // Bounded memory: drop the oldest turns with their evidence; their reports go too.
    while (session.turns.length > MAX_HISTORY) {
      const dropped = session.turns.shift()!;
      const ids = new Set(dropped.evidence.map((e) => e.id));
      for (const [d, e] of session.reports) if (ids.has(e.id)) session.reports.delete(d);
    }
    trace.event("ask.done", label, {
      question, status: answer.status, tool_calls: answer.tool_calls, investigators: answer.investigators,
      cost_usd: answer.usage.cost_usd, llm_calls: answer.usage.llm_calls, seconds: answer.seconds,
      draft_program_failed: answer.checks.draft_program.length, draft_critic_issues: answer.checks.draft_critic.length,
      cut: answer.checks.cut.length,
    });
    return answer;
  }
}

/** What the player reads. */
export function shownAnswer(a: Answer): string {
  return a.notice ? `${a.final}\n${a.notice}` : a.final;
}

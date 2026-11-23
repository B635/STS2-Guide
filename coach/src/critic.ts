// The critic: a separate model call that reads a draft answer next to the
// evidence gathered for it and lists the claims the evidence does not
// support. It runs after the program's own check (numbers and names), which
// it cannot replace: the program catches made-up numbers, the critic catches
// claims built from real numbers that the evidence still does not say (a
// cause of death, a "should have" the engine never gave, an answer to a
// question the engine cannot answer).

import type OpenAI from "openai";
import { costUsd } from "./llm.js";
import type { Trace } from "./trace.js";
import type { Evidence } from "./verifier.js";

/** Original TASK-028 b5 prompt, kept for reproducible baseline comparison. */
export const CRITIC_PROMPT_B5 = `你是审查员。下面有玩家的问题、复盘助手的回答草稿，以及助手能依据的全部证据（一个确定性评分引擎的工具结果，按编号列出）。
证据说明：E0 是这局的概况（程序给出）。tool 为 investigate 的是调查员的报告：headline 和 comparison 是引擎的原始数据；finding 里的 reasons、what_if、advice 是程序已经核对过的结论，也算证据；what_if 是引擎把当时的牌组改一改再重新计分的结果。其余是引擎工具的原始结果。
判断方法：对草稿的每一句，先在证据里找它的关键词；找得到相同或等价的意思就算有依据（概括、换个说法都可以）。只有找不到或和证据矛盾时才列出。
数字和卡牌名称已经由程序逐个核对过，都出现在证据里；你不用再查它们在不在，只看它们有没有被安到错的东西上（例如把这一层的数字说成另一层的）。
草稿里描述引擎能看什么、不能看什么的句子，只要和“引擎的范围”一致就算有依据。
要列出的说法：
- 证据里没有的事实或数字；
- 和证据矛盾的说法（例如把引擎的首选、玩家的选择或分差说反）；
- 关于战斗过程、输赢原因、地图路线、事件的断言（引擎不模拟这些，证据里没有就不能这样说）；
- 把引擎的看法说成定论，或把证据里没有的建议说成引擎的意见。
不算问题的：根据证据给出的方向性建议、对问题的复述、“引擎回答不了”或“没能确认”这类说明、语气词。也不要因为措辞不完整、没有列全、或“可能被误解”而列出：只看说的事有没有依据。
只输出 json：{"doubts":[{"sentence":"草稿里的原句，逐字复制","reason":"先写你在证据里找到或没找到什么","verdict":"supported 或 unsupported"}]}。只列你有疑问的句子；查过之后有依据的写 supported。都没疑问就输出 {"doubts":[]}。`;

/** Calibrated TASK-031 c1 prompt: explicit 7-mode hallucination checklist + anti-over-rejection rubric. */
export const CRITIC_PROMPT_C1 = `你是复盘助手的事实审查员。下面给出“引擎的范围”、玩家的问题、回答草稿，以及助手能依据的全部证据（按编号列出）。
证据说明：E0 是程序给出的整局概况（包含每个决定的玩家选择、引擎首选、分差、判定与汇总计数）。tool 为 investigate 的是调查员报告：headline 和 comparison 是引擎原始数据，finding 里的 reasons、what_if、advice 是程序核对过的结论；what_if 是改动当时牌组后重算的结果。其余是引擎工具原始结果。

请逐句对照证据，重点排查以下 7 类真正的问题（命中任意一项即判 unsupported）：
1. 选择/首选/方向说反：把玩家选择与引擎首选说反、分数归属说反、或把 what_if 结论说反（例如证据说改牌后首选没变，草稿说会改选，或反之）。
2. 跨层/跨选项错配：把某一层或某个选项的真实数字、卡牌、理由安到另一层或另一个选项上。
3. 信心/判定矛盾：证据写明“有把握、差距明显”却说“差距不大/把握不大”；或证据为低/中把握、部分效果未计入，却说“把握非常高/全部算清/明显失误/必须拿”。
4. 超范围与战斗因果编造：引擎不模拟战斗过程、出牌顺序、击杀目标、掉血/阵亡原因、地图路线、事件或未评分决定（not_scored）/未建模遗物。即使草稿先声明“引擎看不到”，只要随后仍断言战斗中“抽不到牌导致暴毙”“能一回合秒怪”“买这件未评分遗物非常划算”，也判 unsupported。
5. 无依据的额外机制或排他断言：声称证据里没出现的维度/协同/效果（如 comparison 无能量项却说“补能量”，或只看了单张牌却说“四个选项里唯一改善循环的”）。
6. 汉字数字与手工计数错误：把汉字写的次数或总数说错（如“五处”“八次”“全部六张”“只有三次一致”与 E0 统计或证据列表不符）。
7. 把证据里没有的事实或数值说成引擎结果。

以下情况属于合理表达，**必须判 supported（不要误杀）**：
- 对证据里完整表格或多项数据的准确概括：例如 deck_trend 每一层都低于需求时说“从第 1 幕一开始就不够，不是某一层突然掉下来的”；或对比中某维度差距最大时说“主要因为防御/主要差在防御”。
- 忠实复述或转述 E0 / get_decision / investigate 里的 verdict、confidence、部分计分提示或 what_if 结论（只要方向与程度不矛盾；注：E0 中 tier 1 决定的 confidence 字段常为 medium，它与 verdict 原文“引擎有把握且差距明显”是一致的，不算矛盾）。
- 中文并列短语计数：例如“这局 20 个选牌、商店和休息点决定”指三类决定合计 20 个，只要与 E0 总数一致就算有依据，不要误读成“20 个选牌”。
- 与“引擎的范围”一致的能力说明、“引擎回答不了/看不出来”的诚实拒答、以及基于证据缺口给出的温和方向性建议（如“后续可留意格挡牌”）。
- 仅因“没列全所有细节”或“换了等价说法”不得判 unsupported。

自检规则：写 reason 时先核对证据原话；如果你自己在 reason 里确认证据确实支持该句，verdict 必须写 "supported"，绝不能自相矛盾地写 "unsupported"。

只输出合法 JSON：
{"doubts":[{"sentence":"草稿里的原句，逐字复制","reason":"先写你在证据里找到或没找到什么","verdict":"supported 或 unsupported"}]}
只列你有疑问的句子；若整段草稿都没有问题，输出 {"doubts":[]}。`;

export const CRITIC_PROMPT = CRITIC_PROMPT_C1;

export interface CriticIssue {
  sentence: string;
  reason: string;
}

export interface CriticVerdict {
  issues: CriticIssue[];
  /** false when the reply was not the asked-for JSON (counted, and treated as no issues). */
  parsed: boolean;
  /** Doubts the critic itself then found support for (not issues). */
  cleared: number;
  cost_usd: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface CritiqueOptions {
  prompt?: string;
  thinking?: boolean;
}

/** Evidence as the critic sees it: one line per result, by its ID. */
export function evidenceText(evidence: Evidence[]): string {
  return evidence.map((e) => `${e.id}（${e.tool}）= ${JSON.stringify(e.result)}`).join("\n");
}

function stripJsonFence(raw: string): string {
  const trimmed = raw.trim();
  const match = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return match ? match[1]!.trim() : trimmed;
}

export async function critique(
  llm: OpenAI,
  model: string,
  question: string,
  draft: string,
  evidence: Evidence[],
  trace: Trace,
  label: string,
  scope = "",
  options: CritiqueOptions = {},
): Promise<CriticVerdict> {
  const systemPrompt = options.prompt ?? CRITIC_PROMPT;
  const thinking = options.thinking ?? false;
  const request: Record<string, unknown> = {
    model,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: `${scope ? `引擎的范围：\n${scope}\n\n` : ""}玩家的问题：${question}\n\n回答草稿：\n${draft}\n\n证据：\n${evidenceText(evidence)}` },
    ],
  };
  if (thinking) {
    request.thinking = { type: "enabled" };
  } else {
    request.temperature = 0;
    request.thinking = { type: "disabled" };
    request.response_format = { type: "json_object" };
  }
  const response = await trace.span(
    "llm.critic", label,
    () => llm.chat.completions.create(request as unknown as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming),
    (r) => ({ model, usage: r.usage }),
  );
  const usage = response.usage;
  const verdict: CriticVerdict = {
    issues: [], parsed: false, cleared: 0,
    cost_usd: usage ? costUsd(model, usage as never) : 0,
    prompt_tokens: usage?.prompt_tokens ?? 0,
    completion_tokens: usage?.completion_tokens ?? 0,
  };
  try {
    // The critic writes its reason before its verdict; only "unsupported" counts (it often lists a doubt and then finds the support).
    const data = JSON.parse(stripJsonFence(response.choices[0]?.message.content ?? "")) as { doubts?: unknown };
    if (Array.isArray(data.doubts)) {
      verdict.parsed = true;
      const doubts = data.doubts
        .filter((x): x is { sentence: unknown; reason?: unknown; verdict?: unknown } => typeof x === "object" && x !== null && "sentence" in x)
        .map((x) => ({ sentence: String(x.sentence).trim(), reason: String(x.reason ?? ""), verdict: String(x.verdict ?? "unsupported") }));
      verdict.cleared = doubts.filter((x) => x.verdict === "supported").length;
      verdict.issues = doubts.filter((x) => x.verdict !== "supported" && x.sentence.length > 0).map(({ sentence, reason }) => ({ sentence, reason }));
    }
  } catch {
    // Unparsable verdict: recorded below; the program check still stands.
  }
  trace.event("critic.verdict", label, { parsed: verdict.parsed, issues: verdict.issues, cleared: verdict.cleared });
  return verdict;
}

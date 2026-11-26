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

/** Calibrated TASK-031 c1 prompt: abstract 7-mode hallucination checklist + anti-over-rejection rubric (zero test-set phrase leakage). */
export const CRITIC_PROMPT_C1 = `你是复盘助手的事实审查员。下面给出“引擎的范围”、玩家的问题、回答草稿，以及助手能依据的全部证据（按编号列出）。
证据说明：E0 是程序给出的整局概况（包含每个决定的玩家选择、引擎首选、分差、判定与汇总计数）。tool 为 investigate 的是调查员报告：headline 和 comparison 是引擎原始数据，finding 里的 reasons、what_if、advice 是程序核对过的结论；what_if 是改动当时牌组后重算的结果。其余是引擎工具原始结果。

请逐句对照证据，重点排查以下 7 类真正的问题（命中任意一项即判 unsupported）：
1. 选择/首选/方向说反：把玩家选择与引擎首选说反、分数归属说反、或把 what_if 结论说反（例如证据说改牌后首选没变，草稿说会改选，或反之）。
2. 跨层/跨选项错配：把某一层或某个选项的真实数字、卡牌、理由安到另一层或另一个选项上，或把多组假设的合计区间安到单次改动上。
3. 信心/判定矛盾：证据写明“有把握、差距明显”却说“差距不大/把握不大”；或证据为低/中把握、部分效果未计入，却说成把握极高、全部算清、绝对定论或明显失误。
4. 超范围与战斗因果编造：引擎不模拟战斗过程、出牌顺序、击杀目标、单回合爆发、掉血/阵亡因果、地图路线、事件或未评分决定（not_scored）/未建模商品。即使草稿先声明“引擎看不到”，只要随后仍对具体战斗因果、战损结果或未评分商品性价比下断言，也判 unsupported。
5. 无依据的额外机制或排他断言：声称证据里没出现的维度、卡牌联动或效果（例如对比里未提及某维度却声称其提升该维度，或未列出全部候选数值却断言某选项是“唯一”满足某条件的）。
6. 汉字数字与手工计数错误：把汉字或阿拉伯数字写的次数、数量、一致数或合计说错，与 E0 统计或证据列表实际计数不符。
7. 把证据里没有的事实或数值说成引擎结果。

以下情况属于合理表达，**必须判 supported（不要误杀）**：
- 对证据里完整表格或多项数据的准确概括：例如多楼层趋势均低于需求时概括为该幕持续不足，或对比中某维度差值最大时概括为主要差在该维度。
- 忠实复述或转述 E0 / get_decision / investigate 里的 verdict、confidence、部分计分提示或 what_if 结论（只要方向、范围与程度不矛盾；注：E0 中 tier 1 决定的 confidence 字段常为 medium，它与 verdict 原文“引擎有把握且差距明显”是一致的，不算矛盾）。
- 中文并列短语计数：例如“N 个 A、B 和 C 类决定”指并列各类决定合计 N 个，只要与 E0 总数一致就算有依据，不要误读成单类数量。
- 与“引擎的范围”一致的能力说明、“引擎回答不了/看不出来”的诚实拒答、以及基于证据中已确认缺口给出的温和方向性建议（未强加定论或虚构数值）。
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
  /** false when the reply was not the asked-for JSON even after retry. */
  parsed: boolean;
  /** Explicit error marker when parsed === false (fail-closed). */
  parse_error?: string;
  /** Number of automatic retries performed due to unparsable JSON. */
  retries: number;
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

/** Extract the first balanced JSON object substring `{...}` respecting string literals. */
function findBalancedJsonObject(raw: string): string | null {
  let start = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]!;
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      if (depth > 0) {
        depth--;
        if (depth === 0 && start >= 0) return raw.slice(start, i + 1);
      }
    }
  }
  return null;
}

/** Extract JSON from any fenced code block or surrounding commentary. */
export function stripJsonFence(raw: string): string {
  const trimmed = raw.trim();
  const fenceMatches = [...trimmed.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)];
  for (const m of fenceMatches) {
    const candidate = (m[1] ?? "").trim();
    if (candidate) return candidate;
  }
  const balanced = findBalancedJsonObject(trimmed);
  if (balanced) return balanced;
  const first = trimmed.indexOf("{");
  const last = trimmed.lastIndexOf("}");
  if (first >= 0 && last > first) return trimmed.slice(first, last + 1).trim();
  return trimmed;
}

function tryParseDoubts(raw: string): Array<{ sentence: string; reason: string; verdict: string }> | null {
  const candidates: string[] = [];
  const trimmed = raw.trim();
  if (trimmed) candidates.push(trimmed);
  for (const m of trimmed.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)) {
    const block = (m[1] ?? "").trim();
    if (block) candidates.push(block);
  }
  const stripped = stripJsonFence(trimmed);
  if (stripped && !candidates.includes(stripped)) candidates.push(stripped);
  const first = trimmed.indexOf("{");
  const last = trimmed.lastIndexOf("}");
  if (first >= 0 && last > first) {
    const span = trimmed.slice(first, last + 1).trim();
    if (span && !candidates.includes(span)) candidates.push(span);
  }
  for (const text of candidates) {
    try {
      const data = JSON.parse(text) as { doubts?: unknown };
      if (typeof data === "object" && data !== null && Array.isArray(data.doubts)) {
        return data.doubts
          .filter((x): x is { sentence: unknown; reason?: unknown; verdict?: unknown } => typeof x === "object" && x !== null && "sentence" in x)
          .map((x) => ({
            sentence: String(x.sentence).trim(),
            reason: String(x.reason ?? ""),
            verdict: String(x.verdict ?? "unsupported"),
          }));
      }
    } catch {
      // try next extraction candidate
    }
  }
  return null;
}

const CRITIC_PARSE_RETRY_LIMIT = 1;
const PARSE_RETRY_PROMPT =
  '你的上一条回复无法解析为合法 JSON（缺少 {"doubts": [...]}）。请只输出严格合法的 JSON 对象：{"doubts":[{"sentence":"...","reason":"...","verdict":"supported 或 unsupported"}]}，不要附带任何额外文字。';

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
  const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
    { role: "system", content: systemPrompt },
    { role: "user", content: `${scope ? `引擎的范围：\n${scope}\n\n` : ""}玩家的问题：${question}\n\n回答草稿：\n${draft}\n\n证据：\n${evidenceText(evidence)}` },
  ];
  const verdict: CriticVerdict = {
    issues: [],
    parsed: false,
    retries: 0,
    cleared: 0,
    cost_usd: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
  };

  for (let attempt = 0; attempt <= CRITIC_PARSE_RETRY_LIMIT; attempt++) {
    const request: Record<string, unknown> = { model, messages: [...messages] };
    if (thinking) {
      request.thinking = { type: "enabled" };
    } else {
      request.temperature = 0;
      request.thinking = { type: "disabled" };
      request.response_format = { type: "json_object" };
    }
    const spanLabel = attempt === 0 ? label : `${label}#retry${attempt}`;
    const response = await trace.span(
      "llm.critic", spanLabel,
      () => llm.chat.completions.create(request as unknown as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming),
      (r) => ({ model, attempt, usage: r.usage }),
    );
    const usage = response.usage;
    if (usage) {
      verdict.cost_usd += costUsd(model, usage as never);
      verdict.prompt_tokens += usage.prompt_tokens;
      verdict.completion_tokens += usage.completion_tokens;
    }
    const rawContent = response.choices[0]?.message.content ?? "";
    // The critic writes its reason before its verdict; only "unsupported" counts (it often lists a doubt and then finds the support).
    const doubts = tryParseDoubts(rawContent);
    if (doubts !== null) {
      verdict.parsed = true;
      verdict.retries = attempt;
      verdict.cleared = doubts.filter((x) => x.verdict === "supported").length;
      verdict.issues = doubts
        .filter((x) => x.verdict !== "supported" && x.sentence.length > 0)
        .map(({ sentence, reason }) => ({ sentence, reason }));
      break;
    }
    verdict.retries = attempt;
    if (attempt < CRITIC_PARSE_RETRY_LIMIT) {
      if (rawContent.trim()) messages.push({ role: "assistant", content: rawContent });
      messages.push({ role: "user", content: PARSE_RETRY_PROMPT });
    }
  }

  if (!verdict.parsed) {
    // Fail-closed: never silently approve an unchecked draft when critic output stays unparsable.
    verdict.parse_error = "critic_json_parse_failed";
    verdict.issues = [
      {
        sentence: "[parse_error]",
        reason: "审查员返回内容无法解析为合法 JSON（自动重试后仍失败），保守拦截以避免放行未核查内容",
      },
    ];
  }
  trace.event("critic.verdict", label, {
    parsed: verdict.parsed,
    parse_error: verdict.parse_error,
    retries: verdict.retries,
    issues: verdict.issues,
    cleared: verdict.cleared,
  });
  return verdict;
}

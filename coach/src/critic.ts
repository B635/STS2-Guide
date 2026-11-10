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

export const CRITIC_PROMPT = `你是审查员。下面有玩家的问题、复盘助手的回答草稿，以及助手能依据的全部证据（一个确定性评分引擎的工具结果，按编号列出）。
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

/** Evidence as the critic sees it: one line per result, by its ID. */
export function evidenceText(evidence: Evidence[]): string {
  return evidence.map((e) => `${e.id}（${e.tool}）= ${JSON.stringify(e.result)}`).join("\n");
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
): Promise<CriticVerdict> {
  const response = await trace.span(
    "llm.critic", label,
    () => llm.chat.completions.create({
      model, temperature: 0, thinking: { type: "disabled" }, response_format: { type: "json_object" },
      messages: [
        { role: "system", content: CRITIC_PROMPT },
        { role: "user", content: `${scope ? `引擎的范围：\n${scope}\n\n` : ""}玩家的问题：${question}\n\n回答草稿：\n${draft}\n\n证据：\n${evidenceText(evidence)}` },
      ],
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming),
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
    const data = JSON.parse(response.choices[0]?.message.content ?? "") as { doubts?: unknown };
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

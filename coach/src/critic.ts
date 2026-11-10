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
判断方法：对草稿的每一句，先在证据里找它的关键词和数字；找得到相同或等价的意思就算有依据（概括、换个说法都可以）。只有找不到或和证据矛盾时才列出。
要列出的说法：
- 证据里没有的事实或数字；
- 和证据矛盾的说法（例如把引擎的首选、玩家的选择或分差说反）；
- 关于战斗过程、输赢原因、地图路线、事件的断言（引擎不模拟这些，证据里没有就不能这样说）；
- 把引擎的看法说成定论，或把证据里没有的建议说成引擎的意见。
不算问题的：根据证据给出的方向性建议、对问题的复述、“引擎回答不了”或“没能确认”这类说明、语气词。
只输出 json：{"unsupported":[{"sentence":"草稿里的原句，逐字复制","reason":"简短的原因"}]}。都有依据就输出 {"unsupported":[]}。`;

export interface CriticIssue {
  sentence: string;
  reason: string;
}

export interface CriticVerdict {
  issues: CriticIssue[];
  /** false when the reply was not the asked-for JSON (counted, and treated as no issues). */
  parsed: boolean;
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
): Promise<CriticVerdict> {
  const response = await trace.span(
    "llm.critic", label,
    () => llm.chat.completions.create({
      model, temperature: 0, thinking: { type: "disabled" }, response_format: { type: "json_object" },
      messages: [
        { role: "system", content: CRITIC_PROMPT },
        { role: "user", content: `玩家的问题：${question}\n\n回答草稿：\n${draft}\n\n证据：\n${evidenceText(evidence)}` },
      ],
    } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming),
    (r) => ({ model, usage: r.usage }),
  );
  const usage = response.usage;
  const verdict: CriticVerdict = {
    issues: [], parsed: false,
    cost_usd: usage ? costUsd(model, usage as never) : 0,
    prompt_tokens: usage?.prompt_tokens ?? 0,
    completion_tokens: usage?.completion_tokens ?? 0,
  };
  try {
    const data = JSON.parse(response.choices[0]?.message.content ?? "") as { unsupported?: unknown };
    if (Array.isArray(data.unsupported)) {
      verdict.parsed = true;
      verdict.issues = data.unsupported
        .filter((x): x is { sentence: unknown; reason?: unknown } => typeof x === "object" && x !== null && "sentence" in x)
        .map((x) => ({ sentence: String(x.sentence).trim(), reason: String(x.reason ?? "") }))
        .filter((x) => x.sentence.length > 0);
    }
  } catch {
    // Unparsable verdict: recorded below; the program check still stands.
  }
  trace.event("critic.verdict", label, { parsed: verdict.parsed, issues: verdict.issues });
  return verdict;
}

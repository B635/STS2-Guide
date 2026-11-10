// The report's one-sentence summary, written from verified findings only.
// It is checked like a finding (numbers and names must come from those
// findings); if it fails twice, a sentence built by the program is used.
// With player notes from earlier runs, the counts of what led earlier flags
// are added to the facts, so the sentence may say a weakness keeps coming back.

import type OpenAI from "openai";
import type { FlagResult } from "./agent.js";
import { factorLabel } from "./labels.js";
import { costUsd } from "./llm.js";
import type { Trace } from "./trace.js";
import { checkText } from "./verifier.js";
import type { PlayerNote } from "./notes.js";
import { recurringFactors } from "./notes.js";

export interface Summary {
  text: string;
  source: "model" | "program";
  cost_usd: number;
}

const PROMPT = `下面是一局《杀戮尖塔 2》里，引擎有把握与玩家不同的几个决定（已核对过的结论）。
用简体中文写一句话（不超过 60 字）：指出这些决定的共同点（例如反复出现的同一种不足），并给一个可以照做的方向。不要罗列卡牌名称。
只能用下面出现过的数字和卡牌名称；不要说“失误”“错误”。只输出这一句话。`;

const EARLIER = `
earlier_runs 是这位玩家以往对局的笔记统计（哪种不足在几局里带头出现过）。如果这局的共同点和以往反复出现的不足一样，可以点出“和以往几局一样”。`;

export function programSummary(verified: FlagResult[], tier1Count: number): string {
  if (tier1Count === 0) return "这局没有引擎有把握的分歧：你的选择和引擎的看法没有明显差距。";
  const counts = new Map<string, number>();
  for (const r of verified) for (const reason of r.finding?.reasons.slice(0, 1) ?? []) {
    counts.set(reason.factor, (counts.get(reason.factor) ?? 0) + 1);
  }
  const [top] = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  if (!top) return `引擎点出 ${tier1Count} 个值得想一想的决定。`;
  return `引擎点出 ${tier1Count} 个值得想一想的决定，其中 ${top[1]} 个主要和${factorLabel(top[0])}有关。`;
}

export async function summarize(
  llm: OpenAI,
  model: string,
  verified: FlagResult[],
  tier1Count: number,
  names: ReadonlySet<string>,
  trace: Trace,
  notes: PlayerNote[] = [],
  runId?: string,
): Promise<Summary> {
  const fallback = { text: programSummary(verified, tier1Count), source: "program" as const, cost_usd: 0 };
  if (verified.length < 2) return fallback; // nothing in common to point out
  const findings = verified.map((r) => ({ decision: r.flag, reasons: r.finding?.reasons.map((x) => ({ factor: factorLabel(x.factor), text: x.text })), advice: r.finding?.advice }));
  // Without notes the request is the same as before notes existed (the evaluation runs that way).
  const earlier = recurringFactors(notes, runId).filter((f) => f.runs >= 2);
  const facts = earlier.length ? { findings, earlier_runs: earlier } : findings;
  const prompt = earlier.length ? PROMPT + EARLIER : PROMPT;
  let cost = 0;
  for (let attempt = 0; attempt < 2; attempt++) {
    const response = await trace.span(
      "llm.summary", undefined,
      () => llm.chat.completions.create({
        model, temperature: 0, thinking: { type: "disabled" },
        messages: [{ role: "system", content: prompt }, { role: "user", content: JSON.stringify(facts) }],
      } as OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming),
      (r) => ({ model, attempt, usage: r.usage }),
    );
    if (response.usage) cost += costUsd(model, response.usage as never);
    const text = (response.choices[0]?.message.content ?? "").trim();
    const errors = text.length > 80 ? ["too long"] : checkText("summary", text, [facts, tier1Count], names);
    trace.event("verify.summary", undefined, { ok: errors.length === 0, errors, text });
    if (text && errors.length === 0) return { text, source: "model", cost_usd: cost };
  }
  return { ...fallback, cost_usd: cost };
}

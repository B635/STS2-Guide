// What the agent hands in for one flagged decision. The headline
// ("引擎认为 X 比你选的 Y 高 N 分") is written by the program from the engine's
// record, never by the model; the model writes the reasons, the optional
// what-if insight and the advice, each tied to the tool result it rests on.
// What the engine could not score is stated by the program from the engine's
// own gaps, not by the model (it guessed at engine internals when asked).

import { z } from "zod";

export const Reason = z.object({
  factor: z.string().describe("engine factor code the reason rests on, e.g. deck_fitness:defense"),
  evidence: z.string().describe("ID of the tool result that shows it, e.g. T0 or T2"),
  text: z.string().min(4).max(120).describe("one short sentence in simplified Chinese"),
});

export const Finding = z.object({
  decision_id: z.string(),
  reasons: z.array(Reason).min(1).max(3).describe("why the engine prefers its pick, largest factor first"),
  what_if: z
    .object({
      evidence: z.string().describe("ID of a what_if tool result"),
      text: z.string().min(4).max(120),
    })
    .optional()
    .describe("only if a what_if call showed something useful, e.g. which earlier pick would have changed this"),
  advice: z.string().min(4).max(120).describe("one sentence: what to look for next time, in simplified Chinese"),
});
export type Finding = z.infer<typeof Finding>;

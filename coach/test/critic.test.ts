import { describe, expect, it } from "vitest";
import { computeCriticMetrics, loadCriticBench } from "../src/criticeval.js";
import type { CriticCaseResult } from "../src/criticeval.js";
import { CRITIC_PROMPT, CRITIC_PROMPT_C1, critique } from "../src/critic.js";
import { Trace } from "../src/trace.js";

describe("critic judge benchmark & calibration", () => {
  it("uses calibrated CRITIC_PROMPT_C1 as default critic prompt", () => {
    expect(CRITIC_PROMPT).toBe(CRITIC_PROMPT_C1);
  });
  it("critic-bench.json has balanced, unique dev (20) and test (40) cases", () => {
    const bench = loadCriticBench();
    expect(bench.dev).toHaveLength(20);
    expect(bench.test).toHaveLength(40);
    const ids = new Set([...bench.dev, ...bench.test].map((c) => c.id));
    expect(ids.size).toBe(60);
    expect(bench.dev.filter((c) => c.expected === "supported")).toHaveLength(10);
    expect(bench.dev.filter((c) => c.expected === "unsupported")).toHaveLength(10);
    expect(bench.test.filter((c) => c.expected === "supported")).toHaveLength(20);
    expect(bench.test.filter((c) => c.expected === "unsupported")).toHaveLength(20);
  });

  it("computes recall, FPR, F1, repeat stability, and spread accurately", () => {
    const rows: CriticCaseResult[] = [
      {
        id: "C1", source_q: "Q01", expected: "unsupported", category: "swapped_pick",
        trials: [
          { flagged: true, parsed: true, issues: [{ sentence: "s", reason: "r" }], cleared: 0, cost_usd: 0.001, ms: 100 },
          { flagged: true, parsed: true, issues: [{ sentence: "s", reason: "r" }], cleared: 0, cost_usd: 0.001, ms: 100 },
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
        ],
      },
      {
        id: "C2", source_q: "Q03", expected: "supported", category: "real_supported",
        trials: [
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
        ],
      },
    ];
    const m = computeCriticMetrics("dev", "c1", "deepseek-flash", false, rows);
    expect(m.tp).toBe(2);
    expect(m.fn).toBe(1);
    expect(m.fp).toBe(0);
    expect(m.tn).toBe(3);
    expect(m.recall).toBeCloseTo(2 / 3, 3);
    expect(m.fpr).toBe(0);
    expect(m.all_agree_rate).toBe(0.5);
    expect(m.flag_rate_spread).toBe(0.5);
    expect(m.majority_recall).toBe(1);
    expect(m.majority_fpr).toBe(0);
  });

  it("parses fenced JSON returned in thinking mode", async () => {
    let seenThinking: unknown;
    const fakeLlm = {
      chat: {
        completions: {
          create: async (req: Record<string, unknown>) => {
            seenThinking = req.thinking;
            return {
              choices: [
                {
                  message: {
                    content: '```json\n{"doubts":[{"sentence":"bad","reason":"not in evidence","verdict":"unsupported"}]}\n```',
                  },
                },
              ],
              usage: { prompt_tokens: 100, completion_tokens: 40 },
            };
          },
        },
      },
    };
    const v = await critique(
      fakeLlm as never, "deepseek-flash", "q", "bad", [{ id: "E0", tool: "overview", result: {} }],
      new Trace(), "t1", "", { prompt: CRITIC_PROMPT_C1, thinking: true },
    );
    expect(seenThinking).toEqual({ type: "enabled" });
    expect(v.parsed).toBe(true);
    expect(v.issues).toHaveLength(1);
  });
});

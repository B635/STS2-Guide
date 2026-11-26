import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { checkAnswer, quotes, sentences } from "../src/ask.js";
import { CRITIC_PROMPT, CRITIC_PROMPT_C1, critique, stripJsonFence } from "../src/critic.js";
import { computeCriticMetrics, evidenceForCase, loadCriticBench, scoreTrialSentences } from "../src/criticeval.js";
import type { CriticCaseResult } from "../src/criticeval.js";
import { EVAL_DIR } from "../src/evaluate.js";
import { entityNames } from "../src/names.js";
import { Trace } from "../src/trace.js";

const LEAKED_PHRASES = [
  "抽不到牌导致暴毙",
  "能一回合秒怪",
  "买这件未评分遗物非常划算",
  "八次",
  "全部六张",
  "只有三次一致",
  "后续可留意格挡牌",
];

describe("critic judge benchmark & calibration", () => {
  it("uses leak-free CRITIC_PROMPT_C1 as default critic prompt", () => {
    expect(CRITIC_PROMPT).toBe(CRITIC_PROMPT_C1);
    for (const phrase of LEAKED_PHRASES) {
      expect(CRITIC_PROMPT_C1).not.toContain(phrase);
    }
  });

  it("critic-bench.json v2 has zero-leakage test set across all 15 split.json test runs and exact sentence ground truth", () => {
    const bench = loadCriticBench();
    const split = JSON.parse(readFileSync(path.join(EVAL_DIR, "split.json"), "utf-8")) as {
      dev: string[];
      test: string[];
    };
    const devRuns = new Set(split.dev);
    const testRuns = new Set(split.test);

    expect(bench.version).toBeGreaterThanOrEqual(2);
    expect(bench.dev).toHaveLength(20);
    expect(bench.test).toHaveLength(40);
    const ids = new Set([...bench.dev, ...bench.test].map((c) => c.id));
    expect(ids.size).toBe(60);
    expect(bench.dev.filter((c) => c.expected === "supported")).toHaveLength(10);
    expect(bench.dev.filter((c) => c.expected === "unsupported")).toHaveLength(10);
    expect(bench.test.filter((c) => c.expected === "supported")).toHaveLength(20);
    expect(bench.test.filter((c) => c.expected === "unsupported")).toHaveLength(20);

    const seenTestRuns = new Set<string>();
    const names = entityNames();
    for (const c of bench.test) {
      expect(c.run_id).toBeTruthy();
      expect(testRuns.has(c.run_id!)).toBe(true);
      expect(devRuns.has(c.run_id!)).toBe(false);
      seenTestRuns.add(c.run_id!);

      for (const phrase of LEAKED_PHRASES) {
        expect(c.draft).not.toContain(phrase);
      }

      const { question, evidence } = evidenceForCase(c, bench, []);
      expect(question.length).toBeGreaterThan(0);
      expect(evidence.length).toBeGreaterThanOrEqual(2);
      expect(evidence[0]?.id).toBe("E0");

      if (c.expected === "supported") {
        expect(c.unsupported_sentences).toEqual([]);
        // Every supported test draft's numbers and entity names must ground in its own evidence pool.
        const programErrors = checkAnswer(c.draft, evidence, names);
        expect(programErrors, `${c.id} program check failed: ${JSON.stringify(programErrors)}`).toEqual([]);
      } else {
        expect(c.unsupported_sentences.length).toBeGreaterThanOrEqual(1);
        const sents = sentences(c.draft);
        for (const target of c.unsupported_sentences) {
          expect(
            sents.some((s) => quotes({ sentence: target, reason: "" }, s)),
            `${c.id} target not matched in sentences: ${target}`,
          ).toBe(true);
        }
      }
    }
    // All 15 held-out test runs must be represented in test!
    expect(seenTestRuns.size).toBe(15);

    // Every dev case must also have exact unsupported_sentences ground truth.
    for (const c of bench.dev) {
      if (c.expected === "supported") {
        expect(c.unsupported_sentences).toEqual([]);
      } else {
        expect(c.unsupported_sentences.length).toBeGreaterThanOrEqual(1);
        const sents = sentences(c.draft);
        for (const target of c.unsupported_sentences) {
          expect(
            sents.some((s) => quotes({ sentence: target, reason: "" }, s)),
            `${c.id} target not matched in sentences: ${target}`,
          ).toBe(true);
        }
      }
    }
  });

  it("computes paragraph metrics AND sentence-level localization metrics accurately", () => {
    const rows: CriticCaseResult[] = [
      {
        id: "C1",
        source_q: "Q01",
        expected: "unsupported",
        category: "swapped_pick",
        draft: "这是真话第一句。这是造假第二句。",
        unsupported_sentences: ["这是造假第二句。"],
        trials: [
          // Trial 1: exact hit on fabricated sentence
          { flagged: true, parsed: true, issues: [{ sentence: "这是造假第二句。", reason: "r" }], cleared: 0, cost_usd: 0.001, ms: 100 },
          // Trial 2: paragraph TP, BUT flagged the TRUE sentence instead of the fabricated one!
          { flagged: true, parsed: true, issues: [{ sentence: "这是真话第一句。", reason: "wrong" }], cleared: 0, cost_usd: 0.001, ms: 100 },
          // Trial 3: missed completely (FN)
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
        ],
      },
      {
        id: "C2",
        source_q: "Q03",
        expected: "supported",
        category: "real_supported",
        draft: "全部有依据的句子。",
        unsupported_sentences: [],
        trials: [
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
          { flagged: false, parsed: true, issues: [], cleared: 0, cost_usd: 0.001, ms: 100 },
        ],
      },
    ];
    const m = computeCriticMetrics("dev", "c1", "deepseek-flash", false, rows);
    // Paragraph level
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
    // Sentence level: 1 TP (trial 1), 1 FP (trial 2 true sentence), 2 FN (trial 2 & 3 missed fabricated sentence)
    expect(m.sentence_tp).toBe(1);
    expect(m.sentence_fp).toBe(1);
    expect(m.sentence_fn).toBe(2);
    expect(m.sentence_precision).toBeCloseTo(0.5, 4);
    expect(m.sentence_recall).toBeCloseTo(1 / 3, 4);
    expect(m.sentence_f1).toBeCloseTo(0.4, 4);
    expect(m.exact_localization_rate).toBeCloseTo(1 / 3, 4);
    expect(m.wrong_sentence_on_tp).toBe(1);

    // Direct helper check for unmatched junk quote
    const junk = scoreTrialSentences("真句。假句。", "unsupported", ["假句。"], [{ sentence: "根本不在草稿里", reason: "x" }]);
    expect(junk).toMatchObject({ tp: 0, fp: 1, fn: 1, exact_unsupported: false, wrong_on_tp: true });
  });

  it("extracts JSON from fenced blocks anywhere or inline commentary via stripJsonFence", () => {
    expect(stripJsonFence('分析完毕如下：\n```json\n{"doubts":[]}\n```\n请查收。')).toBe('{"doubts":[]}');
    expect(stripJsonFence('结论：{"doubts":[{"sentence":"错句","reason":"无据","verdict":"unsupported"}]} 以上')).toBe(
      '{"doubts":[{"sentence":"错句","reason":"无据","verdict":"unsupported"}]}',
    );
  });

  it("parses fenced JSON returned with surrounding commentary in thinking mode", async () => {
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
                    content: '核对如下：\n```json\n{"doubts":[{"sentence":"bad","reason":"not in evidence","verdict":"unsupported"}]}\n```\n完成。',
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
    expect(v.retries).toBe(0);
    expect(v.issues).toHaveLength(1);
  });

  it("retries once on invalid JSON and succeeds if the retry is valid", async () => {
    let calls = 0;
    const fakeLlm = {
      chat: {
        completions: {
          create: async () => {
            calls++;
            return {
              choices: [
                {
                  message: {
                    content: calls === 1 ? "这段回答看起来没有问题。" : '{"doubts":[]}',
                  },
                },
              ],
              usage: { prompt_tokens: 100, completion_tokens: 20 },
            };
          },
        },
      },
    };
    const v = await critique(
      fakeLlm as never, "deepseek-flash", "q", "ok", [{ id: "E0", tool: "overview", result: {} }],
      new Trace(), "t-retry-ok", "",
    );
    expect(calls).toBe(2);
    expect(v.parsed).toBe(true);
    expect(v.parse_error).toBeUndefined();
    expect(v.retries).toBe(1);
    expect(v.issues).toEqual([]);
    expect(v.prompt_tokens).toBe(200);
    expect(v.completion_tokens).toBe(40);
  });

  it("fails closed with explicit parse_error and conservative issue when retry also fails", async () => {
    let calls = 0;
    const fakeLlm = {
      chat: {
        completions: {
          create: async () => {
            calls++;
            return {
              choices: [{ message: { content: "仍然不是JSON" } }],
              usage: { prompt_tokens: 80, completion_tokens: 15 },
            };
          },
        },
      },
    };
    const v = await critique(
      fakeLlm as never, "deepseek-flash", "q", "maybe hallucinated", [{ id: "E0", tool: "overview", result: {} }],
      new Trace(), "t-retry-fail", "",
    );
    expect(calls).toBe(2);
    expect(v.parsed).toBe(false);
    expect(v.parse_error).toBe("critic_json_parse_failed");
    expect(v.retries).toBe(1);
    expect(v.issues.length).toBeGreaterThan(0);
  });
});

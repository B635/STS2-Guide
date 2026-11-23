// Offline & live benchmark runner for calibrating the Critic LLM-as-a-Judge (TASK-031 A2-1).
// Measures Recall, False Positive Rate (over-rejection), Precision, F1,
// and multi-repeat stability (agreement rate & rejection-rate spread) over
// real ask-dev evidence pools without requiring the Python MCP server.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import type OpenAI from "openai";
import { SCOPE } from "./ask.js";
import { loadAskRecorded } from "./askeval.js";
import type { AskRecorded } from "./askeval.js";
import { CRITIC_PROMPT_B5, CRITIC_PROMPT_C1, critique } from "./critic.js";
import type { CriticIssue } from "./critic.js";
import { EVAL_DIR } from "./evaluate.js";
import { DEFAULT_MODEL, createClient } from "./llm.js";
import { Trace } from "./trace.js";
import type { Evidence } from "./verifier.js";

export const CRITIC_BENCH_FILE = path.join(EVAL_DIR, "critic-bench.json");

export type CriticVariant = "b5" | "c1" | "c2";

export interface CriticCase {
  id: string;
  source_q: string;
  expected: "supported" | "unsupported";
  category: string;
  draft: string;
  note: string;
}

export interface CriticBench {
  version: number;
  description: string;
  dev: CriticCase[];
  test: CriticCase[];
}

export interface CriticTrial {
  flagged: boolean;
  parsed: boolean;
  issues: CriticIssue[];
  cleared: number;
  cost_usd: number;
  ms: number;
}

export interface CriticCaseResult {
  id: string;
  source_q: string;
  expected: "supported" | "unsupported";
  category: string;
  trials: CriticTrial[];
}

export interface CriticBenchMetrics {
  set: "dev" | "test";
  variant: CriticVariant;
  model: string;
  thinking: boolean;
  cases: number;
  repeats: number;
  // Aggregate across all case * repeat trials (primary Judge calibration metrics).
  tp: number;
  fn: number;
  fp: number;
  tn: number;
  recall: number;
  fpr: number;
  precision: number;
  f1: number;
  unparsed: number;
  // Repeat-to-repeat stability metrics.
  all_agree_cases: number;
  all_agree_rate: number;
  per_repeat_flag_rates: number[];
  flag_rate_spread: number;
  // Majority-vote metrics (if repeats >= 3, majority of 3 trials per case).
  majority_recall: number;
  majority_fpr: number;
  majority_f1: number;
  by_category: Record<string, { cases: number; trials: number; flagged: number; rate: number }>;
  cost_usd: number;
  cost_per_call: number;
  avg_ms: number;
  errors: Array<{ id: string; category: string; expected: string; flagged_trials: number; sample_reason: string }>;
}

export function loadCriticBench(file = CRITIC_BENCH_FILE): CriticBench {
  return JSON.parse(readFileSync(file, "utf-8")) as CriticBench;
}

function evidenceForQuestion(qid: string, rows: AskRecorded[]): { question: string; evidence: Evidence[] } {
  const row = rows.find((r) => r.id === qid);
  if (!row) throw new Error(`Missing source question ${qid} in recorded ask-dev`);
  const pool = new Map<string, Evidence>([[row.context.id, row.context]]);
  for (const r of rows) if (r.run === row.run) for (const e of r.answer.evidence) pool.set(e.id, e);
  const evidence = row.answer.window.map((id) => pool.get(id)).filter((e): e is Evidence => e !== undefined);
  return { question: row.question, evidence };
}

export function variantOptions(variant: CriticVariant): { prompt: string; thinking: boolean } {
  if (variant === "b5") return { prompt: CRITIC_PROMPT_B5, thinking: false };
  if (variant === "c1") return { prompt: CRITIC_PROMPT_C1, thinking: false };
  return { prompt: CRITIC_PROMPT_C1, thinking: true };
}

const r4 = (x: number) => Math.round(x * 10000) / 10000;

export function computeCriticMetrics(
  set: "dev" | "test",
  variant: CriticVariant,
  model: string,
  thinking: boolean,
  results: CriticCaseResult[],
): CriticBenchMetrics {
  const repeats = results[0]?.trials.length ?? 0;
  let tp = 0;
  let fn = 0;
  let fp = 0;
  let tn = 0;
  let unparsed = 0;
  let allAgree = 0;
  let totalCost = 0;
  let totalMs = 0;
  let totalCalls = 0;
  let majTp = 0;
  let majFn = 0;
  let majFp = 0;
  let majTn = 0;

  const perRepeatFlagged = Array.from({ length: repeats }, () => 0);
  const byCategory: Record<string, { cases: number; trials: number; flagged: number; rate: number }> = {};
  const errors: CriticBenchMetrics["errors"] = [];

  for (const row of results) {
    const cat = (byCategory[row.category] ??= { cases: 0, trials: 0, flagged: 0, rate: 0 });
    cat.cases++;
    let flaggedCount = 0;
    for (let i = 0; i < row.trials.length; i++) {
      const t = row.trials[i]!;
      totalCalls++;
      totalCost += t.cost_usd;
      totalMs += t.ms;
      if (!t.parsed) unparsed++;
      cat.trials++;
      if (t.flagged) {
        flaggedCount++;
        cat.flagged++;
        perRepeatFlagged[i]!++;
      }
      if (row.expected === "unsupported") {
        if (t.flagged) tp++;
        else fn++;
      } else {
        if (t.flagged) fp++;
        else tn++;
      }
    }
    if (flaggedCount === 0 || flaggedCount === row.trials.length) allAgree++;
    const majFlagged = flaggedCount > row.trials.length / 2;
    if (row.expected === "unsupported") {
      if (majFlagged) majTp++;
      else majFn++;
    } else {
      if (majFlagged) majFp++;
      else majTn++;
    }
    const wantFlag = row.expected === "unsupported";
    if ((wantFlag && flaggedCount < row.trials.length) || (!wantFlag && flaggedCount > 0)) {
      const firstIssue = row.trials.flatMap((t) => t.issues)[0];
      errors.push({
        id: row.id,
        category: row.category,
        expected: row.expected,
        flagged_trials: flaggedCount,
        sample_reason: firstIssue ? `${firstIssue.sentence.slice(0, 40)} => ${firstIssue.reason.slice(0, 90)}` : "(not flagged)",
      });
    }
  }

  for (const cat of Object.values(byCategory)) {
    cat.rate = cat.trials ? r4(cat.flagged / cat.trials) : 0;
  }

  const recall = tp + fn ? r4(tp / (tp + fn)) : 0;
  const fpr = fp + tn ? r4(fp / (fp + tn)) : 0;
  const precision = tp + fp ? r4(tp / (tp + fp)) : 0;
  const f1 = precision + recall ? r4((2 * precision * recall) / (precision + recall)) : 0;
  const perRepeatRates = perRepeatFlagged.map((c) => (results.length ? r4(c / results.length) : 0));
  const spread = perRepeatRates.length ? r4(Math.max(...perRepeatRates) - Math.min(...perRepeatRates)) : 0;
  const majRecall = majTp + majFn ? r4(majTp / (majTp + majFn)) : 0;
  const majFpr = majFp + majTn ? r4(majFp / (majFp + majTn)) : 0;
  const majPrec = majTp + majFp ? majTp / (majTp + majFp) : 0;
  const majF1 = majPrec + majRecall ? r4((2 * majPrec * majRecall) / (majPrec + majRecall)) : 0;

  return {
    set,
    variant,
    model,
    thinking,
    cases: results.length,
    repeats,
    tp,
    fn,
    fp,
    tn,
    recall,
    fpr,
    precision,
    f1,
    unparsed,
    all_agree_cases: allAgree,
    all_agree_rate: results.length ? r4(allAgree / results.length) : 0,
    per_repeat_flag_rates: perRepeatRates,
    flag_rate_spread: spread,
    majority_recall: majRecall,
    majority_fpr: majFpr,
    majority_f1: majF1,
    by_category: byCategory,
    cost_usd: r4(totalCost),
    cost_per_call: totalCalls ? Math.round((totalCost / totalCalls) * 100000) / 100000 : 0,
    avg_ms: totalCalls ? Math.round(totalMs / totalCalls) : 0,
    errors,
  };
}

export async function runCriticBench(
  set: "dev" | "test",
  variant: CriticVariant,
  options: { model?: string; repeats?: number; workers?: number; llm?: OpenAI; outFile?: string } = {},
): Promise<{ metrics: CriticBenchMetrics; results: CriticCaseResult[] }> {
  const model = options.model ?? DEFAULT_MODEL;
  const repeats = options.repeats ?? 3;
  const workers = options.workers ?? 4;
  const llm = options.llm ?? createClient();
  const bench = loadCriticBench();
  const cases = bench[set];
  const recorded = loadAskRecorded(DEFAULT_MODEL);
  const { prompt, thinking } = variantOptions(variant);

  const traceDir = path.join(EVAL_DIR, "results", "critic");
  mkdirSync(traceDir, { recursive: true });
  const trace = new Trace(path.join(traceDir, `${set}-${variant}-${model}.trace.jsonl`));

  const results: CriticCaseResult[] = new Array(cases.length);
  let cursor = 0;

  async function worker() {
    while (cursor < cases.length) {
      const idx = cursor++;
      const c = cases[idx]!;
      const { question, evidence } = evidenceForQuestion(c.source_q, recorded);
      const trials: CriticTrial[] = [];
      for (let rep = 0; rep < repeats; rep++) {
        const t0 = Date.now();
        const v = await critique(llm, model, question, c.draft, evidence, trace, `${c.id}#${rep + 1}`, SCOPE, { prompt, thinking });
        trials.push({
          flagged: v.issues.length > 0,
          parsed: v.parsed,
          issues: v.issues,
          cleared: v.cleared,
          cost_usd: v.cost_usd,
          ms: Date.now() - t0,
        });
      }
      results[idx] = { id: c.id, source_q: c.source_q, expected: c.expected, category: c.category, trials };
      const flagged = trials.filter((t) => t.flagged).length;
      process.stderr.write(`${c.id} (${c.expected.padEnd(11)} ${c.category.padEnd(22)}) flagged ${flagged}/${repeats}\n`);
    }
  }

  await Promise.all(Array.from({ length: Math.min(workers, cases.length) }, () => worker()));
  const metrics = computeCriticMetrics(set, variant, model, thinking, results);

  const rawFile = path.join(traceDir, `${set}-${variant}-${model}.json`);
  writeFileSync(rawFile, JSON.stringify({ metrics, results }, null, 1) + "\n", "utf-8");

  const metricsFile = options.outFile ?? path.join(EVAL_DIR, "metrics", `critic-${set}-${variant}-${model}.json`);
  mkdirSync(path.dirname(metricsFile), { recursive: true });
  writeFileSync(metricsFile, JSON.stringify(metrics, null, 1) + "\n", "utf-8");
  return { metrics, results };
}

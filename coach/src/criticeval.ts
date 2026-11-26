// Offline & live benchmark runner for calibrating the Critic LLM-as-a-Judge (TASK-031 A2-1).
// Measures Recall, False Positive Rate (over-rejection), Precision, F1,
// and multi-repeat stability (agreement rate & rejection-rate spread) over
// real ask-dev evidence pools without requiring the Python MCP server.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import type OpenAI from "openai";
import { SCOPE, quotes, sentences } from "./ask.js";
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
  run_id?: string;
  expected: "supported" | "unsupported";
  category: string;
  draft: string;
  /** Exact ground-truth unsupported sentences in `draft` (empty for supported cases). */
  unsupported_sentences: string[];
  note: string;
}

export interface CriticEvidencePool {
  run_id: string;
  question: string;
  evidence: Evidence[];
}

export interface CriticBench {
  version: number;
  description: string;
  /** Per-run E0 overviews for self-contained test evidence pools (keyed by run_id). */
  runs?: Record<string, Evidence>;
  /** Per-question evidence pools for test cases (keyed by source_q). */
  test_questions?: Record<string, CriticEvidencePool>;
  dev: CriticCase[];
  test: CriticCase[];
}

export interface CriticTrial {
  flagged: boolean;
  parsed: boolean;
  parse_error?: string;
  issues: CriticIssue[];
  cleared: number;
  cost_usd: number;
  ms: number;
}

export interface CriticCaseResult {
  id: string;
  source_q: string;
  run_id?: string;
  expected: "supported" | "unsupported";
  category: string;
  draft?: string;
  unsupported_sentences?: string[];
  trials: CriticTrial[];
}

export interface CriticBenchMetrics {
  set: "dev" | "test";
  variant: CriticVariant;
  model: string;
  thinking: boolean;
  cases: number;
  repeats: number;
  // Paragraph-level metrics across all case * repeat trials.
  tp: number;
  fn: number;
  fp: number;
  tn: number;
  recall: number;
  fpr: number;
  precision: number;
  f1: number;
  unparsed: number;
  // Sentence-level localization metrics across all case * repeat trials.
  sentence_tp: number;
  sentence_fp: number;
  sentence_fn: number;
  sentence_precision: number;
  sentence_recall: number;
  sentence_f1: number;
  exact_localization_rate: number;
  wrong_sentence_on_tp: number;
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

export function evidenceForCase(
  c: CriticCase,
  bench: CriticBench,
  recorded: AskRecorded[],
): { question: string; evidence: Evidence[] } {
  const pool = bench.test_questions?.[c.source_q];
  if (pool) {
    const e0 = bench.runs?.[pool.run_id];
    if (!e0) throw new Error(`Missing run overview E0 for test run ${pool.run_id} (${c.source_q})`);
    return { question: pool.question, evidence: [e0, ...pool.evidence] };
  }
  return evidenceForQuestion(c.source_q, recorded);
}

export interface SentenceTrialScore {
  tp: number;
  fp: number;
  fn: number;
  exact_unsupported: boolean;
  wrong_on_tp: boolean;
  counted: boolean;
}

/** Score Critic's flagged issues at sentence granularity against ground-truth unsupported sentences. */
export function scoreTrialSentences(
  draft: string | undefined,
  expected: "supported" | "unsupported",
  unsupportedSentences: string[] | undefined,
  issues: CriticIssue[],
): SentenceTrialScore {
  if (draft !== undefined) {
    const draftSents = sentences(draft);
    const goldIndices = new Set<number>();
    if (expected === "unsupported") {
      for (const target of unsupportedSentences ?? []) {
        const probe: CriticIssue = { sentence: target, reason: "" };
        draftSents.forEach((s, idx) => {
          if (quotes(probe, s)) goldIndices.add(idx);
        });
      }
    }
    const predIndices = new Set<number>();
    let unmatched = 0;
    for (const issue of issues) {
      const hits: number[] = [];
      draftSents.forEach((s, idx) => {
        if (quotes(issue, s)) hits.push(idx);
      });
      if (hits.length === 0) unmatched++;
      else for (const idx of hits) predIndices.add(idx);
    }
    let tp = 0;
    let fp = unmatched;
    for (const idx of predIndices) {
      if (goldIndices.has(idx)) tp++;
      else fp++;
    }
    let fn = 0;
    for (const idx of goldIndices) {
      if (!predIndices.has(idx)) fn++;
    }
    const exactUnsupported = expected === "unsupported" && goldIndices.size > 0 && fn === 0 && fp === 0;
    const wrongOnTp = expected === "unsupported" && issues.length > 0 && tp === 0;
    return { tp, fp, fn, exact_unsupported: exactUnsupported, wrong_on_tp: wrongOnTp, counted: true };
  }

  if (unsupportedSentences !== undefined) {
    const gold = expected === "unsupported" ? unsupportedSentences : [];
    const hitGold = new Set<number>();
    let fp = 0;
    for (const issue of issues) {
      const matched = gold
        .map((g, idx) => (quotes(issue, g) ? idx : -1))
        .filter((idx) => idx >= 0);
      if (matched.length === 0) fp++;
      else for (const idx of matched) hitGold.add(idx);
    }
    const tp = hitGold.size;
    const fn = gold.length - tp;
    const exactUnsupported = expected === "unsupported" && gold.length > 0 && fn === 0 && fp === 0;
    const wrongOnTp = expected === "unsupported" && issues.length > 0 && tp === 0;
    return { tp, fp, fn, exact_unsupported: exactUnsupported, wrong_on_tp: wrongOnTp, counted: true };
  }

  return { tp: 0, fp: 0, fn: 0, exact_unsupported: false, wrong_on_tp: false, counted: false };
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
  let sentenceTp = 0;
  let sentenceFp = 0;
  let sentenceFn = 0;
  let exactUnsupported = 0;
  let unsupportedCountedTrials = 0;
  let wrongSentenceOnTp = 0;
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
    let sentenceMismatch = false;
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
      const sent = scoreTrialSentences(row.draft, row.expected, row.unsupported_sentences, t.issues);
      if (sent.counted) {
        sentenceTp += sent.tp;
        sentenceFp += sent.fp;
        sentenceFn += sent.fn;
        if (row.expected === "unsupported") {
          unsupportedCountedTrials++;
          if (sent.exact_unsupported) exactUnsupported++;
          if (sent.wrong_on_tp) wrongSentenceOnTp++;
          if (sent.fn > 0 || sent.fp > 0) sentenceMismatch = true;
        }
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
    if ((wantFlag && flaggedCount < row.trials.length) || (!wantFlag && flaggedCount > 0) || sentenceMismatch) {
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
  const sentencePrecision = sentenceTp + sentenceFp ? r4(sentenceTp / (sentenceTp + sentenceFp)) : 0;
  const sentenceRecall = sentenceTp + sentenceFn ? r4(sentenceTp / (sentenceTp + sentenceFn)) : 0;
  const sentenceF1 =
    sentencePrecision + sentenceRecall ? r4((2 * sentencePrecision * sentenceRecall) / (sentencePrecision + sentenceRecall)) : 0;
  const exactLocalizationRate = unsupportedCountedTrials ? r4(exactUnsupported / unsupportedCountedTrials) : 0;
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
    sentence_tp: sentenceTp,
    sentence_fp: sentenceFp,
    sentence_fn: sentenceFn,
    sentence_precision: sentencePrecision,
    sentence_recall: sentenceRecall,
    sentence_f1: sentenceF1,
    exact_localization_rate: exactLocalizationRate,
    wrong_sentence_on_tp: wrongSentenceOnTp,
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
  const recorded = set === "dev" ? loadAskRecorded(DEFAULT_MODEL) : [];
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
      const { question, evidence } = evidenceForCase(c, bench, recorded);
      const trials: CriticTrial[] = [];
      for (let rep = 0; rep < repeats; rep++) {
        const t0 = Date.now();
        const v = await critique(llm, model, question, c.draft, evidence, trace, `${c.id}#${rep + 1}`, SCOPE, { prompt, thinking });
        trials.push({
          flagged: v.issues.length > 0,
          parsed: v.parsed,
          ...(v.parse_error ? { parse_error: v.parse_error } : {}),
          issues: v.issues,
          cleared: v.cleared,
          cost_usd: v.cost_usd,
          ms: Date.now() - t0,
        });
      }
      results[idx] = {
        id: c.id,
        source_q: c.source_q,
        ...(c.run_id ? { run_id: c.run_id } : {}),
        expected: c.expected,
        category: c.category,
        draft: c.draft,
        unsupported_sentences: c.unsupported_sentences,
        trials,
      };
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

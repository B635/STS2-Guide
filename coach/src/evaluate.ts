// Evaluation (TASK-028): run the coach over a fixed set of runs, keep every
// output, and compute the metrics from the kept outputs, so they can be
// recomputed later without calling the model.
//
// 1. groundedness: share of handed-in findings that pass the verifier, re-run here
//    on the recorded evidence (target 100% of what reaches the report);
// 2. coverage: share of the shown tier-1 flags that got a verified explanation;
// 3. ratings: the developer's own ratings of the shown flags (useful / not useful / engine wrong);
// 4. cost, time and tool use per review, for comparing models.

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { FlagResult } from "./agent.js";
import { Engine, REPO_ROOT } from "./engine.js";
import { Finding } from "./finding.js";
import { entityNames } from "./names.js";
import type { Prepass } from "./prepass.js";
import { runReview } from "./review.js";
import type { Summary } from "./summary.js";
import { verifyFinding } from "./verifier.js";
import { renderReport } from "./report.js";
import type { ReportMeta } from "./report.js";

export const EVAL_DIR = path.join(REPO_ROOT, "coach", "eval");
export type EvalSet = "dev" | "test";

export interface Recorded {
  run: string;
  set: EvalSet;
  model: string;
  prompt_version: string;
  pre: Prepass;
  explained: FlagResult[];
  summary: Summary;
  meta: ReportMeta;
}

export const PROMPT_VERSION = "p3";

export function resultsDir(set: EvalSet, model: string): string {
  return path.join(EVAL_DIR, "results", set, model);
}

export function runFiles(set: EvalSet): string[] {
  const dir = path.join(EVAL_DIR, "runs", set);
  return readdirSync(dir).filter((f) => f.endsWith(".run")).sort().map((f) => path.join(dir, f));
}

export async function record(set: EvalSet, model: string, thinking = false): Promise<Recorded[]> {
  const out = resultsDir(set, model);
  mkdirSync(out, { recursive: true });
  const engine = await Engine.start();
  const recorded: Recorded[] = [];
  try {
    for (const file of runFiles(set)) {
      const run = path.basename(file, ".run");
      const review = await runReview(engine, file, { model, thinking, traceFile: path.join(out, `${run}.trace.jsonl`) });
      const row: Recorded = { run, set, model, prompt_version: PROMPT_VERSION, pre: review.pre, explained: review.explained, summary: review.summary, meta: review.meta };
      writeFileSync(path.join(out, `${run}.json`), JSON.stringify(row, null, 1) + "\n", "utf-8");
      recorded.push(row);
      process.stderr.write(`${set} ${run}: ${review.explained.filter((r) => r.status === "verified").length}/${review.pre.flags.length} verified, $${review.meta.cost_usd.toFixed(4)}\n`);
    }
  } finally {
    await engine.close();
  }
  return recorded;
}

export function loadRecorded(set: EvalSet, model: string): Recorded[] {
  const dir = resultsDir(set, model);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => JSON.parse(readFileSync(path.join(dir, f), "utf-8")) as Recorded);
}

export type Rating = "useful" | "not_useful" | "engine_wrong";

export function loadRatings(): Map<string, { rating: Rating; note: string }> {
  const dir = path.join(EVAL_DIR, "ratings");
  const ratings = new Map<string, { rating: Rating; note: string }>();
  if (!existsSync(dir)) return ratings;
  for (const f of readdirSync(dir).filter((x) => x.endsWith(".json"))) {
    const data = JSON.parse(readFileSync(path.join(dir, f), "utf-8")) as { run_id: string; ratings: Record<string, { rating: Rating; note: string }> };
    for (const [flag, r] of Object.entries(data.ratings)) ratings.set(`${data.run_id}/${flag}`, r);
  }
  return ratings;
}

/** Parse what the report's "复制评分" button copied and keep it under coach/eval/ratings/. */
export function importRatings(text: string): string {
  const marker = "STS2-COACH-RATINGS ";
  const start = text.indexOf(marker);
  if (start < 0) throw new Error(`no ${marker.trim()} block in the input`);
  const data = JSON.parse(text.slice(start + marker.length).trim().split("\n")[0]!) as { run_id: string; ratings: Record<string, unknown> };
  const dir = path.join(EVAL_DIR, "ratings");
  mkdirSync(dir, { recursive: true });
  const file = path.join(dir, `${data.run_id}.json`);
  writeFileSync(file, JSON.stringify(data, null, 1) + "\n", "utf-8");
  return file;
}

export interface Metrics {
  set: EvalSet;
  model: string;
  prompt_version: string | null;
  runs: number;
  runs_without_flags: number;
  tier1_total: number;
  shown: number;
  verified: number;
  verified_first_try: number;
  dropped: number;
  groundedness: number | null;
  coverage: number | null;
  reverify_failures: string[];
  summary_from_model: number;
  tool_calls_per_flag: number | null;
  what_if_used: number;
  cost_usd: number;
  cost_per_run: number | null;
  seconds_per_run: number | null;
  ratings: Record<Rating, number> & { rated: number };
}

export function metrics(set: EvalSet, model: string, rows: Recorded[]): Metrics {
  const names = entityNames();
  const ratings = loadRatings();
  const flags = rows.flatMap((r) => r.explained.map((e) => ({ row: r, e })));
  const verified = flags.filter((f) => f.e.status === "verified");
  // Re-check every verified finding on its recorded evidence: the report must hold up later too.
  const reverify: string[] = [];
  for (const { row, e } of verified) {
    const detail = row.pre.flags.find((d) => d.id === e.flag)!;
    const parsed = Finding.safeParse(e.finding);
    const errors = parsed.success ? verifyFinding(parsed.data, detail, e.evidence, names).errors : ["schema"];
    if (errors.length) reverify.push(`${row.run}/${e.flag}: ${errors.join("; ")}`);
  }
  const count: Record<Rating, number> & { rated: number } = { useful: 0, not_useful: 0, engine_wrong: 0, rated: 0 };
  for (const { row, e } of verified) {
    const r = ratings.get(`${row.run}/${e.flag}`);
    if (r) {
      count[r.rating]++;
      count.rated++;
    }
  }
  const shown = flags.length;
  const ratio = (a: number, b: number) => (b ? Math.round((a / b) * 1000) / 1000 : null);
  const cost = rows.reduce((a, r) => a + r.meta.cost_usd, 0);
  return {
    set, model, prompt_version: rows[0]?.prompt_version ?? null,
    runs: rows.length,
    runs_without_flags: rows.filter((r) => r.pre.summary.tier1.length === 0).length,
    tier1_total: rows.reduce((a, r) => a + r.pre.summary.tier1.length, 0),
    shown,
    verified: verified.length,
    verified_first_try: verified.filter((f) => f.e.repairs === 0).length,
    dropped: shown - verified.length,
    groundedness: ratio(verified.length - reverify.length, verified.length),
    coverage: ratio(verified.length, shown),
    reverify_failures: reverify,
    summary_from_model: rows.filter((r) => r.summary.source === "model").length,
    tool_calls_per_flag: ratio(flags.reduce((a, f) => a + f.e.toolCalls, 0), shown),
    what_if_used: verified.filter((f) => f.e.finding?.what_if_text).length,
    cost_usd: Math.round(cost * 10000) / 10000,
    cost_per_run: ratio(cost, rows.length),
    seconds_per_run: ratio(rows.reduce((a, r) => a + r.meta.seconds, 0), rows.length),
    ratings: count,
  };
}

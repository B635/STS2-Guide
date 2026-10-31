// One full review: deterministic pre-pass, one agent loop per flag (in
// parallel), the summary sentence. Shared by the CLI and the evaluation.

import { explainFlag } from "./agent.js";
import type { FlagResult } from "./agent.js";
import type { Engine } from "./engine.js";
import { createClient } from "./llm.js";
import { entityNames } from "./names.js";
import { prepass } from "./prepass.js";
import type { Prepass } from "./prepass.js";
import { programSummary, summarize } from "./summary.js";
import type { Summary } from "./summary.js";
import { Trace } from "./trace.js";
import type { ReportMeta } from "./report.js";

export interface ReviewOptions {
  model: string | null; // null: no model, the pre-pass only
  thinking?: boolean;
  traceFile?: string;
}

export interface Review {
  pre: Prepass;
  explained: FlagResult[];
  summary: Summary;
  meta: ReportMeta;
  trace: Trace;
}

export async function runReview(engine: Engine, runPath: string, options: ReviewOptions): Promise<Review> {
  const started = Date.now();
  const trace = new Trace(options.traceFile);
  const pre = await trace.span("prepass", undefined, () => prepass(engine, runPath), (p) => ({
    run_id: p.summary.run_id, tier1: p.summary.tier1, tier2: p.summary.tier2.length,
  }));
  let explained: FlagResult[] = [];
  let summary: Summary = { text: programSummary([], pre.summary.tier1.length), source: "program", cost_usd: 0 };
  if (options.model && pre.flags.length > 0) {
    const llm = createClient();
    const names = entityNames();
    const ctx = { engine, llm, model: options.model, tools: await engine.agentTools(), names, trace, thinking: options.thinking ?? false };
    explained = await Promise.all(pre.flags.map((flag) => explainFlag(ctx, pre.summary, flag)));
    const verified = explained.filter((r) => r.status === "verified");
    summary = await summarize(llm, options.model, verified, pre.summary.tier1.length, names, trace);
  }
  const cost = explained.reduce((a, r) => a + r.usage.cost_usd, 0) + summary.cost_usd;
  const meta: ReportMeta = {
    model: options.model, cost_usd: cost, seconds: (Date.now() - started) / 1000,
    generated_at: new Date().toLocaleString("sv-SE").slice(0, 16), // local time, "YYYY-MM-DD HH:MM"
  };
  trace.event("review.done", undefined, {
    run_id: pre.summary.run_id, model: options.model, cost_usd: cost, seconds: meta.seconds,
    verified: explained.filter((r) => r.status === "verified").map((r) => r.flag),
    dropped: explained.filter((r) => r.status !== "verified").map((r) => r.flag),
    summary_source: summary.source,
  });
  return { pre, explained, summary, meta, trace };
}

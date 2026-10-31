// The deterministic part of a review: score the run, pick the flags to explain.
// No LLM here; everything comes from the engine.

import { Engine } from "./engine.js";
import { DecisionDetail, DecisionList, RunSummary } from "./schemas.js";

export const MAX_FLAGS = 5;

export interface Prepass {
  summary: RunSummary;
  decisions: DecisionList;
  /** Tier-1 flags to explain, largest gap first, at most MAX_FLAGS. */
  flags: DecisionDetail[];
}

export async function prepass(engine: Engine, runPath: string): Promise<Prepass> {
  const summary = await engine.callParsed("review_run", { path: runPath }, RunSummary);
  const decisions = await engine.callParsed("list_decisions", { run_id: summary.run_id }, DecisionList);
  const flags: DecisionDetail[] = [];
  for (const id of summary.tier1.slice(0, MAX_FLAGS)) {
    flags.push(await engine.callParsed("get_decision", { run_id: summary.run_id, decision_id: id }, DecisionDetail));
  }
  return { summary, decisions, flags };
}

// Shapes of the engine's answers (review/run_review.py, review/mcp_server.py).
// Parsed with zod so a change on the Python side fails loudly here.

import { z } from "zod";

export const RunSummary = z.object({
  run_id: z.string(),
  character: z.string(),
  ascension: z.number().nullable(),
  build_id: z.string().nullable(),
  win: z.boolean(),
  killed_by: z.string().nullable(),
  acts_reached: z.number(),
  last_scored_floor: z.number(),
  final_deck_size: z.number(),
  decisions_scored: z.record(z.string(), z.number()),
  decisions_skipped: z.record(z.string(), z.number()),
  agreement: z.object({ agree: z.number(), judged: z.number(), rate: z.number().nullable() }),
  tier1: z.array(z.string()),
  tier2: z.array(z.string()),
  warnings: z.array(z.string()),
});
export type RunSummary = z.infer<typeof RunSummary>;

export const DecisionRow = z.object({
  id: z.string(),
  kind: z.enum(["card_reward", "rest", "shop"]),
  floor: z.number(),
  act: z.number(),
  chosen: z.string().nullable(),
  chosen_name: z.string().nullable(),
  engine_top: z.string().nullable(),
  engine_top_name: z.string().nullable(),
  confidence: z.string(),
  tier: z.number().nullable(),
  gap: z.number().nullable(),
  not_strict_because: z.array(z.string()),
});
export type DecisionRow = z.infer<typeof DecisionRow>;

export const DecisionList = z.object({
  decisions: z.array(DecisionRow),
  skipped: z.array(z.object({ id: z.string(), kind: z.string(), floor: z.number(), reason: z.string() })),
});
export type DecisionList = z.infer<typeof DecisionList>;

export const Factor = z.object({ code: z.string().nullable(), delta: z.number().nullable(), message: z.string().nullable() });

export const OptionDetail = z.object({
  id: z.string(),
  kind: z.string(),
  entity_id: z.string().nullable(),
  name: z.string(),
  score: z.number().nullable(),
  deck_score: z.number().nullable(),
  rank: z.number().nullable(),
  eligible: z.boolean(),
  fully_scored: z.boolean(),
  gaps: z.array(z.string()),
  factors: z.array(Factor),
  dimensions: z.record(z.string(), z.number()),
});

export const DecisionDetail = z.object({
  id: z.string(),
  kind: z.string(),
  floor: z.number(),
  act: z.number(),
  hp: z.number(),
  max_hp: z.number(),
  gold: z.number(),
  engine_top: z.string().nullable(),
  confidence: z.string(),
  chosen: z.string().nullable(),
  tier: z.number().nullable(),
  gap: z.number().nullable(),
  pick_margin: z.number(),
  options: z.array(OptionDetail),
});
export type DecisionDetail = z.infer<typeof DecisionDetail>;

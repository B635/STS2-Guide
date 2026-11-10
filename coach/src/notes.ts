// Player notes across runs: what the engine kept pointing at in this
// player's earlier reviews. Written by the program from verified findings
// only (the leading factor of each verified flag, with where and when), never
// from free text the model wrote, so a note can not carry a made-up claim
// from one run into the next. At most MAX_NOTES; the oldest go first.
//
// The file is the player's own data: it lives under coach/notes/ (git-ignored)
// unless COACH_NOTES_FILE or an explicit path says otherwise (tests, evaluation).

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { FlagResult } from "./agent.js";
import { REPO_ROOT } from "./engine.js";
import { Finding } from "./finding.js";
import { factorLabel } from "./labels.js";
import type { Prepass } from "./prepass.js";
import { verifyFinding } from "./verifier.js";

export const MAX_NOTES = 20;

export interface PlayerNote {
  run_id: string;
  character: string;
  act: number;
  floor: number;
  decision_id: string;
  chosen_name: string;
  engine_top_name: string;
  factor: string;
  factor_label: string;
  /** The run's date (from its start time when the run id is one), YYYY-MM-DD. */
  date: string;
}

export function notesFile(): string {
  return process.env.COACH_NOTES_FILE ?? path.join(REPO_ROOT, "coach", "notes", "player-notes.json");
}

export function loadNotes(file: string = notesFile()): PlayerNote[] {
  if (!existsSync(file)) return [];
  const data = JSON.parse(readFileSync(file, "utf-8")) as { notes?: PlayerNote[] };
  return (data.notes ?? []).slice(-MAX_NOTES);
}

export function saveNotes(notes: PlayerNote[], file: string = notesFile()): void {
  mkdirSync(path.dirname(file), { recursive: true });
  const kept = notes.slice(-MAX_NOTES);
  writeFileSync(file, JSON.stringify({ note: "Written by the coach from verified findings; at most 20.", notes: kept }, null, 1) + "\n", "utf-8");
}

function runDate(runId: string, today: Date): string {
  // Run ids are the run's start time in seconds; anything else gets the review's date.
  const seconds = /^\d{9,11}$/.test(runId) ? Number(runId) : NaN;
  return (Number.isFinite(seconds) ? new Date(seconds * 1000) : today).toISOString().slice(0, 10);
}

/**
 * One note per verified flag of a review. Each finding is checked again on
 * its own evidence here, so a note can only come from a finding that passes.
 */
export function notesFromReview(pre: Prepass, explained: FlagResult[], names: ReadonlySet<string>, today = new Date()): PlayerNote[] {
  const notes: PlayerNote[] = [];
  for (const result of explained) {
    if (result.status !== "verified") continue;
    const detail = pre.flags.find((d) => d.id === result.flag);
    const parsed = Finding.safeParse(result.finding);
    if (!detail || !parsed.success) continue;
    if (!verifyFinding(parsed.data, detail, result.evidence, names).ok) continue;
    const factor = parsed.data.reasons[0]!.factor;
    const name = (id: string | null) => detail.options.find((o) => o.id === id)?.name ?? id ?? "-";
    notes.push({
      run_id: pre.summary.run_id, character: pre.summary.character, act: detail.act, floor: detail.floor,
      decision_id: detail.id, chosen_name: name(detail.chosen), engine_top_name: name(detail.engine_top),
      factor, factor_label: factorLabel(factor), date: runDate(pre.summary.run_id, today),
    });
  }
  return notes;
}

/** Replace a run's notes (a run reviewed again), append, keep the newest MAX_NOTES. */
export function mergeNotes(existing: PlayerNote[], fresh: PlayerNote[]): PlayerNote[] {
  const runs = new Set(fresh.map((n) => n.run_id));
  return [...existing.filter((n) => !runs.has(n.run_id)), ...fresh].slice(-MAX_NOTES);
}

/** How often each factor led a verified flag in runs other than this one, most frequent first. */
export function recurringFactors(notes: PlayerNote[], excludeRun?: string): Array<{ factor: string; runs: number; flags: number }> {
  const byFactor = new Map<string, { runs: Set<string>; flags: number }>();
  for (const n of notes) {
    if (n.run_id === excludeRun) continue;
    const row = byFactor.get(n.factor_label) ?? { runs: new Set<string>(), flags: 0 };
    row.runs.add(n.run_id);
    row.flags++;
    byFactor.set(n.factor_label, row);
  }
  return [...byFactor.entries()]
    .map(([factor, r]) => ({ factor, runs: r.runs.size, flags: r.flags }))
    .sort((a, b) => b.flags - a.flags);
}

/** One line per note, for the model's context (all fields are the program's). */
export function noteLine(n: PlayerNote): string {
  return `${n.date} ${n.character} 对局 ${n.run_id} 第 ${n.act} 幕第 ${n.floor} 层：你选「${n.chosen_name}」，引擎更看好「${n.engine_top_name}」，主要因为${n.factor_label}`;
}

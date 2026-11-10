import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { FlagResult } from "../src/agent.js";
import { MAX_NOTES, loadNotes, mergeNotes, noteLine, notesFromReview, recurringFactors, saveNotes } from "../src/notes.js";
import type { PlayerNote } from "../src/notes.js";
import type { Prepass } from "../src/prepass.js";
import { DECISIONS, FLAG, NAMES, SUMMARY } from "./fixtures.js";

const pre: Prepass = { summary: { ...SUMMARY, run_id: "1773034032" }, decisions: DECISIONS, flags: [FLAG] };
const usage = { prompt_tokens: 0, completion_tokens: 0, cost_usd: 0, llm_calls: 1 };
const finding = {
  decision_id: "F06-card",
  reasons: [{ factor: "deck_fitness:defense", evidence: "T0", text: "血墙把每回合防御从 6.9 提到 8.9。" }],
  advice: "防御低于需求 8.5 时，先补防御牌。",
};
const verified: FlagResult = { flag: "F06-card", status: "verified", finding, errors: [], evidence: [{ id: "T0", tool: "get_decision", result: FLAG }], toolCalls: 0, repairs: 0, usage };

const note = (run: string, factor = "deck_fitness:defense", label = "防御"): PlayerNote => ({
  run_id: run, character: "IRONCLAD", act: 1, floor: 6, decision_id: "F06-card", chosen_name: "剑柄打击",
  engine_top_name: "血墙", factor, factor_label: label, date: "2026-03-09",
});

describe("player notes", () => {
  it("are written from verified findings only, with the program's fields", () => {
    const dropped: FlagResult = { ...verified, status: "dropped" };
    const notes = notesFromReview(pre, [verified, dropped], NAMES);
    expect(notes).toEqual([{
      run_id: "1773034032", character: "IRONCLAD", act: 1, floor: 6, decision_id: "F06-card",
      chosen_name: "剑柄打击", engine_top_name: "血墙", factor: "deck_fitness:defense", factor_label: "防御",
      date: "2026-03-09",
    }]);
    expect(noteLine(notes[0]!)).toContain("主要因为防御");
  });

  it("skip a finding marked verified that does not pass the verifier on its evidence", () => {
    const tampered: FlagResult = { ...verified, finding: { ...finding, advice: "下次优先拿完美打击。" } };
    expect(notesFromReview(pre, [tampered], NAMES)).toEqual([]);
  });

  it("keep at most 20, the newest, and replace a run reviewed again", () => {
    const many = Array.from({ length: 25 }, (_, i) => note(String(i)));
    const merged = mergeNotes([], many);
    expect(merged).toHaveLength(MAX_NOTES);
    expect(merged[0]!.run_id).toBe("5");
    const again = mergeNotes(merged, [note("7", "deck_fitness:aoe", "群伤")]);
    expect(again.filter((n) => n.run_id === "7")).toEqual([note("7", "deck_fitness:aoe", "群伤")]);
    expect(again.at(-1)!.run_id).toBe("7");
  });

  it("round-trip through the file, capped on save and load", () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), "coach-notes-"));
    const file = path.join(dir, "sub", "notes.json");
    expect(loadNotes(file)).toEqual([]);
    saveNotes(Array.from({ length: 30 }, (_, i) => note(String(i))), file);
    expect(JSON.parse(readFileSync(file, "utf-8")).notes).toHaveLength(MAX_NOTES);
    expect(loadNotes(file).map((n) => n.run_id)[0]).toBe("10");
    rmSync(dir, { recursive: true, force: true });
  });

  it("count recurring factors across other runs", () => {
    const notes = [note("a"), note("a"), note("b"), note("c", "deck_fitness:aoe", "群伤"), note("x")];
    expect(recurringFactors(notes, "x")).toEqual([{ factor: "防御", runs: 2, flags: 3 }, { factor: "群伤", runs: 1, flags: 1 }]);
  });
});

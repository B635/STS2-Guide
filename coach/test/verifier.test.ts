import { describe, expect, it } from "vitest";
import type { Finding } from "../src/finding.js";
import type { DecisionDetail } from "../src/schemas.js";
import { mentionedNames, separatingFactors, verifyFinding } from "../src/verifier.js";
import type { Evidence } from "../src/verifier.js";

const option = (id: string, name: string, deckScore: number, factors: Array<[string, number, string]>) => ({
  id, kind: "card", entity_id: id.split(":")[1] ?? null, name, score: deckScore, deck_score: deckScore, rank: 1,
  eligible: true, fully_scored: true, gaps: [], dimensions: {},
  factors: factors.map(([code, delta, message]) => ({ code, delta, message })),
});

const detail: DecisionDetail = {
  id: "F06-card", kind: "card_reward", floor: 6, act: 1, hp: 60, max_hp: 80, gold: 99,
  engine_top: "1:BLOOD_WALL", confidence: "medium", chosen: "0:POMMEL_STRIKE", tier: 1, gap: 2.4, pick_margin: 1,
  options: [
    option("1:BLOOD_WALL", "血墙", 40.1, [
      ["deck_fitness:defense", 3.1, "拿血墙后防御：每回合 6.9 → 8.9（需求 8.5）"],
      ["deck_fitness:frontload", -0.5, "拿血墙后前期输出：每回合 12.5 → 10.7"],
    ]),
    option("0:POMMEL_STRIKE", "剑柄打击", 37.7, [
      ["deck_fitness:defense", -0.6, "拿剑柄打击后防御：每回合 6.9 → 6.4"],
      ["deck_fitness:frontload", 0.4, "拿剑柄打击后前期输出：每回合 12.5 → 14.1"],
      ["deck_fitness:pick_margin", 9, "a policy leaning, never a reason"],
    ]),
  ],
};
const evidence: Evidence[] = [
  { id: "T0", tool: "get_decision", result: detail },
  { id: "T1", tool: "what_if", result: { engine_top_after: "1:BLOOD_WALL", gap_after: 0.6 } },
];
const names = new Set(["血墙", "剑柄打击", "打击", "完美打击"]);
const good: Finding = {
  decision_id: "F06-card",
  reasons: [{ factor: "deck_fitness:defense", evidence: "T0", text: "血墙把每回合防御从 6.9 提到 8.9，剑柄打击降到 6.4。" }],
  what_if: { evidence: "T1", text: "改动后引擎仍选血墙，差距缩到 0.6。" },
  advice: "防御不到需求 8.5 时，先补防御牌。",
};

describe("verifier", () => {
  it("accepts a finding grounded in its evidence", () => {
    expect(verifyFinding(good, detail, evidence, names)).toEqual({ ok: true, errors: [] });
  });

  it("ranks factors by how much they separate the two options, without policy leanings", () => {
    expect(separatingFactors(detail)).toEqual(["deck_fitness:defense", "deck_fitness:frontload"]);
  });

  it("rejects a number that no cited result contains", () => {
    const bad = { ...good, reasons: [{ ...good.reasons[0]!, text: "血墙把防御提高了 5 点。" }] };
    expect(verifyFinding(bad, detail, evidence, names).errors).toEqual(["reasons[0]: the number 5 is not in T0/T0"]);
  });

  it("rejects a card the evidence never mentions", () => {
    const bad = { ...good, advice: "下次优先拿完美打击。" };
    expect(verifyFinding(bad, detail, evidence, names).errors[0]).toContain('"完美打击"');
  });

  it("does not read a shorter name inside a longer one", () => {
    expect(mentionedNames("剑柄打击很好", names)).toEqual(["剑柄打击"]);
  });

  it("rejects reasons on factors that do not separate the options, and unknown evidence", () => {
    const bad = { ...good, reasons: [{ factor: "deck_fitness:pick_margin", evidence: "T9", text: "门槛更低。" }] };
    const errors = verifyFinding(bad, detail, evidence, names).errors;
    expect(errors.some((e) => e.includes("T9"))).toBe(true);
    expect(errors.some((e) => e.includes("not one of the largest differences"))).toBe(true);
  });

  it("rejects a what-if claim that cites another tool, the wrong decision, and banned wording", () => {
    const bad = {
      ...good,
      decision_id: "F07-card",
      what_if: { evidence: "T0", text: "这是一个失误。" },
    };
    const errors = verifyFinding(bad, detail, evidence, names).errors;
    expect(errors).toContain("decision_id must be F06-card, got F07-card");
    expect(errors).toContain("what_if cites T0, which is not a what_if result");
    expect(errors).toContain('what_if: do not use "失误"');
  });
});

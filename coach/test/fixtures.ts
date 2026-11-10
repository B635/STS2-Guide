// Small engine records for the loop tests (shapes of review/mcp_server.py).

import type { DecisionDetail, DecisionList, RunSummary } from "../src/schemas.js";

const option = (id: string, name: string, deckScore: number, rank: number, factors: Array<[string, number, string]>) => ({
  id, kind: id === "skip" ? "skip" : "card", entity_id: id.split(":")[1] ?? null, name, score: deckScore, deck_score: deckScore, rank,
  eligible: true, fully_scored: true, gaps: [], dimensions: {},
  factors: factors.map(([code, delta, message]) => ({ code, delta, message })),
});

export const SUMMARY: RunSummary = {
  run_id: "111", character: "IRONCLAD", ascension: 2, build_id: "v0.98.2", win: false, killed_by: "知识恶魔",
  acts_reached: 2, last_scored_floor: 30, final_deck_size: 12,
  decisions_scored: { card_reward: 2, rest: 1 }, decisions_skipped: {},
  agreement: { agree: 1, judged: 3, rate: 0.333 }, tier1: ["F06-card"], tier2: ["F09-rest"], warnings: [],
};

/** Tier 1: the engine prefers 血墙, the player took 剑柄打击. */
export const FLAG: DecisionDetail = {
  id: "F06-card", kind: "card_reward", floor: 6, act: 1, hp: 60, max_hp: 80, gold: 99,
  engine_top: "1:BLOOD_WALL", confidence: "medium", chosen: "0:POMMEL_STRIKE", tier: 1, gap: 2.4, pick_margin: 1,
  options: [
    option("1:BLOOD_WALL", "血墙", 40.1, 1, [
      ["deck_fitness:defense", 3.1, "拿血墙后防御：每回合 6.9 → 8.9（需求 8.5）"],
      ["deck_fitness:frontload", -0.5, "拿血墙后前期输出：每回合 12.5 → 10.7（需求 25.0）"],
    ]),
    option("0:POMMEL_STRIKE", "剑柄打击", 37.7, 2, [
      ["deck_fitness:defense", -0.6, "拿剑柄打击后防御：每回合 6.9 → 6.4（需求 8.5）"],
      ["deck_fitness:frontload", 0.4, "拿剑柄打击后前期输出：每回合 12.5 → 14.1（需求 25.0）"],
    ]),
    option("skip", "跳过", 35.0, 3, [["deck_fitness:pick_margin", 1, "门槛"]]),
  ],
};

/** Tier 0: the player and the engine both took 耸肩无视; 愤怒 is second. */
export const AGREED: DecisionDetail = {
  id: "F08-card", kind: "card_reward", floor: 8, act: 1, hp: 50, max_hp: 80, gold: 120,
  engine_top: "0:SHRUG_IT_OFF", confidence: "medium", chosen: "0:SHRUG_IT_OFF", tier: 0, gap: 0, pick_margin: 1,
  options: [
    option("0:SHRUG_IT_OFF", "耸肩无视", 45.5, 1, [
      ["deck_fitness:defense", 2.2, "拿耸肩无视后防御：每回合 6.4 → 7.6（需求 8.9）"],
      ["deck_fitness:cycle", 0.6, "拿耸肩无视后循环：每回合额外抽牌 0.25 → 0.46（需求 1.0）"],
    ]),
    option("1:ANGER", "愤怒", 43.0, 2, [
      ["deck_fitness:frontload", 1.1, "拿愤怒后前期输出：每回合 15.7 → 18.0（需求 25.9）"],
    ]),
    option("skip", "跳过", 41.0, 3, [["deck_fitness:pick_margin", 1, "门槛"]]),
  ],
};

/** A rest site the engine only scores by its campfire rule: nothing to compare. */
export const REST: DecisionDetail = {
  id: "F09-rest", kind: "rest", floor: 9, act: 1, hp: 30, max_hp: 80, gold: 120,
  engine_top: "rest:HEAL", confidence: "low", chosen: "rest:SMITH", tier: 2, gap: -12.5, pick_margin: 1,
  options: [
    { ...option("rest:HEAL", "休息", 20, 1, [["campfire_rest_rule", 30, "生命低于一半时优先休息"]]), kind: "rest" },
    { ...option("rest:SMITH", "锻造", 32.5, 2, []), kind: "rest", fully_scored: false },
  ],
};

export const DECISIONS: DecisionList = {
  decisions: [FLAG, AGREED, REST].map((d) => ({
    id: d.id, kind: d.kind as "card_reward" | "rest", floor: d.floor, act: d.act, chosen: d.chosen,
    chosen_name: d.options.find((o) => o.id === d.chosen)?.name ?? null,
    engine_top: d.engine_top, engine_top_name: d.options.find((o) => o.id === d.engine_top)?.name ?? null,
    confidence: d.confidence, tier: d.tier, gap: d.gap, not_strict_because: d.tier === 2 ? ["low_confidence", "partly_scored"] : [],
  })),
  skipped: [{ id: "F35-card", kind: "card_reward", floor: 35, reason: "unknown_boss:DOORMAKER_BOSS" }],
};

export const DETAILS: Record<string, DecisionDetail> = { [FLAG.id]: FLAG, [AGREED.id]: AGREED, [REST.id]: REST };

export function compare(detail: DecisionDetail, a: string, b: string) {
  const byId = new Map(detail.options.map((o) => [o.id, o]));
  const oa = byId.get(a)!, ob = byId.get(b)!;
  const codes = new Set([...oa.factors, ...ob.factors].map((f) => f.code!).filter((c) => c !== "deck_fitness:pick_margin" && c !== "campfire_rest_rule"));
  const d = (o: typeof oa, c: string) => o.factors.find((f) => f.code === c);
  return {
    decision_id: detail.id,
    a: { id: a, name: oa.name, deck_score: oa.deck_score }, b: { id: b, name: ob.name, deck_score: ob.deck_score },
    factors: [...codes].map((c) => ({
      factor: c, a: d(oa, c)?.delta ?? 0, b: d(ob, c)?.delta ?? 0, difference: (d(oa, c)?.delta ?? 0) - (d(ob, c)?.delta ?? 0),
      a_message: d(oa, c)?.message ?? null, b_message: d(ob, c)?.message ?? null,
    })),
  };
}

/** The engine's answers for the fixtures above. */
export const ENGINE_ANSWERS: Record<string, (args: Record<string, unknown>) => unknown> = {
  get_decision: (a) => {
    const d = DETAILS[String(a.decision_id)];
    if (!d) throw new Error(`no scored decision ${String(a.decision_id)}`);
    return d;
  },
  compare_options: (a) => compare(DETAILS[String(a.decision_id)]!, String(a.option_a), String(a.option_b)),
  get_deck: (a) => ({ decision_id: a.decision_id, deck_size: 11, deck: [{ card: "STRIKE_IRONCLAD", name: "打击", upgrades: 0, count: 5 }, { card: "DEFEND_IRONCLAD", name: "防御", upgrades: 0, count: 4 }], relics: [{ relic: "BURNING_BLOOD", name: "燃烧之血" }] }),
  what_if_choices: () => ({ remove: ["STRIKE_IRONCLAD"], upgrade: [], add: [{ card: "ANGER", name: "愤怒" }] }),
  what_if: (a) => ({ decision_id: a.decision_id, changes: a.changes, engine_top_before: "1:BLOOD_WALL", engine_top_after: "1:BLOOD_WALL", pick_changed: false, agrees_with_player_after: false, gap_after: 1.2 }),
  get_card: (a) => ({ id: a.card_id, name: a.card_id === "BLOOD_WALL" ? "血墙" : "剑柄打击", cost: 2, type: "Skill", description: "获得 16 点格挡。" }),
  get_relic: (a) => ({ id: a.relic_id, name: "燃烧之血", description: "战斗结束时回复 6 点生命。" }),
  deck_trend: () => ({ labels: { defense: "防御" }, decisions: [
    { decision_id: "F06-card", floor: 6, act: 1, deck_size: 11, values: { defense: { have: 6.9, need: 8.5, below_need: true } } },
    { decision_id: "F08-card", floor: 8, act: 1, deck_size: 12, values: { defense: { have: 6.4, need: 8.9, below_need: true } } },
  ] }),
};

export const NAMES: ReadonlySet<string> = new Set(["血墙", "剑柄打击", "打击", "防御", "耸肩无视", "愤怒", "燃烧之血", "完美打击", "放血"]);

// Chinese labels for the engine's codes (advisor/deck_fitness.py DIMENSION_LABELS and factor codes).

const FACTORS: Record<string, string> = {
  "deck_fitness:frontload": "前期输出",
  "deck_fitness:aoe": "群伤",
  "deck_fitness:defense": "防御",
  "deck_fitness:scaling": "成长",
  "deck_fitness:cycle": "循环（过牌）",
  "deck_fitness:energy": "能量",
  "deck_fitness:synergy": "配合",
  "deck_fitness:lookahead": "后续选牌",
  "merchant:deck": "牌组变化",
  "merchant:price": "花掉的金币",
};

export function factorLabel(code: string): string {
  return FACTORS[code] ?? code;
}

const NOT_STRICT: Record<string, string> = {
  partly_scored: "有选项只算了部分效果",
  low_confidence: "引擎把握不大",
  small_gap: "差距不大",
  choice_not_scored: "你的选择没法评分",
  engine_no_pick: "引擎没给出建议",
  choice_unknown: "记录里看不出你的选择",
};

export function notStrictLabel(reasons: string[]): string {
  return reasons.map((r) => NOT_STRICT[r] ?? r).join("、");
}

const SKIPPED: Record<string, string> = {
  no_boss: "这一幕没走到 Boss，记录里没有 Boss 可以对照",
  unknown_boss: "Boss 不在现有资料里（旧版本）",
  unknown_card: "有卡牌不在现有资料里（旧版本）",
  rest_option_not_modeled: "休息点用了引擎没建模的选项",
};

export function skippedLabel(reason: string): string {
  return SKIPPED[reason.split(":")[0] ?? reason] ?? reason;
}

export const KIND_LABELS: Record<string, string> = { card_reward: "选牌", rest: "休息点", shop: "商店" };

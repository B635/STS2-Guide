// Every card, relic and potion name in the game's catalog: the verifier uses
// them to find names in the model's text.

import { readFileSync } from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./engine.js";

interface Named {
  id?: string;
  name?: string;
  description?: string | null;
  upgrade_description?: string | null;
  cost?: number | null;
  type?: string | null;
  rarity?: string | null;
  star_cost?: number | null;
  vars?: Record<string, number> | null;
  upgrade?: Record<string, unknown> | null;
}

function applyUpgradeDelta(base: number | undefined, upgrade: Record<string, unknown> | null | undefined, key: string): number | undefined {
  if (base === undefined || !upgrade || !(key in upgrade)) return base;
  const raw = upgrade[key];
  if (typeof raw === "number") return Math.trunc(base + raw);
  if (typeof raw === "string") {
    const t = raw.trim();
    if (/^[+-]\d+$/.test(t)) return base + Number.parseInt(t, 10);
  }
  return base;
}

// The catalog's descriptions lost the game's inline icons, so "获得[2 stars]。" reads "获得。".
// The amounts are still in vars/upgrade: put them back as text (matching review/engine.py).
export function restoreIcons(
  rawText: string | null | undefined,
  vars?: Record<string, number> | null,
  upgrade?: Record<string, unknown> | null,
  upgraded = false,
): string {
  let text = (rawText ?? "").replace(/\s*\n\s*/g, " ");
  if (!text) return text;
  let stars = vars?.["Stars"];
  let energy = vars?.["Energy"];
  if (upgraded) {
    stars = applyUpgradeDelta(stars, upgrade, "stars");
    energy = applyUpgradeDelta(energy, upgrade, "energy");
  }
  if (stars !== undefined && energy !== undefined) {
    text = text.replace(/获得与(?=[。，；]|$)/g, `获得 ${energy} 点能量与 ${stars} 颗星`);
    const gains = [`${stars} 颗星`, `${energy} 点能量`];
    const matches = [...text.matchAll(/获得(?=[。，；并]|$)/g)];
    if (matches.length >= 2) {
      let idx = 0;
      text = text.replace(/获得(?=[。，；并]|$)/g, () => {
        const piece = gains[idx] ?? `${energy} 点能量与 ${stars} 颗星`;
        idx++;
        return `获得 ${piece}`;
      });
    } else if (matches.length === 1) {
      text = text.replace(/获得(?=[。，；并]|$)/g, `获得 ${energy} 点能量与 ${stars} 颗星`);
    }
  } else if (stars !== undefined) {
    text = text.replace(/获得(?=[。，；并]|$)/g, `获得 ${stars} 颗星`);
    text = text.replace(new RegExp(`获得${stars}(?=[。，；并]|$)`, "g"), `获得 ${stars} 颗星`);
  } else if (energy !== undefined) {
    text = text.replace(/获得(?=[。，；并]|$)/g, `获得 ${energy} 点能量`);
    text = text.replace(new RegExp(`获得${energy}(?=[。，；并]|$)`, "g"), `获得 ${energy} 点能量`);
  }
  return text;
}

let cached: Set<string> | undefined;
let knowledge: Record<string, Named[]> | undefined;

function load(): Record<string, Named[]> {
  knowledge ??= JSON.parse(readFileSync(path.join(REPO_ROOT, "data", "knowledge.json"), "utf-8")) as Record<string, Named[]>;
  return knowledge;
}

export function characterName(id: string): string {
  return load()["characters"]?.find((c) => c.id === id)?.name ?? id;
}

export function entityNames(): ReadonlySet<string> {
  if (!cached) {
    const all = load();
    cached = new Set(["cards", "relics", "potions"].flatMap((kind) => (all[kind] ?? []).map((e) => e.name ?? "")).filter(Boolean));
  }
  return cached;
}

/** A card's or relic's ID from its ID or its Chinese name (the model may give either). */
export function lookupId(kind: "cards" | "relics", text: string): string | undefined {
  const rows = load()[kind] ?? [];
  const t = text.trim();
  return rows.find((r) => r.id === t.toUpperCase())?.id ?? rows.find((r) => r.name === t)?.id;
}

const KIND_WORDS: Record<string, string> = { cards: "", relics: "遗物", potions: "药水" };

/** What a card, relic or potion does, in the game's own words, by its Chinese name (for the reader of a report). */
export function describe(name: string): { name: string; meta: string; description: string; upgrade_description?: string } | undefined {
  for (const kind of ["cards", "relics", "potions"]) {
    const row = (load()[kind] ?? []).find((r) => r.name === name);
    if (!row) continue;
    const meta =
      kind === "cards"
        ? [row.cost === null || row.cost === undefined ? null : `${row.cost} 费`, row.star_cost ? `消耗 ${row.star_cost} 颗星` : null, row.type, row.rarity].filter(Boolean).join(" · ")
        : [KIND_WORDS[kind], row.rarity].filter(Boolean).join(" · ");
    const description = restoreIcons(row.description, row.vars, row.upgrade, false);
    const upgradeDescription = row.upgrade_description
      ? restoreIcons(row.upgrade_description, row.vars, row.upgrade, true)
      : undefined;
    return {
      name,
      meta,
      description,
      ...(upgradeDescription ? { upgrade_description: upgradeDescription } : {}),
    };
  }
  return undefined;
}

// Every card, relic and potion name in the game's catalog: the verifier uses
// them to find names in the model's text.

import { readFileSync } from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./engine.js";

interface Named {
  id?: string;
  name?: string;
  description?: string | null;
  cost?: number | null;
  type?: string | null;
  rarity?: string | null;
  star_cost?: number | null;
  vars?: Record<string, number> | null;
}

// The catalog's descriptions lost the game's inline icons, so "获得[2 stars]。" reads "获得。".
// The amount is still in the card's vars: put it back as text.
function restoreIcons(row: Named): string {
  const text = (row.description ?? "").replace(/\s*\n\s*/g, " ");
  const stars = row.vars?.["Stars"];
  const energy = row.vars?.["Energy"];
  const gain = stars !== undefined ? `${stars} 颗星` : energy !== undefined ? `${energy} 点能量` : null;
  return gain ? text.replace(/获得(?=[。，；]|$)/g, `获得 ${gain}`) : text;
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
export function describe(name: string): { name: string; meta: string; description: string } | undefined {
  for (const kind of ["cards", "relics", "potions"]) {
    const row = (load()[kind] ?? []).find((r) => r.name === name);
    if (!row) continue;
    const meta =
      kind === "cards"
        ? [row.cost === null || row.cost === undefined ? null : `${row.cost} 费`, row.star_cost ? `消耗 ${row.star_cost} 颗星` : null, row.type, row.rarity].filter(Boolean).join(" · ")
        : [KIND_WORDS[kind], row.rarity].filter(Boolean).join(" · ");
    return { name, meta, description: (row.description ?? "").replace(/\s*\n\s*/g, " ") };
  }
  return undefined;
}

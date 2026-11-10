// Every card, relic and potion name in the game's catalog: the verifier uses
// them to find names in the model's text.

import { readFileSync } from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./engine.js";

interface Named {
  id?: string;
  name?: string;
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

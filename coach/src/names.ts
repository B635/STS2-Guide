// Every card, relic and potion name in the game's catalog: the verifier uses
// them to find names in the model's text.

import { readFileSync } from "node:fs";
import path from "node:path";
import { REPO_ROOT } from "./engine.js";

interface Named {
  name?: string;
}

let cached: Set<string> | undefined;

export function entityNames(): ReadonlySet<string> {
  if (!cached) {
    const knowledge = JSON.parse(readFileSync(path.join(REPO_ROOT, "data", "knowledge.json"), "utf-8")) as Record<
      string,
      Named[]
    >;
    cached = new Set(
      ["cards", "relics", "potions"].flatMap((kind) => (knowledge[kind] ?? []).map((e) => e.name ?? "")).filter(Boolean),
    );
  }
  return cached;
}

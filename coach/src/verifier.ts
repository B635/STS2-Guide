// Checks one finding against the tool results it cites. Deterministic: the
// same finding and evidence always give the same verdict, so the pass rate
// is also evaluation metric 1 ("no fabrication").
//
// Rules:
// - the finding is about the flagged decision;
// - every reason names a factor that is among the largest differences
//   between the engine's pick and the player's choice, and cites a result
//   that contains that factor;
// - a what-if claim cites a what_if result;
// - every number and every card / relic / potion name in the text appears in
//   the cited results (the advice may use any result of this review);
// - no banned wording.

import type { DecisionDetail } from "./schemas.js";
import type { Finding } from "./finding.js";

export const TOP_FACTORS = 3;
export const BANNED_WORDS = ["失误", "错误", "愚蠢", "你错了", "犯错"];
const POLICY_LEANINGS = new Set(["deck_fitness:pick_margin", "campfire_rest_rule"]);

export interface Evidence {
  id: string; // T0 is the decision detail given up front; T1.. are tool calls
  tool: string;
  result: unknown;
}

export interface Verdict {
  ok: boolean;
  errors: string[];
}

/** Factor codes by how much they separate the engine's pick from the player's choice. */
export function separatingFactors(detail: DecisionDetail): string[] {
  const top = detail.options.find((o) => o.id === detail.engine_top);
  const chosen = detail.options.find((o) => o.id === detail.chosen);
  if (!top || !chosen) return [];
  const delta = (o: typeof top, code: string) => o.factors.find((f) => f.code === code)?.delta ?? 0;
  const codes = new Set([...top.factors, ...chosen.factors].map((f) => f.code).filter((c): c is string => !!c));
  return [...codes]
    .filter((c) => !POLICY_LEANINGS.has(c))
    .map((c) => ({ c, d: Math.abs(delta(top, c) - delta(chosen, c)) }))
    .filter((x) => x.d > 0)
    .sort((a, b) => b.d - a.d)
    .map((x) => x.c);
}

function numbersIn(text: string): string[] {
  return (text.match(/\d+(?:\.\d+)?/g) ?? []).map(normalizeNumber);
}

function normalizeNumber(n: string): string {
  return String(Number(n));
}

function evidenceNumbers(results: unknown[]): Set<string> {
  const found = new Set<string>();
  for (const r of results) for (const n of JSON.stringify(r).match(/-?\d+(?:\.\d+)?/g) ?? []) {
    found.add(normalizeNumber(n.replace("-", "")));
  }
  return found;
}

/** Names of game entities mentioned in a text, longest first so 打击 inside 双重打击 is not counted twice. */
export function mentionedNames(text: string, names: ReadonlySet<string>): string[] {
  const hits: string[] = [];
  let rest = text;
  for (const name of [...names].sort((a, b) => b.length - a.length)) {
    if (name.length < 2) continue;
    if (rest.includes(name)) {
      hits.push(name);
      rest = rest.split(name).join(" ");
    }
  }
  return hits;
}

export function verifyFinding(
  finding: Finding,
  detail: DecisionDetail,
  evidence: Evidence[],
  entityNames: ReadonlySet<string>,
): Verdict {
  const errors: string[] = [];
  const byId = new Map(evidence.map((e) => [e.id, e]));
  if (finding.decision_id !== detail.id) errors.push(`decision_id must be ${detail.id}, got ${finding.decision_id}`);

  const allowed = separatingFactors(detail).slice(0, TOP_FACTORS);
  for (const [i, reason] of finding.reasons.entries()) {
    const cited = byId.get(reason.evidence);
    if (!cited) errors.push(`reasons[${i}] cites ${reason.evidence}, which is not a tool result of this review`);
    else if (!JSON.stringify(cited.result).includes(`"${reason.factor}"`)) {
      errors.push(`reasons[${i}]: ${reason.evidence} does not contain factor ${reason.factor}`);
    }
    if (!allowed.includes(reason.factor)) {
      errors.push(`reasons[${i}]: ${reason.factor} is not one of the largest differences here (${allowed.join(", ")})`);
    }
  }
  const factors = finding.reasons.map((r) => r.factor);
  if (new Set(factors).size !== factors.length) errors.push("each reason must name a different factor");

  const texts: Array<{ where: string; text: string; cites: string[] }> = finding.reasons.map((r, i) => ({
    where: `reasons[${i}]`,
    text: r.text,
    cites: [r.evidence, "T0"],
  }));
  if (finding.what_if) {
    const cited = byId.get(finding.what_if.evidence);
    if (!cited || cited.tool !== "what_if") errors.push(`what_if cites ${finding.what_if.evidence}, which is not a what_if result`);
    texts.push({ where: "what_if", text: finding.what_if.text, cites: [finding.what_if.evidence, "T0"] });
  }
  const everything = evidence.map((e) => e.id);
  texts.push({ where: "advice", text: finding.advice, cites: everything });

  for (const { where, text, cites } of texts) {
    const results = cites.map((id) => byId.get(id)?.result).filter((r) => r !== undefined);
    const numbers = evidenceNumbers(results);
    for (const n of numbersIn(text)) {
      if (!numbers.has(n)) errors.push(`${where}: the number ${n} is not in ${cites.join("/")}`);
    }
    const cited = results.map((r) => JSON.stringify(r)).join("\n");
    for (const name of mentionedNames(text, entityNames)) {
      if (!cited.includes(name)) errors.push(`${where}: "${name}" is not in ${cites.join("/")}`);
    }
    for (const word of BANNED_WORDS) if (text.includes(word)) errors.push(`${where}: do not use "${word}"`);
  }
  return { ok: errors.length === 0, errors };
}

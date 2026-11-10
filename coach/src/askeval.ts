// Evaluation of the follow-up conversation (part B): fixed questions over the
// development runs, every answer kept with its evidence and checks, metrics
// computed from the kept outputs only (re-checked without the model), and a
// one-page sheet of sampled answers for the owner to rate.
//
// Development set only: the questions were written after looking at the dev
// runs, and the test runs are not used for part B at all.

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { UNCONFIRMED, ask, checkAnswer, newSession, shownAnswer } from "./ask.js";
import type { Answer } from "./ask.js";
import { Engine } from "./engine.js";
import { EVAL_DIR, loadRecorded, runFiles } from "./evaluate.js";
import { createClient } from "./llm.js";
import { entityNames } from "./names.js";
import { mergeNotes, notesFromReview } from "./notes.js";
import type { PlayerNote } from "./notes.js";
import { STYLE, ratingScript } from "./report.js";
import { DecisionList, RunSummary } from "./schemas.js";
import { Trace } from "./trace.js";
import type { Evidence } from "./verifier.js";

/** Bump when the lead, critic or investigator prompts change. */
export const ASK_PROMPT_VERSION = "b4";
export const QUESTIONS_FILE = path.join(EVAL_DIR, "ask-questions-dev.json");
export const SHEET_SIZE = 10;
const SHEET_SEED = 28;

export type QuestionKind = "decision" | "follow_up" | "pattern" | "card" | "out_of_scope" | "notes";

export interface EvalQuestion {
  id: string;
  run: string;
  kind: QuestionKind;
  question: string;
  /** What a good answer does, for the rater (written with the question). */
  expect: string;
}

export interface AskRecorded {
  id: string;
  run: string;
  kind: QuestionKind;
  question: string;
  expect: string;
  model: string;
  prompt_version: string;
  context: Evidence; // E0 of the session
  answer: Answer;
}

export function loadQuestions(file = QUESTIONS_FILE): EvalQuestion[] {
  return (JSON.parse(readFileSync(file, "utf-8")) as { questions: EvalQuestion[] }).questions;
}

export function askResultsDir(model: string): string {
  return path.join(EVAL_DIR, "results", "ask-dev", model);
}

/** The player notes the evaluation uses: written by the program from the kept part-A dev results. */
export function devNotes(model: string): PlayerNote[] {
  const names = entityNames();
  let notes: PlayerNote[] = [];
  for (const row of loadRecorded("dev", model)) notes = mergeNotes(notes, notesFromReview(row.pre, row.explained, names));
  return notes;
}

export async function recordAsk(model: string, only?: Set<string>): Promise<AskRecorded[]> {
  const questions = loadQuestions().filter((q) => !only || only.has(q.run) || only.has(q.id));
  const out = askResultsDir(model);
  mkdirSync(out, { recursive: true });
  const notes = devNotes(model);
  writeFileSync(path.join(out, "notes.json"), JSON.stringify({ notes }, null, 1) + "\n", "utf-8");
  const files = new Map(runFiles("dev").map((f) => [path.basename(f, ".run"), f]));
  const engine = await Engine.start();
  const llm = createClient();
  const names = entityNames();
  const recorded: AskRecorded[] = [];
  try {
    const tools = await engine.agentTools();
    // One session per run, questions in file order: follow-ups see the turns before them.
    for (const run of [...new Set(questions.map((q) => q.run))]) {
      const file = files.get(run);
      if (!file) throw new Error(`${run} is not a development run`);
      const trace = new Trace(path.join(out, `${run}.trace.jsonl`));
      const summary = await engine.callParsed("review_run", { path: file }, RunSummary);
      const decisions = await engine.callParsed("list_decisions", { run_id: summary.run_id }, DecisionList);
      const session = newSession(summary, decisions, notes);
      const ctx = { engine, llm, model, tools, names, trace, thinking: false };
      for (const q of questions.filter((x) => x.run === run)) {
        const answer = await ask(ctx, session, q.question);
        const row: AskRecorded = { ...q, model, prompt_version: ASK_PROMPT_VERSION, context: session.context, answer };
        writeFileSync(path.join(out, `${q.id}.json`), JSON.stringify(row, null, 1) + "\n", "utf-8");
        recorded.push(row);
        process.stderr.write(`${q.id} ${run} ${answer.status} tools ${answer.tool_calls} inv ${answer.investigators.length} $${answer.usage.cost_usd.toFixed(4)} ${answer.seconds.toFixed(0)}s\n`);
      }
    }
  } finally {
    await engine.close();
  }
  return recorded;
}

export function loadAskRecorded(model: string): AskRecorded[] {
  const dir = askResultsDir(model);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => /^Q\d+\.json$/.test(f))
    .sort()
    .map((f) => JSON.parse(readFileSync(path.join(dir, f), "utf-8")) as AskRecorded);
}

/** Every result an answer was checked against, rebuilt from the kept outputs of its run. */
function windowOf(row: AskRecorded, rows: AskRecorded[]): Evidence[] {
  const pool = new Map<string, Evidence>([[row.context.id, row.context]]);
  for (const r of rows) if (r.run === row.run) for (const e of r.answer.evidence) pool.set(e.id, e);
  return row.answer.window.map((id) => pool.get(id)).filter((e): e is Evidence => e !== undefined);
}

// Phrases of an answer that says the engine cannot answer (for the out-of-scope questions).
const DECLINES = /回答不了|看不出|没法|无法|不能回答|没有.*数据|不模拟|没有评|没能确认/;

export interface AskMetrics {
  model: string;
  prompt_version: string | null;
  questions: number;
  by_kind: Record<string, number>;
  status: Record<Answer["status"], number>;
  /** 1. program check of the final answers, re-run on the kept evidence. */
  final_program_pass: number | null;
  final_program_failures: string[];
  /** Drafts: how many passed the program check, how many the critic flagged, how many were sent back. */
  draft_program_pass: number | null;
  draft_critic_flagged: number | null;
  sent_back: number | null;
  /** Of the rewrites: share where something still failed and was cut (or all of it). */
  still_unsupported_after_rewrite: number | null;
  critic_unparsed: number;
  out_of_scope_declined: string;
  tool_calls_per_question: number | null;
  investigators_per_question: number | null;
  investigations_reused: number;
  llm_calls_per_question: number | null;
  cost_usd: number;
  cost_per_question: number | null;
  cost_by_role: Record<string, number>;
  seconds_per_question: number | null;
  ratings: Record<string, number>;
}

export function loadAskRatings(): Map<string, { rating: string; note: string }> {
  const file = path.join(EVAL_DIR, "ratings", "ask-dev.json");
  if (!existsSync(file)) return new Map();
  const data = JSON.parse(readFileSync(file, "utf-8")) as { ratings: Record<string, { rating: string; note: string }> };
  return new Map(Object.entries(data.ratings));
}

export function askMetrics(model: string, rows: AskRecorded[]): AskMetrics {
  const names = entityNames();
  const n = rows.length;
  const ratio = (a: number, b: number) => (b ? Math.round((a / b) * 1000) / 1000 : null);
  const failures: string[] = [];
  for (const row of rows) {
    const failed = checkAnswer(row.answer.final, windowOf(row, rows), names);
    if (failed.length) failures.push(`${row.id}: ${failed.flatMap((f) => f.errors).join("; ")}`);
  }
  const answers = rows.map((r) => r.answer);
  const sentBack = answers.filter((a) => a.checks.draft_program.length || a.checks.draft_critic.length);
  const rewritten = answers.filter((a) => a.rewrite !== null);
  const outOfScope = rows.filter((r) => r.kind === "out_of_scope");
  const status = { passed: 0, rewritten: 0, cut: 0, unconfirmed: 0 } as Record<Answer["status"], number>;
  for (const a of answers) status[a.status]++;
  const byKind: Record<string, number> = {};
  for (const r of rows) byKind[r.kind] = (byKind[r.kind] ?? 0) + 1;
  const cost = answers.reduce((s, a) => s + a.usage.cost_usd, 0);
  const byRole: Record<string, number> = { lead: 0, investigator: 0, critic: 0 };
  for (const a of answers) for (const [k, v] of Object.entries(a.usage.by_role)) byRole[k] = Math.round(((byRole[k] ?? 0) + v) * 10000) / 10000;
  const ratings: Record<string, number> = { rated: 0 };
  const kept = loadAskRatings();
  for (const r of rows) {
    const x = kept.get(r.id);
    if (x) {
      ratings[x.rating] = (ratings[x.rating] ?? 0) + 1;
      ratings.rated!++;
    }
  }
  return {
    model, prompt_version: rows[0]?.prompt_version ?? null, questions: n, by_kind: byKind, status,
    final_program_pass: ratio(n - failures.length, n),
    final_program_failures: failures,
    draft_program_pass: ratio(answers.filter((a) => a.checks.draft_program.length === 0 && a.draft).length, answers.filter((a) => a.draft).length),
    draft_critic_flagged: ratio(answers.filter((a) => a.checks.draft_critic.length > 0).length, answers.filter((a) => a.draft).length),
    sent_back: ratio(sentBack.length, answers.filter((a) => a.draft).length),
    still_unsupported_after_rewrite: ratio(rewritten.filter((a) => a.status === "cut" || a.status === "unconfirmed").length, rewritten.length),
    critic_unparsed: answers.reduce((s, a) => s + a.checks.critic_unparsed, 0),
    out_of_scope_declined: `${outOfScope.filter((r) => DECLINES.test(shownAnswer(r.answer)) || r.answer.final === UNCONFIRMED).length}/${outOfScope.length}`,
    tool_calls_per_question: ratio(answers.reduce((s, a) => s + a.tool_calls, 0), n),
    investigators_per_question: ratio(answers.reduce((s, a) => s + a.investigators.filter((i) => !i.cached && i.relation !== "none").length, 0), n),
    investigations_reused: answers.reduce((s, a) => s + a.investigators.filter((i) => i.cached).length, 0),
    llm_calls_per_question: ratio(answers.reduce((s, a) => s + a.usage.llm_calls, 0), n),
    cost_usd: Math.round(cost * 10000) / 10000,
    cost_per_question: ratio(cost, n) === null ? null : Math.round((cost / n) * 10000) / 10000,
    cost_by_role: byRole,
    seconds_per_question: ratio(answers.reduce((s, a) => s + a.seconds, 0), n),
    ratings,
  };
}

/** A fixed sample (seeded), so re-rendering the sheet shows the same questions. */
export function sample<T>(items: T[], k: number, seed = SHEET_SEED): T[] {
  let state = seed >>> 0;
  const next = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [copy[i], copy[j]] = [copy[j]!, copy[i]!];
  }
  return copy.slice(0, k);
}

const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const STATUS_LABELS: Record<Answer["status"], string> = {
  passed: "草稿一次通过核对",
  rewritten: "审查后重写一次",
  cut: "重写后仍有说法对不上，已删去",
  unconfirmed: "没能确认",
};

export function renderAskSheet(rows: AskRecorded[]): string {
  const picked = sample(rows, SHEET_SIZE).sort((a, b) => a.id.localeCompare(b.id));
  const cards = picked.map((r, i) => {
    const s = r.context.result as { run?: { character?: string; result?: string } };
    const earlier = rows.filter((x) => x.run === r.run && x.id < r.id).map((x) => x.question);
    return `<section class="flag" data-flag="ask-dev/${esc(r.id)}">
  <div class="where">${esc(r.id)} · ${esc(s.run?.character ?? "")} · ${esc(s.run?.result ?? "")} · 对局 ${esc(r.run)}${earlier.length ? ` · 同一段对话里前面问过：${esc(earlier.join(" / "))}` : ""}</div>
  <h3>${i + 1}. ${esc(r.question)}</h3>
  <p class="note">计划：${esc(r.answer.plan)}</p>
  <p class="advice" style="white-space:pre-wrap">${esc(shownAnswer(r.answer))}</p>
  <p class="caveat">${esc(STATUS_LABELS[r.answer.status])}；查询 ${r.answer.tool_calls} 次，调查员 ${r.answer.investigators.length} 个。好的回答应该：${esc(r.expect)}</p>
  <div class="rate" role="group" aria-label="这个回答">
    <button data-v="useful">有用</button><button data-v="not_useful">没用</button><button data-v="wrong">说错了或答非所问</button>
    <input type="text" placeholder="备注（可不填）" maxlength="200">
  </div>
</section>`;
  });
  const first = rows[0];
  const payload = { run_id: "SET:ask-dev", model: first?.model ?? null, prompt_version: first?.prompt_version ?? null };
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>追问评分 · dev</title><style>${STYLE}</style></head><body><main>
<h1>复盘追问评分表（${picked.length} 题）</h1>
<p class="note">从开发集的 ${rows.length} 个固定问题里固定抽出 ${picked.length} 题。每题按你的判断选一个：<b>有用</b>（回答对，而且对你有帮助）、<b>没用</b>（不算错，但没帮助）、<b>说错了或答非所问</b>。可以加一句备注。全部评完后按最下面的“复制评分”，把复制的文字贴回对话（<code>coach ratings import</code> 会存成 <code>coach/eval/ratings/ask-dev.json</code>）。</p>
${cards.join("\n")}
<button id="copy">复制评分</button><span id="copied" hidden>已复制</span>
</main><script>${ratingScript(payload)}</script></body></html>
`;
}

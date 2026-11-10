#!/usr/bin/env node
// coach review <file.run> [--no-llm] [--model M] [--thinking] [--out DIR] [--json] [--no-notes]
//   score one finished run, explain its flags and write the HTML report; the verified
//   flags become player notes (coach/notes/, at most 20) that later reviews and `coach ask` read
// coach eval run <dev|test> [--model M] [--thinking]   review every run of the set and keep the outputs
// coach eval metrics <dev|test> [--model M]            metrics from the kept outputs (no model calls)
// coach ratings import <file>                          keep the text the report's "复制评分" copied
// coach ask <file.run> [--question Q]... [--questions FILE] [--json] [--no-notes] [--out DIR]
//   follow-up questions about one run: a terminal chat, or the given questions answered in order
// coach eval ask run|metrics|sheet dev [--only RUN_OR_ID,...]   the fixed follow-up questions (dev runs only)

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { openSession } from "./ask.js";
import { askMetrics, loadAskRecorded, recordAsk, renderAskSheet } from "./askeval.js";
import { askAll, chat, parseQuestions } from "./chat.js";
import { Engine } from "./engine.js";
import { EVAL_DIR, importRatings, loadRecorded, metrics, record } from "./evaluate.js";
import type { EvalSet } from "./evaluate.js";
import { DEFAULT_MODEL, createClient } from "./llm.js";
import { Trace } from "./trace.js";
import { entityNames } from "./names.js";
import { loadNotes, mergeNotes, notesFile, notesFromReview, saveNotes } from "./notes.js";
import type { Prepass } from "./prepass.js";
import { renderReport } from "./report.js";
import { renderSheet } from "./sheet.js";
import { runReview } from "./review.js";

const USAGE = [
  "usage: coach review <file.run> [--no-llm] [--model M] [--thinking] [--out DIR] [--json] [--no-notes]",
  "       coach eval run|metrics|sheet <dev|test> [--model M] [--thinking]",
  "       coach ratings import <file>",
  "       coach ask <file.run> [--question Q]... [--questions FILE] [--json] [--no-notes] [--out DIR]",
  "       coach eval ask run|metrics|sheet dev [--only RUN_OR_ID,...]",
].join("\n");

async function askCommand(runFile: string, options: { model: string; out: string; json: boolean; notes: boolean; questions: string[] }): Promise<number> {
  const engine = await Engine.start();
  try {
    const run = path.basename(runFile, path.extname(runFile));
    const trace = new Trace(path.join(options.out, "traces", `ask-${run}.jsonl`));
    const session = await openSession(engine, path.resolve(runFile), options.notes ? loadNotes() : []);
    const ctx = { engine, llm: createClient(), model: options.model, tools: await engine.agentTools(), names: entityNames(), trace, thinking: false };
    const answers = options.questions.length
      ? await askAll(ctx, session, options.questions, options.json)
      : await chat(ctx, session);
    if (options.json) console.log(JSON.stringify(answers, null, 1));
    const cost = answers.reduce((s, a) => s + a.usage.cost_usd, 0);
    process.stderr.write(`${answers.length} answers, $${cost.toFixed(4)}; trace: ${path.resolve(options.out, "traces", `ask-${run}.jsonl`)}\n`);
    return 0;
  } finally {
    await engine.close();
  }
}

async function evalAsk(action: string | undefined, model: string, only: string | undefined): Promise<number> {
  if (action === "run") await recordAsk(model, only ? new Set(only.split(",")) : undefined);
  else if (action === "sheet") {
    const file = path.join(EVAL_DIR, "rating-sheet-ask-dev.html");
    writeFileSync(file, renderAskSheet(loadAskRecorded(model)), "utf-8");
    console.log(`rating sheet: ${file}`);
    return 0;
  } else if (action !== "metrics") {
    console.error(USAGE);
    return 2;
  }
  const result = askMetrics(model, loadAskRecorded(model));
  const file = path.join(EVAL_DIR, "metrics", `ask-dev-${model}.json`);
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(result, null, 1) + "\n", "utf-8");
  console.log(JSON.stringify(result, null, 1));
  return 0;
}

function printPrepass({ summary, decisions, flags }: Prepass): void {
  const s = summary;
  const rate = s.agreement.rate === null ? "-" : `${Math.round(s.agreement.rate * 100)}%`;
  console.log(`${s.character} A${s.ascension ?? 0} ${s.build_id ?? "?"} ${s.win ? "胜利" : `败于 ${s.killed_by ?? "?"}`}`);
  console.log(
    `scored ${JSON.stringify(s.decisions_scored)}, skipped ${JSON.stringify(s.decisions_skipped)}, ` +
      `engine agrees ${s.agreement.agree}/${s.agreement.judged} (${rate})`,
  );
  console.log(`tier 1: ${flags.length} shown of ${s.tier1.length}; tier 2: ${s.tier2.length}`);
  for (const f of flags) {
    const name = (id: string | null) => f.options.find((o) => o.id === id)?.name ?? id ?? "-";
    console.log(`  ${f.id}  chose ${name(f.chosen)}, engine ${name(f.engine_top)} (+${f.gap}, ${f.confidence})`);
  }
  for (const d of decisions.decisions) {
    const mark = d.tier === 1 ? "!" : d.tier === 2 ? "?" : d.tier === 0 ? " " : "-";
    console.log(`${mark} ${d.id.padEnd(9)} ${d.chosen_name ?? d.chosen ?? "-"} / ${d.engine_top_name ?? "-"}`);
  }
}

async function main(argv: string[]): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      json: { type: "boolean", default: false },
      "no-llm": { type: "boolean", default: false },
      model: { type: "string", default: DEFAULT_MODEL },
      thinking: { type: "boolean", default: false },
      out: { type: "string", default: "out" },
      "no-notes": { type: "boolean", default: false },
      question: { type: "string", multiple: true, default: [] },
      questions: { type: "string" },
      only: { type: "string" },
    },
  });
  const [command, runFile] = positionals;
  if (command === "eval" && runFile === "ask") {
    // Part B is evaluated on the development runs only.
    if (positionals[3] !== "dev") {
      console.error("coach eval ask works on the dev set only\n" + USAGE);
      return 2;
    }
    return evalAsk(positionals[2], values.model, values.only);
  }
  if (command === "ask" && runFile) {
    const questions = [...values.question, ...(values.questions ? parseQuestions(readFileSync(values.questions, "utf-8")) : [])];
    return askCommand(runFile, { model: values.model, out: values.out, json: values.json, notes: !values["no-notes"], questions });
  }
  if (command === "eval" && (positionals[2] === "dev" || positionals[2] === "test")) {
    const set = positionals[2] as EvalSet;
    if (runFile === "run") await record(set, values.model, values.thinking);
    else if (runFile === "sheet") {
      const file = path.join(EVAL_DIR, `rating-sheet-${set}.html`);
      writeFileSync(file, renderSheet(loadRecorded(set, values.model)), "utf-8");
      console.log(`rating sheet: ${file}`);
      return 0;
    } else if (runFile !== "metrics") {
      console.error(USAGE);
      return 2;
    }
    const result = metrics(set, values.model, loadRecorded(set, values.model));
    const file = path.join(EVAL_DIR, "metrics", `${set}-${values.model}.json`);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify(result, null, 1) + "\n", "utf-8");
    console.log(JSON.stringify(result, null, 1));
    return 0;
  }
  if (command === "ratings" && runFile === "import" && positionals[2]) {
    console.log(`kept ${importRatings(readFileSync(positionals[2], "utf-8"))}`);
    return 0;
  }
  if (command !== "review" || !runFile) {
    console.error(USAGE);
    return 2;
  }
  const engine = await Engine.start();
  try {
    const run = path.basename(runFile, path.extname(runFile));
    const review = await runReview(engine, path.resolve(runFile), {
      model: values["no-llm"] ? null : values.model,
      thinking: values.thinking,
      traceFile: path.join(values.out, "traces", `${run}.jsonl`),
      notes: values["no-notes"] ? [] : loadNotes(),
    });
    if (!values["no-notes"] && review.explained.length > 0) {
      const fresh = notesFromReview(review.pre, review.explained, entityNames());
      saveNotes(mergeNotes(loadNotes(), fresh));
      if (!values.json) console.log(`player notes: ${fresh.length} from this run (${notesFile()})`);
    }
    if (values.json) {
      console.log(JSON.stringify({ ...review.pre, explained: review.explained, summary: review.summary, meta: review.meta }, null, 1));
      return 0;
    }
    printPrepass(review.pre);
    for (const r of review.explained) {
      console.log(`${r.flag}: ${r.status} (tools ${r.toolCalls}, repairs ${r.repairs}, $${r.usage.cost_usd.toFixed(4)})${r.status === "verified" ? "" : " " + r.errors.join("; ")}`);
    }
    const file = path.join(values.out, "reports", `${run}.html`);
    mkdirSync(path.dirname(file), { recursive: true });
    writeFileSync(file, renderReport(review.pre, review.explained, review.summary, review.meta), "utf-8");
    console.log(`summary (${review.summary.source}): ${review.summary.text}`);
    console.log(`report: ${path.resolve(file)}  ($${review.meta.cost_usd.toFixed(4)}, ${review.meta.seconds.toFixed(0)} s)`);
    return 0;
  } finally {
    await engine.close();
  }
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (error: unknown) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  },
);

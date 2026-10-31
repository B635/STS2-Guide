#!/usr/bin/env node
// coach review <file.run> [--no-llm] [--model M] [--thinking] [--out DIR] [--json]
//   score one finished run, explain its flags and write the HTML report

import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { Engine } from "./engine.js";
import { DEFAULT_MODEL } from "./llm.js";
import type { Prepass } from "./prepass.js";
import { renderReport } from "./report.js";
import { runReview } from "./review.js";

const USAGE = "usage: coach review <file.run> [--no-llm] [--model M] [--thinking] [--out DIR] [--json]";

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
    },
  });
  const [command, runFile] = positionals;
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
    });
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

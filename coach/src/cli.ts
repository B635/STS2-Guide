#!/usr/bin/env node
// coach review <file.run> [--json]   score one finished run and list its flags

import path from "node:path";
import { parseArgs } from "node:util";
import { Engine } from "./engine.js";
import { prepass } from "./prepass.js";
import type { Prepass } from "./prepass.js";

const USAGE = "usage: coach review <file.run> [--json]";

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
    options: { json: { type: "boolean", default: false } },
  });
  const [command, runFile] = positionals;
  if (command !== "review" || !runFile) {
    console.error(USAGE);
    return 2;
  }
  const engine = await Engine.start();
  try {
    const result = await prepass(engine, path.resolve(runFile));
    if (values.json) console.log(JSON.stringify(result, null, 1));
    else printPrepass(result);
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

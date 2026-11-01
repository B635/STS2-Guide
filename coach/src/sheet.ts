// One rating page for a whole evaluation set: every shown flag of every run,
// so the rater copies once. Keys are "<run>/<flag>".

import type { Recorded } from "./evaluate.js";
import { STYLE, flagCard, ratingScript } from "./report.js";
import { characterName } from "./names.js";

const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

export function renderSheet(rows: Recorded[]): string {
  let n = 0;
  const parts: string[] = [];
  for (const row of rows) {
    const results = new Map(row.explained.map((r) => [r.flag, r]));
    const shown = row.pre.flags.filter((d) => results.get(d.id)?.status === "verified");
    if (shown.length === 0) continue;
    const s = row.pre.summary;
    parts.push(`<h2>${esc(characterName(s.character))} · ${esc(s.build_id ?? "")} · 对局 ${esc(row.run)}</h2>`);
    for (const d of shown) parts.push(flagCard(d, results.get(d.id), n++, `${row.run}/${d.id}`));
  }
  const first = rows[0];
  const payload = { run_id: `SET:${first?.set ?? "?"}`, model: first?.model ?? null, prompt_version: first?.prompt_version ?? null };
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>复盘评分 · ${esc(first?.set ?? "")}</title><style>${STYLE}</style></head><body><main>
<h1>复盘教练评分表（${n} 条）</h1>
<p class="note">每条按你的判断选一个：<b>有用</b>（说得对，而且对你有帮助）、<b>没用</b>（说得不算错，但没帮助）、<b>引擎错了</b>（你认为这里你的选择更好）。可以加一句备注。全部评完后按最下面的“复制评分”，把复制的文字贴回对话。</p>
${parts.join("\n")}
<button id="copy">复制评分</button><span id="copied" hidden>已复制</span>
</main><script>${ratingScript(payload)}</script></body></html>
`;
}

// The review as one self-contained HTML file (simplified Chinese, like the
// game's names). Written by the program from the engine's records and the
// verified findings; the model never writes HTML.

import type { FlagResult } from "./agent.js";
import { KIND_LABELS, factorLabel, notStrictLabel, skippedLabel } from "./labels.js";
import { characterName } from "./names.js";
import type { Prepass } from "./prepass.js";
import type { DecisionDetail } from "./schemas.js";
import type { Summary } from "./summary.js";

export const CURRENT_BUILD = "v0.111.0";

export interface ReportMeta {
  model: string | null;
  cost_usd: number;
  seconds: number;
  generated_at: string;
}

const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

function optionName(detail: DecisionDetail, id: string | null): string {
  return detail.options.find((o) => o.id === id)?.name ?? id ?? "-";
}

export function headline(detail: DecisionDetail): string {
  return `引擎认为「${optionName(detail, detail.engine_top)}」比你选的「${optionName(detail, detail.chosen)}」高 ${detail.gap} 分`;
}

function flagCard(detail: DecisionDetail, result: FlagResult | undefined, index: number): string {
  const f = result?.finding;
  if (!f) return "";
  const main = f.reasons[0] ? `，主要因为${factorLabel(f.reasons[0].factor)}` : "";
  const reasons = f.reasons.map((r) => `<li><span class="tag">${esc(factorLabel(r.factor))}</span>${esc(r.text)}</li>`).join("");
  const partly = detail.options.filter((o) => !o.fully_scored).map((o) => o.name);
  const caveat = partly.length
    ? `<p class="caveat">引擎没能完整计算：${esc(partly.join("、"))}（部分效果未计入，分数是下限）。</p>`
    : "";
  return `<section class="flag" data-flag="${esc(detail.id)}">
  <div class="where">第 ${detail.act} 幕 · 第 ${detail.floor} 层 · ${esc(KIND_LABELS[detail.kind] ?? detail.kind)} · 生命 ${detail.hp}/${detail.max_hp} · 金币 ${detail.gold}</div>
  <h3>${index + 1}. ${esc(headline(detail))}${esc(main)}</h3>
  <ul class="reasons">${reasons}</ul>
  ${f.what_if ? `<p class="whatif"><b>如果当时…</b>${esc(f.what_if.text)}</p>` : ""}
  <p class="advice"><b>下次可以注意：</b>${esc(f.advice)}</p>
  ${caveat}
  <div class="rate" role="group" aria-label="这条建议">
    <button data-v="useful">有用</button><button data-v="not_useful">没用</button><button data-v="engine_wrong">引擎错了</button>
    <input type="text" placeholder="备注（可不填）" maxlength="200">
  </div>
</section>`;
}

export function renderReport(pre: Prepass, explained: FlagResult[], summary: Summary, meta: ReportMeta): string {
  const s = pre.summary;
  const results = new Map(explained.map((r) => [r.flag, r]));
  const shown = pre.flags.filter((d) => results.get(d.id)?.status === "verified");
  const dropped = explained.filter((r) => r.status !== "verified").map((r) => r.flag);
  const rate = s.agreement.rate === null ? "-" : `${Math.round(s.agreement.rate * 100)}%`;
  const skippedCount = Object.values(s.decisions_skipped).reduce((a, b) => a + b, 0);
  const scoredCount = Object.values(s.decisions_scored).reduce((a, b) => a + b, 0);
  const versionNote =
    s.build_id && s.build_id !== CURRENT_BUILD
      ? `<p class="note">这局是 ${esc(s.build_id)} 版打的，引擎数据是 ${CURRENT_BUILD} 版，个别卡牌数值可能已经改过。</p>`
      : "";
  const notShown = s.tier1.length - pre.flags.length;
  const rows = pre.decisions.decisions
    .map((d) => {
      const tier = d.tier === 1 ? "值得想一想" : d.tier === 2 ? `不同（${notStrictLabel(d.not_strict_because)}）` : d.tier === 0 ? "一致" : "未判断";
      return `<tr class="t${d.tier ?? "x"}"><td>${d.floor}</td><td>${esc(KIND_LABELS[d.kind])}</td><td>${esc(d.chosen_name ?? d.chosen ?? "-")}</td><td>${esc(d.engine_top_name ?? "-")}</td><td>${d.gap ?? ""}</td><td>${esc(tier)}</td></tr>`;
    })
    .join("\n");
  const skipped = pre.decisions.skipped
    .map((x) => `<li>第 ${x.floor} 层${esc(KIND_LABELS[x.kind] ?? x.kind)}：${esc(skippedLabel(x.reason))}</li>`)
    .join("");
  const payload = { run_id: s.run_id, model: meta.model, generated_at: meta.generated_at, flags: shown.map((d) => d.id) };

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>复盘 · ${esc(characterName(s.character))} · ${esc(s.run_id)}</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1b18;--muted:#6b665e;--card:#fff;--line:#e4e0d8;--accent:#9a4d1f;--tag:#f1e7dc;--ok:#2f6b3a}
@media (prefers-color-scheme:dark){:root{--bg:#171614;--fg:#ece8e1;--muted:#a39d93;--card:#211f1c;--line:#36332e;--accent:#e09a62;--tag:#3a2c20;--ok:#7fc28c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 system-ui,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px}
h3{font-size:16px;margin:4px 0 8px}.meta,.where,.note,.caveat,footer{color:var(--muted);font-size:13px}
.summary{font-size:16px;padding:12px 14px;border-left:3px solid var(--accent);background:var(--card)}
.flag{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0}
.reasons{padding-left:18px;margin:6px 0}.tag{display:inline-block;background:var(--tag);border-radius:4px;padding:0 6px;margin-right:6px;font-size:12px}
.rate{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.rate button{border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;padding:3px 10px;cursor:pointer}
.rate button.on{background:var(--accent);color:#fff;border-color:var(--accent)}.rate input{flex:1;min-width:140px;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:6px;padding:3px 8px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}tr.t1 td{font-weight:600}
.table{overflow-x:auto}#copy{margin-top:8px;padding:6px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
#copied{color:var(--ok);margin-left:8px}
</style></head><body><main>
<h1>${esc(characterName(s.character))} · 进阶 ${s.ascension ?? 0} · ${s.win ? "胜利" : `败于 ${esc(s.killed_by ?? "?")}`}</h1>
<p class="meta">${esc(s.build_id ?? "")} · 评了 ${scoredCount} 个决定，${skippedCount} 个没法评 · 引擎和你的选择一致 ${s.agreement.agree}/${s.agreement.judged}（${rate}）</p>
<p class="note">这些是一个确定性评分引擎的看法，不是定论：在你过去的对局里，引擎和你的选择大约只有一半一致。下面只列出引擎有把握、而且差距明显的决定。</p>
${versionNote}
<p class="summary">${esc(summary.text)}</p>
<h2>值得想一想（${shown.length}）</h2>
${shown.map((d, i) => flagCard(d, results.get(d.id), i)).join("\n") || "<p>这局没有需要特别注意的决定。</p>"}
${notShown > 0 ? `<p class="note">另外还有 ${notShown} 个类似的决定没有展开，见附录。</p>` : ""}
${dropped.length ? `<p class="note">${dropped.length} 条说明没有通过核对，没有列出（${esc(dropped.join("、"))}）。</p>` : ""}
${shown.length ? `<button id="copy">复制评分</button><span id="copied" hidden>已复制，贴回聊天就好</span>` : ""}
<h2>附录：全部决定</h2>
<p class="note">休息点的升级价值只算得到下限，目前不会被列为“值得想一想”。商店只比较买或离开。</p>
<div class="table"><table><thead><tr><th>层</th><th>类型</th><th>你的选择</th><th>引擎首选</th><th>分差</th><th>结果</th></tr></thead><tbody>
${rows}
</tbody></table></div>
${skipped ? `<p class="note">没法评的决定：</p><ul class="note">${skipped}</ul>` : ""}
<footer><p>STS2 Guide 复盘教练 · ${esc(meta.generated_at)} · ${meta.model ? `模型 ${esc(meta.model)}，费用约 $${meta.cost_usd.toFixed(4)}，` : "未使用模型，"}用时 ${meta.seconds.toFixed(0)} 秒 · 每条说明都已和引擎的工具结果核对</p></footer>
</main>
<script>
const REVIEW = ${JSON.stringify(payload).replace(/</g, "\\u003c")};
const ratings = {};
document.querySelectorAll(".flag").forEach((card) => {
  const flag = card.dataset.flag, note = card.querySelector("input");
  card.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    card.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    ratings[flag] = { rating: b.dataset.v, note: note.value };
  }));
  note.addEventListener("input", () => { if (ratings[flag]) ratings[flag].note = note.value; });
});
const copy = document.getElementById("copy");
if (copy) copy.addEventListener("click", async () => {
  const text = "STS2-COACH-RATINGS " + JSON.stringify({ ...REVIEW, ratings });
  try { await navigator.clipboard.writeText(text); }
  catch { const t = document.createElement("textarea"); t.value = text; document.body.appendChild(t); t.select(); document.execCommand("copy"); t.remove(); }
  document.getElementById("copied").hidden = false;
});
</script></body></html>
`;
}

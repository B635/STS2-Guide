// The lead loop with a scripted model: plan, parallel investigators, budgets,
// the program check and the critic, the rewrite and the cuts.
import { describe, expect, it } from "vitest";
import { MAX_HISTORY, MAX_INVESTIGATORS, MAX_LEAD_TOOL_CALLS, UNCONFIRMED, ask, checkAnswer, newSession, sentences } from "../src/ask.js";
import type { AskEvent } from "../src/ask.js";
import { Trace } from "../src/trace.js";
import { fakeEngine, fakeLlm, TOOL_SPECS } from "./fakes.js";
import type { Scripted } from "./fakes.js";
import { DECISIONS, ENGINE_ANSWERS, NAMES, SUMMARY } from "./fixtures.js";

type Req = { messages: Array<{ role: string; content?: string | null }>; stream?: boolean; response_format?: unknown; tools?: Array<{ function: { name: string } }>; tool_choice?: unknown };

const FINDINGS: Record<string, unknown> = {
  "F06-card": {
    decision_id: "F06-card",
    reasons: [{ factor: "deck_fitness:defense", evidence: "T1", text: "血墙把每回合防御从 6.9 提到 8.9，需求 8.5。" }],
    advice: "防御低于需求 8.5 时，先补防御牌。",
  },
  "F08-card": {
    decision_id: "F08-card",
    reasons: [{ factor: "deck_fitness:defense", evidence: "T1", text: "耸肩无视把每回合防御从 6.4 提到 7.6。" }],
    advice: "防御低于需求 8.9 时，耸肩无视这类牌值得拿。",
  },
};

const GOOD_DRAFT = "第 6 层引擎更看好血墙：拿了它防御每回合 6.9 → 8.9，需求 8.5。你选的剑柄打击防御反而降到 6.4。";

interface Script {
  lead: (req: Req, turn: number) => Scripted;
  critic?: (draft: string, call: number) => Array<{ sentence: string; reason: string }>;
  investigator?: (decision: string) => Promise<void> | void;
}

function setup(script: Script) {
  let leadTurn = 0, criticCall = 0;
  const { llm, requests } = fakeLlm(async (raw) => {
    const req = raw as unknown as Req;
    if (req.response_format) {
      const user = String(req.messages[1]?.content ?? "");
      const draft = user.split("回答草稿：\n")[1]?.split("\n\n证据：")[0] ?? "";
      return { content: JSON.stringify({ unsupported: script.critic?.(draft, criticCall++) ?? [] }) };
    }
    if (req.tools?.some((t) => t.function.name === "submit_finding")) {
      const decision = /要解释的决定：(F\d+-\w+)/.exec(String(req.messages[1]?.content))![1]!;
      await script.investigator?.(decision);
      return { calls: [{ name: "submit_finding", args: FINDINGS[decision] }] };
    }
    return script.lead(req, leadTurn++);
  });
  const { engine, calls } = fakeEngine(ENGINE_ANSWERS);
  const trace = new Trace();
  const ctx = { engine, llm, model: "deepseek-flash", tools: TOOL_SPECS, names: NAMES, trace, thinking: false };
  const session = newSession(SUMMARY, DECISIONS, []);
  const events: AskEvent[] = [];
  return { ctx, session, events, requests, calls, trace, onEvent: (e: AskEvent) => events.push(e) };
}

const plan = (text = "先看第 6 层和第 8 层的选牌。", decisions = ["F06-card", "F08-card"]): Scripted => ({ calls: [{ name: "plan", args: { text, decisions } }] });

describe("lead agent", () => {
  it("plans, investigates several decisions in parallel, and answers when both checks pass", async () => {
    // Each investigator waits until both have started: run one after the other, this would time out.
    let started = 0;
    let release!: () => void;
    const both = new Promise<void>((r) => (release = r));
    const t = setup({
      lead: (_req, turn) =>
        turn === 0 ? plan()
        : turn === 1 ? { calls: [{ name: "investigate", args: { decision_ids: ["F06-card", "F08-card"], focus: "为什么" } }] }
        : { content: GOOD_DRAFT },
      investigator: async () => {
        if (++started === 2) release();
        await both;
      },
    });
    const answer = await ask(t.ctx, t.session, "第 6 层为什么引擎不同意我？", t.onEvent);
    expect(answer.status).toBe("passed");
    expect(answer.final).toBe(GOOD_DRAFT);
    expect(answer.plan).toBe("先看第 6 层和第 8 层的选牌。");
    expect(answer.investigators.map((i) => [i.decision, i.status, i.relation])).toEqual([["F06-card", "verified", "flag"], ["F08-card", "verified", "agree"]]);
    expect(answer.evidence.map((e) => e.tool)).toEqual(["investigate", "investigate"]);
    // The plan turn is forced; the draft was streamed.
    const leadRequests = t.requests.filter((r) => r.stream);
    expect(leadRequests[0]!.tool_choice).toEqual({ type: "function", function: { name: "plan" } });
    expect(t.events.filter((e) => e.type === "draft").map((e) => (e as { delta: string }).delta).join("")).toBe(GOOD_DRAFT);
    const names = t.trace.spans.map((s) => s.name);
    for (const n of ["plan", "llm.lead", "investigate", "check.program", "llm.critic", "critic.verdict", "ask.done"]) expect(names).toContain(n);
  }, 2000);

  it("sends a draft with a made-up number back once; a clean rewrite is the answer", async () => {
    const t = setup({
      lead: (_req, turn) =>
        turn === 0 ? plan()
        : turn === 1 ? { calls: [{ name: "investigate", args: { decision_ids: ["F06-card"], focus: "为什么" } }] }
        : turn === 2 ? { content: "第 6 层引擎更看好血墙，防御每回合多 5 点。" }
        : { content: GOOD_DRAFT },
    });
    const answer = await ask(t.ctx, t.session, "第 6 层怎么回事？", t.onEvent);
    expect(answer.checks.draft_program[0]).toContain("the number 5");
    expect(answer.status).toBe("rewritten");
    expect(answer.final).toBe(GOOD_DRAFT);
    const rewriteRequest = t.requests.filter((r) => r.stream).at(-1)!;
    expect(rewriteRequest.tool_choice).toBe("none");
    expect(String(rewriteRequest.messages.at(-1)!.content)).toContain("防御每回合多 5 点");
  });

  it("cuts what the critic still quotes after the rewrite and says so", async () => {
    const claim = "你输掉这局就是因为第 6 层没拿血墙。";
    const t = setup({
      lead: (_req, turn) =>
        turn === 0 ? plan()
        : turn === 1 ? { calls: [{ name: "investigate", args: { decision_ids: ["F06-card"], focus: "为什么" } }] }
        : { content: GOOD_DRAFT + claim },
      critic: () => [{ sentence: claim.replace("。", ""), reason: "引擎不模拟战斗" }],
    });
    const answer = await ask(t.ctx, t.session, "我为什么输了？", t.onEvent);
    expect(answer.checks.draft_critic).toHaveLength(1);
    expect(answer.status).toBe("cut");
    expect(answer.final).toBe(GOOD_DRAFT);
    expect(answer.checks.cut).toEqual([claim]);
    expect(answer.notice).toContain("1 句");
  });

  it("answers that it could not confirm when nothing survives", async () => {
    const t = setup({
      lead: (_req, turn) => (turn === 0 ? plan("先看概况。", []) : { content: "你在第 99 层打出了 120 点伤害。" }),
    });
    const answer = await ask(t.ctx, t.session, "我最高打出多少伤害？", t.onEvent);
    expect(answer.status).toBe("unconfirmed");
    expect(answer.final).toBe(UNCONFIRMED);
    expect(answer.checks.rewrite_program.length).toBeGreaterThan(0);
  });

  it("replaces a plan that does not hold up with the program's own", async () => {
    const t = setup({ lead: (_req, turn) => (turn === 0 ? plan("先看第 77 层。", ["F06-card"]) : { content: "引擎的结果里看不出来。" }) });
    const answer = await ask(t.ctx, t.session, "？", t.onEvent);
    expect(answer.plan_source).toBe("program");
    expect(answer.plan).toBe("先看第 6 层的选牌。");
  });

  it("keeps to the tool and investigator budgets", async () => {
    const t = setup({
      lead: (req, turn) => {
        if (turn === 0) return plan();
        if (turn === 1) return { calls: [{ name: "investigate", args: { decision_ids: ["F06-card", "F08-card", "F06-card2", "F07-card", "F11-card", "F12-card"], focus: "x" } }] };
        if (req.tool_choice === "none") return { content: "引擎的结果里看不出来。" };
        return { calls: [{ name: "get_deck", args: { decision_id: "F06-card" } }, { name: "get_deck", args: { decision_id: "F08-card" } }] };
      },
    });
    const answer = await ask(t.ctx, t.session, "整局怎么样？", t.onEvent);
    expect(answer.tool_calls).toBe(MAX_LEAD_TOOL_CALLS);
    expect(t.calls.filter((c) => c.name === "get_decision").length).toBeLessThanOrEqual(MAX_INVESTIGATORS);
    const reports = answer.evidence.filter((e) => e.tool === "investigate").map((e) => e.result as { status: string; note?: string });
    expect(reports.filter((r) => r.note?.includes("budget"))).toHaveLength(2);
    expect(answer.final).toBe("引擎的结果里看不出来。");
  });

  it("reuses an earlier investigation and keeps only the last turns", async () => {
    const t = setup({
      lead: (req, turn) => {
        if (req.tool_choice && typeof req.tool_choice === "object") return plan();
        const last = req.messages.at(-1)!;
        if (last.role === "user") return { calls: [{ name: "investigate", args: { decision_ids: ["F06-card"], focus: "x" } }] };
        return { content: GOOD_DRAFT };
      },
    });
    for (let i = 0; i < MAX_HISTORY + 2; i++) await ask(t.ctx, t.session, `问题 ${i}`, t.onEvent);
    expect(t.session.turns).toHaveLength(MAX_HISTORY);
    expect(t.session.turns[0]!.question).toBe("问题 2");
    // One investigator ran for F06-card; the next questions reused its report while it was in memory.
    const runs = t.trace.spans.filter((s) => s.name === "investigate").length;
    expect(runs).toBeLessThan(MAX_HISTORY + 2);
    const lastLead = t.requests.filter((r) => r.stream).at(-1)!;
    expect(lastLead.messages.filter((m) => m.role === "user" && String(m.content).startsWith("问题")).length).toBe(MAX_HISTORY + 1);
  });
});

describe("program check of an answer", () => {
  it("splits sentences and keeps their punctuation", () => {
    expect(sentences("第一句。第二句！\n第三句")).toEqual(["第一句。", "第二句！\n", "第三句"]);
  });

  it("flags only the sentences with numbers or names the evidence lacks", () => {
    const evidence = [{ id: "E1", tool: "x", result: { a: "血墙 8.5" } }];
    expect(checkAnswer("血墙需求 8.5。完美打击更好。", evidence, NAMES)).toEqual([
      { sentence: "完美打击更好。", errors: ['answer: "完美打击" is not in the evidence of this conversation'] },
    ]);
  });
});

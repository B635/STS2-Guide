// The lead loop with a scripted model: plan, parallel investigators, budgets,
// the program check and the critic, the rewrite and the cuts.
import { describe, expect, it } from "vitest";
import {
  MAX_HISTORY,
  MAX_INVESTIGATORS,
  MAX_LEAD_TOOL_CALLS,
  UNCONFIRMED,
  ask,
  checkAnswer,
  investigationHeadline,
  newSession,
  overview,
  sentences,
  withoutCut,
} from "../src/ask.js";
import type { AskEvent } from "../src/ask.js";
import type { DecisionDetail, DecisionList } from "../src/schemas.js";
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
      const issues = (script.critic?.(draft, criticCall++) ?? []).map((i) => ({ ...i, verdict: "unsupported" }));
      // A doubt the critic clears itself never counts.
      return { content: JSON.stringify({ doubts: [...issues, { sentence: "第 6 层引擎更看好血墙", reason: "E1 有", verdict: "supported" }] }) };
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

  it("states a plan itself when the model skips the forced plan call", async () => {
    const t = setup({
      lead: (_req, turn) => (turn === 0 ? { calls: [{ name: "get_deck", args: { decision_id: "F08-card" } }] } : { content: "引擎的结果里看不出来。" }),
    });
    const answer = await ask(t.ctx, t.session, "第 8 层的牌组？", t.onEvent);
    expect(answer.plan_source).toBe("program");
    expect(answer.plan).toBe("先看第 8 层的选牌。");
    expect(t.events[0]).toEqual({ type: "plan", text: "先看第 8 层的选牌。" });
    expect(answer.tool_calls).toBe(1);
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

  it("renews cached investigation evidence so turn 6 still validates after turn 1 drops out", async () => {
    // Turn 1 investigates F06-card (adding 6.9 / 8.9 / 8.5 to evidence).
    // Turns 2-4 ask unrelated questions without investigating F06-card.
    // Turn 5 reuses cached F06-card (must call keep() so Turn 5 owns a fresh evidence ID).
    // Turn 6 cites 6.9 / 8.9 / 8.5 directly without calling tools -> must pass checkAnswer!
    let qIdx = 0;
    let qStep = 0;
    const t = setup({
      lead: (req) => {
        if (req.tool_choice && typeof req.tool_choice === "object") return plan("先看概况。", ["F06-card"]);
        qStep++;
        if ((qIdx === 1 || qIdx === 5) && qStep === 1) {
          return { calls: [{ name: "investigate", args: { decision_ids: ["F06-card"], focus: "防御" } }] };
        }
        if (qIdx === 1 || qIdx === 5 || qIdx === 6) {
          return { content: GOOD_DRAFT };
        }
        return { content: "引擎的结果里看不出来。" };
      },
    });
    for (let i = 1; i <= 5; i++) {
      qIdx = i;
      qStep = 0;
      const ans = await ask(t.ctx, t.session, `追问 ${i}`, t.onEvent);
      expect(ans.status).toBe("passed");
      if (i === 5) {
        expect(ans.investigators[0]?.cached).toBe(true);
        expect(ans.evidence.some((e) => e.tool === "investigate")).toBe(true);
      }
    }
    // Turn 1 has now been evicted from the 4-turn sliding window, while Turn 5 remains.
    expect(t.session.turns).toHaveLength(MAX_HISTORY);
    expect(t.session.turns[0]!.question).toBe("追问 2");
    expect(t.session.reports.has("F06-card")).toBe(true);

    qIdx = 6;
    qStep = 0;
    const turn6 = await ask(t.ctx, t.session, "第 6 层刚才说的防御数字是多少？", t.onEvent);
    expect(turn6.checks.draft_program).toEqual([]);
    expect(turn6.status).toBe("passed");
    expect(turn6.final).toBe(GOOD_DRAFT);
  });

  it("never claims player and engine bought the same shop item when tier===0 but chosen!==engine_top", () => {
    const shopDetail: DecisionDetail = {
      id: "F15-shop",
      kind: "shop",
      floor: 15,
      act: 1,
      hp: 55,
      max_hp: 80,
      gold: 210,
      engine_top: "card:2:OFFERING",
      confidence: "medium",
      chosen: "removal",
      tier: 0,
      gap: 4.2,
      pick_margin: 1,
      options: [
        {
          id: "card:2:OFFERING",
          kind: "card",
          entity_id: "OFFERING",
          name: "祭品",
          score: 48.2,
          deck_score: 48.2,
          rank: 1,
          eligible: true,
          fully_scored: true,
          gaps: [],
          factors: [{ code: "merchant:deck", delta: 5.2, message: "买祭品" }],
          dimensions: {},
        },
        {
          id: "removal",
          kind: "other",
          entity_id: null,
          name: "删牌",
          score: 44.0,
          deck_score: 44.0,
          rank: 2,
          eligible: true,
          fully_scored: true,
          gaps: [],
          factors: [{ code: "merchant:deck", delta: 1.0, message: "删打击" }],
          dimensions: {},
        },
      ],
    };
    const head = investigationHeadline(shopDetail, { relation: "disagree", pair: { a: "card:2:OFFERING", b: "removal" } });
    expect(head).not.toContain("都选了「祭品」");
    expect(head).toContain("都在这家商店选择了消费而非离开，具体购买项不同");
    expect(head).toContain("你选了「删牌」");
    expect(head).toContain("引擎首选「祭品」");

    const shopList: DecisionList = {
      decisions: [
        {
          id: "F15-shop",
          kind: "shop",
          floor: 15,
          act: 1,
          chosen: "removal",
          chosen_name: "删牌",
          engine_top: "card:2:OFFERING",
          engine_top_name: "祭品",
          confidence: "medium",
          tier: 0,
          gap: 4.2,
          not_strict_because: [],
        },
        {
          id: "F22-shop",
          kind: "shop",
          floor: 22,
          act: 2,
          chosen: "bought_unscored",
          chosen_name: null,
          engine_top: "card:1:SOW",
          engine_top_name: "播种",
          confidence: "medium",
          tier: 0,
          gap: null,
          not_strict_because: [],
        },
      ],
      skipped: [],
    };
    const e0 = overview(SUMMARY, shopList, []) as { decisions: Array<{ player: string; engine: string; verdict: string }> };
    expect(e0.decisions[0]!.player).toBe("删牌");
    expect(e0.decisions[0]!.engine).toBe("祭品");
    expect(e0.decisions[0]!.verdict).toContain("都在这家商店选择了消费而非离开，具体购买项不同");
    expect(e0.decisions[1]!.player).toBe("已购买（未计分项）");
    expect(e0.decisions[1]!.verdict).toContain("都在这家商店选择了消费而非离开，具体购买项不同");
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

  it("cuts a list item whole, and single sentences elsewhere", () => {
    const list = sentences("开头一句。\n- 第 21 层：甲。乙。\n- 第 22 层：丙。\n结尾一句。另一句。");
    expect(withoutCut(list, new Set(["- 第 21 层：甲。", "另一句。"])).join("")).toBe("开头一句。\n- 第 22 层：丙。\n结尾一句。");
  });

  it("rejects internal IDs in what the player reads", () => {
    const evidence = [{ id: "E1", tool: "x", result: { a: "DEFEND_NECROBINDER F15-card 15" } }];
    expect(checkAnswer("只算了 DEFEND_NECROBINDER。", evidence, NAMES)[0]!.errors).toEqual(["answer: write names, not the internal ID DEFEND_NECROBINDER"]);
    expect(checkAnswer("见F15-card。", evidence, NAMES)).toHaveLength(1);
    expect(checkAnswer("引擎的 what_if 里试过。", evidence, NAMES)).toHaveLength(1);
    expect(checkAnswer("第 15 层的选牌。", evidence, NAMES)).toEqual([]);
  });
});

import { describe, expect, it } from "vitest";
import { explainFlag, investigate, subjectOf } from "../src/agent.js";
import type { Subject } from "../src/agent.js";
import { separatingFactors } from "../src/verifier.js";
import { Trace } from "../src/trace.js";
import { fakeEngine, fakeLlm, TOOL_SPECS } from "./fakes.js";
import { AGREED, ENGINE_ANSWERS, FLAG, NAMES, REST, SUMMARY } from "./fixtures.js";

const GOOD = {
  decision_id: "F06-card",
  reasons: [{ factor: "deck_fitness:defense", evidence: "T1", text: "血墙把每回合防御从 6.9 提到 8.9，需求 8.5。" }],
  advice: "防御低于需求 8.5 时，先补防御牌。",
};

function context(respond: Parameters<typeof fakeLlm>[0]) {
  const { llm, requests } = fakeLlm(respond);
  const { engine, calls } = fakeEngine(ENGINE_ANSWERS);
  const trace = new Trace();
  return { ctx: { engine, llm, model: "deepseek-flash", tools: TOOL_SPECS, names: NAMES, trace, thinking: false }, requests, calls, trace };
}

describe("flag agent (part A)", () => {
  it("sends the same request for a tier-1 flag as before the investigator was generalized", async () => {
    const { ctx, requests } = context(() => ({ calls: [{ name: "submit_finding", args: GOOD }] }));
    const result = await explainFlag(ctx, SUMMARY, FLAG);
    expect(result.status).toBe("verified");
    expect(requests[0]!.messages).toMatchSnapshot();
    expect((requests[0]!.tools as Array<{ function: { name: string } }>).map((t) => t.function.name)).toEqual(["what_if", "get_card", "get_relic", "submit_finding"]);
  });

  it("sends a rejected finding back once, then drops it", async () => {
    const bad = { ...GOOD, advice: "下次优先拿完美打击。" };
    const { ctx, requests, trace } = context(() => ({ calls: [{ name: "submit_finding", args: bad }] }));
    const result = await explainFlag(ctx, SUMMARY, FLAG);
    expect(result.status).toBe("dropped");
    expect(result.repairs).toBe(1);
    expect(requests).toHaveLength(2);
    expect(trace.spans.filter((s) => s.name === "verify").map((s) => s.attributes.ok)).toEqual([false, false]);
  });
});

describe("investigator (part B)", () => {
  it("compares an agreed pick with the runner-up and checks reasons against that pair", async () => {
    const finding = {
      decision_id: "F08-card",
      reasons: [{ factor: "deck_fitness:defense", evidence: "T1", text: "耸肩无视把每回合防御从 6.4 提到 7.6。" }],
      advice: "防御低于需求 8.9 时，耸肩无视这类牌值得拿。",
    };
    const subject = subjectOf(AGREED);
    expect(subject).toEqual({ relation: "agree", pair: { a: "0:SHRUG_IT_OFF", b: "1:ANGER" } });
    const { ctx, requests, calls } = context(() => ({ calls: [{ name: "submit_finding", args: finding }] }));
    const result = await investigate(ctx, SUMMARY, AGREED, subject as Subject, { focus: "第 8 层为什么拿耸肩无视？" });
    expect(result.status).toBe("verified");
    expect(calls.find((c) => c.name === "compare_options")!.args).toMatchObject({ option_a: "0:SHRUG_IT_OFF", option_b: "1:ANGER" });
    const [system, user] = requests[0]!.messages as Array<{ content: string }>;
    expect(system!.content).toContain("首选和玩家的选择相同");
    expect(system!.content).toContain("pick_changed");
    expect(user!.content).toContain("第 8 层为什么拿耸肩无视？");
    // The verifier's allowed factors are the ones between the pick and the runner-up.
    expect(separatingFactors(AGREED, (subject as Subject).pair)).toEqual(["deck_fitness:defense", "deck_fitness:frontload", "deck_fitness:cycle"]);
  });

  it("marks a tier-2 disagreement and passes the program's note on why the engine is less sure", async () => {
    const tier2 = { ...FLAG, tier: 2 };
    const subject = subjectOf(tier2) as Subject;
    expect(subject.relation).toBe("disagree");
    const { ctx, requests } = context(() => ({ calls: [{ name: "submit_finding", args: GOOD }] }));
    await investigate(ctx, SUMMARY, tier2, subject, { note: "差距不大" });
    const [system, user] = requests[0]!.messages as Array<{ content: string }>;
    expect(system!.content).toContain("把握不大");
    expect(user!.content).toContain("引擎对这个分歧没那么有把握：差距不大。");
  });

  it("has nothing to compare when no factor separates the two options", () => {
    expect(subjectOf(REST)).toEqual({ relation: "none", reason: "no_factor_difference" });
    expect(subjectOf({ ...FLAG, engine_top: null })).toEqual({ relation: "none", reason: "engine_no_pick" });
  });
});

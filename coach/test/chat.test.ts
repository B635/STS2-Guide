// The terminal side of `coach ask`: question files, what gets printed, and
// the non-interactive path end to end with a scripted model.
import { afterEach, describe, expect, it, vi } from "vitest";
import { newSession } from "../src/ask.js";
import type { Answer } from "../src/ask.js";
import { askAll, parseQuestions, printer } from "../src/chat.js";
import { Trace } from "../src/trace.js";
import { fakeEngine, fakeLlm, TOOL_SPECS } from "./fakes.js";
import { DECISIONS, ENGINE_ANSWERS, NAMES, SUMMARY } from "./fixtures.js";

const sink = () => {
  const parts: string[] = [];
  return { out: { write: (t: string) => void parts.push(t) }, text: () => parts.join("") };
};

afterEach(() => vi.restoreAllMocks());

describe("question files", () => {
  it("take a JSON array or one question per line, without blanks and comments", () => {
    expect(parseQuestions('["第 6 层？", " ", "第 8 层？"]')).toEqual(["第 6 层？", "第 8 层？"]);
    expect(parseQuestions("# 注释\n第 6 层？\r\n\n第 8 层？\n")).toEqual(["第 6 层？", "第 8 层？"]);
  });
});

describe("printer", () => {
  const answer = (status: Answer["status"], final: string, notice = ""): Answer => ({ status, final, notice } as Answer);

  it("streams the draft, shows progress lines, and only confirms a draft that passed", () => {
    const progress = sink(), text = sink();
    const p = printer(progress.out, text.out, false);
    p.event({ type: "plan", text: "先看第 6 层的选牌。" });
    p.event({ type: "investigate", decisions: ["F06-card"] });
    p.event({ type: "draft", delta: "引擎更看好" });
    p.event({ type: "draft", delta: "血墙。" });
    p.event({ type: "check", stage: "draft", failed: [] });
    p.event({ type: "critic", stage: "draft", issues: [] });
    p.done(answer("passed", "引擎更看好血墙。"), true);
    expect(text.text()).toBe("\n回答（核对中）：\n引擎更看好血墙。\n");
    expect(progress.text()).toContain("计划：先看第 6 层的选牌。");
    expect(progress.text()).toContain("· 调查 F06-card");
    expect(progress.text()).toContain("[已核对");
  });

  it("prints the checked version after a rewrite, with the program's notice", () => {
    const progress = sink(), text = sink();
    const p = printer(progress.out, text.out, false);
    p.event({ type: "draft", delta: "草稿。" });
    p.event({ type: "critic", stage: "draft", issues: [{ sentence: "草稿。", reason: "x" }] });
    p.event({ type: "rewrite", delta: "重写。" });
    p.done(answer("cut", "重写。", "（有 1 句说法没能和引擎的结果核对上，已经删去。）"), true);
    expect(text.text()).toContain("核对后的回答：\n重写。\n（有 1 句");
    expect(progress.text()).toContain("审查：1 处说法没有依据");
  });
});

describe("non-interactive ask", () => {
  it("answers each question in order in one conversation and prints question and answer", async () => {
    const written: string[] = [];
    vi.spyOn(process.stdout, "write").mockImplementation((t) => (written.push(String(t)), true));
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const drafts = ["第 6 层引擎更看好血墙。", "第 8 层你和引擎都选了耸肩无视。"];
    let n = 0;
    const { llm } = fakeLlm((req) => {
      const r = req as { tool_choice?: unknown; response_format?: unknown };
      if (r.response_format) return { content: '{"doubts":[]}' };
      if (r.tool_choice && typeof r.tool_choice === "object") return { calls: [{ name: "plan", args: { text: "先看概况。", decisions: [] } }] };
      return { content: drafts[n++]! };
    });
    const { engine } = fakeEngine(ENGINE_ANSWERS);
    const ctx = { engine, llm, model: "deepseek-flash", tools: TOOL_SPECS, names: NAMES, trace: new Trace(), thinking: false };
    const session = newSession(SUMMARY, DECISIONS, []);
    const questions = ["第 6 层？", "第 8 层？"];
    const answers = await askAll(ctx, session, questions);
    expect(answers.map((a) => a.final)).toEqual(drafts);
    expect(session.turns.map((t) => t.question)).toEqual(questions);
    expect(written.join("")).toContain("问：第 8 层？\n答：第 8 层你和引擎都选了耸肩无视。");
  });
});

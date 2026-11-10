// Integration: the real Python engine over MCP (needs the .venv-coach environment).
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Engine, EngineToolError, REPO_ROOT } from "../src/engine.js";
import { prepass } from "../src/prepass.js";

const RUN = path.join(REPO_ROOT, "coach", "eval", "runs", "test", "1791474137.run");

describe("engine over MCP", () => {
  let engine: Engine;
  beforeAll(async () => {
    engine = await Engine.start();
  }, 60_000);
  afterAll(async () => {
    await engine.close();
  });

  it("hides the orchestrator tools from the agent", async () => {
    const names = (await engine.agentTools()).map((t) => t.name).sort();
    expect(names).toEqual(["compare_options", "deck_trend", "get_card", "get_decision", "get_deck", "get_relic", "what_if", "what_if_choices"]);
  });

  it("scores a run and lists its strict flags with details", async () => {
    const result = await prepass(engine, RUN);
    expect(result.summary.tier1).toEqual(["F15-card", "F09-card", "F02-card"]);
    expect(result.flags.map((f) => f.id)).toEqual(result.summary.tier1);
    for (const flag of result.flags) {
      expect(flag.options.map((o) => o.id)).toContain(flag.chosen);
      expect(flag.engine_top).not.toBe(flag.chosen);
    }
  }, 60_000);

  it("turns tool errors into EngineToolError", async () => {
    await expect(engine.call("get_decision", { run_id: "1791474137", decision_id: "F99-card" })).rejects.toThrow(
      EngineToolError,
    );
  });
});

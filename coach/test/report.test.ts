import { describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe as describeEntity, restoreIcons } from "../src/names.js";
import { CURRENT_BUILD, readCurrentBuild, renderReport } from "../src/report.js";
import type { Prepass } from "../src/prepass.js";
import { SUMMARY } from "./fixtures.js";

describe("report build version and shop rendering", () => {
  it("reads CURRENT_BUILD from packaging/compatibility.json and falls back safely", () => {
    expect(CURRENT_BUILD).toBe("v0.113.1");
    expect(readCurrentBuild("/nonexistent/compatibility.json")).toBe("v0.113.1");
    const tmp = mkdtempSync(path.join(os.tmpdir(), "sts2-compat-"));
    try {
      const custom = path.join(tmp, "compatibility.json");
      writeFileSync(custom, JSON.stringify({ game: { version: "0.114.0" } }), "utf-8");
      expect(readCurrentBuild(custom)).toBe("v0.114.0");
      writeFileSync(custom, JSON.stringify({ game: { version: "v0.115.2" } }), "utf-8");
      expect(readCurrentBuild(custom)).toBe("v0.115.2");
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("renders shop tier===0 with distinct purchase faithfully without claiming identical item", () => {
    const pre: Prepass = {
      summary: SUMMARY,
      decisions: {
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
      },
      flags: [],
    };
    const html = renderReport(pre, [], { text: "总结", source: "program" }, { model: null, cost_usd: 0, seconds: 1, generated_at: "2026-11-26" });
    expect(html).toContain("<td>15</td><td>商店</td><td>删牌</td><td>祭品</td><td>4.2</td><td>一致（均选择消费，具体购买项不同）</td>");
    expect(html).toContain("<td>22</td><td>商店</td><td>已购买（未计分项）</td><td>播种</td><td></td><td>一致（均选择消费，具体购买项不同）</td>");
    expect(html).toContain(`引擎数据是 ${CURRENT_BUILD} 版`);
  });
});

describe("names icon restoration", () => {
  it("restores both Stars and Energy simultaneously and covers upgrade_description", () => {
    // Dual Stars + Energy in separate clauses (BIG_BANG)
    expect(restoreIcons("抽1张牌。\n获得。\n获得。\n铸造5。", { Stars: 1, Energy: 1 })).toBe(
      "抽1张牌。 获得 1 颗星。 获得 1 点能量。 铸造5。",
    );
    // Dual Stars + Energy joined by 与 (CONVERGENCE) + upgrade delta on stars
    expect(restoreIcons("在本回合保留你的手牌。\n在下个回合，\n获得与。", { Stars: 1, Energy: 1 }, { stars: "+1" }, false)).toBe(
      "在本回合保留你的手牌。 在下个回合， 获得 1 点能量与 1 颗星。",
    );
    expect(restoreIcons("在本回合保留你的手牌。\n在下个回合，\n获得与。", { Stars: 1, Energy: 1 }, { stars: "+1" }, true)).toBe(
      "在本回合保留你的手牌。 在下个回合， 获得 1 点能量与 2 颗星。",
    );
    // Energy followed by 并 (THE_STAR)
    expect(restoreIcons("在下个回合获得并抽3张牌。", { Energy: 3 }, { energy: "+1" }, false)).toBe(
      "在下个回合获得 3 点能量并抽3张牌。",
    );
    expect(restoreIcons("在下个回合获得并抽4张牌。", { Energy: 3 }, { energy: "+1" }, true)).toBe(
      "在下个回合获得 4 点能量并抽4张牌。",
    );

    // Real catalog lookup via describe()
    const bigBang = describeEntity("大爆炸");
    expect(bigBang?.description).toContain("1 颗星");
    expect(bigBang?.description).toContain("1 点能量");

    const convergence = describeEntity("汇流");
    expect(convergence?.description).toContain("1 点能量与 1 颗星");
    expect(convergence?.upgrade_description).toContain("1 点能量与 2 颗星");

    const bloodletting = describeEntity("放血");
    expect(bloodletting?.description).toContain("2 点能量");
    expect(bloodletting?.upgrade_description).toContain("3 点能量");
  });
});

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it, vi } from "vitest";

describe("ratings import", () => {
  it("splits a rating sheet into one file per run", async () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), "coach-ratings-"));
    vi.resetModules();
    vi.doMock("../src/engine.js", () => ({ REPO_ROOT: dir, Engine: class {} }));
    const { importRatings, EVAL_DIR } = await import("../src/evaluate.js");
    const text = 'noise STS2-COACH-RATINGS {"run_id":"SET:test","model":"m","ratings":{"111/F02-card":{"rating":"useful","note":""},"222/F09-card":{"rating":"engine_wrong","note":"x"}}}';
    importRatings(text);
    const a = JSON.parse(readFileSync(path.join(EVAL_DIR, "ratings", "111.json"), "utf-8"));
    expect(a).toMatchObject({ run_id: "111", ratings: { "F02-card": { rating: "useful" } } });
    const b = JSON.parse(readFileSync(path.join(EVAL_DIR, "ratings", "222.json"), "utf-8"));
    expect(b.ratings["F09-card"].rating).toBe("engine_wrong");
    rmSync(dir, { recursive: true, force: true });
  });
});

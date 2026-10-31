// One JSONL line per step of a review: model calls, tool calls, verdicts.
// Field names follow OpenTelemetry spans (name, start, duration, attributes)
// so the file could be exported to a tracing backend later.

import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

export interface Span {
  name: string; // llm.call | tool.call | verify | flag.done | review.done ...
  flag?: string;
  start: string;
  duration_ms: number;
  attributes: Record<string, unknown>;
}

export class Trace {
  readonly spans: Span[] = [];

  constructor(private readonly file?: string) {
    if (file) {
      mkdirSync(path.dirname(file), { recursive: true });
      writeFileSync(file, "", "utf-8"); // one trace per review
    }
  }

  async span<T>(name: string, flag: string | undefined, run: () => Promise<T>, attributes: (r: T) => Record<string, unknown>): Promise<T> {
    const start = new Date();
    const result = await run();
    this.record({ name, ...(flag ? { flag } : {}), start: start.toISOString(), duration_ms: Date.now() - start.getTime(), attributes: attributes(result) });
    return result;
  }

  record(span: Span): void {
    this.spans.push(span);
    if (this.file) appendFileSync(this.file, JSON.stringify(span) + "\n", "utf-8");
  }

  event(name: string, flag: string | undefined, attributes: Record<string, unknown>): void {
    this.record({ name, ...(flag ? { flag } : {}), start: new Date().toISOString(), duration_ms: 0, attributes });
  }
}

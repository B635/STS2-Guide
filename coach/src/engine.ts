// The Python review engine, reached over MCP (stdio). The coach starts it as a
// child process; nothing listens on a port.

import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport, getDefaultEnvironment } from "@modelcontextprotocol/sdk/client/stdio.js";
import { z } from "zod";

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

// Tools the deterministic orchestrator calls itself; the agent never sees them.
export const ORCHESTRATOR_TOOLS = new Set(["review_run", "list_decisions"]);

export class EngineToolError extends Error {}

export interface ToolSpec {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

function defaultPython(): string {
  if (process.env.COACH_PYTHON) return process.env.COACH_PYTHON;
  const venv = path.join(REPO_ROOT, ".venv-coach");
  const candidate =
    process.platform === "win32" ? path.join(venv, "Scripts", "python.exe") : path.join(venv, "bin", "python");
  return existsSync(candidate) ? candidate : "python3";
}

export class Engine {
  private constructor(private readonly client: Client) {}

  static async start(python: string = defaultPython()): Promise<Engine> {
    const transport = new StdioClientTransport({
      command: python,
      args: ["-m", "review.mcp_server"],
      cwd: REPO_ROOT,
      env: { ...getDefaultEnvironment(), PYTHONIOENCODING: "utf-8" },
      stderr: "pipe", // the server logs every request to stderr; keep it off the user's terminal
    });
    const client = new Client({ name: "sts2-review-coach", version: "0.1.0" });
    await client.connect(transport);
    return new Engine(client);
  }

  async close(): Promise<void> {
    await this.client.close();
  }

  /** The tools the agent may call, as the server describes them. */
  async agentTools(): Promise<ToolSpec[]> {
    const { tools } = await this.client.listTools();
    return tools
      .filter((t) => !ORCHESTRATOR_TOOLS.has(t.name))
      .map((t) => ({ name: t.name, description: t.description ?? "", inputSchema: t.inputSchema }));
  }

  /** Call one tool; the result is the JSON the engine wrote. Tool errors throw EngineToolError. */
  async call(name: string, args: Record<string, unknown>): Promise<unknown> {
    const result = await this.client.callTool({ name, arguments: args });
    const content = (result.content ?? []) as Array<{ type: string; text?: string }>;
    const text = content.find((c) => c.type === "text")?.text ?? "";
    if (result.isError) throw new EngineToolError(text);
    if (result.structuredContent !== undefined && result.structuredContent !== null) return result.structuredContent;
    return JSON.parse(text);
  }

  async callParsed<T>(name: string, args: Record<string, unknown>, schema: z.ZodType<T>): Promise<T> {
    return schema.parse(await this.call(name, args));
  }
}

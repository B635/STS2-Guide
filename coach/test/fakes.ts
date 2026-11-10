// Test doubles: a scripted model and a canned engine, so the agent loops run
// without the network or the Python engine.

import type OpenAI from "openai";
import type { Engine } from "../src/engine.js";
import { EngineToolError } from "../src/engine.js";
import type { ToolSpec } from "../src/engine.js";

type Request = Record<string, unknown> & { messages: Array<Record<string, unknown>>; stream?: boolean };
type ToolCall = { name: string; args: unknown };
export type Scripted = { content?: string; calls?: ToolCall[] };

let callCounter = 0;

function completion(reply: Scripted): OpenAI.Chat.Completions.ChatCompletion {
  const tool_calls = (reply.calls ?? []).map((c) => ({
    id: `call_${++callCounter}`,
    type: "function" as const,
    function: { name: c.name, arguments: typeof c.args === "string" ? c.args : JSON.stringify(c.args) },
  }));
  return {
    id: "x", object: "chat.completion", created: 0, model: "fake",
    choices: [{
      index: 0, logprobs: null, finish_reason: tool_calls.length ? "tool_calls" : "stop",
      message: { role: "assistant", content: reply.content ?? null, refusal: null, ...(tool_calls.length ? { tool_calls } : {}) },
    }],
    usage: { prompt_tokens: 1000, completion_tokens: 100, total_tokens: 1100 },
  } as OpenAI.Chat.Completions.ChatCompletion;
}

async function* chunks(reply: Scripted): AsyncGenerator<unknown> {
  // Content in two pieces, then every tool call in one delta, then usage: enough to exercise the accumulator.
  const text = reply.content ?? "";
  const half = Math.ceil(text.length / 2);
  for (const piece of [text.slice(0, half), text.slice(half)]) {
    if (piece) yield { choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] };
  }
  const calls = reply.calls ?? [];
  if (calls.length) {
    yield {
      choices: [{
        index: 0, finish_reason: null,
        delta: {
          tool_calls: calls.map((c, index) => ({
            index, id: `call_${++callCounter}`, type: "function",
            function: { name: c.name, arguments: JSON.stringify(c.args) },
          })),
        },
      }],
    };
  }
  yield { choices: [{ index: 0, delta: {}, finish_reason: calls.length ? "tool_calls" : "stop" }] };
  yield { choices: [], usage: { prompt_tokens: 1000, completion_tokens: 100, total_tokens: 1100 } };
}

/**
 * A model that answers from a script. `respond` sees each request and returns
 * the reply; requests are kept for assertions.
 */
export function fakeLlm(respond: (request: Request, index: number) => Scripted | Promise<Scripted>) {
  const requests: Request[] = [];
  const llm = {
    chat: {
      completions: {
        create: async (request: Request) => {
          // Keep a copy: the loops push onto the same messages array afterwards.
          requests.push(structuredClone(request));
          const reply = await respond(request, requests.length - 1);
          return request.stream ? chunks(reply) : completion(reply);
        },
      },
    },
  };
  return { llm: llm as unknown as OpenAI, requests };
}

/** An engine that answers each tool from a table; unknown tools throw like the real one. */
export function fakeEngine(answers: Record<string, (args: Record<string, unknown>) => unknown>) {
  const calls: Array<{ name: string; args: Record<string, unknown> }> = [];
  const engine = {
    call: async (name: string, args: Record<string, unknown>) => {
      calls.push({ name, args });
      const answer = answers[name];
      if (!answer) throw new EngineToolError(`no tool ${name}`);
      return answer(args);
    },
  };
  return { engine: engine as unknown as Engine, calls };
}

export const TOOL_SPECS: ToolSpec[] = ["what_if", "get_card", "get_relic", "get_decision", "get_deck", "deck_trend"].map((name) => ({
  name,
  description: `${name} (fake)`,
  inputSchema: { type: "object", properties: { run_id: { type: "string" }, decision_id: { type: "string" } }, required: ["run_id"] },
}));

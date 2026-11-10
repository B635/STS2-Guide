// The model: DeepSeek through its OpenAI-compatible API. Another compatible
// endpoint only needs a different baseURL and model name.

import { existsSync } from "node:fs";
import path from "node:path";
import { loadEnvFile } from "node:process";
import OpenAI from "openai";
import { REPO_ROOT } from "./engine.js";

export const DEFAULT_MODEL = "deepseek-flash";
export const COMPARISON_MODEL = "deepseek-v4-pro";
export const DEEPSEEK_BASE_URL = "https://api.deepseek.com";

// USD per million tokens at peak price (api-docs.deepseek.com, 2026-11): input cache hit, input miss, output.
export const PRICES: Record<string, { hit: number; miss: number; output: number }> = {
  "deepseek-flash": { hit: 0.006, miss: 0.3, output: 1.2 },
  "deepseek-v4-pro": { hit: 0.044, miss: 1.32, output: 3.96 },
};

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  prompt_cache_hit_tokens?: number;
}

export function costUsd(model: string, usage: Usage): number {
  const price = PRICES[model];
  if (!price) return 0;
  const hit = usage.prompt_cache_hit_tokens ?? 0;
  return ((usage.prompt_tokens - hit) * price.miss + hit * price.hit + usage.completion_tokens * price.output) / 1e6;
}

export function createClient(): OpenAI {
  if (!process.env.DEEPSEEK_API_KEY) {
    const envFile = path.join(REPO_ROOT, ".env");
    if (existsSync(envFile)) loadEnvFile(envFile);
  }
  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) throw new Error("DEEPSEEK_API_KEY is not set (environment or the repository's .env)");
  return new OpenAI({ apiKey, baseURL: process.env.COACH_BASE_URL ?? DEEPSEEK_BASE_URL });
}

export interface Streamed {
  message: { role: "assistant"; content: string | null; tool_calls?: OpenAI.Chat.Completions.ChatCompletionMessageFunctionToolCall[] };
  usage: Usage | undefined;
  finish_reason: string | null;
}

/**
 * One chat call, streamed: content deltas go to onContent as they arrive,
 * tool-call deltas are put back together. Returns the whole message and the
 * usage (asked for with stream_options.include_usage).
 */
export async function streamChat(
  llm: OpenAI,
  request: Record<string, unknown>,
  onContent?: (delta: string) => void,
): Promise<Streamed> {
  const stream = (await llm.chat.completions.create({
    ...request, stream: true, stream_options: { include_usage: true },
  } as unknown as OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming)) as AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>;
  let content = "";
  let usage: Usage | undefined;
  let finish: string | null = null;
  const calls: Array<{ id: string; name: string; args: string }> = [];
  for await (const chunk of stream) {
    if (chunk.usage) usage = chunk.usage as Usage;
    const choice = chunk.choices[0];
    if (!choice) continue;
    if (choice.finish_reason) finish = choice.finish_reason;
    const delta = choice.delta;
    if (delta?.content) {
      content += delta.content;
      onContent?.(delta.content);
    }
    for (const t of delta?.tool_calls ?? []) {
      const call = (calls[t.index] ??= { id: "", name: "", args: "" });
      if (t.id) call.id = t.id;
      if (t.function?.name) call.name += t.function.name;
      if (t.function?.arguments) call.args += t.function.arguments;
    }
  }
  const tool_calls = calls.filter(Boolean).map((c) => ({ id: c.id, type: "function" as const, function: { name: c.name, arguments: c.args } }));
  return {
    message: { role: "assistant", content: content || null, ...(tool_calls.length ? { tool_calls } : {}) },
    usage,
    finish_reason: finish,
  };
}

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

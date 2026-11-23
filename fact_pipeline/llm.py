"""OpenAI-compatible LLM client (DeepSeek default) with exact cost tracking and fake client for tests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "deepseek-flash"
COMPARISON_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# USD per million tokens (aligned with coach/src/llm.ts, DeepSeek 2026-11 peak pricing):
# (prompt_cache_hit, prompt_cache_miss, completion)
PRICES: Dict[str, tuple] = {
    "deepseek-flash": (0.006, 0.3, 1.2),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_cache_hit_tokens=self.prompt_cache_hit_tokens + other.prompt_cache_hit_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass(frozen=True)
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: Optional[str] = None
    latency_ms: int = 0


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: int = 0,
) -> float:
    price = PRICES.get(model)
    if price is None:
        return 0.0
    hit_price, miss_price, out_price = price
    hit = min(max(0, prompt_cache_hit_tokens), max(0, prompt_tokens))
    miss = max(0, prompt_tokens - hit)
    return (miss * miss_price + hit * hit_price + max(0, completion_tokens) * out_price) / 1_000_000.0


class LLMClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> LLMResponse: ...


class OpenAIChatClient:
    """Thin wrapper over openai.OpenAI for DeepSeek / OpenAI-compatible endpoints."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 90.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key and not os.environ.get("DEEPSEEK_API_KEY"):
            env_file = ROOT_DIR / ".env"
            if env_file.is_file():
                load_dotenv(env_file)
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (environment or repository .env)")
        resolved_base = base_url or os.environ.get("COACH_BASE_URL") or DEEPSEEK_BASE_URL
        from openai import OpenAI

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base, timeout=timeout)
        self._max_retries = max_retries

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if not thinking:
            kwargs["temperature"] = temperature
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if model.startswith("deepseek"):
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            if model.startswith("deepseek"):
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                choice = resp.choices[0]
                msg = choice.message
                calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments or "{}",
                    )
                    for tc in (msg.tool_calls or [])
                ]
                raw_usage = resp.usage
                prompt_tok = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
                comp_tok = int(getattr(raw_usage, "completion_tokens", 0) or 0)
                cache_hit = int(getattr(raw_usage, "prompt_cache_hit_tokens", 0) or 0)
                if cache_hit == 0:
                    details = getattr(raw_usage, "prompt_tokens_details", None)
                    if details is not None:
                        cache_hit = int(getattr(details, "cached_tokens", 0) or 0)
                usage = LLMUsage(
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    prompt_cache_hit_tokens=cache_hit,
                    cost_usd=cost_usd(model, prompt_tok, comp_tok, cache_hit),
                )
                return LLMResponse(
                    content=msg.content,
                    tool_calls=calls,
                    usage=usage,
                    finish_reason=choice.finish_reason,
                    latency_ms=elapsed_ms,
                )
            except Exception as exc:  # pragma: no cover - network error path
                last_error = exc
                if attempt >= self._max_retries:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(str(last_error))


class ScriptedLLMClient:
    """Deterministic fake LLM client for unit tests."""

    def __init__(
        self,
        responses: Sequence[LLMResponse] | Callable[[Dict[str, Any]], LLMResponse],
    ) -> None:
        self._responses = list(responses) if not callable(responses) else None
        self._handler = responses if callable(responses) else None
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> LLMResponse:
        recorded = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": [dict(t) for t in (tools or [])],
            "tool_choice": tool_choice,
            "temperature": temperature,
            "thinking": thinking,
        }
        self.calls.append(recorded)
        if self._handler is not None:
            return self._handler(recorded)
        assert self._responses is not None
        if not self._responses:
            raise AssertionError("ScriptedLLMClient ran out of scripted responses")
        return self._responses.pop(0)


def make_tool_response(
    name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str = "call_1",
    prompt_tokens: int = 500,
    completion_tokens: int = 120,
    model: str = DEFAULT_MODEL,
) -> LLMResponse:
    """Helper for tests to build a single tool-call response."""
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments, ensure_ascii=False))],
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd(model, prompt_tokens, completion_tokens),
        ),
        finish_reason="tool_calls",
        latency_ms=10,
    )

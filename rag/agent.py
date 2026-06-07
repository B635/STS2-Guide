"""Lightweight tool-using agent for the STS2 RAG pipeline.

The agent does not answer from its own memory. It chooses one retrieval tool,
executes it, then delegates grounded generation to ``rag_chat``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import (
    BM25_TOP_N,
    DEEPSEEK_MODEL,
    MAX_SUB_QUERIES,
    MULTI_QUERY_PER_SUB_N,
    RETRIEVE_TOP_N,
    RERANKER_CANDIDATE_N,
    RRF_K,
    VECTOR_TOP_N_FOR_HYBRID,
)
from rag.chat import rag_chat
from rag.hyde import generate_hypothetical
from rag.query_planner import decompose_query
from rag.query_rewriter import rewrite_query
from rag.retriever import (
    format_context,
    hybrid_retrieve,
    multi_query_retrieve,
    retrieve,
)
from rag.router import structured_query
from rag.verifier import VerificationResult, format_verification_summary, verify_answer
from rag.vector_store import VectorStore


AGENT_TOOLS = {
    "structured_lookup",
    "hybrid_search",
    "hyde_hybrid_search",
    "multi_query_search",
    "vector_search",
}
PLANNER_TOOLS = AGENT_TOOLS - {"structured_lookup"}
AGENT_FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Default factual retrieval over the STS2 knowledge base using BM25 + vector search + RRF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for the retrieval tool."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["", "cards", "relics", "potions", "monsters", "characters"],
                            }
                        },
                    },
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hyde_hybrid_search",
            "description": "Retrieval for descriptive or open-ended questions where user wording may differ from knowledge entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query used to generate the hypothetical document."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["", "cards", "relics", "potions", "monsters", "characters"],
                            }
                        },
                    },
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_query_search",
            "description": "Retrieval for comparison, multi-entity, multi-dimensional, or strategy questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Main search query."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["", "cards", "relics", "potions", "monsters", "characters"],
                            }
                        },
                    },
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Independent sub-queries for complex questions, at most {MAX_SUB_QUERIES}.",
                    },
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Semantic vector-search fallback when BM25/hybrid search is unavailable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for vector retrieval."},
                    "top_n": {"type": "integer", "description": "Number of final documents to return, 1-10."},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["", "cards", "relics", "potions", "monsters", "characters"],
                            }
                        },
                    },
                    "reason": {"type": "string", "description": "Brief reason for choosing this tool."},
                },
                "required": ["query", "reason"],
            },
        },
    },
]

AGENT_PLANNER_PROMPT = """你是杀戮尖塔2 RAG 系统的工具选择器。

根据用户问题选择一个最合适的工具：
- hybrid_search：默认知识库检索，适合事实问答
- hyde_hybrid_search：描述性、开放式、用户表达和知识库条目可能差异较大的问题
- multi_query_search：比较、多实体、多维度、策略分析类问题
- vector_search：BM25 不可用时的语义检索兜底

注意：结构化查询已经由系统规则提前处理。这里不要选择 structured_lookup。

只输出 JSON：
{{
  "tool": "工具名",
  "query": "实际用于该工具的检索问题，通常等于用户问题；必要时可改成更适合检索的短句",
  "top_n": 5,
  "filters": {{"type": "cards/relics/potions/monsters/characters 或空字符串"}},
  "sub_queries": ["复杂问题拆解后的子问题；仅 multi_query_search 需要，最多 {max_sub_queries} 个"],
  "reason": "一句话理由"
}}

用户问题：
{query}"""


@dataclass
class AgentConfig:
    top_n: int = RETRIEVE_TOP_N
    candidate_n: int = RERANKER_CANDIDATE_N
    vector_n: int = VECTOR_TOP_N_FOR_HYBRID
    bm25_n: int = BM25_TOP_N
    rrf_k: int = RRF_K
    verification_max_retries: int = 1
    repair_top_n: int = max(RETRIEVE_TOP_N + 2, RETRIEVE_TOP_N * 2)


@dataclass
class AgentStep:
    tool: str
    detail: str
    observation: str


@dataclass
class AgentResult:
    answer: str
    results: List[Dict]
    retrieve_query: str
    selected_tool: str
    reason: str
    verification: Optional[VerificationResult] = None
    steps: List[AgentStep] = field(default_factory=list)


def _heuristic_tool(query: str, has_bm25: bool) -> str:
    if not has_bm25:
        return "vector_search"
    hyde_keywords = ("怎么", "如何", "为什么", "思路", "策略", "搭配", "适合", "回血", "续航")
    multi_keywords = ("对比", "区别", "哪个", "哪些", "有哪些", "推荐", "打法", "强在哪")
    if any(kw in query for kw in multi_keywords):
        return "multi_query_search"
    if any(kw in query for kw in hyde_keywords):
        return "hyde_hybrid_search"
    return "hybrid_search"


def _clean_sub_queries(sub_queries, fallback_query: str) -> List[str]:
    if not isinstance(sub_queries, list):
        return []
    cleaned = []
    for item in sub_queries:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    cleaned = cleaned[:MAX_SUB_QUERIES]
    if cleaned == [fallback_query]:
        return []
    return cleaned


def _normalize_plan(payload: Dict, query: str, fallback_tool: str, has_bm25: bool) -> Dict:
    tool = str(payload.get("tool", "")).strip()
    reason = str(payload.get("reason", "")).strip() or "Planner selected retrieval tool."
    if tool not in PLANNER_TOOLS:
        tool = fallback_tool
        reason = "Planner selected an unavailable tool; used rule-guard fallback."
    if tool in {"hybrid_search", "hyde_hybrid_search"} and not has_bm25:
        tool = "vector_search"
        reason = "BM25 index unavailable; used vector search fallback."
    if tool == "vector_search" and has_bm25:
        tool = fallback_tool
        reason = "BM25 index is available; rule guard avoided unnecessary pure vector fallback."
    if fallback_tool in {"multi_query_search", "hyde_hybrid_search"} and tool == "hybrid_search":
        tool = fallback_tool
        reason = "Rule guard selected a more specific retrieval strategy for this query shape."

    tool_query = str(payload.get("query", "")).strip() or query
    if len(tool_query) < 2:
        tool_query = query

    top_n = payload.get("top_n")
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        top_n = None
    if top_n is not None:
        top_n = max(1, min(top_n, 10))

    filters = payload.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    return {
        "tool": tool,
        "query": tool_query,
        "top_n": top_n,
        "filters": filters,
        "sub_queries": _clean_sub_queries(payload.get("sub_queries"), query),
        "reason": reason,
    }


def _payload_from_tool_call(tool_call: Any) -> Optional[Dict]:
    function = getattr(tool_call, "function", None)
    if function is None:
        return None

    tool_name = str(getattr(function, "name", "")).strip()
    raw_args = getattr(function, "arguments", "{}") or "{}"
    try:
        args = json.loads(raw_args)
    except (TypeError, json.JSONDecodeError):
        args = {}
    if not isinstance(args, dict):
        args = {}

    args["tool"] = tool_name
    return args


def _plan_with_function_call(query: str, client) -> Optional[Dict]:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是杀戮尖塔2 RAG 系统的工具选择器。结构化查询已经由系统规则提前处理，"
                    "不要选择 structured_lookup。请根据用户问题选择一个最合适的检索工具。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "根据问题选择检索工具并填写函数参数。"
                    f"复杂问题最多拆成 {MAX_SUB_QUERIES} 个 sub_queries。\n"
                    f"用户问题：{query}"
                ),
            },
        ],
        tools=AGENT_FUNCTION_TOOLS,
        tool_choice="auto",
    )
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None
    return _payload_from_tool_call(tool_calls[0])


def _plan_with_json_prompt(query: str, client) -> Dict:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": AGENT_PLANNER_PROMPT.format(
            query=query,
            max_sub_queries=MAX_SUB_QUERIES,
        )}],
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return payload if isinstance(payload, dict) else {}


def plan_tool(query: str, client, has_bm25: bool) -> Dict:
    """Choose one retrieval tool. Falls back to heuristics on any LLM issue."""
    fallback = _heuristic_tool(query, has_bm25)
    fallback_plan = _normalize_plan(
        {"tool": fallback, "query": query, "reason": "LLM planner unavailable; used heuristic routing."},
        query,
        fallback,
        has_bm25,
    )
    if client is None:
        return fallback_plan

    try:
        payload = _plan_with_function_call(query, client)
        if payload is None:
            payload = _plan_with_json_prompt(query, client)
        return _normalize_plan(payload, query, fallback, has_bm25)
    except Exception:
        return _normalize_plan(
            {"tool": fallback, "query": query, "reason": "Planner failed; used heuristic routing."},
            query,
            fallback,
            has_bm25,
        )


def _maybe_rerank(query: str, candidates: List[Dict], reranker, top_n: int) -> List[Dict]:
    if reranker is None:
        return candidates[:top_n]
    from rag.reranker import rerank

    return rerank(query, candidates, reranker, top_n=top_n)


def repair_retrieve(
    query: str,
    docs: List[str],
    store: VectorStore,
    model,
    bm25_index,
    reranker,
    cfg: AgentConfig,
) -> tuple:
    """Retrieve a broader repair context after answer verification fails."""
    repair_top_n = max(cfg.top_n, cfg.repair_top_n)
    repair_candidate_n = max(cfg.candidate_n, repair_top_n)

    if bm25_index is not None:
        candidates = hybrid_retrieve(
            query,
            docs,
            store,
            model,
            bm25_index,
            vector_n=max(cfg.vector_n, repair_top_n * 4),
            bm25_n=max(cfg.bm25_n, repair_top_n * 4),
            rrf_k=cfg.rrf_k,
            top_n=repair_candidate_n if reranker is not None else repair_top_n,
        )
        results = _maybe_rerank(query, candidates, reranker, repair_top_n)
        return "hybrid_search_repair", results, len(candidates)

    candidates = retrieve(query, docs, store, model, n=repair_candidate_n)
    results = _maybe_rerank(query, candidates, reranker, repair_top_n)
    return "vector_search_repair", results, len(candidates)


def run_agent(
    question: str,
    history: List[Dict],
    docs: List[str],
    items: List[Dict],
    index: Dict,
    store: VectorStore,
    model,
    client,
    bm25_index=None,
    reranker=None,
    config: Optional[AgentConfig] = None,
) -> AgentResult:
    """Run a single-turn tool-agent step and grounded RAG generation."""
    cfg = config or AgentConfig()
    steps: List[AgentStep] = []
    tool_query = ""
    planned_sub_queries: List[str] = []
    selected_top_n = cfg.top_n

    retrieve_query = rewrite_query(question, history, client, index)
    if retrieve_query != question:
        steps.append(AgentStep(
            "query_rewrite",
            question,
            f"Rewritten for retrieval: {retrieve_query}",
        ))

    routed = structured_query(retrieve_query, index, items)
    if routed is not None:
        results = routed[:cfg.top_n]
        tool = "structured_lookup"
        reason = "Structured entity/count query matched the knowledge index."
        steps.append(AgentStep(
            "structured_lookup",
            retrieve_query,
            f"Returned {len(results)} structured result(s).",
        ))
    else:
        plan = plan_tool(retrieve_query, client, has_bm25=bm25_index is not None)
        tool = plan["tool"]
        reason = plan["reason"]
        tool_query = plan["query"]
        planned_sub_queries = plan["sub_queries"]
        selected_top_n = plan["top_n"] or cfg.top_n
        plan_detail = f"query={tool_query}; top_n={plan['top_n']}; filters={plan['filters']}"
        if planned_sub_queries:
            plan_detail += f"; sub_queries={' | '.join(planned_sub_queries)}"
        steps.append(AgentStep(
            "tool_plan",
            retrieve_query,
            plan_detail,
        ))

    if routed is not None:
        pass
    elif tool == "multi_query_search":
        sub_queries = planned_sub_queries or decompose_query(tool_query, client)
        candidates = multi_query_retrieve(
            sub_queries,
            docs,
            store,
            model,
            n_per_query=MULTI_QUERY_PER_SUB_N,
        )
        results = _maybe_rerank(tool_query, candidates, reranker, selected_top_n)
        steps.append(AgentStep(
            "multi_query_search",
            " | ".join(sub_queries),
            f"Merged {len(candidates)} candidate(s), returned {len(results)}.",
        ))
    elif tool == "hyde_hybrid_search":
        vector_query = generate_hypothetical(tool_query, client)
        if bm25_index is not None:
            candidates = hybrid_retrieve(
                tool_query,
                docs,
                store,
                model,
                bm25_index,
                vector_n=cfg.vector_n,
                bm25_n=cfg.bm25_n,
                rrf_k=cfg.rrf_k,
                top_n=cfg.candidate_n if reranker is not None else selected_top_n,
                vector_query=vector_query,
            )
        else:
            candidates = retrieve(vector_query, docs, store, model, n=cfg.candidate_n)
        results = _maybe_rerank(tool_query, candidates, reranker, selected_top_n)
        steps.append(AgentStep(
            "hyde_hybrid_search",
            vector_query[:120],
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ))
    elif tool == "hybrid_search" and bm25_index is not None:
        candidates = hybrid_retrieve(
            tool_query,
            docs,
            store,
            model,
            bm25_index,
            vector_n=cfg.vector_n,
            bm25_n=cfg.bm25_n,
            rrf_k=cfg.rrf_k,
            top_n=cfg.candidate_n if reranker is not None else selected_top_n,
        )
        results = _maybe_rerank(tool_query, candidates, reranker, selected_top_n)
        steps.append(AgentStep(
            "hybrid_search",
            tool_query,
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ))
    else:
        candidates = retrieve(
            tool_query or retrieve_query,
            docs,
            store,
            model,
            n=cfg.candidate_n if reranker is not None else selected_top_n,
        )
        results = _maybe_rerank(tool_query or retrieve_query, candidates, reranker, selected_top_n)
        tool = "vector_search"
        steps.append(AgentStep(
            "vector_search",
            tool_query or retrieve_query,
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ))

    context = format_context(results)
    answer = rag_chat(question, context, history, client)
    steps.append(AgentStep("grounded_generation", question, f"Used {len(results)} source(s)."))
    verification = verify_answer(answer, results)
    steps.append(AgentStep(
        "verify_answer",
        question,
        format_verification_summary(verification),
    ))
    for attempt in range(cfg.verification_max_retries):
        if verification.passed:
            break

        if routed is not None:
            repair_results = results
            repair_observation = "Verification failed; regenerating with existing structured result."
        else:
            repair_tool, repair_results, candidate_count = repair_retrieve(
                tool_query or retrieve_query,
                docs,
                store,
                model,
                bm25_index,
                reranker,
                cfg,
            )
            results = repair_results
            repair_observation = (
                f"Verification failed; {repair_tool} returned "
                f"{len(repair_results)} result(s) from {candidate_count} candidate(s)."
            )

        steps.append(AgentStep(
            "verification_repair",
            retrieve_query,
            repair_observation,
        ))
        context = format_context(repair_results)
        answer = rag_chat(question, context, history, client)
        steps.append(AgentStep(
            "grounded_generation",
            question,
            f"Repair attempt {attempt + 1} used {len(repair_results)} source(s).",
        ))
        verification = verify_answer(answer, repair_results)
        steps.append(AgentStep(
            "verify_answer",
            question,
            format_verification_summary(verification),
        ))

    return AgentResult(
        answer=answer,
        results=results,
        retrieve_query=retrieve_query,
        selected_tool=tool,
        reason=reason,
        verification=verification,
        steps=steps,
    )


def format_agent_trace(result: AgentResult) -> str:
    lines = [f"[Agent] tool={result.selected_tool} reason={result.reason}"]
    for step in result.steps:
        lines.append(f"  - {step.tool}: {step.observation}")
    return "\n".join(lines)

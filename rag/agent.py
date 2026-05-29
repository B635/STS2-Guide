"""Lightweight tool-using agent for the STS2 RAG pipeline.

The agent does not answer from its own memory. It chooses one retrieval tool,
executes it, then delegates grounded generation to ``rag_chat``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    BM25_TOP_N,
    DEEPSEEK_MODEL,
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
from rag.vector_store import VectorStore


AGENT_TOOLS = {
    "structured_lookup",
    "hybrid_search",
    "hyde_hybrid_search",
    "multi_query_search",
    "vector_search",
}
PLANNER_TOOLS = AGENT_TOOLS - {"structured_lookup"}

AGENT_PLANNER_PROMPT = """你是杀戮尖塔2 RAG 系统的工具选择器。

根据用户问题选择一个最合适的工具：
- hybrid_search：默认知识库检索，适合事实问答
- hyde_hybrid_search：描述性、开放式、用户表达和知识库条目可能差异较大的问题
- multi_query_search：比较、多实体、多维度、策略分析类问题
- vector_search：BM25 不可用时的语义检索兜底

注意：结构化查询已经由系统规则提前处理。这里不要选择 structured_lookup。

只输出 JSON：
{{"tool": "工具名", "reason": "一句话理由"}}

用户问题：
{query}"""


@dataclass
class AgentConfig:
    top_n: int = RETRIEVE_TOP_N
    candidate_n: int = RERANKER_CANDIDATE_N
    vector_n: int = VECTOR_TOP_N_FOR_HYBRID
    bm25_n: int = BM25_TOP_N
    rrf_k: int = RRF_K


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


def plan_tool(query: str, client, has_bm25: bool) -> Dict[str, str]:
    """Choose one retrieval tool. Falls back to heuristics on any LLM issue."""
    fallback = _heuristic_tool(query, has_bm25)
    if client is None:
        return {"tool": fallback, "reason": "LLM planner unavailable; used heuristic routing."}

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": AGENT_PLANNER_PROMPT.format(query=query)}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        tool = str(payload.get("tool", "")).strip()
        reason = str(payload.get("reason", "")).strip() or "Planner selected retrieval tool."
        if tool not in PLANNER_TOOLS:
            tool = fallback
            reason = "Planner selected an unavailable tool; used rule-guard fallback."
        if tool in {"hybrid_search", "hyde_hybrid_search"} and not has_bm25:
            tool = "vector_search"
            reason = "BM25 index unavailable; used vector search fallback."
        if tool == "vector_search" and has_bm25:
            tool = fallback
            reason = "BM25 index is available; rule guard avoided unnecessary pure vector fallback."
        if fallback in {"multi_query_search", "hyde_hybrid_search"} and tool == "hybrid_search":
            tool = fallback
            reason = "Rule guard selected a more specific retrieval strategy for this query shape."
        return {"tool": tool, "reason": reason}
    except Exception:
        return {"tool": fallback, "reason": "Planner failed; used heuristic routing."}


def _maybe_rerank(query: str, candidates: List[Dict], reranker, top_n: int) -> List[Dict]:
    if reranker is None:
        return candidates[:top_n]
    from rag.reranker import rerank

    return rerank(query, candidates, reranker, top_n=top_n)


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

    if routed is not None:
        pass
    elif tool == "multi_query_search":
        sub_queries = decompose_query(retrieve_query, client)
        candidates = multi_query_retrieve(
            sub_queries,
            docs,
            store,
            model,
            n_per_query=MULTI_QUERY_PER_SUB_N,
        )
        results = _maybe_rerank(retrieve_query, candidates, reranker, cfg.top_n)
        steps.append(AgentStep(
            "multi_query_search",
            " | ".join(sub_queries),
            f"Merged {len(candidates)} candidate(s), returned {len(results)}.",
        ))
    elif tool == "hyde_hybrid_search":
        vector_query = generate_hypothetical(retrieve_query, client)
        if bm25_index is not None:
            candidates = hybrid_retrieve(
                retrieve_query,
                docs,
                store,
                model,
                bm25_index,
                vector_n=cfg.vector_n,
                bm25_n=cfg.bm25_n,
                rrf_k=cfg.rrf_k,
                top_n=cfg.candidate_n if reranker is not None else cfg.top_n,
                vector_query=vector_query,
            )
        else:
            candidates = retrieve(vector_query, docs, store, model, n=cfg.candidate_n)
        results = _maybe_rerank(retrieve_query, candidates, reranker, cfg.top_n)
        steps.append(AgentStep(
            "hyde_hybrid_search",
            vector_query[:120],
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ))
    elif tool == "hybrid_search" and bm25_index is not None:
        candidates = hybrid_retrieve(
            retrieve_query,
            docs,
            store,
            model,
            bm25_index,
            vector_n=cfg.vector_n,
            bm25_n=cfg.bm25_n,
            rrf_k=cfg.rrf_k,
            top_n=cfg.candidate_n if reranker is not None else cfg.top_n,
        )
        results = _maybe_rerank(retrieve_query, candidates, reranker, cfg.top_n)
        steps.append(AgentStep(
            "hybrid_search",
            retrieve_query,
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ))
    else:
        candidates = retrieve(
            retrieve_query,
            docs,
            store,
            model,
            n=cfg.candidate_n if reranker is not None else cfg.top_n,
        )
        results = _maybe_rerank(retrieve_query, candidates, reranker, cfg.top_n)
        tool = "vector_search"
        steps.append(AgentStep(
            "vector_search",
            retrieve_query,
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ))

    context = format_context(results)
    answer = rag_chat(question, context, history, client)
    steps.append(AgentStep("grounded_generation", question, f"Used {len(results)} source(s)."))

    return AgentResult(
        answer=answer,
        results=results,
        retrieve_query=retrieve_query,
        selected_tool=tool,
        reason=reason,
        steps=steps,
    )


def format_agent_trace(result: AgentResult) -> str:
    lines = [f"[Agent] tool={result.selected_tool} reason={result.reason}"]
    for step in result.steps:
        lines.append(f"  - {step.tool}: {step.observation}")
    return "\n".join(lines)

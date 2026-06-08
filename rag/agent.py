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
from rag.agent_schema import AGENT_FUNCTION_TOOLS
from rag.hyde import generate_hypothetical
from rag.query_planner import decompose_query
from rag.query_rewriter import rewrite_query
from rag.retriever import (
    format_context,
    hybrid_retrieve,
    multi_query_retrieve,
    retrieve,
)
from rag.router import STRATEGY_KEYWORDS, structured_query
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

ROLE_QUERY_ALIASES = {
    "故障机器人": "defect",
    "铁甲战士": "ironclad",
    "亡灵契约师": "necrobinder",
    "储君": "regent",
    "静默猎手": "silent",
}

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


def expand_query_aliases(query: str) -> str:
    """Add internal character color tags used by card/relic docs."""
    additions = [alias for name, alias in ROLE_QUERY_ALIASES.items() if name in query and alias not in query.lower()]
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _role_aliases_in_query(query: str) -> List[str]:
    lowered = query.lower()
    return [
        alias
        for name, alias in ROLE_QUERY_ALIASES.items()
        if name in query or alias in lowered
    ]


def _is_strategy_query(query: str) -> bool:
    return any(keyword in query for keyword in STRATEGY_KEYWORDS)


def role_strategy_context(query: str, items: List[Dict], limit: int = 8) -> List[Dict]:
    """Return deterministic role-related context for open strategy questions."""
    aliases = set(_role_aliases_in_query(query))
    if not aliases or not _is_strategy_query(query):
        return []

    preferred_terms = ("仆从", "生成", "无色", "力量", "格挡", "消耗", "抽", "易伤", "虚弱")
    candidates = []
    for idx, item in enumerate(items):
        text = item.get("embed_text", "")
        item_type = item.get("_type", "")
        color = str(item.get("color", "")).lower()
        item_id = str(item.get("id", "")).lower()
        name = item.get("name", "")

        score = 0.0
        if item_type == "characters" and item_id.lower() in aliases:
            score += 4.0
        if color in aliases:
            score += 2.0
            if item_type == "cards":
                score += 0.8
            if item_type == "relics":
                score += 0.4
        if "regent" in aliases and item_type == "cards" and "Minion" in text:
            score += 2.4
        if name in query:
            score += 1.0
        for term in preferred_terms:
            if term in text:
                score += 0.35

        if score > 0:
            candidates.append({
                "text": text,
                "score": score,
                "index": idx,
                "source": "role_strategy_context",
            })

    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:limit]


def merge_results(*result_lists: List[Dict]) -> List[Dict]:
    merged: Dict[int, Dict] = {}
    order = 0
    for results in result_lists:
        for row in results:
            idx = int(row.get("index", -1))
            key = idx if idx >= 0 else -(order + 1)
            order += 1
            current = merged.get(key)
            if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                merged[key] = row
    return sorted(merged.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)


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
    role_context: List[Dict] = []

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
        planner_query = expand_query_aliases(retrieve_query)
        if planner_query != retrieve_query:
            steps.append(AgentStep(
                "query_alias_expand",
                retrieve_query,
                f"Expanded for retrieval: {planner_query}",
            ))
        plan = plan_tool(planner_query, client, has_bm25=bm25_index is not None)
        tool = plan["tool"]
        reason = plan["reason"]
        tool_query = plan["query"]
        planned_sub_queries = plan["sub_queries"]
        selected_top_n = plan["top_n"] or cfg.top_n
        role_context = role_strategy_context(tool_query, items)
        if role_context:
            selected_top_n = max(selected_top_n, min(len(role_context), 8))
            steps.append(AgentStep(
                "role_strategy_context",
                tool_query,
                f"Added {len(role_context)} role-specific card/relic context item(s).",
            ))
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
        candidates = merge_results(role_context, candidates)
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
        candidates = merge_results(role_context, candidates)
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
        candidates = merge_results(role_context, candidates)
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
        candidates = merge_results(role_context, candidates)
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
            repair_results = merge_results(role_context, repair_results)[:max(cfg.repair_top_n, selected_top_n)]
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

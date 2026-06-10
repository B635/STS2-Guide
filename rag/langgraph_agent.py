"""Optional LangGraph implementation of the tool-using RAG agent."""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from rag.agent import (
    AgentConfig,
    AgentResult,
    AgentStep,
    expand_query_aliases,
    merge_results,
    plan_tool,
    repair_retrieve,
    role_strategy_context,
)
from rag.chat import rag_chat
from rag.hyde import generate_hypothetical
from rag.query_planner import decompose_query
from rag.query_rewriter import rewrite_query
from rag.retriever import format_context, hybrid_retrieve, multi_query_retrieve, retrieve
from rag.router import structured_query
from rag.verifier import VerificationResult, format_verification_summary, verify_answer
from rag.vector_store import VectorStore


class AgentGraphState(TypedDict, total=False):
    question: str
    history: List[Dict]
    docs: List[str]
    items: List[Dict]
    index: Dict
    store: VectorStore
    model: object
    client: object
    bm25_index: object
    reranker: object
    config: AgentConfig
    retrieve_query: str
    tool_query: str
    planned_sub_queries: List[str]
    tool_top_n: int
    tool_filters: Dict
    role_context: List[Dict]
    selected_tool: str
    reason: str
    results: List[Dict]
    answer: str
    verification: VerificationResult
    verification_attempts: int
    steps: List[AgentStep]


def _append_step(state: AgentGraphState, tool: str, detail: str, observation: str) -> List[AgentStep]:
    return list(state.get("steps", [])) + [AgentStep(tool, detail, observation)]


def _maybe_rerank(query: str, candidates: List[Dict], reranker, top_n: int) -> List[Dict]:
    if reranker is None:
        return candidates[:top_n]
    from rag.reranker import rerank

    return rerank(query, candidates, reranker, top_n=top_n)


def _rewrite_node(state: AgentGraphState) -> Dict:
    question = state["question"]
    history = state.get("history", [])
    retrieve_query = rewrite_query(question, history, state["client"], state["index"])
    updates = {"retrieve_query": retrieve_query}
    if retrieve_query != question:
        updates["steps"] = _append_step(
            state,
            "query_rewrite",
            question,
            f"Rewritten for retrieval: {retrieve_query}",
        )
    return updates


def _route_or_plan_node(state: AgentGraphState) -> Dict:
    cfg = state["config"]
    retrieve_query = state["retrieve_query"]
    routed = structured_query(retrieve_query, state["index"], state["items"])
    if routed is not None:
        results = routed[:cfg.top_n]
        return {
            "selected_tool": "structured_lookup",
            "reason": "Structured entity/count query matched the knowledge index.",
            "results": results,
            "steps": _append_step(
                state,
                "structured_lookup",
                retrieve_query,
                f"Returned {len(results)} structured result(s).",
            ),
        }

    planner_query = expand_query_aliases(retrieve_query)
    steps = state.get("steps", [])
    if planner_query != retrieve_query:
        steps = _append_step(
            state,
            "query_alias_expand",
            retrieve_query,
            f"Expanded for retrieval: {planner_query}",
        )

    plan = plan_tool(planner_query, state["client"], has_bm25=state.get("bm25_index") is not None)
    role_context = role_strategy_context(plan["query"], state["items"])
    tool_top_n = plan["top_n"] or state["config"].top_n
    if role_context:
        tool_top_n = max(tool_top_n, min(len(role_context), 8))
        steps = _append_step(
            {**state, "steps": steps},
            "role_strategy_context",
            plan["query"],
            f"Added {len(role_context)} role-specific card/relic context item(s).",
        )

    plan_detail = f"query={plan['query']}; top_n={plan['top_n']}; filters={plan['filters']}"
    if plan["sub_queries"]:
        plan_detail += f"; sub_queries={' | '.join(plan['sub_queries'])}"
    return {
        "selected_tool": plan["tool"],
        "reason": plan["reason"],
        "tool_query": plan["query"],
        "planned_sub_queries": plan["sub_queries"],
        "tool_top_n": tool_top_n,
        "tool_filters": plan["filters"],
        "role_context": role_context,
        "steps": _append_step(
            {**state, "steps": steps},
            "tool_plan",
            planner_query,
            plan_detail,
        ),
    }


def _next_after_route(state: AgentGraphState) -> str:
    return "generate" if state.get("results") is not None else "execute_tool"


def _execute_tool_node(state: AgentGraphState) -> Dict:
    cfg = state["config"]
    query = state.get("tool_query", state["retrieve_query"])
    top_n = state.get("tool_top_n", cfg.top_n)
    docs = state["docs"]
    store = state["store"]
    model = state["model"]
    bm25_index = state.get("bm25_index")
    reranker = state.get("reranker")
    tool = state.get("selected_tool", "vector_search")
    role_context = state.get("role_context", [])

    if tool == "multi_query_search":
        sub_queries = state.get("planned_sub_queries") or decompose_query(query, state["client"])
        candidates = multi_query_retrieve(
            sub_queries,
            docs,
            store,
            model,
            n_per_query=cfg.top_n,
        )
        candidates = merge_results(role_context, candidates)
        results = _maybe_rerank(query, candidates, reranker, top_n)
        return {
            "results": results,
            "steps": _append_step(
                state,
                "multi_query_search",
                " | ".join(sub_queries),
                f"Merged {len(candidates)} candidate(s), returned {len(results)}.",
            ),
        }

    if tool == "hyde_hybrid_search":
        vector_query = generate_hypothetical(query, state["client"])
        if bm25_index is not None:
            candidates = hybrid_retrieve(
                query,
                docs,
                store,
                model,
                bm25_index,
                vector_n=cfg.vector_n,
                bm25_n=cfg.bm25_n,
                rrf_k=cfg.rrf_k,
                top_n=cfg.candidate_n if reranker is not None else top_n,
                vector_query=vector_query,
            )
        else:
            candidates = retrieve(vector_query, docs, store, model, n=cfg.candidate_n)
        candidates = merge_results(role_context, candidates)
        results = _maybe_rerank(query, candidates, reranker, top_n)
        return {
            "results": results,
            "steps": _append_step(
                state,
                "hyde_hybrid_search",
                vector_query[:120],
                f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
            ),
        }

    if tool == "hybrid_search" and bm25_index is not None:
        candidates = hybrid_retrieve(
            query,
            docs,
            store,
            model,
            bm25_index,
            vector_n=cfg.vector_n,
            bm25_n=cfg.bm25_n,
            rrf_k=cfg.rrf_k,
            top_n=cfg.candidate_n if reranker is not None else top_n,
        )
        candidates = merge_results(role_context, candidates)
        results = _maybe_rerank(query, candidates, reranker, top_n)
        return {
            "results": results,
            "steps": _append_step(
                state,
                "hybrid_search",
                query,
                f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
            ),
        }

    candidates = retrieve(
        query,
        docs,
        store,
        model,
        n=cfg.candidate_n if reranker is not None else top_n,
    )
    candidates = merge_results(role_context, candidates)
    results = _maybe_rerank(query, candidates, reranker, top_n)
    return {
        "selected_tool": "vector_search",
        "results": results,
        "steps": _append_step(
            state,
            "vector_search",
            query,
            f"Retrieved {len(candidates)} candidate(s), returned {len(results)}.",
        ),
    }


def _generate_node(state: AgentGraphState) -> Dict:
    results = state.get("results", [])
    context = format_context(results)
    answer = rag_chat(state["question"], context, state.get("history", []), state["client"])
    attempts = state.get("verification_attempts", 0)
    observation = (
        f"Repair attempt {attempts} used {len(results)} source(s)."
        if attempts
        else f"Used {len(results)} source(s)."
    )
    return {
        "answer": answer,
        "steps": _append_step(
            state,
            "grounded_generation",
            state["question"],
            observation,
        ),
    }


def _verify_node(state: AgentGraphState) -> Dict:
    verification = verify_answer(state.get("answer", ""), state.get("results", []))
    attempts = state.get("verification_attempts", 0) + 1
    return {
        "verification": verification,
        "verification_attempts": attempts,
        "steps": _append_step(
            state,
            "verify_answer",
            state["question"],
            format_verification_summary(verification),
        ),
    }


def _next_after_verify(state: AgentGraphState) -> str:
    verification = state.get("verification")
    if verification is None or verification.passed:
        return "end"
    attempts = state.get("verification_attempts", 0)
    if attempts <= state["config"].verification_max_retries:
        return "repair"
    return "end"


def _repair_node(state: AgentGraphState) -> Dict:
    if state.get("selected_tool") == "structured_lookup":
        return {
            "steps": _append_step(
                state,
                "verification_repair",
                state["retrieve_query"],
                "Verification failed; regenerating with existing structured result.",
            ),
        }

    repair_tool, results, candidate_count = repair_retrieve(
        state.get("tool_query", state["retrieve_query"]),
        state["docs"],
        state["store"],
        state["model"],
        state.get("bm25_index"),
        state.get("reranker"),
        state["config"],
    )
    role_context = state.get("role_context", [])
    if role_context:
        results = merge_results(role_context, results)[:max(state["config"].repair_top_n, state.get("tool_top_n", state["config"].top_n))]
    return {
        "results": results,
        "steps": _append_step(
            state,
            "verification_repair",
            state["retrieve_query"],
            f"Verification failed; {repair_tool} returned {len(results)} result(s) from {candidate_count} candidate(s).",
        ),
    }


def build_langgraph_agent():
    """Build the LangGraph state machine.

    Import LangGraph lazily so the rest of the project can run without the
    optional dependency installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Run `pip install langgraph` or use the default agent."
        ) from exc

    graph = StateGraph(AgentGraphState)
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("route_or_plan", _route_or_plan_node)
    graph.add_node("execute_tool", _execute_tool_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("repair", _repair_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "route_or_plan")
    graph.add_conditional_edges(
        "route_or_plan",
        _next_after_route,
        {
            "execute_tool": "execute_tool",
            "generate": "generate",
        },
    )
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        _next_after_verify,
        {
            "repair": "repair",
            "end": END,
        },
    )
    graph.add_edge("repair", "generate")
    return graph.compile()


def run_langgraph_agent(
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
    app = build_langgraph_agent()
    cfg = config or AgentConfig()
    final_state = app.invoke({
        "question": question,
        "history": history,
        "docs": docs,
        "items": items,
        "index": index,
        "store": store,
        "model": model,
        "client": client,
        "bm25_index": bm25_index,
        "reranker": reranker,
        "config": cfg,
        "verification_attempts": 0,
        "steps": [],
    })
    return AgentResult(
        answer=final_state["answer"],
        results=final_state.get("results", []),
        retrieve_query=final_state.get("retrieve_query", question),
        selected_tool=final_state.get("selected_tool", "vector_search"),
        reason=final_state.get("reason", "LangGraph workflow completed."),
        verification=final_state.get("verification"),
        steps=final_state.get("steps", []),
    )

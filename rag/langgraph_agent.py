"""Optional LangGraph implementation of the tool-using RAG agent."""
from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from rag.agent import AgentConfig, AgentResult, AgentStep, plan_tool
from rag.chat import rag_chat
from rag.hyde import generate_hypothetical
from rag.query_planner import decompose_query
from rag.query_rewriter import rewrite_query
from rag.retriever import format_context, hybrid_retrieve, multi_query_retrieve, retrieve
from rag.router import structured_query
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
    selected_tool: str
    reason: str
    results: List[Dict]
    answer: str
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

    plan = plan_tool(retrieve_query, state["client"], has_bm25=state.get("bm25_index") is not None)
    return {
        "selected_tool": plan["tool"],
        "reason": plan["reason"],
    }


def _next_after_route(state: AgentGraphState) -> str:
    return "generate" if state.get("results") is not None else "execute_tool"


def _execute_tool_node(state: AgentGraphState) -> Dict:
    cfg = state["config"]
    query = state["retrieve_query"]
    docs = state["docs"]
    store = state["store"]
    model = state["model"]
    bm25_index = state.get("bm25_index")
    reranker = state.get("reranker")
    tool = state.get("selected_tool", "vector_search")

    if tool == "multi_query_search":
        sub_queries = decompose_query(query, state["client"])
        candidates = multi_query_retrieve(
            sub_queries,
            docs,
            store,
            model,
            n_per_query=cfg.top_n,
        )
        results = _maybe_rerank(query, candidates, reranker, cfg.top_n)
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
                top_n=cfg.candidate_n if reranker is not None else cfg.top_n,
                vector_query=vector_query,
            )
        else:
            candidates = retrieve(vector_query, docs, store, model, n=cfg.candidate_n)
        results = _maybe_rerank(query, candidates, reranker, cfg.top_n)
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
            top_n=cfg.candidate_n if reranker is not None else cfg.top_n,
        )
        results = _maybe_rerank(query, candidates, reranker, cfg.top_n)
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
        n=cfg.candidate_n if reranker is not None else cfg.top_n,
    )
    results = _maybe_rerank(query, candidates, reranker, cfg.top_n)
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
    return {
        "answer": answer,
        "steps": _append_step(
            state,
            "grounded_generation",
            state["question"],
            f"Used {len(results)} source(s).",
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
    graph.add_edge("generate", END)
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
        "steps": [],
    })
    return AgentResult(
        answer=final_state["answer"],
        results=final_state.get("results", []),
        retrieve_query=final_state.get("retrieve_query", question),
        selected_tool=final_state.get("selected_tool", "vector_search"),
        reason=final_state.get("reason", "LangGraph workflow completed."),
        steps=final_state.get("steps", []),
    )

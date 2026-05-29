import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import RERANKER_CANDIDATE_N
from rag.agent import AgentConfig, plan_tool
from rag.bm25 import build_bm25_index
from rag.chat import create_client
from rag.embedder import load_model, load_or_compute_embeddings
from rag.hyde import generate_hypothetical
from rag.knowledge import load_knowledge
from rag.query_planner import decompose_query
from rag.retriever import hybrid_retrieve, multi_query_retrieve, retrieve
from rag.router import structured_query


DEFAULT_EVAL_FILE = "./data/agent_eval.json"


def load_eval_cases(eval_file: str) -> List[Dict]:
    with open(eval_file, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def doc_matches_case(doc: str, case: Dict) -> bool:
    for term_group in case.get("expected_any", []):
        if all(term in doc for term in term_group):
            return True
    return False


def find_first_hit_rank(results: List[Dict], case: Dict) -> Optional[int]:
    for i, result in enumerate(results, start=1):
        if doc_matches_case(result["text"], case):
            return i
    return None


def select_and_retrieve(
    query: str,
    docs: List[str],
    items: List[Dict],
    index: Dict,
    store,
    model,
    bm25,
    client,
    cfg: AgentConfig,
    reranker=None,
) -> Tuple[str, str, List[Dict]]:
    routed = structured_query(query, index, items)
    if routed is not None:
        return "structured_lookup", "Structured entity/count query matched.", routed[:cfg.top_n]

    plan = plan_tool(query, client, has_bm25=bm25 is not None)
    tool = plan["tool"]
    reason = plan["reason"]

    if tool == "multi_query_search":
        sub_queries = decompose_query(query, client)
        candidates = multi_query_retrieve(sub_queries, docs, store, model, n_per_query=cfg.top_n)
    elif tool == "hyde_hybrid_search":
        vector_query = generate_hypothetical(query, client)
        candidates = hybrid_retrieve(
            query, docs, store, model, bm25,
            vector_n=cfg.vector_n, bm25_n=cfg.bm25_n,
            rrf_k=cfg.rrf_k, top_n=cfg.candidate_n,
            vector_query=vector_query,
        )
    elif tool == "hybrid_search":
        candidates = hybrid_retrieve(
            query, docs, store, model, bm25,
            vector_n=cfg.vector_n, bm25_n=cfg.bm25_n,
            rrf_k=cfg.rrf_k, top_n=cfg.candidate_n,
        )
    else:
        tool = "vector_search"
        candidates = retrieve(query, docs, store, model, n=cfg.candidate_n)

    if reranker is not None:
        from rag.reranker import rerank
        results = rerank(query, candidates, reranker, top_n=cfg.top_n)
    else:
        results = candidates[:cfg.top_n]
    return tool, reason, results


def evaluate_cases(cases: List[Dict], top_k: int, use_reranker: bool, candidate_n: int) -> None:
    docs, items, index = load_knowledge()
    model = load_model()
    store = load_or_compute_embeddings(docs, model)
    bm25 = build_bm25_index(docs)
    client = create_client()
    cfg = AgentConfig(top_n=top_k, candidate_n=candidate_n)

    reranker = None
    if use_reranker:
        from rag.reranker import load_reranker
        reranker = load_reranker()

    tool_correct = 0
    tool_expected_total = 0
    hit_counts = {k: 0 for k in range(1, top_k + 1)}
    reciprocal_rank_sum = 0.0
    failures = []

    for case in cases:
        query = case["query"]
        tool, reason, results = select_and_retrieve(
            query, docs, items, index, store, model, bm25, client, cfg, reranker=reranker,
        )

        expected_tool = case.get("expected_tool")
        if expected_tool:
            tool_expected_total += 1
            if tool == expected_tool:
                tool_correct += 1

        rank = find_first_hit_rank(results, case)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
            for k in range(rank, top_k + 1):
                hit_counts[k] += 1
        else:
            failures.append({
                "query": query,
                "tool": tool,
                "reason": reason,
                "top_results": [
                    {"score": round(r.get("rerank_score", r["score"]), 4), "text": r["text"][:120]}
                    for r in results[:3]
                ],
            })

        mark = "OK" if not expected_tool or tool == expected_tool else f"expected {expected_tool}"
        print(f"[{mark}] tool={tool} | {query}")
        print(f"  reason: {reason}")

    total = max(len(cases), 1)
    print("\n" + "=" * 52)
    print(f"Agent 评测完成，总样本数: {len(cases)}")
    if tool_expected_total:
        print(f"Tool Accuracy: {tool_correct}/{tool_expected_total} = {tool_correct / tool_expected_total:.2%}")
    for k, count in hit_counts.items():
        print(f"Hit@{k}: {count / total:.2%}")
    print(f"MRR: {reciprocal_rank_sum / total:.4f}")

    if failures:
        print("-" * 52)
        print("失败样本:")
        for fail in failures:
            print(f"  问题: {fail['query']} | tool={fail['tool']}")
            for idx, item in enumerate(fail["top_results"], start=1):
                print(f"    Top{idx} | score={item['score']:.4f} | {item['text']}")
    print("=" * 52)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tool-Using Agent routing and retrieval quality.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--candidate-n", type=int, default=RERANKER_CANDIDATE_N)
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file)
    evaluate_cases(cases, args.top_k, args.reranker, args.candidate_n)


if __name__ == "__main__":
    main()

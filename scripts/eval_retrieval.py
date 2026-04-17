import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import KNOWLEDGE_FILE, BM25_TOP_N, VECTOR_TOP_N_FOR_HYBRID, RRF_K
from rag.embedder import load_model, load_or_compute_embeddings
from rag.knowledge import load_knowledge
from rag.retriever import retrieve, hybrid_retrieve
from rag.router import structured_query
from rag.chat import create_client
from rag.hyde import generate_hypothetical


DEFAULT_EVAL_FILE = "./data/retrieval_eval.json"


def load_eval_cases(eval_file: str) -> List[Dict]:
    with open(eval_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["cases"]


def doc_matches_case(doc: str, case: Dict) -> bool:
    expected_any = case.get("expected_any", [])
    if not expected_any:
        return False

    for term_group in expected_any:
        if all(term in doc for term in term_group):
            return True

    return False


def find_first_hit_rank(results: List[Dict], case: Dict) -> Optional[int]:
    for i, result in enumerate(results, start=1):
        if doc_matches_case(result["text"], case):
            return i
    return None


def evaluate_cases(
    cases: List[Dict],
    docs: List[str],
    items: List[Dict],
    index: Dict,
    top_k: int,
    use_reranker: bool = False,
    candidate_n: int = 20,
    use_router: bool = False,
    use_hybrid: bool = False,
    use_hyde: bool = False,
) -> Tuple[Dict[int, float], float, List[Dict]]:
    model = load_model()
    doc_embeddings = load_or_compute_embeddings(docs, model)

    reranker = None
    if use_reranker:
        from rag.reranker import load_reranker, rerank
        reranker = load_reranker()

    bm25_idx = None
    if use_hybrid:
        from rag.bm25 import build_bm25_index
        bm25_idx = build_bm25_index(docs)

    client = create_client() if use_hyde else None

    hit_counts = {k: 0 for k in range(1, top_k + 1)}
    reciprocal_rank_sum = 0.0
    failures = []

    for case in cases:
        query = case["query"]

        routed = structured_query(query, index, items) if use_router else None

        if routed is not None:
            results = routed[:top_k]
        else:
            vector_query = generate_hypothetical(query, client) if use_hyde else None

            if use_hybrid and use_reranker:
                candidates = hybrid_retrieve(
                    query, docs, doc_embeddings, model, bm25_idx,
                    vector_n=VECTOR_TOP_N_FOR_HYBRID, bm25_n=BM25_TOP_N,
                    rrf_k=RRF_K, top_n=candidate_n,
                    vector_query=vector_query,
                )
                results = rerank(query, candidates, reranker, top_n=top_k)
            elif use_hybrid:
                results = hybrid_retrieve(
                    query, docs, doc_embeddings, model, bm25_idx,
                    vector_n=VECTOR_TOP_N_FOR_HYBRID, bm25_n=BM25_TOP_N,
                    rrf_k=RRF_K, top_n=top_k,
                    vector_query=vector_query,
                )
            elif use_reranker:
                candidates = retrieve(vector_query or query, docs, doc_embeddings, model, n=candidate_n)
                results = rerank(query, candidates, reranker, top_n=top_k)
            else:
                results = retrieve(vector_query or query, docs, doc_embeddings, model, n=top_k)

        rank = find_first_hit_rank(results, case)

        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
            for k in range(rank, top_k + 1):
                hit_counts[k] += 1
        else:
            failures.append(
                {
                    "query": query,
                    "top_results": [
                        {"score": round(r.get("rerank_score", r["score"]), 4), "text": r["text"][:120]}
                        for r in results[:3]
                    ],
                }
            )

    total = max(len(cases), 1)
    hit_rates = {k: hit_counts[k] / total for k in range(1, top_k + 1)}
    mrr = reciprocal_rank_sum / total
    return hit_rates, mrr, failures


def print_summary(
    hit_rates: Dict[int, float],
    mrr: float,
    total_cases: int,
    failures: List[Dict],
    max_failures: int,
    label: str = "",
) -> None:
    print("=" * 48)
    if label:
        print(f"[{label}]")
    print(f"检索评测完成，总样本数: {total_cases}")
    print("-" * 48)
    print("命中率:")
    for k, rate in hit_rates.items():
        print(f"  Hit@{k}: {rate:.2%}")
    print(f"MRR: {mrr:.4f}")

    if failures:
        print("-" * 48)
        print(f"失败样本（最多展示 {max_failures} 条）:")
        for fail in failures[:max_failures]:
            print(f"  问题: {fail['query']}")
            for idx, item in enumerate(fail["top_results"], start=1):
                print(f"    Top{idx} | score={item['score']:.4f} | {item['text']}")
    print("=" * 48)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a labeled query set.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE, help="Path to retrieval eval json file")
    parser.add_argument("--top-k", type=int, default=5, help="Top K to evaluate")
    parser.add_argument("--max-failures", type=int, default=10, help="Max failed cases to print")
    parser.add_argument("--reranker", action="store_true", help="Enable reranker and show comparison")
    parser.add_argument("--candidate-n", type=int, default=20, help="Candidate pool size for reranker")
    parser.add_argument("--router", action="store_true", help="Enable structured query router")
    parser.add_argument("--hybrid", action="store_true", help="Enable Hybrid retrieval (BM25 + Vector with RRF)")
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE: LLM generates a hypothetical doc for the dense side")
    args = parser.parse_args()

    docs, items, index = load_knowledge()
    cases = load_eval_cases(args.eval_file)

    def _label(base: str) -> str:
        parts = [base]
        if args.hybrid:
            parts.append("Hybrid")
        if args.hyde:
            parts.append("HyDE")
        if args.router:
            parts.append("Router")
        return " + ".join(parts)

    if args.reranker:
        print("\n--- 基线（无 Reranker）---")
        hit_rates, mrr, failures = evaluate_cases(
            cases, docs, items, index, args.top_k,
            use_reranker=False, use_router=args.router, use_hybrid=args.hybrid,
            use_hyde=args.hyde,
        )
        print_summary(hit_rates, mrr, len(cases), failures, args.max_failures, label=_label("Baseline"))

        print("\n--- 加入 Reranker ---")
        rr_hit_rates, rr_mrr, rr_failures = evaluate_cases(
            cases, docs, items, index, args.top_k,
            use_reranker=True, candidate_n=args.candidate_n,
            use_router=args.router, use_hybrid=args.hybrid,
            use_hyde=args.hyde,
        )
        print_summary(rr_hit_rates, rr_mrr, len(cases), rr_failures, args.max_failures, label=_label("Reranker"))

        print("\n--- 指标提升对比 ---")
        for k in sorted(hit_rates.keys()):
            diff = rr_hit_rates[k] - hit_rates[k]
            sign = "+" if diff >= 0 else ""
            print(f"  Hit@{k}: {hit_rates[k]:.2%} → {rr_hit_rates[k]:.2%}  ({sign}{diff:.2%})")
        mrr_diff = rr_mrr - mrr
        sign = "+" if mrr_diff >= 0 else ""
        print(f"  MRR:   {mrr:.4f} → {rr_mrr:.4f}  ({sign}{mrr_diff:.4f})")
    else:
        hit_rates, mrr, failures = evaluate_cases(
            cases, docs, items, index, args.top_k,
            use_router=args.router, use_hybrid=args.hybrid,
            use_hyde=args.hyde,
        )
        print_summary(hit_rates, mrr, len(cases), failures, args.max_failures, label=_label("Baseline"))


if __name__ == "__main__":
    main()

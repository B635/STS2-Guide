import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import KNOWLEDGE_FILE
from rag.embedder import load_model, load_or_compute_embeddings
from rag.retriever import retrieve


DEFAULT_EVAL_FILE = "./data/retrieval_eval.json"


def load_docs() -> List[str]:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["docs"]


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


def evaluate_cases(cases: List[Dict], docs: List[str], top_k: int) -> Tuple[Dict[int, float], float, List[Dict]]:
    model = load_model()
    doc_embeddings = load_or_compute_embeddings(docs, model)

    hit_counts = {k: 0 for k in range(1, top_k + 1)}
    reciprocal_rank_sum = 0.0
    failures = []

    for case in cases:
        query = case["query"]
        results = retrieve(query, docs, doc_embeddings, model, n=top_k)
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
                        {"score": round(r["score"], 4), "text": r["text"][:120]} for r in results[:3]
                    ],
                }
            )

    total = max(len(cases), 1)
    hit_rates = {k: hit_counts[k] / total for k in range(1, top_k + 1)}
    mrr = reciprocal_rank_sum / total
    return hit_rates, mrr, failures


def print_summary(hit_rates: Dict[int, float], mrr: float, total_cases: int, failures: List[Dict], max_failures: int) -> None:
    print("=" * 48)
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
    args = parser.parse_args()

    docs = load_docs()
    cases = load_eval_cases(args.eval_file)

    hit_rates, mrr, failures = evaluate_cases(cases, docs, args.top_k)
    print_summary(hit_rates, mrr, len(cases), failures, args.max_failures)


if __name__ == "__main__":
    main()

"""Evaluate whether guide-oriented questions retrieve guide chunks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    BM25_TOP_N,
    RELATIONAL_DB_FILE,
    RRF_K,
    VECTOR_TOP_N_FOR_HYBRID,
)
from rag.bm25 import build_bm25_index
from rag.knowledge import load_runtime_knowledge
from rag.retriever import attach_result_metadata, hybrid_retrieve
from storage.relational import RelationalRepository


DEFAULT_EVAL_FILE = "./data/guide_eval.json"


def first_expected_rank(results: List[Dict], expected_guides: List[str]) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        source_id = str(result.get("source_id") or "")
        if any(source_id.startswith(f"guide:{slug}:") for slug in expected_guides):
            return rank
    return None


def evaluate(eval_file: str, top_k: int, bm25_only: bool) -> int:
    with open(eval_file, "r", encoding="utf-8") as file:
        cases = json.load(file)["cases"]

    repository = RelationalRepository(RELATIONAL_DB_FILE)
    docs, items, _ = load_runtime_knowledge(repository)
    bm25 = build_bm25_index(docs)
    model = store = None
    if not bm25_only:
        from rag.embedder import load_model, load_or_compute_embeddings

        model = load_model()
        store = load_or_compute_embeddings(docs, model)

    hits = 0
    reciprocal_rank = 0.0
    for case in cases:
        query = case["query"]
        if bm25_only:
            candidates = bm25.retrieve(query, n=top_k)
        else:
            candidates = hybrid_retrieve(
                query,
                docs,
                store,
                model,
                bm25,
                vector_n=VECTOR_TOP_N_FOR_HYBRID,
                bm25_n=BM25_TOP_N,
                rrf_k=RRF_K,
                top_n=top_k,
            )
        results = attach_result_metadata(candidates, items)
        rank = first_expected_rank(results, case["expected_guides"])
        mark = "PASS" if rank is not None else "FAIL"
        print(f"[{mark}] rank={rank or '-'} | {query}")
        for result in results[:3]:
            print(
                f"  {result.get('source_id') or result.get('source_type')} | "
                f"{result.get('title') or result['text'][:40]}"
            )
        if rank is not None:
            hits += 1
            reciprocal_rank += 1.0 / rank

    total = max(len(cases), 1)
    print(f"Guide Hit@{top_k}: {hits}/{total} = {hits / total:.2%}")
    print(f"Guide MRR: {reciprocal_rank / total:.4f}")
    return 0 if hits == len(cases) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate guide retrieval.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Run a fast lexical smoke test without loading embedding models.",
    )
    args = parser.parse_args()
    raise SystemExit(evaluate(args.eval_file, args.top_k, args.bm25_only))


if __name__ == "__main__":
    main()

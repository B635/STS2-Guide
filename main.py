import os
os.environ["HF_HOME"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

import argparse
from config import (
    KNOWLEDGE_FILE, RETRIEVE_TOP_N, RERANKER_CANDIDATE_N, MULTI_QUERY_PER_SUB_N,
    BM25_TOP_N, VECTOR_TOP_N_FOR_HYBRID, RRF_K,
)
from rag.embedder import load_model, load_or_compute_embeddings
from rag.chat import create_client, rag_chat
from rag.retriever import retrieve, adaptive_retrieve, multi_query_retrieve, hybrid_retrieve, format_context, format_sources
from rag.knowledge import load_knowledge
from rag.router import structured_query
from rag.query_planner import decompose_query
from rag.hyde import generate_hypothetical
from rag.errors import handle_api_error, handle_file_error


def main():
    parser = argparse.ArgumentParser(description="杀戮尖塔2攻略助手")
    parser.add_argument("--reranker", action="store_true", help="启用 Reranker 两阶段精排")
    parser.add_argument("--multi-query", action="store_true", help="启用 Query 分解 + 多次检索")
    parser.add_argument("--hybrid", action="store_true", help="启用 Hybrid 检索（BM25 + 向量 RRF 融合）")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE：LLM 生成假设文档给向量侧用（与 --multi-query 互斥）")
    args = parser.parse_args()

    if args.hyde and args.multi_query:
        print("⚠️  --hyde 与 --multi-query 互斥（都会改写 query），请只开一个")
        return

    try:
        docs, items, index = load_knowledge()
    except Exception as e:
        print(handle_file_error(e, KNOWLEDGE_FILE))
        return

    model = load_model()
    doc_embeddings = load_or_compute_embeddings(docs, model)
    client = create_client()

    reranker = None
    if args.reranker:
        from rag.reranker import load_reranker, rerank
        reranker = load_reranker()
        print("✓ Reranker 已启用")

    bm25_idx = None
    if args.hybrid:
        from rag.bm25 import build_bm25_index
        bm25_idx = build_bm25_index(docs)
        print("✓ Hybrid 检索已启用（BM25 + 向量 RRF）")

    history = []
    print("杀戮尖塔2攻略助手已启动，输入'quit'退出")
    mode_parts = []
    if args.multi_query:
        mode_parts.append("Query 分解")
    if args.hyde:
        mode_parts.append("HyDE 改写")
    if args.hybrid:
        mode_parts.append("Hybrid 检索")
    mode_parts.append("Reranker 精排" if args.reranker else "向量检索")
    print(f"模式：{' + '.join(mode_parts)}")
    print("=" * 40)

    while True:
        question = input("\n你的问题：")
        if question.lower() == "quit":
            print("再见！")
            break
        if not question.strip():
            continue

        try:
            routed = structured_query(question, index, items)
            if routed is not None:
                print(f"  [结构化路由] 命中 {len(routed)} 条")
                results = routed[:RETRIEVE_TOP_N]
            elif args.multi_query:
                sub_queries = decompose_query(question, client)
                if len(sub_queries) > 1:
                    print(f"  [Query 分解] → {sub_queries}")
                candidates = multi_query_retrieve(
                    sub_queries, docs, doc_embeddings, model,
                    n_per_query=MULTI_QUERY_PER_SUB_N,
                )
                if reranker is not None:
                    from rag.reranker import rerank
                    results = rerank(question, candidates, reranker, top_n=RETRIEVE_TOP_N)
                else:
                    results = candidates[:RETRIEVE_TOP_N]
            elif args.hybrid:
                vector_query = None
                if args.hyde:
                    vector_query = generate_hypothetical(question, client)
                    print(f"  [HyDE] 假设文档：{vector_query[:80]}...")
                pool_n = RERANKER_CANDIDATE_N if reranker is not None else RETRIEVE_TOP_N
                candidates = hybrid_retrieve(
                    question, docs, doc_embeddings, model, bm25_idx,
                    vector_n=VECTOR_TOP_N_FOR_HYBRID, bm25_n=BM25_TOP_N,
                    rrf_k=RRF_K, top_n=pool_n,
                    vector_query=vector_query,
                )
                if reranker is not None:
                    from rag.reranker import rerank
                    results = rerank(question, candidates, reranker, top_n=RETRIEVE_TOP_N)
                else:
                    results = candidates[:RETRIEVE_TOP_N]
            elif reranker is not None:
                vector_query = question
                if args.hyde:
                    vector_query = generate_hypothetical(question, client)
                    print(f"  [HyDE] 假设文档：{vector_query[:80]}...")
                from rag.reranker import rerank
                candidates = retrieve(vector_query, docs, doc_embeddings, model, n=RERANKER_CANDIDATE_N)
                results = rerank(question, candidates, reranker, top_n=RETRIEVE_TOP_N)
            elif args.hyde:
                vector_query = generate_hypothetical(question, client)
                print(f"  [HyDE] 假设文档：{vector_query[:80]}...")
                results = retrieve(vector_query, docs, doc_embeddings, model, n=RETRIEVE_TOP_N)
            else:
                results = adaptive_retrieve(question, docs, doc_embeddings, model, client)

            context = format_context(results)
            answer = rag_chat(question, context, history, client)

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

            print(f"\n回答：{answer}")
            print(f"\n参考来源：\n{format_sources(results)}")

        except Exception as e:
            print(handle_api_error(e))


if __name__ == "__main__":
    main()
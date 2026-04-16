import os
os.environ["HF_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

import argparse
import json
from config import KNOWLEDGE_FILE, RETRIEVE_TOP_N, RERANKER_CANDIDATE_N
from rag.embedder import load_model, load_or_compute_embeddings
from rag.chat import create_client, rag_chat
from rag.retriever import retrieve, adaptive_retrieve, format_context, format_sources
from rag.errors import handle_api_error, handle_file_error


def load_knowledge() -> list:
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["docs"]


def main():
    parser = argparse.ArgumentParser(description="杀戮尖塔2攻略助手")
    parser.add_argument("--reranker", action="store_true", help="启用 Reranker 两阶段精排")
    args = parser.parse_args()

    try:
        docs = load_knowledge()
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

    history = []
    print("杀戮尖塔2攻略助手已启动，输入'quit'退出")
    if args.reranker:
        print("模式：向量检索 + Reranker 精排")
    print("=" * 40)

    while True:
        question = input("\n你的问题：")
        if question.lower() == "quit":
            print("再见！")
            break
        if not question.strip():
            continue

        try:
            if reranker is not None:
                from rag.reranker import rerank
                candidates = retrieve(question, docs, doc_embeddings, model, n=RERANKER_CANDIDATE_N)
                results = rerank(question, candidates, reranker, top_n=RETRIEVE_TOP_N)
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
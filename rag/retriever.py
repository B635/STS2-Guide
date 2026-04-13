import numpy as np
from sentence_transformers import SentenceTransformer
from config import RETRIEVE_TOP_N


def retrieve(query: str, docs: list, doc_embeddings: np.ndarray, model: SentenceTransformer, n: int = RETRIEVE_TOP_N):
    query_embedding = model.encode([query])
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = similarities.argsort()[-n:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "text": docs[i],
            "score": float(similarities[i]),
            "index": int(i)
        })

    return results


def format_context(results: list) -> str:
    return "\n".join([r["text"] for r in results])


def format_sources(results: list) -> str:
    lines = []
    for i, r in enumerate(results):
        lines.append(f"  [{i+1}] (相似度{r['score']:.2f}) {r['text'][:30]}...")
    return "\n".join(lines)

def adaptive_retrieve(query: str, docs: list, doc_embeddings: np.ndarray, model: SentenceTransformer, client, n: int = RETRIEVE_TOP_N):
    results = retrieve(query, docs, doc_embeddings, model, n)
    context = format_context(results)

    from openai import OpenAI
    from config import DEEPSEEK_MODEL
    check_response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "判断以下背景知识是否足够回答问题，只回答'够'或'不够'"},
            {"role": "user", "content": f"问题：{query}\n背景知识：{context}"}
        ]
    )

    if "不够" in check_response.choices[0].message.content:
        print("  [自适应检索] 信息不足，扩大检索范围...")
        results = retrieve(query, docs, doc_embeddings, model, n=len(docs))

    return results
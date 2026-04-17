import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from config import (
    RETRIEVE_TOP_N,
    ADAPTIVE_SCORE_THRESHOLD,
    ADAPTIVE_RETRIEVE_STAGES,
    ADAPTIVE_CHECK_CONTEXT_CHARS,
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_norm = b / norm(b)
    return np.dot(a, b_norm.T).flatten()


def get_retrieve_n(query: str) -> int:
    if any(kw in query for kw in ["所有", "全部", "列举", "有哪些", "几个"]):
        return 10
    if any(kw in query for kw in ["区别", "对比", "哪个好", "推荐"]):
        return 6
    return RETRIEVE_TOP_N


def _lexical_boost(query: str, doc: str) -> float:
    boost = 0.0
    domain_keywords = ["角色", "卡牌", "遗物", "药水", "怪物", "boss", "BOSS"]
    for kw in domain_keywords:
        if kw in query and kw in doc:
            boost += 0.08
    return boost


def retrieve(query: str, docs: list, doc_embeddings: np.ndarray, model: SentenceTransformer, n: int = None):
    if n is None:
        n = get_retrieve_n(query)

    query_embedding = model.encode([query])
    dense_scores = cosine_similarity(doc_embeddings, query_embedding[0])

    combined_scores = []
    for i, doc in enumerate(docs):
        combined_scores.append(dense_scores[i] + _lexical_boost(query, doc))
    combined_scores = np.array(combined_scores)

    top_indices = combined_scores.argsort()[-n:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "text": docs[i],
            "score": float(combined_scores[i]),
            "index": int(i)
        })
    return results


def rrf_fuse(ranked_lists: list, k: int, top_n: int, docs: list) -> list:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each doc index picks up 1/(k+rank) from every list it appears in.
    Returns top_n dicts shaped like the other retrievers' outputs.
    """
    scores = {}
    for results in ranked_lists:
        for rank, r in enumerate(results, start=1):
            idx = r["index"]
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        {"text": docs[idx], "score": float(score), "index": int(idx)}
        for idx, score in ordered
    ]


def hybrid_retrieve(
    query: str,
    docs: list,
    doc_embeddings: np.ndarray,
    model: SentenceTransformer,
    bm25_index,
    vector_n: int,
    bm25_n: int,
    rrf_k: int,
    top_n: int,
    vector_query: str = None,
) -> list:
    # HyDE path: caller pre-computed a hypothetical doc for the dense side.
    # BM25 always runs on the raw query so proper nouns / digits stay lexical.
    vq = vector_query if vector_query else query
    vector_results = retrieve(vq, docs, doc_embeddings, model, n=vector_n)
    bm25_results = bm25_index.retrieve(query, n=bm25_n)
    return rrf_fuse([vector_results, bm25_results], k=rrf_k, top_n=top_n, docs=docs)


def multi_query_retrieve(
    sub_queries: list,
    docs: list,
    doc_embeddings: np.ndarray,
    model: SentenceTransformer,
    n_per_query: int,
) -> list:
    seen = {}
    for sub_q in sub_queries:
        results = retrieve(sub_q, docs, doc_embeddings, model, n=n_per_query)
        for r in results:
            idx = r["index"]
            if idx not in seen or r["score"] > seen[idx]["score"]:
                seen[idx] = r
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


def format_context(results: list) -> str:
    # Prefix each doc with [n] so the LLM can cite by index; n is 1-based.
    return "\n".join([f"[{i + 1}] {r['text']}" for i, r in enumerate(results)])


def format_sources(results: list) -> str:
    lines = []
    for i, r in enumerate(results):
        lines.append(f"  [{i+1}] (相似度{r['score']:.2f}) {r['text'][:30]}...")
    return "\n".join(lines)


def _build_adaptive_stages(initial_n: int, docs_count: int) -> list:
    stages = [max(1, min(initial_n, docs_count))]
    for stage in ADAPTIVE_RETRIEVE_STAGES:
        bounded_stage = max(1, min(int(stage), docs_count))
        if bounded_stage > stages[-1]:
            stages.append(bounded_stage)

    if stages[-1] < docs_count:
        stages.append(docs_count)
    return stages


def _is_context_enough(query: str, results: list, client) -> bool:
    if client is None:
        return True

    context = format_context(results)
    if len(context) > ADAPTIVE_CHECK_CONTEXT_CHARS:
        context = context[:ADAPTIVE_CHECK_CONTEXT_CHARS]

    from config import DEEPSEEK_MODEL
    check_response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "判断以下背景知识是否足够回答问题，只回答'够'或'不够'"},
            {"role": "user", "content": f"问题：{query}\n背景知识：{context}"},
        ],
    )
    return "不够" not in check_response.choices[0].message.content


def adaptive_retrieve(query: str, docs: list, doc_embeddings: np.ndarray, model: SentenceTransformer, client, n: int = None):
    initial_n = n if n is not None else get_retrieve_n(query)
    stages = _build_adaptive_stages(initial_n, len(docs))

    last_results = []
    for idx, stage_n in enumerate(stages):
        results = retrieve(query, docs, doc_embeddings, model, n=stage_n)
        last_results = results

        top_score = results[0]["score"] if results else 0.0
        if top_score >= ADAPTIVE_SCORE_THRESHOLD:
            return results

        if _is_context_enough(query, results, client):
            return results

        if idx < len(stages) - 1:
            print(f"  [自适应检索] 信息不足，扩大检索范围到 Top-{stages[idx + 1]}...")

    return last_results
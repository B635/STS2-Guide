from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, EMBEDDING_CACHE_DIR

_reranker_instance = None


def load_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        print(f"加载 Reranker 模型：{RERANKER_MODEL}")
        _reranker_instance = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker_instance


def rerank(query: str, results: list, reranker: CrossEncoder, top_n: int) -> list:
    """
    Rerank candidate results using a CrossEncoder.
    results: list of dicts with 'text', 'score', 'index'
    Returns top_n results sorted by reranker score.
    """
    if not results:
        return results

    pairs = [[query, r["text"]] for r in results]
    rerank_scores = reranker.predict(pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = float(rerank_scores[i])
        r["retrieval_score"] = r["score"]

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]

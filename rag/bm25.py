"""BM25 sparse retrieval over the same `docs` used by the vector index.

Docs are tokenized with jieba so Chinese entity names get kept as single tokens.
Returns results aligned by index with the vector side so the RRF fusion can key on
the position integer directly.
"""
from typing import List
import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


class BM25Index:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.tokenized = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query: str, n: int) -> List[dict]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[-n:][::-1]
        return [
            {"text": self.docs[i], "score": float(scores[i]), "index": int(i)}
            for i in top_indices
        ]


def build_bm25_index(docs: List[str]) -> BM25Index:
    return BM25Index(docs)

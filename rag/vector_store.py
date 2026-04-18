"""Vector store abstraction over the dense index.

The abstraction isolates the index strategy (flat / IVF / HNSW) from the
retriever, so upgrading to a larger-scale backend (IVF, HNSW, or an external
service like Milvus) is a one-line swap at construction time.

Current strategies:
  - `flat`  : `IndexFlatIP`, exact inner product. 100% recall, O(N) search.
              Right for ≤ 10k vectors.
  - `ivf`   : `IndexIVFFlat`, coarse-quantization clustering. Searches only the
              `nprobe` nearest clusters. Trades a little recall for 10-100x
              speedup. Right for 10k-1M vectors. Requires training.

Embeddings are assumed L2-normalized upstream (see `embedder.py`), so inner
product == cosine similarity.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import faiss


class VectorStore:
    """Minimal interface every dense index implementation must satisfy."""

    dim: int
    size: int

    def search(self, query_vec: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for the top-n nearest neighbors.

        `query_vec` is a 1-D array of shape (dim,) OR a 2-D array of shape (1, dim).
        Returned arrays are 1-D, length n.
        """
        raise NotImplementedError


class FaissFlatIP(VectorStore):
    """Exact inner-product search. Use for small corpora (≤ 10k)."""

    def __init__(self, embeddings: np.ndarray):
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dim = vectors.shape[1]
        self.size = vectors.shape[0]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        n = min(n, self.size)
        scores, indices = self.index.search(q, n)
        return indices[0], scores[0]


class FaissIVFFlat(VectorStore):
    """Coarse-quantization search. Not used by default — kept as a drop-in
    for when the corpus outgrows flat search.

    `nlist` is the number of clusters (sqrt(N) is a common starting point);
    `nprobe` is how many clusters to scan at query time (higher = better recall,
    slower query).
    """

    def __init__(self, embeddings: np.ndarray, nlist: int = 64, nprobe: int = 8):
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.dim = vectors.shape[1]
        self.size = vectors.shape[0]

        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = nprobe

    def search(self, query_vec: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(query_vec, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        n = min(n, self.size)
        scores, indices = self.index.search(q, n)
        return indices[0], scores[0]


def build_vector_store(embeddings: np.ndarray, strategy: str = "flat") -> VectorStore:
    """Factory. Callers should use this instead of instantiating directly,
    so the strategy choice lives in one place."""
    if strategy == "flat":
        return FaissFlatIP(embeddings)
    if strategy == "ivf":
        return FaissIVFFlat(embeddings)
    raise ValueError(f"Unknown vector store strategy: {strategy}")

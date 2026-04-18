import hashlib
import json
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR, EMBEDDINGS_FILE, DOCS_HASH_FILE, VECTOR_STORE_STRATEGY
from rag.vector_store import build_vector_store, VectorStore

NORMALIZED_EMBEDDINGS_FILE = "./embeddings_normalized.npy"


def get_docs_hash(docs: list) -> str:
    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "docs": docs,
    }
    stable_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(stable_json.encode("utf-8")).hexdigest()


def _compute_normalized(docs: list, model: SentenceTransformer) -> np.ndarray:
    import os
    current_hash = get_docs_hash(docs)

    if os.path.exists(NORMALIZED_EMBEDDINGS_FILE) and os.path.exists(DOCS_HASH_FILE):
        with open(DOCS_HASH_FILE, "r") as f:
            saved_hash = f.read()
        if saved_hash == current_hash:
            print("从缓存加载归一化向量...")
            return np.load(NORMALIZED_EMBEDDINGS_FILE)

    print("计算向量并缓存...")
    embeddings = model.encode(docs, show_progress_bar=True)
    normalized = embeddings / norm(embeddings, axis=1, keepdims=True)
    np.save(NORMALIZED_EMBEDDINGS_FILE, normalized)
    np.save(EMBEDDINGS_FILE, embeddings)

    with open(DOCS_HASH_FILE, "w") as f:
        f.write(current_hash)

    return normalized


def load_or_compute_embeddings(docs: list, model: SentenceTransformer) -> VectorStore:
    """Compute (or cache-load) normalized embeddings and wrap them in a VectorStore.

    Returns a VectorStore whose `search()` does ANN lookup via FAISS. The
    underlying strategy is picked from `VECTOR_STORE_STRATEGY` in config; for
    ≤ 10k docs `flat` gives exact results at negligible cost.
    """
    normalized = _compute_normalized(docs, model)
    print(f"构建向量索引 (strategy={VECTOR_STORE_STRATEGY}, dim={normalized.shape[1]}, N={normalized.shape[0]})...")
    return build_vector_store(normalized, strategy=VECTOR_STORE_STRATEGY)


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=EMBEDDING_CACHE_DIR)

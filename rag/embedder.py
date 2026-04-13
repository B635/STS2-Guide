import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR, EMBEDDINGS_FILE, DOCS_HASH_FILE


def get_docs_hash(docs: list) -> str:
    return hashlib.md5(str(docs).encode()).hexdigest()


def load_or_compute_embeddings(docs: list, model: SentenceTransformer) -> np.ndarray:
    import os
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(DOCS_HASH_FILE):
        with open(DOCS_HASH_FILE, "r") as f:
            saved_hash = f.read()
        if saved_hash == get_docs_hash(docs):
            print("从缓存加载向量...")
            return np.load(EMBEDDINGS_FILE)

    print("计算向量并缓存...")
    embeddings = model.encode(docs)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(DOCS_HASH_FILE, "w") as f:
        f.write(get_docs_hash(docs))
    return embeddings


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=EMBEDDING_CACHE_DIR)
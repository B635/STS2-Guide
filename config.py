import os
from dotenv import load_dotenv

load_dotenv()

# Keep model and hub cache inside the project to avoid ~/.cache permission issues.
# HF_HOME alone covers transformers + datasets + hub caches; TRANSFORMERS_CACHE is deprecated.
os.environ.setdefault("HF_HOME", "./models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "./models")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_CACHE_DIR = "./models"
EMBEDDINGS_FILE = "./embeddings.npy"
DOCS_HASH_FILE = "./docs_hash.txt"
KNOWLEDGE_FILE = "./data/knowledge.json"

RETRIEVE_TOP_N = 3
ADAPTIVE_SCORE_THRESHOLD = 0.5
ADAPTIVE_RETRIEVE_STAGES = [3, 8, 15]
ADAPTIVE_CHECK_CONTEXT_CHARS = 1800

MAX_HISTORY = 10

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_CANDIDATE_N = 20

# Multi-query (query decomposition + multi-retrieval)
MAX_SUB_QUERIES = 3
MULTI_QUERY_PER_SUB_N = 10

# Hybrid retrieval (BM25 + Vector with RRF fusion)
BM25_TOP_N = 20
VECTOR_TOP_N_FOR_HYBRID = 20
RRF_K = 60

# Vector store backend — "flat" (exact, ≤10k docs) or "ivf" (approximate, ≥10k docs)
VECTOR_STORE_STRATEGY = "flat"
# Oversample multiplier when lexical_boost is applied on dense candidates.
# FAISS returns top-K by pure cosine; boost may re-rank inside that pool, so
# we ask FAISS for more than we need and let the boost reshuffle.
LEXICAL_BOOST_OVERSAMPLE = 3
import os
from dotenv import load_dotenv

load_dotenv()

# Keep model and hub cache inside the project to avoid ~/.cache permission issues.
os.environ.setdefault("HF_HOME", "./models")
os.environ.setdefault("TRANSFORMERS_CACHE", "./models")
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
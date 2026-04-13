import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = "./models"
EMBEDDINGS_FILE = "./embeddings.npy"
DOCS_HASH_FILE = "./docs_hash.txt"
KNOWLEDGE_FILE = "./data/knowledge.json"

RETRIEVE_TOP_N = 3

MAX_HISTORY = 10
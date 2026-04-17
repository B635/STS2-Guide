import json
from config import KNOWLEDGE_FILE

ENTITY_TYPES = ("characters", "cards", "relics", "potions", "monsters")


def load_knowledge(path=None):
    """Return (docs, items, index).

    - docs:  list[str] of embed_text, aligned by index with items.
    - items: list[dict] of full structured entries with an injected `_type` field.
    - index: dict[type -> list[dict]] for structured (non-vector) queries.
    """
    path = path or KNOWLEDGE_FILE
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    index = {}
    items = []
    for t in ENTITY_TYPES:
        bucket = payload.get(t, [])
        typed = []
        for raw in bucket:
            row = dict(raw)
            row["_type"] = t
            typed.append(row)
            items.append(row)
        index[t] = typed

    docs = [it["embed_text"] for it in items]
    return docs, items, index

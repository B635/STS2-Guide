import json
import os

from config import (
    GUIDE_CHUNK_MAX_CHARS,
    GUIDE_CHUNK_OVERLAP_CHARS,
    GUIDES_FILE,
    KNOWLEDGE_FILE,
)
from rag.guide_knowledge import build_guide_items

ENTITY_TYPES = ("characters", "cards", "relics", "potions", "monsters")


def load_knowledge(path=None, guides_path=None, include_guides=True):
    """Return (docs, items, index).

    - docs:  list[str] of embed_text, aligned by index with items.
    - items: list[dict] of facts followed by optional long-form guide chunks.
    - index: dict[type -> list[dict]] for structured (non-vector) queries.
    """
    path = path or KNOWLEDGE_FILE
    guides_path = guides_path or GUIDES_FILE
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

    if include_guides and os.path.exists(guides_path):
        with open(guides_path, "r", encoding="utf-8") as f:
            guide_payload = json.load(f)
        items.extend(
            build_guide_items(
                guide_payload,
                items,
                max_chars=GUIDE_CHUNK_MAX_CHARS,
                overlap_chars=GUIDE_CHUNK_OVERLAP_CHARS,
            )
        )

    docs = [it["embed_text"] for it in items]
    return docs, items, index


def load_runtime_knowledge(repository, path=None, guides_path=None):
    """Load the product runtime with a strict relational/vector boundary.

    Structured entities are imported into SQLite and returned only through
    ``index``. Only unstructured guide chunks are aligned with ``docs`` and
    therefore embedded in the vector store.

    ``load_knowledge`` remains available for legacy evaluation scripts that
    compare historical all-in-one retrieval behavior.
    """
    path = path or KNOWLEDGE_FILE
    guides_path = guides_path or GUIDES_FILE
    repository.sync_catalog(path)
    index = repository.load_catalog_index()
    fact_items = [
        item
        for entity_type in ENTITY_TYPES
        for item in index.get(entity_type, [])
    ]

    guide_items = []
    if os.path.exists(guides_path):
        with open(guides_path, "r", encoding="utf-8") as f:
            guide_payload = json.load(f)
        guide_items = build_guide_items(
            guide_payload,
            fact_items,
            max_chars=GUIDE_CHUNK_MAX_CHARS,
            overlap_chars=GUIDE_CHUNK_OVERLAP_CHARS,
        )

    docs = [item["embed_text"] for item in guide_items]
    return docs, guide_items, index

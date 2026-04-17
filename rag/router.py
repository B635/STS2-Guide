"""Structured query router.

Answers count/list/filter queries and entity-attribute lookups directly from the
per-type index, bypassing vector retrieval. Returns None when the query doesn't
fit a structured pattern — the caller should fall back to vector search.

Results use the same shape as retriever results: {"text", "score", "index"}.
"""
from typing import Optional, List, Dict
from rag.knowledge import ENTITY_TYPES


COUNT_KEYWORDS = ("几个", "多少", "数量", "几位", "几名", "几种", "几张")
TYPE_LABEL = {
    "characters": "角色",
    "cards": "卡牌",
    "relics": "遗物",
    "potions": "药水",
    "monsters": "怪物",
}
TYPE_UNIT = {
    "characters": "个角色",
    "cards": "张卡牌",
    "relics": "个遗物",
    "potions": "个药水",
    "monsters": "个怪物",
}


def _is_pure_count_query(query: str) -> Optional[str]:
    """Return entity type if this is a bare '有几个X' style question, else None."""
    if not any(kw in query for kw in COUNT_KEYWORDS):
        return None
    for t, label in TYPE_LABEL.items():
        if label in query:
            return t
    return None


def _find_entities(query: str, index: Dict) -> List[Dict]:
    """Return items whose name appears in the query, longest-name first.

    Skips names shorter than 2 chars to avoid pathological matches.
    """
    hits = []
    for t in ENTITY_TYPES:
        for item in index.get(t, []):
            name = item.get("name") or ""
            if len(name) >= 2 and name in query:
                hits.append(item)
    hits.sort(key=lambda it: len(it["name"]), reverse=True)

    # Drop matches whose name is a strict substring of an already-accepted longer match.
    accepted = []
    for h in hits:
        if any(h["name"] in a["name"] and h["name"] != a["name"] for a in accepted):
            continue
        accepted.append(h)
    return accepted


def _count_result(entity_type: str, index: Dict) -> Dict:
    count = len(index.get(entity_type, []))
    unit = TYPE_UNIT[entity_type]
    text = f"共有{count}{unit}"
    return {"text": text, "score": 1.0, "index": -1, "source": "structured"}


def _entity_results(matches: List[Dict]) -> List[Dict]:
    return [
        {
            "text": m["embed_text"],
            "score": 1.0,
            "index": -1,
            "source": "structured",
            "item": m,
        }
        for m in matches
    ]


def structured_query(query: str, index: Dict, items: List[Dict]) -> Optional[List[Dict]]:
    """Try to answer from structured data. Return results or None to defer."""
    matches = _find_entities(query, index)

    count_type = _is_pure_count_query(query)
    if count_type is not None and not matches:
        return [_count_result(count_type, index)]

    if not matches:
        return None

    return _entity_results(matches)

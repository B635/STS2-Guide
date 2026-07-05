"""Normalize and chunk long-form community guides for retrieval."""
from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Sequence, Tuple


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WIKI_LINK_RE = re.compile(r"\[\[(?:[^:\]]+:)?([^\]|]+)(?:\|[^\]]+)?\]\]")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

GUIDE_CATEGORY_LABELS = {
    "character": "角色攻略",
    "general": "新手攻略",
    "strategy": "战斗策略",
    "deckbuilding": "构筑攻略",
}

GUIDE_TAG_LABELS = {
    "beginner": "新手 新手入门 入门攻略",
    "basics": "基础 基础玩法",
    "tutorial": "教程 新手教程",
    "loops": "循环",
    "combos": "连段 组合",
    "scaling": "成长",
    "deckbuilding": "牌组构筑 构筑牌组 选牌",
    "relics": "遗物",
    "pathing": "路线规划 规划路线",
    "strategy": "策略",
    "offensive": "进攻 进攻构筑",
    "lethality": "斩杀",
    "burst": "爆发 爆发构筑",
    "encounters": "遭遇 怪物遭遇",
    "monsters": "怪物 敌人 敌人行动模式",
    "boss": "Boss Boss攻略 首领",
}


def _normalize_alias(value: str) -> str:
    return NON_ALNUM_RE.sub("", str(value).lower())


def _clean_heading(value: str) -> str:
    return re.sub(r"\s+#+\s*$", "", value).strip()


def split_markdown_sections(markdown: str) -> List[Tuple[str, str]]:
    """Split Markdown into heading-aware sections while preserving body text."""
    sections: List[Tuple[str, str]] = []
    heading_stack: List[str] = []
    body: List[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(heading_stack) or "Overview", text))
        body.clear()

    for line in str(markdown or "").splitlines():
        match = HEADING_RE.match(line)
        if not match:
            body.append(line)
            continue

        flush()
        level = len(match.group(1))
        heading = _clean_heading(match.group(2))
        heading_stack[:] = heading_stack[: level - 1]
        heading_stack.append(heading)

    flush()
    return sections


def _find_break(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    lower_bound = start + max((target_end - start) // 2, 1)
    candidates = [
        text.rfind("\n", lower_bound, target_end),
        text.rfind(". ", lower_bound, target_end),
        text.rfind("。", lower_bound, target_end),
        text.rfind(" ", lower_bound, target_end),
    ]
    split_at = max(candidates)
    return split_at + 1 if split_at >= lower_bound else target_end


def split_section_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split one section on nearby semantic boundaries with bounded overlap."""
    normalized = re.sub(r"\n{3,}", "\n\n", str(text or "")).strip()
    if not normalized:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = _find_break(normalized, start, min(start + max_chars, len(normalized)))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break

        next_start = max(end - overlap_chars, start + 1)
        while next_start < end and not normalized[next_start].isspace():
            next_start += 1
        start = min(next_start + 1, end)

    return chunks


def chunk_markdown(
    markdown: str,
    max_chars: int,
    overlap_chars: int,
) -> List[Dict]:
    chunks: List[Dict] = []
    ordinal = 0
    for section, body in split_markdown_sections(markdown):
        for content in split_section_text(body, max_chars, overlap_chars):
            chunks.append(
                {
                    "ordinal": ordinal,
                    "section": section,
                    "content": content,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            )
            ordinal += 1
    return chunks


def build_entity_aliases(items: Sequence[Dict]) -> Dict[str, List[str]]:
    """Map normalized game IDs to official Chinese display names."""
    aliases: Dict[str, List[str]] = {}
    for item in items:
        entity_id = _normalize_alias(item.get("id", ""))
        name = str(item.get("name", "")).strip()
        if not entity_id or not name:
            continue
        aliases.setdefault(entity_id, [])
        if name not in aliases[entity_id]:
            aliases[entity_id].append(name)
    return aliases


def _linked_entity_names(guide: Dict, text: str, aliases: Dict[str, List[str]]) -> List[str]:
    keys = {_normalize_alias(match) for match in WIKI_LINK_RE.findall(text)}
    keys.update(_normalize_alias(tag) for tag in guide.get("tags", []) or [])
    keys.add(_normalize_alias(guide.get("character", "")))

    names = {
        name
        for key in keys
        if key
        for name in aliases.get(key, [])
    }
    return sorted(names)


def _iter_guides(payload) -> Iterable[Dict]:
    if isinstance(payload, dict):
        guides = payload.get("guides", [])
    else:
        guides = payload
    if not isinstance(guides, list):
        return []
    return (guide for guide in guides if isinstance(guide, dict))


def build_guide_items(
    payload,
    fact_items: Sequence[Dict],
    max_chars: int,
    overlap_chars: int,
) -> List[Dict]:
    """Convert fetched guide documents into retrieval-aligned chunk items."""
    aliases = build_entity_aliases(fact_items)
    items: List[Dict] = []

    for guide in _iter_guides(payload):
        slug = str(guide.get("slug") or guide.get("id") or "").strip()
        title = str(guide.get("title") or slug).strip()
        content = str(guide.get("content") or "").strip()
        if not slug or not content:
            continue

        category = str(guide.get("category") or "strategy").strip()
        category_label = GUIDE_CATEGORY_LABELS.get(category, "攻略")
        tag_labels = [
            GUIDE_TAG_LABELS[tag_key]
            for tag in guide.get("tags", []) or []
            if (tag_key := _normalize_alias(tag)) in GUIDE_TAG_LABELS
        ]
        source_url = f"https://spire-codex.com/guides/{slug}"
        original_url = str(guide.get("website") or "").strip() or None

        for chunk in chunk_markdown(content, max_chars, overlap_chars):
            entity_names = _linked_entity_names(guide, chunk["content"], aliases)
            chinese_terms = " ".join(entity_names[:5])
            retrieval_header = f"杀戮尖塔2 攻略 {category_label}"
            if tag_labels:
                retrieval_header += f" {' '.join(tag_labels)}"
            if chinese_terms:
                retrieval_header += f" {chinese_terms}"

            items.append(
                {
                    "id": f"guide:{slug}:{chunk['ordinal']}",
                    "name": title,
                    "_type": "guides",
                    "embed_text": f"{retrieval_header}\n{chunk['content']}",
                    "guide_slug": slug,
                    "chunk_ordinal": chunk["ordinal"],
                    "chunk_hash": chunk["content_hash"],
                    "section": chunk["section"],
                    "source_title": title,
                    "source_author": str(guide.get("author") or "").strip() or None,
                    "source_url": source_url,
                    "original_url": original_url,
                    "source_language": "eng",
                    "published_at": guide.get("date"),
                    "updated_at": guide.get("updated"),
                    "category": category,
                    "difficulty": guide.get("difficulty"),
                    "character": guide.get("character"),
                    "tags": list(guide.get("tags") or []),
                    "linked_entities": entity_names,
                }
            )

    return items

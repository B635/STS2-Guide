"""History-aware query rewriting for multi-turn retrieval.

Problem: a follow-up like "他的血量呢" has no entity name, so BM25 finds
nothing and dense retrieval drifts. The LLM sees the history in the final
generation call, but the retrieval pipeline does not.

Solution: before retrieval, ask the LLM to rewrite the query into a
standalone form using the last few turns of history. If the question is
already self-contained, the rewriter returns it unchanged.

Post-validation is two-layer:
  1. entity-level: any entity name appearing in the rewrite must also
     appear in the original query or history. Catches the LLM inventing
     a brand-new entity.
  2. token-level: any 2+ char noun added by the rewrite (present in
     rewrite but not in original) must be an entity name from the
     knowledge index, or a substring of one. Catches the LLM splicing
     an unrelated noun from an earlier assistant reply when the current
     query was already standalone (e.g. "有哪些角色" → "初手牌有哪些角色").
     Non-entity nouns are the tell-tale signal of this failure mode.

Either layer failing → fall back to original query. A slightly worse
retrieval beats confidently retrieving the wrong thing.
"""
import jieba
import jieba.posseg as pseg
from config import DEEPSEEK_MODEL
from rag.router import _find_entities

REWRITE_SYSTEM = "你是查询改写助手，只输出改写后的问题，不加任何解释、不加引号、不加前缀。"

REWRITE_USER_TEMPLATE = """把用户最新的问题改写成不依赖对话历史的独立问题，以便检索系统能正确理解。

规则：
1. 如果最新问题本身已经是一个完整独立的问句（含有明确的主题词，如"有哪些 X"、"什么是 X"、"X 是多少"），**原样返回**，一个字都不要改。
2. 只有当最新问题包含代词（他/她/它/这个/那个/这些）或省略了主语时，才结合对话历史把这些指代替换成具体的实体名。
3. **禁止从上一轮助手的回复中摘取与"主语补全"无关的其他名词**——你能从助手回复里提取的只有实体名（角色/卡牌/遗物/药水/怪物），其他词语一律不要带进新问题。
4. 只输出改写后的问题本身，不加解释、不加引号。

对话历史：
{history}

最新问题：{query}"""


def _format_history(history: list, n_recent: int = 4) -> str:
    if not history:
        return "(无)"
    recent = history[-n_recent:]
    lines = []
    for msg in recent:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _entity_names_in(text: str, index: dict) -> set:
    return {it["name"] for it in _find_entities(text, index)}


def _all_entity_names(index: dict) -> set:
    return {it["name"] for bucket in index.values() for it in bucket}


def _validate(rewritten: str, original: str, history: list, index: dict) -> bool:
    """Two-layer hallucination guard.

    Layer 1 (entity-level): any entity from the knowledge index that appears
    in the rewrite must also appear in the original query or history.

    Layer 2 (token-level): any 2+ char noun introduced by the rewrite but
    not present in the original query must be an entity name (or a
    substring of one). Blocks the rewriter from splicing unrelated nouns
    out of an earlier assistant reply when the current query was already
    self-contained.
    """
    # Layer 1
    rewrite_ents = _entity_names_in(rewritten, index)
    allowed_text = original + " " + " ".join(m["content"] for m in history)
    allowed_ents = _entity_names_in(allowed_text, index)
    if not rewrite_ents.issubset(allowed_ents):
        return False

    # Layer 2
    all_entities = _all_entity_names(index)
    original_tokens = {t for t in jieba.lcut(original) if len(t) >= 2}

    for word, flag in pseg.cut(rewritten):
        if len(word) < 2 or not flag.startswith("n"):
            continue
        if word in original_tokens:
            continue
        if word in all_entities:
            continue
        if any(word in ent for ent in all_entities):
            continue
        return False

    return True


def rewrite_query(query: str, history: list, client, index: dict = None) -> str:
    """Return a standalone version of `query`, or `query` itself if no rewrite needed.

    `index` is optional; when provided, post-validation rejects rewrites that
    smuggle in entity names the LLM invented.
    """
    if not history:
        return query

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": REWRITE_USER_TEMPLATE.format(
                history=_format_history(history), query=query,
            )},
        ],
    )
    rewritten = response.choices[0].message.content.strip()

    if not rewritten or rewritten == query:
        return query

    if index is not None and not _validate(rewritten, query, history, index):
        return query

    return rewritten

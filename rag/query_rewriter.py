"""History-aware query rewriting for multi-turn retrieval.

Problem: a follow-up like "他的血量呢" has no entity name, so BM25 finds
nothing and dense retrieval drifts. The LLM sees the history in the final
generation call, but the retrieval pipeline does not.

Solution: before retrieval, ask the LLM to rewrite the query into a
standalone form using the last few turns of history. If the question is
already self-contained, the rewriter returns it unchanged.

Post-validation: the rewriter is itself an LLM and can hallucinate entity
names. After rewriting, we check that any entities appearing in the rewrite
also appear in the original query OR in the history. If a new entity was
injected from the model's prior, we fall back to the original query — a
slightly worse retrieval beats confidently retrieving the wrong entity.
"""
from config import DEEPSEEK_MODEL
from rag.router import _find_entities

REWRITE_SYSTEM = "你是查询改写助手，只输出改写后的问题，不加任何解释、不加引号、不加前缀。"

REWRITE_USER_TEMPLATE = """把用户最新的问题改写成不依赖对话历史的独立问题，以便检索系统能正确理解。

规则：
1. 如果最新问题本身已经完整独立（含有完整的实体名、没有代词引用），**原样返回**，一个字都不要改。
2. 如果最新问题包含代词（他/她/它/这个/那个/这些）或省略了主语，结合对话历史把这些指代替换成具体的实体名。
3. 只输出改写后的问题本身，不要加解释。

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


def _validate(rewritten: str, original: str, history: list, index: dict) -> bool:
    """Reject rewrites that introduce entity names not present in original or history."""
    rewrite_ents = _entity_names_in(rewritten, index)
    if not rewrite_ents:
        return True

    allowed_text = original + " " + " ".join(m["content"] for m in history)
    allowed_ents = _entity_names_in(allowed_text, index)
    return rewrite_ents.issubset(allowed_ents)


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

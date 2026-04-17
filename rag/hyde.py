"""HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Idea: the raw user query and the document share little surface overlap, so
the query embedding lands far from the relevant `embed_text`. We ask the LLM
to write a *hypothetical* document that *would* answer the query, then
retrieve with that. The hypothetical doc lives in the same distribution as
the real docs, so cosine similarity gets a much straighter shot.

The hybrid pipeline pairs HyDE with BM25: HyDE text goes to the dense side
(semantic matching on a doc-shaped string), raw query goes to BM25 (keeps
the original proper nouns / digits intact). See `hybrid_retrieve(vector_query=...)`.
"""
from config import DEEPSEEK_MODEL

HYDE_PROMPT = (
    "你是杀戮尖塔2知识库的摘要员。根据用户问题，写一段 30-80 字的假设性知识条目"
    "（可以编造具体数值，只求形式像数据库条目即可）。输出格式参照：\n"
    "  卡牌XX（角色，稀有度，类型，费用N）：描述。\n"
    "  遗物XX（稀有度遗物，shared）：描述。\n"
    "  药水XX（稀有度）：描述。\n"
    "  角色XX：初始血量N点，初始遗物XX……\n"
    "  怪物XX（Boss/Elite/Normal），血量N，技能：XX，遭遇：XX。\n"
    "只输出条目正文，不要解释、问候或列序号。"
)


def generate_hypothetical(query: str, client) -> str:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content.strip()

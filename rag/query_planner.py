import json
from config import DEEPSEEK_MODEL, MAX_SUB_QUERIES


DECOMPOSE_PROMPT = """判断下面的用户问题是否需要拆分成多个子问题来检索游戏知识库（杀戮尖塔2）。

【判断标准】
- 简单事实问题（"X的血量"、"Y是什么"）→ 不拆分，sub_queries 只含原问题
- 涉及多个实体或多个维度（"X和Y怎么对比"、"X打Y怎么配卡"）→ 拆成 2-3 个子问题
- 综合分析问题（"X适合什么打法"、"X强在哪"）→ 拆成属性/卡牌/对策等多个角度

【输出格式】
只输出 JSON：{{"sub_queries": ["子问题1", "子问题2", ...]}}
最多 {max_n} 个子问题。每个子问题应能独立在知识库中检索。

【用户问题】
{query}"""


def decompose_query(query: str, client) -> list:
    if client is None:
        return [query]

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{
                "role": "user",
                "content": DECOMPOSE_PROMPT.format(query=query, max_n=MAX_SUB_QUERIES),
            }],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        sub_queries = data.get("sub_queries", [])

        if not isinstance(sub_queries, list) or not sub_queries:
            return [query]

        cleaned = [str(q).strip() for q in sub_queries if str(q).strip()]
        return cleaned[:MAX_SUB_QUERIES] if cleaned else [query]

    except Exception:
        return [query]

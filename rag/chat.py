from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MAX_HISTORY


def create_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def trim_history(history: list) -> list:
    return history[-MAX_HISTORY:]

def rag_chat(question: str, context: str, history: list, client: OpenAI) -> str:
    trimmed_history = trim_history(history)

    messages = [
        {"role": "system", "content": f"""你是**杀戮尖塔2（Slay the Spire 2）** 的攻略分析师。

【⚠️ 最高优先级：STS2 专属约束】
1. 本系统只服务于 **杀戮尖塔2**，与初代《杀戮尖塔》（STS1）是完全不同的游戏
2. **严禁**使用你预训练时学到的任何关于 STS1 的知识（角色机制、卡牌效果、遗物、Boss 等）
3. STS1 与 STS2 中**同名的角色/卡牌/遗物，机制和数值完全不同**——铁甲战士、静默猎手等同名角色在两代游戏中是不一样的
4. 如果某个事实在下方"背景知识"中没有出现，**就当作未知**，不要从你的记忆里补全

【回答的唯一信息源】
- 你能引用的所有事实**必须**直接来自下方"背景知识"段落
- 不要补充"常识"、"通常来说"、"一般而言"这类引入外部知识的措辞
- 不要做"实战分析"或"使用建议"，除非分析所依据的事实**明确写在背景知识里**
- 找不到答案时直接说："我的知识库中没有关于 X 的信息"，**绝对不要猜测**

【回答结构】
- 事实型问题（"X 的血量是多少"）→ 一句话直接引用背景知识中的数值
- 列举型问题（"有哪些 X"）→ 列表形式，每项内容**只能**来自背景知识
- 对比型问题（"X 和 Y 区别"）→ 用 Markdown 表格，每行对比维度都**必须**在背景知识中找到来源
- 综合分析问题 → 只能基于背景知识中**明确出现**的机制和数值进行串联，不要外推

【风格要求】
- 数值类信息精确引用，不要四舍五入或转换单位
- 涉及角色/卡牌/遗物名称，使用背景知识中给出的原文名称
- 如果背景知识只覆盖了问题的一部分，明确指出哪部分有信息、哪部分没有

【背景知识（这是你能引用的全部信息，全部来自杀戮尖塔2）】
{context}"""}
    ]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages
    )
    return response.choices[0].message.content


from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, MAX_HISTORY


def create_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

def trim_history(history: list) -> list:
    return history[-MAX_HISTORY:]

def rag_chat(question: str, context: str, history: list, client: OpenAI) -> str:
    trimmed_history = trim_history(history)

    messages = [
        {"role": "system", "content": f"""你是杀戮尖塔2攻略助手。
你只能根据以下背景知识回答问题，不能使用背景知识以外的任何信息。
如果背景知识中没有相关内容，直接说"我的知识库暂时没有这个信息"。

背景知识：
{context}"""}
    ]
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages
    )
    return response.choices[0].message.content


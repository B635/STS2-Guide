import os
os.environ["HF_HOME"] = "./models"
os.environ["TRANSFORMERS_CACHE"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

import json
import streamlit as st
from config import KNOWLEDGE_FILE, RETRIEVE_TOP_N, RERANKER_CANDIDATE_N
from rag.embedder import load_model, load_or_compute_embeddings
from rag.chat import create_client, rag_chat
from rag.retriever import retrieve, adaptive_retrieve, format_context, format_sources
from rag.errors import handle_api_error, handle_file_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="杀戮尖塔2 攻略助手",
    page_icon="🗡️",
    layout="wide",
)

# ── Load resources (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载知识库和模型...")
def load_resources():
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)["docs"]
    model = load_model()
    embeddings = load_or_compute_embeddings(docs, model)
    client = create_client()
    return docs, model, embeddings, client


@st.cache_resource(show_spinner="正在加载 Reranker 模型...")
def load_reranker_cached():
    from rag.reranker import load_reranker
    return load_reranker()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 设置")

    use_reranker = st.toggle("启用 Reranker 精排", value=True, help="使用 Cross-Encoder 对候选结果重新排序，提升检索精度")
    use_adaptive = st.toggle("启用自适应检索", value=True, help="不确定时自动扩大检索范围")

    top_n = st.slider("返回文档数", min_value=1, max_value=10, value=RETRIEVE_TOP_N)
    candidate_n = st.slider(
        "Reranker 候选池大小",
        min_value=5,
        max_value=50,
        value=RERANKER_CANDIDATE_N,
        disabled=not use_reranker,
    )

    st.divider()
    st.caption("📚 数据来源：Spire Codex")
    st.caption("🤖 生成模型：DeepSeek")
    st.caption("🔍 嵌入模型：multilingual-MiniLM-L12")
    if use_reranker:
        st.caption("⚡ Reranker：BAAI/bge-reranker-base")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ── Main UI ──────────────────────────────────────────────────────────────────
st.title("🗡️ 杀戮尖塔2 攻略助手")
st.caption("基于 RAG 架构，支持角色、卡牌、遗物、药水、怪物查询")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Load resources
try:
    docs, model, embeddings, client = load_resources()
except Exception as e:
    st.error(handle_file_error(e, KNOWLEDGE_FILE))
    st.stop()

reranker = None
if use_reranker:
    try:
        reranker = load_reranker_cached()
    except Exception as e:
        st.warning(f"Reranker 加载失败，将使用基础检索：{e}")
        use_reranker = False

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 参考来源", expanded=False):
                for src in msg["sources"]:
                    score_label = f"rerank: {src['rerank_score']:.3f}" if "rerank_score" in src else f"score: {src['score']:.3f}"
                    st.markdown(f"**[{score_label}]** {src['text'][:100]}...")

# Chat input
if question := st.chat_input("输入你的问题，例如：铁甲战士初始血量多少？"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieval + generation
    with st.chat_message("assistant"):
        with st.status("🔍 正在检索相关知识...", expanded=True) as status:
            try:
                # Step 1: Retrieve candidates
                if use_adaptive and not use_reranker:
                    results = adaptive_retrieve(question, docs, embeddings, model, client, n=top_n)
                    status.write(f"自适应检索完成，获取 {len(results)} 条文档")
                elif use_reranker:
                    from rag.reranker import rerank as do_rerank
                    candidates = retrieve(question, docs, embeddings, model, n=candidate_n)
                    status.write(f"向量检索：召回 {len(candidates)} 条候选文档")
                    results = do_rerank(question, candidates, reranker, top_n=top_n)
                    status.write(f"Reranker 精排：筛选出 Top-{top_n} 文档")
                else:
                    results = retrieve(question, docs, embeddings, model, n=top_n)
                    status.write(f"向量检索完成，获取 {len(results)} 条文档")

                # Step 2: Build context and generate answer
                context = format_context(results)
                status.write("📝 正在生成回答...")
                answer = rag_chat(question, context, st.session_state.history, client)
                status.update(label="✅ 完成", state="complete", expanded=False)

            except Exception as e:
                answer = handle_api_error(e)
                results = []
                status.update(label="❌ 出错", state="error")

        # Display answer
        st.markdown(answer)

        # Display sources
        if results:
            with st.expander("📄 参考来源", expanded=True):
                for i, r in enumerate(results, 1):
                    if "rerank_score" in r:
                        score_str = f"rerank: **{r['rerank_score']:.3f}** | retrieval: {r['retrieval_score']:.3f}"
                    else:
                        score_str = f"score: **{r['score']:.3f}**"
                    st.markdown(f"`[{i}]` {score_str}  \n{r['text'][:150]}...")

    # Update state
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})

    source_data = [
        {k: v for k, v in r.items() if k in ("text", "score", "rerank_score", "retrieval_score")}
        for r in results
    ]
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": source_data,
    })

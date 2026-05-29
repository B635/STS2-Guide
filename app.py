import os
os.environ["HF_HOME"] = "./models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

import re
import streamlit as st
from config import (
    KNOWLEDGE_FILE, RETRIEVE_TOP_N, RERANKER_CANDIDATE_N, MULTI_QUERY_PER_SUB_N,
    BM25_TOP_N, VECTOR_TOP_N_FOR_HYBRID, RRF_K,
)


def format_citations(answer: str) -> str:
    # [?] is the model's "no evidence" marker — render in red.
    # [n] becomes a monospace chip aligned with the sources list below.
    answer = answer.replace("[?]", " :red[**[?无依据]**]")
    return re.sub(r"\[(\d+)\]", r" `[\1]`", answer)
from rag.embedder import load_model, load_or_compute_embeddings
from rag.chat import create_client, rag_chat
from rag.retriever import retrieve, adaptive_retrieve, multi_query_retrieve, hybrid_retrieve, format_context, format_sources
from rag.knowledge import load_knowledge
from rag.router import structured_query
from rag.query_planner import decompose_query
from rag.hyde import generate_hypothetical
from rag.query_rewriter import rewrite_query
from rag.errors import handle_api_error, handle_file_error
from rag.agent import AgentConfig, format_agent_trace, run_agent
from rag.langgraph_agent import run_langgraph_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="杀戮尖塔2 攻略助手",
    page_icon="🗡️",
    layout="wide",
)

# ── Load resources (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="正在加载知识库和模型...")
def load_resources():
    docs, items, index = load_knowledge()
    model = load_model()
    embeddings = load_or_compute_embeddings(docs, model)
    client = create_client()
    return docs, items, index, model, embeddings, client


@st.cache_resource(show_spinner="正在加载 Reranker 模型...")
def load_reranker_cached():
    from rag.reranker import load_reranker
    return load_reranker()


@st.cache_resource(show_spinner="正在构建 BM25 索引...")
def load_bm25_cached(docs):
    from rag.bm25 import build_bm25_index
    return build_bm25_index(list(docs))


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 设置")

    use_agent = True
    st.caption("默认使用 Tool-Using Agent 自动选择结构化查询、Hybrid、HyDE 或多查询检索工具")
    use_langgraph = st.toggle("使用 LangGraph 工作流", value=False, help="将 Query 改写、路由/规划、工具执行和生成建模为 LangGraph 状态图")
    use_reranker = st.toggle("启用 Reranker 精排", value=True, help="使用 Cross-Encoder 对候选结果重新排序，提升检索精度")
    use_router = False
    use_hybrid = False
    use_multi_query = False
    use_hyde = False
    use_adaptive = False

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
    docs, items, index, model, embeddings, client = load_resources()
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

bm25_idx = None
if use_agent:
    try:
        bm25_idx = load_bm25_cached(tuple(docs))
    except Exception as e:
        st.warning(f"BM25 索引构建失败：{e}")
        st.warning("Agent 将回退到不依赖 BM25 的工具。")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if msg["role"] == "assistant":
            content = format_citations(content)
        st.markdown(content)
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
                if use_agent:
                    runner = run_langgraph_agent if use_langgraph else run_agent
                    agent_result = runner(
                        question,
                        st.session_state.history,
                        docs,
                        items,
                        index,
                        embeddings,
                        model,
                        client,
                        bm25_index=bm25_idx,
                        reranker=reranker,
                        config=AgentConfig(top_n=top_n, candidate_n=candidate_n),
                    )
                    answer = agent_result.answer
                    results = agent_result.results
                    for line in format_agent_trace(agent_result).splitlines():
                        status.write(line)
                else:
                    # Step 0: History-aware query rewrite. Follow-ups like "他的血量呢"
                    # have no entity for retrieval to latch onto; rewrite uses the
                    # last few turns to produce a standalone query. Generation still
                    # uses the user's original wording (LLM already has history).
                    retrieve_query = rewrite_query(question, st.session_state.history, client, index)
                    if retrieve_query != question:
                        status.write(f"✍️ Query 改写：{question} → {retrieve_query}")

                    # Step 1: Try structured router, fall back to vector pipeline
                    routed = structured_query(retrieve_query, index, items) if use_router else None
                    if routed is not None:
                        status.write(f"🎯 结构化路由命中：{len(routed)} 条")
                        results = routed[:top_n]
                    elif use_multi_query:
                        sub_queries = decompose_query(retrieve_query, client)
                        if len(sub_queries) > 1:
                            status.write(f"🧩 Query 拆解为 {len(sub_queries)} 个子问题：")
                            for sq in sub_queries:
                                status.write(f"  • {sq}")
                        else:
                            status.write("Query 无需拆解，按单查询处理")
                        candidates = multi_query_retrieve(
                            sub_queries, docs, embeddings, model,
                            n_per_query=MULTI_QUERY_PER_SUB_N,
                        )
                        status.write(f"多查询合并：去重后 {len(candidates)} 条候选")
                        if use_reranker:
                            from rag.reranker import rerank as do_rerank
                            results = do_rerank(retrieve_query, candidates, reranker, top_n=top_n)
                            status.write(f"Reranker 精排：筛选出 Top-{top_n} 文档")
                        else:
                            results = candidates[:top_n]
                    elif use_hybrid:
                        vector_query = None
                        if use_hyde:
                            vector_query = generate_hypothetical(retrieve_query, client)
                            status.write(f"🧠 HyDE 假设文档：{vector_query[:60]}...")
                        pool_n = candidate_n if use_reranker else top_n
                        candidates = hybrid_retrieve(
                            retrieve_query, docs, embeddings, model, bm25_idx,
                            vector_n=VECTOR_TOP_N_FOR_HYBRID, bm25_n=BM25_TOP_N,
                            rrf_k=RRF_K, top_n=pool_n,
                            vector_query=vector_query,
                        )
                        status.write(f"🔀 Hybrid 检索（BM25+向量 RRF）：融合后 {len(candidates)} 条候选")
                        if use_reranker:
                            from rag.reranker import rerank as do_rerank
                            results = do_rerank(retrieve_query, candidates, reranker, top_n=top_n)
                            status.write(f"Reranker 精排：筛选出 Top-{top_n} 文档")
                        else:
                            results = candidates[:top_n]
                    elif use_adaptive and not use_reranker and not use_hyde:
                        results = adaptive_retrieve(retrieve_query, docs, embeddings, model, client, n=top_n)
                        status.write(f"自适应检索完成，获取 {len(results)} 条文档")
                    elif use_reranker:
                        vector_query = retrieve_query
                        if use_hyde:
                            vector_query = generate_hypothetical(retrieve_query, client)
                            status.write(f"🧠 HyDE 假设文档：{vector_query[:60]}...")
                        from rag.reranker import rerank as do_rerank
                        candidates = retrieve(vector_query, docs, embeddings, model, n=candidate_n)
                        status.write(f"向量检索：召回 {len(candidates)} 条候选文档")
                        results = do_rerank(retrieve_query, candidates, reranker, top_n=top_n)
                        status.write(f"Reranker 精排：筛选出 Top-{top_n} 文档")
                    elif use_hyde:
                        vector_query = generate_hypothetical(retrieve_query, client)
                        status.write(f"🧠 HyDE 假设文档：{vector_query[:60]}...")
                        results = retrieve(vector_query, docs, embeddings, model, n=top_n)
                        status.write(f"向量检索完成，获取 {len(results)} 条文档")
                    else:
                        results = retrieve(retrieve_query, docs, embeddings, model, n=top_n)
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

        # Display answer with citation chips
        st.markdown(format_citations(answer))

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

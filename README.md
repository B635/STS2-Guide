# STS2-Guide：杀戮尖塔2 RAG 攻略助手

基于 **RAG（Retrieval-Augmented Generation）+ Tool-Using Agent** 架构的游戏攻略问答系统。用户用自然语言提问，系统从 1048 条结构化知识中检索相关内容，可经过"结构化路由 → HyDE 改写（可选） → Hybrid 检索（BM25 + 向量 RRF 融合） → Reranker 精排"管线后交由 LLM **带引用**生成回答（每句标注 `[n]`，不可追溯的结论强制标 `[?]`）。Agent 模式下，系统会先根据问题类型自动选择结构化查询、Hybrid 检索、HyDE 增强或 Query 分解等工具，再执行检索、生成和规则化校验；校验失败时会触发一次补救检索/重生成闭环。同时提供可选 **LangGraph** 状态图实现，将 Query 改写、路由/规划、工具执行、带引用生成、答案校验和失败修复封装为可观测工作流。

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 向量模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 维多语言 Bi-Encoder，支持中英文 |
| Reranker | `BAAI/bge-reranker-base` | Cross-Encoder 精排模型，提升检索精度 |
| 稀疏检索 | `rank-bm25` + `jieba` 分词 | BM25Okapi 词频打分，捕捉专名/数值等向量易丢失的字面信号 |
| 融合策略 | Reciprocal Rank Fusion (k=60) | 只用排名不用分数，天然解决 BM25 与余弦尺度不可比的问题 |
| 向量索引 | FAISS (IndexFlatIP / IndexIVFFlat) | 通过 VectorStore 抽象层封装，策略可切换 |
| 生成模型 | DeepSeek API | OpenAI 兼容接口，支持中文 |
| Web UI | Vue 3 + Vite | 前后端分离聊天界面，展示工具轨迹、引用来源和校验结果 |
| API 服务 | FastAPI | 将 Agent 能力封装为 `/chat` 接口，供前端或外部系统调用 |
| 知识库 | 按类型结构化 JSON（1048 条，含 `embed_text`） | 通过 Spire Codex API 自动拉取 |
| 语言 | Python 3.9（推荐 conda 虚拟环境） + Vue 3 | |

## 核心功能

### 检索能力
- **Tool-Using Agent / Function Calling**：新增轻量级 Agent 编排层，优先通过 OpenAI-compatible `tools/tool_calls` 让 LLM 在 `hybrid_search`、`hyde_hybrid_search`、`multi_query_search`、`vector_search` 中选择工具并填写结构化参数；若模型接口不支持 tool calls，则自动回退到 JSON Planner + 启发式兜底。Planner 输出检索 query、top_n、过滤提示、复杂问题 sub_queries，并记录可解释执行轨迹
- **LangGraph 工作流（可选）**：将 `rewrite → route_or_plan → execute_tool → generate → verify → repair` 建模为状态图，校验失败时最多触发一次补救检索并重新生成，便于观察 Agent 每一步状态流转，并保留默认手写 Agent 作为无框架兜底
- **结构化路由**：识别查询中的实体名或"有几个/多少"类计数问题时，直接从按类型切分的知识库返回确定性答案，绕过向量检索以避免在 1000+ 相似短文档中的召回噪声
- **多轮 Query 改写（History-aware Retrieval）**：follow-up 问题（"他的血量呢"）先由 LLM 结合历史重写成独立 query 再检索，并通过"实体白名单"后置校验拦截模型幻觉出的新实体，失败时回退原 query
- **HyDE 假设文档改写**：LLM 先生成一段与 `embed_text` 同构的假设性条目做向量侧检索 query，原始 query 仍留给 BM25 —— 解决"描述性提问 vs 卡片化文档"的分布鸿沟
- **Hybrid 检索（BM25 + Vector + RRF）**：jieba 中文分词后的 BM25 稀疏召回与稠密向量召回并行取 Top-20，再用 Reciprocal Rank Fusion 按排名融合，修正纯向量对专名/数值的弱召回
- **语义检索**：路由未命中时走向量召回，基于余弦相似度 + 词法加权
- **Reranker 两阶段精排**：先用 Bi-Encoder（或 Hybrid）粗召回 Top-20 候选，再用 Cross-Encoder 逐对精排取 Top-N
- **复杂任务拆解 / Query 分解**：Planner 可为比较、多实体、多维度问题直接输出 1-3 个子问题；缺省时回退到 Query Planner，再分别检索、合并去重（旧版固定 pipeline 中 `--multi-query` 仍可单独使用）
- **自适应检索**：调用 LLM 判断当前上下文是否充足，不足时自动扩大检索范围（3 → 8 → 15 → 全量）

### 生成与对话
- **多轮对话**：保留对话历史，支持上下文关联的连续提问
- **历史截断**：自动截断过长对话历史，避免 token 超限
- **来源透明**：每次回答显示参考来源、相似度分数及 Reranker 分数
- **句级引用与答案校验（Citation / Grounding）**：每个陈述句末尾带 `[n]` 指向背景知识编号，无依据的结论强制用 `[?]` 标注；在线 verifier 节点检查引用覆盖、引用编号有效性和数字可追溯性，失败时扩大检索上下文并重新生成，离线评测脚本量化"引用编号有效率 / 数字可追溯率"

### 工程能力
- **FastAPI + Vue 前后端分离**：`api.py` 将 `run_agent` / `run_langgraph_agent` 封装为标准 `/chat` 接口，Vue 前端负责聊天交互、参数控制、工具轨迹、引用来源和 Verifier 结果展示
- **向量缓存**：归一化向量持久化存储，知识库不变时跳过重复计算
- **知识库自动更新**：`fetch_knowledge.py` 从公开 API 自动拉取最新游戏数据
- **量化评测体系**：53 条标注测试集（含开放类问题），支持 Hit@K、MRR 指标评测及 Baseline/Reranker/Router 对比
- **错误处理**：覆盖 API 限流、认证失败、网络异常等常见错误

## 快速开始

### 1. 安装依赖

推荐使用 conda：

```bash
conda create -n sts2 python=3.9 -y
conda activate sts2
pip install -r requirements.txt
```

macOS 环境如遇 `faiss-cpu` 源码编译失败，可优先安装二进制 wheel：

```bash
pip install --prefer-binary "faiss-cpu>=1.8.0"
pip install -r requirements.txt
```

### 2. 配置 API Key

拷贝 `.env.example` 为 `.env` 并填入你的 DeepSeek key：

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 拉取知识库（可选，已包含默认知识库）

```bash
python scripts/fetch_knowledge.py
```

### 4. 运行

**Web 应用模式**（推荐，FastAPI + Vue 前后端分离）：

启动后端：
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

启动前端：
```bash
cd frontend
npm install
npm run dev
```

浏览器打开 Vite 输出的地址，默认是 `http://127.0.0.1:5173`。Vue 前端会请求 FastAPI 的 `/chat` 接口，并展示回答、引用来源、Agent 工具选择轨迹和 Verifier 结果。

Web 前端默认关闭 Reranker，以避免首次提问时下载 Cross-Encoder 模型造成长时间等待；需要最高检索精度时可在侧边栏手动开启。

**CLI 模式**（默认 Tool-Using Agent，自动选择检索工具）：
```bash
python main.py
```

**CLI 模式**（启用 Reranker 两阶段精排）：
```bash
python main.py --reranker
```

**CLI 模式**（启用 Hybrid 检索 + Reranker，推荐组合）：
```bash
python main.py --hybrid --reranker
```

**CLI 模式**（启用全栈 HyDE + Hybrid + Reranker，评测最佳配置）：
```bash
python main.py --hyde --hybrid --reranker
```

**CLI 模式**（Agent + Reranker 精排，推荐交互配置）：
```bash
python main.py --reranker
```

**CLI 模式**（LangGraph Agent 状态图）：
```bash
python main.py --langgraph --reranker
```

如需对照旧版固定 pipeline，可使用 `--legacy` 搭配 `--hybrid`、`--hyde`、`--multi-query` 等参数。

**Legacy Streamlit Demo**（保留用于快速本地演示）：
```bash
pip install streamlit
streamlit run app.py
```

主 Web 入口已替换为 Vue + FastAPI。Streamlit 版本仍可用于快速调试和对照展示。

## 检索评测

### 运行 Baseline 评测

```bash
python scripts/eval_retrieval.py --top-k 5
```

### 运行 Baseline vs Reranker 对比评测

```bash
python scripts/eval_retrieval.py --reranker --candidate-n 20 --top-k 5
```

### 启用结构化路由对比（推荐）

```bash
python scripts/eval_retrieval.py --reranker --router --top-k 5
```

### 启用 Hybrid 检索对比（BM25 + 向量 RRF 融合）

```bash
python scripts/eval_retrieval.py --hybrid --reranker --router --top-k 5
```

### 启用 HyDE 全栈对比（最佳配置）

```bash
python scripts/eval_retrieval.py --hyde --hybrid --reranker --router --top-k 5
```

### 实测指标（53 条标注用例，含 18 条开放/对比类问题，Top-5）

| 配置 | Hit@1 | Hit@5 | MRR |
|------|-------|-------|-----|
| 纯向量（Baseline） | 24.53% | 54.72% | 0.3475 |
| + Reranker | 56.60% | 66.04% | 0.6069 |
| + Router | 73.58% | 88.68% | 0.7865 |
| + Router + Reranker | 81.13% | 86.79% | 0.8365 |
| + Hybrid | 54.72% | 69.81% | 0.6038 |
| + Hybrid + Reranker | 64.15% | 79.25% | 0.7035 |
| + Hybrid + Router | 77.36% | 86.79% | 0.8113 |
| + Hybrid + Router + Reranker | 79.25% | 88.68% | 0.8292 |
| + HyDE | 33.96% | 56.60% | 0.4211 |
| + HyDE + Reranker | 64.15% | 71.70% | 0.6761 |
| + HyDE + Router + Reranker | 79.25% | 84.91% | 0.8176 |
| **+ HyDE + Hybrid + Router + Reranker** | **83.02%** | **88.68%** | **0.8538** |

### 评测指标说明

| 指标 | 含义 |
|------|------|
| Hit@K | 前 K 条结果中包含正确答案的比例 |
| MRR (Mean Reciprocal Rank) | 正确答案排位的倒数的平均值，越高说明正确结果越靠前 |

## 数据来源

游戏数据由 [Spire Codex](https://spire-codex.com/) 提供，涵盖卡牌、遗物、药水、怪物、角色等完整游戏内容，支持中文在内的 14 种语言。感谢 Spire Codex 团队提供的开放 API。

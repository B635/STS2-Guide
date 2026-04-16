# STS2-Guide：杀戮尖塔2 RAG 攻略助手

基于 **RAG（Retrieval-Augmented Generation）** 架构的游戏攻略问答系统。用户用自然语言提问，系统从 1000+ 条结构化知识中检索相关内容，经过两阶段精排后交由 LLM 生成回答。

## 系统架构

```
用户提问
  │
  ▼
┌──────────────────┐
│  Embedding 编码   │  paraphrase-multilingual-MiniLM-L12-v2 (384 维)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  向量召回 Top-N   │  余弦相似度 + 词法加权 (lexical boost)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Reranker 精排    │  BAAI/bge-reranker-base (Cross-Encoder)
│  (可选)           │  对 query-doc 对直接打分，重排序
└────────┬─────────┘
         ▼
┌──────────────────┐
│  自适应检索判断    │  LLM 判断上下文是否充足，不足则扩大召回
│  (可选)           │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  LLM 生成回答     │  DeepSeek API (OpenAI 兼容)
└──────────────────┘
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 向量模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 维多语言 Bi-Encoder，支持中英文 |
| Reranker | `BAAI/bge-reranker-base` | Cross-Encoder 精排模型，提升检索精度 |
| 向量检索 | NumPy 余弦相似度 + 词法加权 | 轻量高效，无需外部向量数据库 |
| 生成模型 | DeepSeek API | OpenAI 兼容接口，支持中文 |
| Web UI | Streamlit | 交互式聊天界面，支持参数动态调整 |
| 知识库 | JSON（1035 条文档） | 通过 Spire Codex API 自动拉取 |
| 语言 | Python 3.9 | |

## 核心功能

### 检索能力
- **语义检索**：基于余弦相似度检索最相关知识，而非关键词匹配
- **词法加权**：对领域关键词和计数类问题做针对性加权，弥补纯语义检索的不足
- **Reranker 两阶段精排**：先用 Bi-Encoder 粗召回 Top-20 候选，再用 Cross-Encoder 逐对精排取 Top-N，显著提升检索精度
- **自适应检索**：调用 LLM 判断当前上下文是否充足，不足时自动扩大检索范围（3 → 8 → 15 → 全量）
- **动态检索数量**：根据问题类型自动调整检索条数（列举类/对比类/普通问题）

### 生成与对话
- **多轮对话**：保留对话历史，支持上下文关联的连续提问
- **历史截断**：自动截断过长对话历史，避免 token 超限
- **来源透明**：每次回答显示参考来源、相似度分数及 Reranker 分数

### 工程能力
- **向量缓存**：归一化向量持久化存储，知识库不变时跳过重复计算
- **知识库自动更新**：`fetch_knowledge.py` 从公开 API 自动拉取最新游戏数据
- **量化评测体系**：35 条标注测试集，支持 Hit@K、MRR 指标评测及 Baseline/Reranker 对比
- **错误处理**：覆盖 API 限流、认证失败、网络异常等常见错误

## 项目结构

```
RagDemo/
├── main.py                    # CLI 入口（支持 --reranker 参数）
├── app.py                     # Streamlit Web UI
├── config.py                  # 全局配置（模型、阈值、检索参数）
├── rag/
│   ├── embedder.py            # 向量编码与缓存管理
│   ├── retriever.py           # 向量检索、词法加权、自适应检索
│   ├── reranker.py            # Cross-Encoder Reranker 精排
│   ├── chat.py                # LLM 对话（DeepSeek API）
│   └── errors.py              # 错误处理
├── scripts/
│   ├── fetch_knowledge.py     # 从 Spire Codex API 拉取知识库
│   └── eval_retrieval.py      # 检索质量评测（Hit@K / MRR）
├── data/
│   ├── knowledge.json         # 知识库（1035 条文档）
│   └── retrieval_eval.json    # 检索评测集（35 条标注用例）
├── models/                    # 模型缓存目录
├── embeddings.npy             # 原始向量缓存
├── embeddings_normalized.npy  # 归一化向量缓存
└── docs_hash.txt              # 知识库哈希（用于缓存失效判断）
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv vnev
source vnev/bin/activate
pip install openai sentence-transformers numpy python-dotenv requests streamlit
```

### 2. 配置 API Key

新建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 拉取知识库（可选，已包含默认知识库）

```bash
python scripts/fetch_knowledge.py
```

### 4. 运行

**CLI 模式**（基础检索 + 自适应）：
```bash
python main.py
```

**CLI 模式**（启用 Reranker 两阶段精排）：
```bash
python main.py --reranker
```

**Web UI 模式**（Streamlit，支持所有功能动态切换）：
```bash
streamlit run app.py
```

## 检索评测

### 运行 Baseline 评测

```bash
python scripts/eval_retrieval.py --top-k 5
```

### 运行 Baseline vs Reranker 对比评测

```bash
python scripts/eval_retrieval.py --reranker --candidate-n 20 --top-k 5
```

### 实测指标（35 条标注用例，Top-5）

| 指标 | Baseline | + Reranker | 提升 |
|------|----------|------------|------|
| Hit@1 | 40.00% | **65.71%** | **+25.71%** |
| Hit@3 | 51.43% | **71.43%** | **+20.00%** |
| Hit@5 | 62.86% | **71.43%** | +8.57% |
| MRR | 0.4733 | **0.6857** | **+0.2124** |

Reranker 对 Top-1 精度提升最明显，说明 Cross-Encoder 的深层交互打分能有效把"最相关"的文档挤到最前面。

### 评测指标说明

| 指标 | 含义 |
|------|------|
| Hit@K | 前 K 条结果中包含正确答案的比例 |
| MRR (Mean Reciprocal Rank) | 正确答案排位的倒数的平均值，越高说明正确结果越靠前 |

## Reranker 技术方案

### 为什么需要两阶段检索？

| | Bi-Encoder（向量检索） | Cross-Encoder（Reranker） |
|---|---|---|
| 输入 | query 和 doc **分别**编码为向量 | query 和 doc **拼接**后一起输入 |
| 速度 | 快（可预计算 doc 向量） | 慢（每对 query-doc 需单独推理） |
| 精度 | 较低（独立编码损失交互信息） | 高（能捕捉 query-doc 之间的深层交互） |
| 适用场景 | 粗召回（从 1000+ 文档中筛选候选） | 精排序（对 20 个候选重新排序） |

两阶段方案兼顾了速度与精度：先用 Bi-Encoder 快速从 1035 篇文档中召回 Top-20 候选，再用 Cross-Encoder 对这 20 个候选逐一精排，取 Top-N 作为最终结果。

## 数据来源

游戏数据由 [Spire Codex](https://spire-codex.com/) 提供，涵盖卡牌、遗物、药水、怪物、角色等完整游戏内容，支持中文在内的 14 种语言。感谢 Spire Codex 团队提供的开放 API。

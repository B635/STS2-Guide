# STS2-Guide：杀戮尖塔2 RAG 攻略助手

基于 **RAG（Retrieval-Augmented Generation）** 架构的游戏攻略问答系统。用户用自然语言提问，系统从 1048 条结构化知识中检索相关内容，经过"结构化路由 → HyDE 改写（可选） → Hybrid 检索（BM25 + 向量 RRF 融合） → Reranker 精排"四级管线后交由 LLM **带引用**生成回答（每句标注 `[n]`，不可追溯的结论强制标 `[?]`）。

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 向量模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 维多语言 Bi-Encoder，支持中英文 |
| Reranker | `BAAI/bge-reranker-base` | Cross-Encoder 精排模型，提升检索精度 |
| 稀疏检索 | `rank-bm25` + `jieba` 分词 | BM25Okapi 词频打分，捕捉专名/数值等向量易丢失的字面信号 |
| 融合策略 | Reciprocal Rank Fusion (k=60) | 只用排名不用分数，天然解决 BM25 与余弦尺度不可比的问题 |
| 向量索引 | FAISS (IndexFlatIP / IndexIVFFlat) | 通过 VectorStore 抽象层封装，策略可切换 |
| 生成模型 | DeepSeek API | OpenAI 兼容接口，支持中文 |
| Web UI | Streamlit | 交互式聊天界面，支持参数动态调整 |
| 知识库 | 按类型结构化 JSON（1048 条，含 `embed_text`） | 通过 Spire Codex API 自动拉取 |
| 语言 | Python 3.9（推荐 conda 虚拟环境） | |

## 核心功能

### 检索能力
- **结构化路由**：识别查询中的实体名或"有几个/多少"类计数问题时，直接从按类型切分的知识库返回确定性答案，绕过向量检索以避免在 1000+ 相似短文档中的召回噪声
- **多轮 Query 改写（History-aware Retrieval）**：follow-up 问题（"他的血量呢"）先由 LLM 结合历史重写成独立 query 再检索，并通过"实体白名单"后置校验拦截模型幻觉出的新实体，失败时回退原 query
- **HyDE 假设文档改写**：LLM 先生成一段与 `embed_text` 同构的假设性条目做向量侧检索 query，原始 query 仍留给 BM25 —— 解决"描述性提问 vs 卡片化文档"的分布鸿沟
- **Hybrid 检索（BM25 + Vector + RRF）**：jieba 中文分词后的 BM25 稀疏召回与稠密向量召回并行取 Top-20，再用 Reciprocal Rank Fusion 按排名融合，修正纯向量对专名/数值的弱召回
- **语义检索**：路由未命中时走向量召回，基于余弦相似度 + 词法加权
- **Reranker 两阶段精排**：先用 Bi-Encoder（或 Hybrid）粗召回 Top-20 候选，再用 Cross-Encoder 逐对精排取 Top-N
- **Query 分解**：LLM 将复杂问题拆成 1-3 个子问题分别检索后合并去重（`--multi-query`，与 HyDE 互斥）
- **自适应检索**：调用 LLM 判断当前上下文是否充足，不足时自动扩大检索范围（3 → 8 → 15 → 全量）

### 生成与对话
- **多轮对话**：保留对话历史，支持上下文关联的连续提问
- **历史截断**：自动截断过长对话历史，避免 token 超限
- **来源透明**：每次回答显示参考来源、相似度分数及 Reranker 分数
- **句级引用（Citation / Grounding）**：每个陈述句末尾带 `[n]` 指向背景知识编号，无依据的结论强制用 `[?]` 标注，前端渲染为高亮 chip/红色警告；规则化评测脚本验证"引用编号有效率 / 数字可追溯率"

### 工程能力
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
# Web UI 额外需要 Streamlit
pip install streamlit
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

**CLI 模式**（基础检索 + 自适应）：
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

# STS2-Guide：杀戮尖塔2 RAG 攻略助手

基于 **RAG（Retrieval-Augmented Generation）** 架构的游戏攻略问答系统。用户用自然语言提问，系统从 1048 条结构化知识中检索相关内容，经过"结构化路由 → 向量检索 → Reranker 精排"三级管线后交由 LLM 生成回答。

## 系统架构

```
用户提问
  │
  ▼
┌──────────────────┐
│  结构化路由       │  识别实体名/计数问题，命中时直接返回结构化答案
│  (可选)           │  不命中则下钻到向量检索
└────────┬─────────┘
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
│  Query 分解 /     │  LLM 拆解复杂问题或判断上下文是否充足
│  自适应检索（可选）│
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
| 知识库 | 按类型结构化 JSON（1048 条，含 `embed_text`） | 通过 Spire Codex API 自动拉取 |
| 语言 | Python 3.9（推荐 conda 虚拟环境） | |

## 核心功能

### 检索能力
- **结构化路由**：识别查询中的实体名或"有几个/多少"类计数问题时，直接从按类型切分的知识库返回确定性答案，绕过向量检索以避免在 1000+ 相似短文档中的召回噪声
- **语义检索**：路由未命中时走向量召回，基于余弦相似度 + 词法加权
- **Reranker 两阶段精排**：先用 Bi-Encoder 粗召回 Top-20 候选，再用 Cross-Encoder 逐对精排取 Top-N
- **Query 分解**：LLM 将复杂问题拆成 1-3 个子问题分别检索后合并去重（`--multi-query`）
- **自适应检索**：调用 LLM 判断当前上下文是否充足，不足时自动扩大检索范围（3 → 8 → 15 → 全量）

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
STS2-Guide/
├── main.py                    # CLI 入口（--reranker / --multi-query）
├── app.py                     # Streamlit Web UI
├── config.py                  # 全局配置（模型、阈值、检索参数）
├── requirements.txt           # pip 依赖清单
├── .env.example               # API Key 模板
├── rag/
│   ├── knowledge.py           # 结构化知识库加载器
│   ├── router.py              # 结构化查询路由（实体名 / 计数）
│   ├── embedder.py            # 向量编码与缓存
│   ├── retriever.py           # 向量检索 + 词法加权 + 自适应检索
│   ├── reranker.py            # Cross-Encoder 精排
│   ├── query_planner.py       # LLM Query 分解
│   ├── chat.py                # LLM 对话（DeepSeek API）
│   └── errors.py              # 错误处理
├── scripts/
│   ├── fetch_knowledge.py     # 拉取知识库（按类型结构化输出）
│   ├── probe_api.py           # 探测 Spire Codex API 字段清单
│   └── eval_retrieval.py      # 检索评测（Hit@K / MRR，支持 --router / --reranker）
├── data/
│   ├── knowledge.json         # 按类型结构化的知识库（1048 条）
│   └── retrieval_eval.json    # 检索评测集（35 条标注用例）
├── models/                    # 模型缓存目录（HF_HOME 标准布局）
└── *.npy / docs_hash.txt      # 向量缓存和知识库哈希
```

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

### 实测指标（35 条标注用例，Top-5）

| 指标 | Baseline | + Reranker | + Reranker + Router |
|------|----------|------------|---------------------|
| Hit@1 | 28.57% | 62.86% | **100.00%** |
| Hit@3 | 31.43% | 68.57% | **100.00%** |
| Hit@5 | 34.29% | 68.57% | **100.00%** |
| MRR | 0.2595 | 0.6524 | **1.0000** |

当前评测集以单实体属性查询为主，路由层基本都能确定性命中，因此分数接近饱和。Reranker 在路由未命中的开放问题上仍会发挥作用。

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

两阶段方案兼顾了速度与精度：先用 Bi-Encoder 快速从 1048 篇文档中召回 Top-20 候选，再用 Cross-Encoder 对这 20 个候选逐一精排，取 Top-N 作为最终结果。

## 结构化路由（Router）

当知识库中存在大量外形相似的短文档（比如 576 张卡牌都以"卡牌XXX（色，稀有度，类型，费用）"开头）时，纯向量检索对"中和多少费"这种短问题召回很不稳定。路由层利用按类型切分的索引做两件事：

1. **计数问题**（"角色有几个"）→ 直接 `len(index["characters"])` 返回 `共有5个角色`。
2. **实体名精确匹配**（"铁甲战士初始血量多少"）→ 扫描 query 中出现的最长 entity name，命中则返回该实体的 `embed_text`。

没有实体名且不是计数问题时，路由返回 `None`，管线自然回落到向量检索 + Reranker。知识库文件按 `characters / cards / relics / potions / monsters` 分区存储，每条保留原始字段之外还附一个 `embed_text` 供向量召回使用。

## 数据来源

游戏数据由 [Spire Codex](https://spire-codex.com/) 提供，涵盖卡牌、遗物、药水、怪物、角色等完整游戏内容，支持中文在内的 14 种语言。感谢 Spire Codex 团队提供的开放 API。

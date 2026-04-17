# STS2-Guide：杀戮尖塔2 RAG 攻略助手

基于 **RAG（Retrieval-Augmented Generation）** 架构的游戏攻略问答系统。用户用自然语言提问，系统从 1048 条结构化知识中检索相关内容，经过"结构化路由 → HyDE 改写（可选） → Hybrid 检索（BM25 + 向量 RRF 融合） → Reranker 精排"四级管线后交由 LLM **带引用**生成回答（每句标注 `[n]`，不可追溯的结论强制标 `[?]`）。

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
│  HyDE 改写        │  LLM 先生成一段"假设性条目"作为向量侧检索 query
│  (可选)           │  BM25 仍用原始 query，两侧互补
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Hybrid 双路召回  │  BM25（jieba 分词）+ 向量（MiniLM-L12 384 维）
│  (可选)           │  两路 Top-N 用 Reciprocal Rank Fusion 融合
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
│  LLM 生成回答     │  DeepSeek API，System Prompt 强制每句末尾带 [n]
│                  │  不可归因的结论必须标 [?]，严禁编造不存在的编号
└──────────────────┘
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 向量模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 维多语言 Bi-Encoder，支持中英文 |
| Reranker | `BAAI/bge-reranker-base` | Cross-Encoder 精排模型，提升检索精度 |
| 稀疏检索 | `rank-bm25` + `jieba` 分词 | BM25Okapi 词频打分，捕捉专名/数值等向量易丢失的字面信号 |
| 融合策略 | Reciprocal Rank Fusion (k=60) | 只用排名不用分数，天然解决 BM25 与余弦尺度不可比的问题 |
| 向量检索 | NumPy 余弦相似度 + 词法加权 | 轻量高效，无需外部向量数据库 |
| 生成模型 | DeepSeek API | OpenAI 兼容接口，支持中文 |
| Web UI | Streamlit | 交互式聊天界面，支持参数动态调整 |
| 知识库 | 按类型结构化 JSON（1048 条，含 `embed_text`） | 通过 Spire Codex API 自动拉取 |
| 语言 | Python 3.9（推荐 conda 虚拟环境） | |

## 核心功能

### 检索能力
- **结构化路由**：识别查询中的实体名或"有几个/多少"类计数问题时，直接从按类型切分的知识库返回确定性答案，绕过向量检索以避免在 1000+ 相似短文档中的召回噪声
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
│   ├── retriever.py           # 向量检索 + 词法加权 + Hybrid (RRF) + 自适应检索
│   ├── bm25.py                # BM25 稀疏召回（jieba 分词 + rank_bm25）
│   ├── hyde.py                # HyDE 假设文档生成（LLM 改写 query）
│   ├── reranker.py            # Cross-Encoder 精排
│   ├── query_planner.py       # LLM Query 分解
│   ├── chat.py                # LLM 对话（DeepSeek API）
│   └── errors.py              # 错误处理
├── scripts/
│   ├── fetch_knowledge.py     # 拉取知识库（按类型结构化输出）
│   ├── probe_api.py           # 探测 Spire Codex API 字段清单
│   ├── eval_retrieval.py      # 检索评测（Hit@K / MRR，支持 --router / --reranker / --hybrid / --hyde）
│   └── eval_citation.py       # 引用质量评测（Citation Rate / Source Validity / Number Grounding）
├── data/
│   ├── knowledge.json         # 按类型结构化的知识库（1048 条）
│   └── retrieval_eval.json    # 检索评测集（53 条标注用例，含开放类问题）
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

**观察：**
- Hybrid 单独上阵把纯向量 Hit@1 拉升 **+30.19%**（24.53% → 54.72%），证明 BM25 的字面匹配对中文专名/数值查询有独立价值
- Hybrid + Reranker 比单独 Reranker 再 +7.55% Hit@1，融合后候选池覆盖更全，精排才有东西可选
- **HyDE 是"必须配 BM25 才能起正作用"的典型**：单独 HyDE + Router + Reranker **-1.88% Hit@1**（79.25% vs 81.13%），因为 LLM 在假设文档里编造的错误实体名会欺骗向量侧；但一旦并上 Hybrid，BM25 用原始 query 保住了实体锚点，HyDE 只负责补全描述向量，全栈组合反超 **+1.89% Hit@1**（83.02%）
- 剩余失败集中在"最值"排序语义（"伤害最高的卡"）和"类型约束"（"血量最高的 Boss"）两类——RAG 对全局统计问题的固有短板，需要结构化聚合或 Agent 工具调用补位

### 评测指标说明

| 指标 | 含义 |
|------|------|
| Hit@K | 前 K 条结果中包含正确答案的比例 |
| MRR (Mean Reciprocal Rank) | 正确答案排位的倒数的平均值，越高说明正确结果越靠前 |

## Hybrid 检索（BM25 + Vector + RRF）

### 问题：单一向量召回的语义偏差

纯 Bi-Encoder 在中文短文档上有两个系统性弱点：

1. **专名被稀释**：query 里的"铁甲战士"与文档里的"铁甲战士"token 拆解后混进整句嵌入，具体名字的贡献被其他词摊薄——向量更关心"话题"而非"是谁"
2. **数值/稀有字符**：像"9999"、"失重"、"三相"这类低频 token 在嵌入空间几乎不起作用，但在字面匹配里是决定性的

评测里这表现为：问"血量最高的 Boss"时，`Normal` 型 9999HP 的"佩尔的士兵"会盖过真正的 Boss——嵌入无法区分 `Boss`/`Elite`/`Normal` 标签。

### 方案：并行召回 + Reciprocal Rank Fusion

- **BM25 稀疏召回**：用 `jieba` 中文分词器切 query 和 docs，`rank_bm25` 的 BM25Okapi 算 TF/IDF 分数，Top-20
- **向量稠密召回**：复用现有 MiniLM 通道，Top-20
- **RRF 融合**：每条文档最终分数 = Σ 1/(k + rank)，`k=60`（工业默认）

RRF 的关键优势是**只用排名不用分数**——BM25 是正数无界、余弦是 [-1, 1]，直接加权必须先做尺度对齐；RRF 对任何打分方式都鲁棒。两侧都打进 Top-N 的文档会被双重加分自然浮到最前，这正是 Hybrid 想要的"证据叠加"。

### 实现

- [rag/bm25.py](rag/bm25.py)：jieba 分词 + BM25Okapi 索引，`docs` 与向量侧对齐（同样的 `embed_text`），`index` 字段保持一致以便 RRF 用整数 key 融合
- [rag/retriever.py](rag/retriever.py) `rrf_fuse()` / `hybrid_retrieve()`：纯 Python dict 聚合，~30 行
- [config.py](config.py) 新增 `BM25_TOP_N=20` / `VECTOR_TOP_N_FOR_HYBRID=20` / `RRF_K=60`
- CLI / eval / Streamlit 三处统一通过 `--hybrid` 开关接入

## 句级引用 / Grounding

### 问题：LLM 即使有上下文也会夹带私货

RAG 系统最常见的幻觉源是"Prompt 喂了 A，LLM 混着脑子里的 A' 一起输出"。本项目早期就踩过——DeepSeek 在回答 STS2 时总会悄悄混入 STS1 预训练知识（同名角色机制完全不同）。System Prompt 里加"只能用背景知识"只解决了一半——问题是**用户看不出哪句被污染了**。

### 方案：每句必须带 `[n]`，无依据则强制标 `[?]`

把 `format_context()` 改成 `[1] ... [2] ... [3] ...`，让 LLM 在生成时把编号回贴到每句话末尾。System Prompt 里加入三条硬规则：

1. 每个陈述句末尾必须有 `[n]`，多来源写成 `[1][3]`
2. **不可归因**的结论必须标 `[?]` 并在句中说明"依据不足"
3. **禁止编造不存在的编号**（只给 3 条就不准出现 `[4]`）

前端用正则解析 `\[(\d+|\?)\]`：数字编号渲染为 monospace chip 和下方 sources 对齐，`[?]` 用 Streamlit 的 `:red[]` 渲染为红色警告标签。

### 自动化评测（零 LLM-as-Judge）

```bash
python scripts/eval_citation.py
```

规则化跑完 53 条 eval case 之后的结果：

| 指标 | 定义 | 当前值 |
|------|------|--------|
| **Citation Rate** | 有引用标注的实质句 / 总实质句 | **88.24%** (60/68) |
| **Source Validity** | 引用编号指向合法 ID / 总引用数 | **100.00%** (64/64) |
| **Number Grounding** | 句中的数字能在被引用 source 里找到 / 含数字的被引用句 | **91.30%** (42/46) |

**Source Validity 100% 是关键信号**：DeepSeek 在整个 eval set 上**零次**编造不存在的引用编号——说明配合带编号的 Context + 显式 Prompt 约束后，引用机制本身是可靠的，剩余 8 条漏引发生在连带总结语上（不是 fact 句）。Number Grounding 的 4 条失败主要是句中出现了游戏名 "杀戮尖塔2" 被当作事实数字误判（已在 TODO 里记下作为规则级 eval 的已知局限）。

## HyDE 假设文档改写

### 问题：query 与 doc 的分布鸿沟

用户自然语言 query（"哪张卡抽牌最多"）和 `embed_text` 的结构化模板（`卡牌XX（角色，稀有度，类型，费用N）：描述`）在嵌入空间里离得很远——即使 Bi-Encoder 懂"抽牌"这个概念，query 本身的句法就不像一条 doc，余弦距离先天不利。

### 方案：让 LLM 先"编一条答案"

HyDE（Hypothetical Document Embeddings，Gao et al. 2022）的思路：先让 LLM 针对 query 生成一条**假设性文档**（可以是编造的、不需准确），再拿这段文本去做向量检索。假设文档与真实 doc 同分布，余弦对齐好得多。

Prompt 里明确要求输出格式：

```
卡牌XX（角色，稀有度，类型，费用N）：描述。
遗物XX（稀有度遗物，shared）：描述。
怪物XX（Boss/Elite/Normal），血量N，技能：XX，遭遇：XX。
...
```

实测 LLM 会生成类似 `卡牌杂技（铁甲战士，稀有，技能，费用1）：丢弃1张牌，抽3张牌。` —— 即使"杂技"属于哪个角色实际是错的，**描述部分已经足够相似**来把真实的"抽牌类稀有技能"拉进 Top-K。

### 与 Hybrid 的协同

HyDE 单独接 Router+Reranker 反而 -1.88% Hit@1，原因正是 LLM 会虚构实体名（比如把"杂技"的归属搞错），向量侧被误导到错的 doc 上。**解法是 BM25 走原始 query**：原始 query 里的"抽牌"、"最多"等关键词在 BM25 视角上锚住方向，再由 HyDE 的向量侧补全语义，两路 RRF 融合后准确度才真正落地：

- 向量侧 query = HyDE 假设文档（负责语义覆盖）
- BM25 侧 query = 原始 query（负责字面锚定）

这也是评测里 **HyDE + Hybrid + Router + Reranker 反而优于 Router + Reranker** 的根本原因：三种机制分别覆盖了"确定性实体匹配"、"字面信号捕捉"、"语义分布对齐"三个正交维度。

### 成本与触发条件

HyDE 需要一次额外 LLM 调用（~300-500ms）。当前设计在 Router 未命中时才触发，命中的确定性 case（单实体、计数、对比）直接跳过。UI 中与 Query 分解互斥（两者都是 query 改写，冲突时自动关闭 Query 分解）。

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

当知识库中存在大量外形相似的短文档（比如 576 张卡牌都以"卡牌XXX（色，稀有度，类型，费用）"开头）时，纯向量检索对"中和多少费"这种短问题召回很不稳定。路由层利用按类型切分的索引做三件事：

1. **计数问题**（"角色有几个"）→ 直接 `len(index["characters"])` 返回 `共有5个角色`。
2. **单实体精确匹配**（"铁甲战士初始血量多少"）→ 扫描 query 中出现的最长 entity name，命中则返回该实体的 `embed_text`。
3. **多实体对比**（"铁甲战士和储君血量哪个高"）→ 把查询中命名的所有实体一次性返回，让 LLM 在完整事实上做对比。

没有实体名且不是计数问题时，路由返回 `None`，管线自然回落到向量检索 + Reranker。知识库文件按 `characters / cards / relics / potions / monsters` 分区存储，每条保留原始字段之外还附一个 `embed_text` 供向量召回使用。

## 数据来源

游戏数据由 [Spire Codex](https://spire-codex.com/) 提供，涵盖卡牌、遗物、药水、怪物、角色等完整游戏内容，支持中文在内的 14 种语言。感谢 Spire Codex 团队提供的开放 API。

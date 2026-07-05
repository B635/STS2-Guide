# STS2-Guide：杀戮尖塔2 实时决策 Agent

一个由游戏状态驱动的《杀戮尖塔2》本地实时决策 Agent。C# 只读 Mod 感知选牌事件，
Python 后台 EXE（`python -m realtime.host`）负责协议校验、当前局检查点和本地推荐器，
Godot 游戏内面板显示推荐分。用户正常启动游戏和后台 EXE 即可，不依赖 Vue/FastAPI/网络。

P0 数据边界：结构化卡牌/遗物数据存入 SQLite；当前局仅保留单一可替换检查点
`active-run.json`；对局结束后只向 `run_summaries` 写入最终摘要。P0 实时管线不向
`run_states`、`decision_events`、`decision_outcomes` 追加历史。

Vue、FastAPI、RAG 和 LLM 属于可选旧入口或 P2 攻略解释层，不进入 P0 实时选牌链路。

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 向量模型 | `paraphrase-multilingual-MiniLM-L12-v2` | 384 维多语言 Bi-Encoder，支持中英文 |
| Reranker | `BAAI/bge-reranker-base` | Cross-Encoder 精排模型，提升检索精度 |
| 稀疏检索 | `rank-bm25` + `jieba` 分词 | BM25Okapi 词频打分，捕捉专名/数值等向量易丢失的字面信号 |
| 融合策略 | Reciprocal Rank Fusion (k=60) | 只用排名不用分数，天然解决 BM25 与余弦尺度不可比的问题 |
| 向量索引 | FAISS (IndexFlatIP / IndexIVFFlat) | 通过 VectorStore 抽象层封装，策略可切换 |
| 关系数据库 | SQLite（Repository 抽象） | 结构化实体、来源快照、社区统计和最终对局摘要；P0 不逐事件写入 |
| 生成模型 | DeepSeek API | OpenAI 兼容接口，支持中文 |
| Web UI | Vue 3 + Vite | 前后端分离聊天界面，展示工具轨迹、引用来源和校验结果 |
| API 服务 | FastAPI | 将 Agent 能力封装为 `/chat` 接口，供前端或外部系统调用 |
| 游戏感知 | C# / Harmony Postfix | 独立只读 Mod，只观察事件并原子写入版本化本地 JSON |
| 实时桥接 | Pydantic + 文件监听 | 严格协议校验、内容哈希、事件幂等去重与工具分发 |
| 知识库 | SQLite 结构化目录 + 章节化攻略向量索引 | 原始 JSON 仅作为可重建的导入快照 |
| 语言 | Python 3.9（推荐 conda 虚拟环境） + C# + Vue 3 | |

## 核心功能

### P0 实时决策 Agent（当前主链路）

- **只读 Mod 感知**：P0 观察选牌事件，读取角色、牌组、遗物、药水和奖励候选；只注册 Harmony `Postfix`，不改返回值、不自动点击、不访问网络
- **版本化状态协议**：`protocol/state-event.schema.json` 是 Mod 与 Agent 的唯一契约
- **当前局检查点**：`active-run.json` 是唯一可替换检查点；保存退出后可恢复同一 Run ID 和递增 sequence
- **本地推荐器**：透明局面 baseline + 低权重社区先验；动态跳过作为正式第四候选
- **游戏内 Godot 面板**：Mod 创建锚定选牌界面的左侧半透明抽屉，显示候选牌和跳过的推荐分（0–100）
- **最终摘要**：胜/负/放弃后只向 SQLite `run_summaries` 写入一条最终摘要，清理检查点和临时事件

### 可选旧入口（非 P0 运行依赖）

- **Vue + FastAPI**：保留为以后攻略问答或历史查看的候选界面，不进入 P0 实时链路
- **RAG / LLM**：P2 攻略解释层，不决定实时分数
- **`main.py` / `app.py`**：CLI 和 Streamlit 旧入口，保留调试使用

### 检索能力（非 P0）
- **Tool-Using Agent / Function Calling**：新增轻量级 Agent 编排层，优先通过 OpenAI-compatible `tools/tool_calls` 让 LLM 在 `hybrid_search`、`hyde_hybrid_search`、`multi_query_search`、`vector_search` 中选择工具并填写结构化参数；若模型接口不支持 tool calls，则自动回退到 JSON Planner + 启发式兜底。Planner 输出检索 query、top_n、过滤提示、复杂问题 sub_queries，并记录可解释执行轨迹
- **LangGraph 工作流（可选）**：将 `rewrite → route_or_plan → execute_tool → generate → verify → repair` 建模为状态图，校验失败时最多触发一次补救检索并重新生成，便于观察 Agent 每一步状态流转，并保留默认手写 Agent 作为无框架兜底
- **结构化路由**：识别查询中的实体名或"有几个/多少"类计数问题时，直接从 SQLite 目录返回确定性答案，绕过向量检索
- **多轮 Query 改写（History-aware Retrieval）**：follow-up 问题（"他的血量呢"）先由 LLM 结合历史重写成独立 query 再检索，并通过"实体白名单"后置校验拦截模型幻觉出的新实体，失败时回退原 query
- **HyDE 假设文档改写**：LLM 先生成一段与 `embed_text` 同构的假设性条目做向量侧检索 query，原始 query 仍留给 BM25 —— 解决"描述性提问 vs 卡片化文档"的分布鸿沟
- **Hybrid 检索（BM25 + Vector + RRF）**：jieba 中文分词后的 BM25 稀疏召回与稠密向量召回并行取 Top-20，再用 Reciprocal Rank Fusion 按排名融合，修正纯向量对专名/数值的弱召回
- **语义检索**：路由未命中时走向量召回，基于余弦相似度 + 词法加权
- **Reranker 两阶段精排**：先用 Bi-Encoder（或 Hybrid）粗召回 Top-20 候选，再用 Cross-Encoder 逐对精排取 Top-N
- **复杂任务拆解 / Query 分解**：Planner 可为比较、多实体、多维度问题直接输出 1-3 个子问题；缺省时回退到 Query Planner，再分别检索、合并去重（旧版固定 pipeline 中 `--multi-query` 仍可单独使用）
- **自适应检索**：调用 LLM 判断当前上下文是否充足，不足时自动扩大检索范围（3 → 8 → 15 → 全量）

### 生成与对话（非 P0）
- **多轮对话**：保留对话历史，支持上下文关联的连续提问
- **历史截断**：自动截断过长对话历史，避免 token 超限
- **来源透明**：每次回答显示参考来源、相似度分数及 Reranker 分数
- **句级引用与答案校验（Citation / Grounding）**：每个陈述句末尾带 `[n]` 指向背景知识编号，无依据的结论强制用 `[?]` 标注；在线 verifier 节点检查引用覆盖、引用编号有效性和数字可追溯性，失败时扩大检索上下文并重新生成，离线评测脚本量化"引用编号有效率 / 数字可追溯率"

### 攻略问答实验能力（非 P0）
- **FastAPI + Vue 前后端分离**：`api.py` 将 `run_agent` / `run_langgraph_agent` 封装为标准 `/chat` 接口，Vue 前端负责聊天交互、参数控制、工具轨迹、引用来源和 Verifier 结果展示
- **向量缓存**：归一化向量持久化存储，知识库不变时跳过重复计算
- **知识库自动更新**：`fetch_knowledge.py` 从公开 API 自动拉取最新游戏数据
- **量化评测体系**：53 条标注测试集（含开放类问题），支持 Hit@K、MRR 指标评测及 Baseline/Reranker/Router 对比
- **错误处理**：覆盖 API 限流、认证失败、网络异常等常见错误
- **状态事件接口**：`POST /events/game-state` 接收只读 Mod 事件，`GET /events/game-state/latest` 供实时界面读取；文件桥接与 HTTP 接口共用同一个处理器

## 数据存储边界

| 数据 | 存储 | 原因 |
|------|------|------|
| 攻略正文与章节 Chunk | FAISS + 向量缓存 | 需要语义相似检索 |
| 卡牌、遗物、角色、药水、怪物属性 | SQLite | 字段明确，需要过滤、关联和确定性查询 |
| Spire Codex 卡牌/遗物/药水社区分数 | SQLite | 有来源快照、样本量和抓取时间；只作相关性先验 |
| 当前局完整状态 | 单一原子 JSON 检查点（`active-run.json`） | 当前局需要恢复，不需要历史查询 |
| 待处理事件 | 临时事件队列 | 处理后删除，避免快速事件覆盖 |
| 胜负、最终楼层、最终牌组/遗物/药水 | SQLite `run_summaries` | P0 仅保存最终对局摘要；不逐事件持久化 |
| `knowledge.json`、`guides.json` | 导入快照 | 用于重建数据库和索引，不作为在线查询层 |

产品运行入口使用严格分层的 `load_runtime_knowledge`。旧评测脚本仍可调用 `load_knowledge` 重放历史“全部文档向量化”基线，避免失去前后对照；这不是产品运行路径。

完整表职责、数据流和后续迁移约束见 [`docs/data-architecture.md`](docs/data-architecture.md)，外部接口取舍与条款审计见 [`docs/data-source-audit.md`](docs/data-source-audit.md)。
实时产品边界、事件协议和 P0/P1/P2 见 [`docs/realtime-architecture.md`](docs/realtime-architecture.md)。

## P0 快速开始

P0 只需要本地结构化推荐器，不使用 API Key、Vue、FastAPI、RAG、LLM 或网络。

### 1. 安装 P0 依赖

推荐使用 conda：

```bash
conda create -n sts2 python=3.9 -y
conda activate sts2
pip install -r requirements-p0.txt
```

### 2. 运行

启动后台 EXE（开发阶段可直接用模块运行）：

```bash
python -m realtime.host
```

1. 启动游戏后 C# 只读 Mod 将选牌事件写入检查点文件；
2. Python 后台读取文件、完成协议校验与本地推荐器评分；
3. Godot 游戏内面板在选牌界面显示各候选和跳过的推荐分（0–100）。

也可以先用仓库示例验证链路：

```bash
python -m realtime.host --once --input protocol/state-event.example.json
```

C# 源码和构建说明位于
[`mod/STS2Guide.ReadOnlyExporter`](mod/STS2Guide.ReadOnlyExporter/README.md)。
当前适配层已针对 STS2 public-beta `v0.107.1` 与 .NET 9 编译，完成真实游戏
选牌状态到本地建议的首轮验证。协议 v2 的药水、遗物动态状态、卡牌附加状态
以及实际选择/跳过监听仍需在游戏内逐项回归；构建默认只更新项目内 artifacts，
不会自动覆盖游戏 `mods` 目录。

### 非 P0 旧实验入口

以下入口不属于 P0 实时选牌链路，仅保留用于调试、历史对照和 P2 攻略问答实验。
它们需要完整依赖和可选 API Key：

```bash
pip install -r requirements.txt
```

如需调用 DeepSeek，拷贝 `.env.example` 为 `.env` 并填写本机 API Key。`.env` 已被
Git 忽略，不得提交。

可选数据更新命令：

```bash
python scripts/fetch_knowledge.py
python scripts/fetch_guides.py
python scripts/fetch_community_scores.py
```

`fetch_guides.py` 使用 Spire Codex 的公开社区 API，攻略原文为英文，中文实体名只作为
检索辅助元数据。正式分发前仍应确认原攻略作者授权。`slaythespire-2.com` 不自动抓取，
只作为人工外链参考。

**Web 应用（Vue + FastAPI，旧实验入口）**：

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

浏览器打开 Vite 输出的地址，默认是 `http://127.0.0.1:5173`。Vue 前端提供”攻略问答”和”选牌建议”两个工作区。

**CLI 模式（默认 Tool-Using Agent，自动选择检索工具）**：
```bash
python main.py
```

**CLI 模式（启用 Reranker 两阶段精排）**：
```bash
python main.py --reranker
```

**CLI 模式（启用 Hybrid 检索 + Reranker）**：
```bash
python main.py --hybrid --reranker
```

**CLI 模式（启用全栈 HyDE + Hybrid + Reranker，评测最佳配置）**：
```bash
python main.py --hyde --hybrid --reranker
```

**CLI 模式（LangGraph Agent 状态图）**：
```bash
python main.py --langgraph --reranker
```

如需对照旧版固定 pipeline，可使用 `--legacy` 搭配 `--hybrid`、`--hyde`、`--multi-query` 等参数。

**Legacy Streamlit Demo**（保留用于快速本地演示）：
```bash
pip install streamlit
streamlit run app.py
```

Web 前端默认关闭 Reranker，以避免首次提问时下载 Cross-Encoder 模型造成长时间等待。以上 CLI、Web 和 Streamlit 入口均不进入 P0 实时选牌链路。

## 检索评测

### 攻略检索烟测

无需加载向量模型的中文 BM25 烟测：

```bash
python scripts/eval_guides.py --bm25-only --top-k 10
```

完整 Hybrid 攻略检索：

```bash
python scripts/eval_guides.py --top-k 10
```

### P0 选牌场景评测（当前推荐器基线）

```bash
python scripts/eval_card_reward_scenarios.py
```

使用 `data/advisor_eval_scenarios.json` 中的 26 个固定场景，离线检查推荐结果、动态跳过和关键评分因子。评测器仅使用临时 SQLite 和 `data/knowledge.json`，不读取历史选择、本机 tier 或网络。

支持 `--strict` 在断言失败时以非零退出，`--json` 输出结构化报告：

```bash
python scripts/eval_card_reward_scenarios.py --strict
python scripts/eval_card_reward_scenarios.py --json --output data/advisor_eval_report.json
```

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

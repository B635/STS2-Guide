# 跨机开发待办

> 此文件用于多机同步进度，**全部完成后可删除**。

---

## ✅ 本次 push 已完成（知识库结构化 + 路由层）

- **Add**: `.env.example` / `requirements.txt` / `scripts/probe_api.py`（环境与 API 字段探测）
- **Change**: 知识库从扁平字符串改为按类型结构化 JSON，每条附 `embed_text`（[scripts/fetch_knowledge.py](scripts/fetch_knowledge.py)、[data/knowledge.json](data/knowledge.json)、[rag/knowledge.py](rag/knowledge.py)）
- **Add**: 结构化查询路由 [rag/router.py](rag/router.py)：实体名精确匹配 + 计数问题直接从索引返回，命中时跳过向量检索
- **Update**: 把路由接入 [main.py](main.py)、[app.py](app.py)、[scripts/eval_retrieval.py](scripts/eval_retrieval.py)（新增 `--router` 开关；Web UI 新增"启用结构化路由"开关）
- **Delete**: `rag/retriever.py` 中已失效的"总览文档"词法加权分支
- **Update**: README 与 TODO 同步到新架构

## ✅ 历史 push 已完成

- **Update**: 收紧 RAG system prompt，避免 LLM 用 STS1 预训练知识回答（[rag/chat.py](rag/chat.py)）
- **Add**: Query 分解 + 多查询检索功能
  - 新文件 [rag/query_planner.py](rag/query_planner.py)：LLM 把复杂问题拆成 1-3 个子问题
  - [rag/retriever.py](rag/retriever.py) 加 `multi_query_retrieve()`：多子问题召回 → 去重 → 排序
  - CLI 用法：`python main.py --multi-query [--reranker]`
  - Web UI：侧边栏新增"启用 Query 分解"开关
- **Update**: Reranker 强制 CPU（避免 macOS 12 MPS warning）[rag/reranker.py](rag/reranker.py)
- **Update**: 删除已弃用的 `TRANSFORMERS_CACHE` 环境变量，仅保留 `HF_HOME`

---

## ⚠️ 在另一台机器 git pull 之后

由于本次删掉了 `TRANSFORMERS_CACHE`，HuggingFace 改用 `HF_HOME` 标准布局 `models/hub/`。
如果另一台之前模型缓存在 `models/models--XXX/`（旧布局），运行时会**重新下载 ~1.6GB**。

**手动迁移避免重下载：**

```bash
mkdir -p models/hub
mv models/models--BAAI--bge-reranker-base models/hub/ 2>/dev/null
mv models/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 models/hub/ 2>/dev/null
mv models/models--sentence-transformers--all-MiniLM-L6-v2 models/hub/ 2>/dev/null
```

---

## ⏳ 待开发功能

### 评测集扩展（优先级高）
- 当前 35 条评测集以单实体属性查询为主，路由层命中率过高
- 需补充开放类问题（如"哪张卡抽牌最多"、"储君推荐什么流派"）来真实衡量向量检索与 Reranker 的贡献

### 层级 2：Chain-of-Thought 推理（中等优先级）
- 让 LLM 在 `<thinking>` 标签内先分析，再在 `<answer>` 输出最终回答
- 改 [rag/chat.py](rag/chat.py) 的 system prompt
- 前端可选展示/隐藏 thinking 过程
- 适合复杂综合分析（如"哪个角色最适合新手"）

### 层级 4：结构化 JSON 输出（高级）
- LLM 输出 JSON 格式：`{answer, key_facts, analysis, recommendation, confidence, sources_used}`
- DeepSeek API 启用 `response_format={"type": "json_object"}`
- 改 [rag/chat.py](rag/chat.py) + 前端按字段卡片化渲染
- 让回答从"聊天"升级为"分析报告"

---

**全部完成后**：删除本文件 → `git rm TODO.md && git commit -m "Delete: 跨机开发同步文件"`

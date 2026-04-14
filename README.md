# STS2-Guide：杀戮尖塔2攻略助手

基于 RAG（检索增强生成）架构的游戏攻略问答系统，用户可以用自然语言提问，系统自动从知识库中检索相关内容并生成回答。

## 技术栈

- **向量模型**：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **向量检索**：numpy 余弦相似度
- **LLM**：DeepSeek API
- **知识库**：JSON（自动从外部 API 拉取）
- **语言**：Python 3.9

## 核心功能

- **语义检索**：基于余弦相似度检索最相关知识，而非关键词匹配
- **自适应检索**：自动判断检索结果是否充足，不足时扩大检索范围
- **动态检索数量**：根据问题类型自动调整检索条数（列举类、对比类、普通问题）
- **多轮对话**：保留对话历史，支持上下文关联的连续提问
- **历史截断**：自动截断过长的对话历史，避免 token 超限
- **来源透明**：每次回答显示参考来源及相似度分数
- **向量缓存**：归一化向量持久化存储，避免重复计算
- **知识库自动更新**：通过脚本从公开 API 自动拉取最新游戏数据
- **错误处理**：覆盖 API 限流、认证失败、网络异常等常见错误

## 快速开始

**1. 安装依赖**
```bash
pip install openai sentence-transformers numpy python-dotenv requests
```

**2. 配置 API key**

新建 `.env` 文件：
**3. 拉取知识库（可选，已包含默认知识库）**
```bash
python scripts/fetch_knowledge.py
```

**4. 运行**
```bash
python main.py
```

## 检索评测（建议每次改检索后执行）

运行评测脚本：

```bash
python scripts/eval_retrieval.py --top-k 5
```

默认评测集在 `data/retrieval_eval.json`，支持按你自己的常见问题持续扩充。

建议重点观察：

- Hit@1：检索第一条是否直接命中正确知识
- Hit@3/Hit@5：前几条是否覆盖正确知识
- MRR：正确知识平均排位是否靠前

## 数据来源

游戏数据由 [Spire Codex](https://spire-codex.com/) 提供，涵盖卡牌、遗物、药水、怪物、角色等完整游戏内容，支持中文在内的 14 种语言。感谢 Spire Codex 团队提供的开放 API。
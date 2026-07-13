# TASK-002：P0 边界与安全冗余清理

- 状态：已验收
- 创建者：Codex
- 实现者：Codex（清理）+ DeepSeek（独立复核）
- 产品依据：`docs/product-spec.md` 的 P0-1、P0-3 与 P0 运行边界
- 前置任务：TASK-001（已验收）

## 目标

删除已经确认无 P0 运行时引用、与当前架构冲突或被正式入口替代的旧代码和生成物，
同时把 README 的 P0 安装/启动说明与可选 RAG 实验入口彻底分开。

本任务只做可证明安全的入口层清理，不修改推荐评分、协议、Mod、SQLite schema 或
可选 RAG 实现。

## 当前证据

- `advisor/map_path.py` 没有生产代码引用，且包含无真实边时连接下一行节点的禁止性
  fallback；
- `advisor/iteration.py` 只被旧历史选择权重脚本和对应测试引用；
- `scripts/watch_game_state.py` 的正式功能已被 `python -m realtime.host` 覆盖；
- 根目录 `p0-host*.log` 和 `data/advisor_iteration_report.json` 是生成物；
- README 的 P0 运行命令已修正，但安装步骤仍先要求完整 RAG 依赖和 API Key；
- `rag/`、`api.py`、`app.py`、`main.py` 和 `frontend/` 不属于 P0，但仍是可选攻略解释
  实验入口，不在本任务删除。

## 必须删除

- `advisor/map_path.py`
- `advisor/iteration.py`
- `scripts/eval_advisor_iterations.py`
- `tests/test_advisor_iteration.py`
- `scripts/watch_game_state.py`
- `data/advisor_iteration_report.json`
- 根目录 `p0-host.out.log`
- 根目录 `p0-host.err.log`

如果某个文件在实施时已经不存在，应在报告中记录，不得报错后扩大删除范围。

## 必须修改

### README

- P0 快速开始首先使用 `requirements-p0.txt`，明确无需 API Key、Vue、FastAPI、
  RAG、LLM 或网络；
- P0 开发启动入口只保留 `python -m realtime.host`；
- 删除 `watch_game_state.py` 和历史选择权重迭代脚本的命令与能力宣称；
- 完整 `requirements.txt`、`.env` API Key、Vue/FastAPI 和 RAG 命令整体放入明确的
  “可选攻略解释实验入口（非 P0）”章节；
- 不删除可选 RAG 功能本身，也不继续把它写成 P0 默认安装步骤。

### Git 忽略

- 忽略根目录 `p0-host*.log`；
- 忽略显式生成的 `data/advisor_eval_report.json`；
- 保留现有数据库、本机 tier 和构建目录忽略规则；
- 不使用过宽的 `*.json`、`data/*` 或 `*.log` 规则隐藏正常项目文件。

### 文档引用

- 删除或改写对本任务已删除文件的当前能力引用；
- 历史任务单中的文件名属于审查记录，不需要篡改；
- `docs/product-spec.md` 已由 Codex 更新，不得改变产品范围。

## 禁止事项

- 不得修改 `advisor/card_reward.py`、`advisor/contextual_scoring.py` 或权重；
- 不得修改协议 schema、生产者、消费者或 Mod；
- 不得删除或修改 SQLite 表、迁移和 repository 历史方法；
- 不得删除 `rag/`、`api.py`、`app.py`、`main.py`、`frontend/` 或其测试；
- 不得修复 `packaging/sts2-guide.spec`；该构建阻塞属于后续 EXE 任务；
- 不得顺手处理 TASK-001 的唯一失败场景；
- 不得使用模糊批量删除命令扩大范围。

## 验收标准

### 静态检查

- [ ] 必须删除的文件均不存在；
- [ ] 生产代码和当前 README 不再引用已删除模块/脚本；
- [ ] `rg` 只允许在历史任务记录中出现旧文件名；
- [ ] P0 安装和启动说明不要求 API Key 或完整 RAG 依赖；
- [ ] 可选 RAG/Web 代码仍在，但明确不属于 P0。

### 自动验证

- [ ] `python -m realtime.host --help` 正常返回；
- [ ] TASK-001 评测专项测试通过；
- [ ] 默认固定场景评测仍为 25/26，且不生成报告文件；
- [ ] 完整 Python 测试通过；删除旧迭代测试导致测试数减少属于预期；
- [ ] `git diff` 中没有评分、协议、Mod、SQLite schema 或 RAG 实现修改。

## 实施报告

### 删除内容

以下文件已按任务单删除：

- `advisor/map_path.py`
- `advisor/iteration.py`
- `scripts/eval_advisor_iterations.py`
- `tests/test_advisor_iteration.py`
- `scripts/watch_game_state.py`
- `data/advisor_iteration_report.json`
- `p0-host.out.log`
- `p0-host.err.log`

前两个模块没有 P0 生产入口引用；旧迭代脚本依赖禁止写入的历史决策数据；旧 watch
脚本已被 `python -m realtime.host` 替代；其余均为本机生成物。

### 修改内容

- README 已把 P0 快速开始与可选 RAG/Web 实验入口分开；
- `.gitignore` 已增加 P0 host 日志、评测报告和 Godot 编辑器缓存规则；
- `docs/product-spec.md` 与 `docs/realtime-architecture.md` 已移除旧入口的当前能力引用；
- 发布审查额外确认 `data/guides.json` 为缺少授权信息的本机攻略正文快照，因此只在
  本机保留并加入忽略规则；对应边界已写入 `docs/data-source-audit.md`。

### 验证证据

- `python -m realtime.host --help`：正常返回；
- TASK-001 专项测试：62/62 通过；
- 默认固定场景：25/26 场景、92/93 断言，且未生成评测报告；
- 完整 Python 测试：122/122 通过；
- `git diff --check`：通过；
- 当前 README 和生产代码不再引用已删除入口，旧文件名只保留在历史任务记录中。

### 范围外发现

- `packaging/sts2-guide.spec` 的根目录解析仍需在后续 EXE 任务修复；
- 弹珠袋易伤窗口仍是固定场景评测的唯一已知缺口。

## DeepSeek 验证报告

### 删除目标确认

| 文件 | 状态 |
|---|---|
| `advisor/map_path.py` | 已删除 |
| `advisor/iteration.py` | 已删除 |
| `scripts/eval_advisor_iterations.py` | 已删除 |
| `tests/test_advisor_iteration.py` | 已删除 |
| `scripts/watch_game_state.py` | 已删除 |
| `data/advisor_iteration_report.json` | 已删除 |
| `p0-host.out.log` | 已删除 |
| `p0-host.err.log` | 已删除 |

### 残留引用检查

`rg` 搜索 `map_path|eval_advisor_iterations|watch_game_state|advisor_iteration_report|test_advisor_iteration` 仅命中任务单文件（TASK-001、TASK-002），生产代码和 README 中无任何残留引用。

### .gitignore 确认

- `p0-host*.log` ✓
- `data/advisor_eval_report.json` ✓
- `data/advisor_iteration_report.json` ✓
- `data/guides.json` ✓
- 未使用过宽的 `*.json`、`data/*` 或 `*.log` 规则

### README 确认

- P0 安装使用 `requirements-p0.txt`，开篇声明 "不使用 API Key、Vue、FastAPI、RAG、LLM 或网络" ✓
- P0 启动只保留 `python -m realtime.host` ✓
- 旧 `scripts/watch_game_state.py` 和历史选择权重迭代命令已移除 ✓
- 完整 RAG/Web/FastAPI/Vue 均在 "非 P0 旧实验入口" 章节 ✓

### 自动验证

| 检查项 | 结果 |
|---|---|
| `python -m realtime.host --help` | 正常返回 |
| TASK-001 专项测试 | 62/62 OK |
| 固定场景评测 | 25/26 场景、92/93 断言 |
| 完整 Python 测试 | 122/122 OK（从 TASK-001 的 125 降至 122，来自按计划删除的 3 个旧迭代测试） |
| `git diff` | 空（无评分、协议、Mod、SQLite schema 或 RAG 实现修改） |
| `data/advisor_eval_report.json` | 不存在（默认运行不生成） |

### 禁止事项遵守确认

- 未修改 `advisor/card_reward.py`、`advisor/contextual_scoring.py` 或评分权重 ✓
- 未修改协议 schema、生产者、消费者或 Mod ✓
- 未删除或修改 SQLite 表、迁移和 repository 历史方法 ✓
- 未删除 `rag/`、`api.py`、`app.py`、`main.py`、`frontend/` 或其测试 ✓
- 未修复 `packaging/sts2-guide.spec` ✓
- 未处理 TASK-001 的唯一失败场景 ✓
- 未使用模糊批量删除命令 ✓

## Codex 审查

### 结论

通过。清理没有修改推荐权重、协议、Mod、SQLite schema 或可选 RAG 实现。

### 证据与问题

删除目标、入口帮助、专项测试、固定场景、完整测试和 diff 静态检查均满足验收标准。
测试总数由 125 降为 122，只来自按计划删除的 3 个旧历史迭代测试。

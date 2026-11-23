# TASK-030：自动化程序集 IL 事实提取与对抗复核管线（A1）

- 状态：待审查（M1–M4 已完成，待确认合入 `main` 与推送 CI）
- 创建日期：2026-11-23（用户确认“先 A1 再 A2”及四项架构选型）
- 完成日期：2026-11-23（分支 `feature/fact-pipeline`）
- 分支：`feature/fact-pipeline`（基于已合入 `TASK-028` 的 `main` `6a3f37b`）
- 代码位置：`fact_pipeline/`（Python）+ `tests/test_fact_pipeline.py`
- 关联文档：[`docs/dev/mechanic-facts.md`](../../dev/mechanic-facts.md)、[`scripts/merge_fact_reviews.py`](../../../scripts/merge_fact_reviews.py)、[`tools/Sts2FactExtractor/Program.cs`](../../../tools/Sts2FactExtractor/Program.cs)

## 1. 目标与边界

将原本依赖人工在对话窗口里分批起草、另开会话复核、手工比对程序集 IL 事实的流程，工程化为一套**可自动运行、可离线评测、可安全分流人机协作**的多 Agent 流水线：

1. **直接服务真实痛点**：游戏每次更新（例如本次 `0.113.1` 仍有 126 个待审能力、17 件新遗物）时，自动完成绝大多数实体的 IL 提取、结构化起草、确定性校验和独立对抗复核，只把有分歧或需要新增词表的少数条目交给人裁决。
2. **建立可复现的工程评测基准**：利用仓库里已由人工/交叉审核裁决完成的事实作为标准答案，切分固定 dev/test 集，量化评估自动通过率、自动通过准确率、复核拦截率、返修净收益与成本。
3. **严格边界**：
   - 只读调用 `Sts2FactExtractor` 读取元数据/IL，不加载或执行游戏程序集；
   - 开发/离线工具，不进入实时 Host、Mod 或发行 EXE；
   - 完整游戏反编译 IL 只缓存在被 Git 忽略的目录（如 `build/fact-pipeline/il-cache/`），不提交进仓库；
   - 单元测试使用脚本化假模型与最小合成 IL 夹具，不依赖 API Key 或本机 `sts2.dll`，可直接进入 CI。

## 2. 首批范围

- **实体类型**：卡牌（`cards:`）、能力（`powers:`）、遗物（`relics:`）三类共用主框架，按类型注入对应字段规范与校验规则。
- **版本无关输入**：通过 CLI 参数显式接收 `--extract`、`--facts`、`--knowledge`、`--dll`（及 `--il-cache`）。代码留在干净的 `main` 演进分支上保持 CI 全绿，同时可直接读取 `0.113.1` 数据与本机 `sts2.dll` 完成评测和实战跑批。

## 3. 管线架构

```text
输入（entity_keys + extract + facts + knowledge + sts2.dll/il-cache）
  │
  ▼
程序预取（零工具开销）
  ├─ extract 结构化字段（overrides / calls / canonical_vars / card_ctor / upgrade_deltas / fingerprint）
  ├─ 目标类完整 IL（Sts2FactExtractor --dump <ClassName>，磁盘缓存）
  └─ Spire Codex 目录条目（仅作交叉核对，冲突以程序集为准）
  │
  ▼
Stage 1：起草 Agent（Drafter，最多 6 次工具调用）
  ├─ 只读工具：dump_class / read_enum / lookup_fact
  └─ 提交工具：submit_fact → 直接跑 merge_fact_reviews 确定性校验
       └─ 校验失败：报错原样退回自修（最多 2 次格式修复）
  │
  ▼
Stage 2：独立对抗复核 Agent（Auditor，全新上下文，最多 4 次工具调用）
  ├─ 只看原始 IL + extract + 规范 + 候选事实（不看 Drafter 推理过程）
  └─ 提交工具：submit_audit(verdict, issues, needs_new_vocabulary, notes)
  │
  ▼
Stage 3：分流与受控返修
  ├─ agree 且无新词表需求且无未解释 Codex 冲突 → auto_approved
  ├─ disagree 且无新词表需求 → 带复核意见返修 1 轮（Drafter → 确定性校验 → 全新 Auditor 复审）
  │    ├─ 复审 agree → auto_approved（标记 revised=true）
  │    └─ 仍 disagree → needs_human
  └─ 触发 needs_new_vocabulary / 校验失败 / 超预算 → needs_human
  │
  ▼
输出
  ├─ auto_approved.json（兼容 scripts/merge_fact_reviews.py，含 audit 元数据）
  ├─ needs_human.json + review-queue.md（人工裁决队列，附分歧原因与 IL 证据）
  └─ per-entity trace.jsonl（调用、工具、token、费用、校验与分流全记录，支持断点续跑）
```

### 技术选型（2026-11-23 已确认）

- **语言**：Python（`fact_pipeline/`），直接复用 `storage/assembly_facts.py` 与 `scripts/merge_fact_reviews.py` 的校验和词表常量，消除跨语言重复定义。
- **模型接口**：OpenAI 兼容 SDK（默认 `deepseek-flash`，沿用 `coach/src/llm.ts` 的定价表与 `thinking: {"type": "disabled"}` 以保障强制 `tool_choice` 生效；固定系统提示前缀以充分利用 prompt cache；保留 `--auditor-model` 参数以便做同模型 vs 跨模型对比）。

## 4. 模块设计

| 模块 | 职责 |
|---|---|
| `fact_pipeline/extractor.py` | 封装 `Sts2FactExtractor`（`--dump`、`--dump-any`、`--enum`）与本地缓存，提供可注入协议以便单测 |
| `fact_pipeline/validator.py` | 复用 `merge_fact_reviews.validate_entry` 与 `codex_conflicts`，补齐三类实体必需字段的结构校验 |
| `fact_pipeline/llm.py` | OpenAI/DeepSeek 客户端封装、费用核算、强制 tool_choice、可替换假客户端接口 |
| `fact_pipeline/prompts.py` | 共享可缓存系统规范前缀、预取上下文组装、起草/返修/复核三类提示 |
| `fact_pipeline/pipeline.py` | 单实体状态机（Drafter → Gate → Auditor → 最多 1 轮返修 → Triage）与批量断点续跑 |
| `fact_pipeline/compare.py` | 事实语义比较器（规范化比较核心结构字段，忽略 `evidence` 措辞与列表顺序噪音） |
| `fact_pipeline/evaluate.py` | 分层 dev/test 切分、从留存输出离线重算核心指标、生成人工裁决报告 |
| `fact_pipeline/cli.py` | CLI：`run`、`eval split`、`eval run`、`eval metrics`、`report` |
| `tests/test_fact_pipeline.py` | 全离线单元测试（校验门禁、状态机全分支、比较器语义等价/差异、指标重算） |

## 5. 离线评测计划

- **数据源**：与本机 `sts2.dll` 一致的 `0.113.1` 已审事实（从 `feature/v0113` 导出到隔离目录 `build/fact-pipeline/v0113-inputs/`，不污染 `main` 下的 `0.111.0` 数据）。
- **防泄露切分**：调 prompt 前按固定种子分层切分：
  - `dev`（约 24 条）：覆盖简单/复杂卡牌、愚者新卡、基线/触发/局外遗物、能力，仅用于调 prompt 与校准比较器；
  - `test`（约 60 条）：分层抽样且专门纳入此前手工复核出现过分歧的高难条目；prompt 定稿后只跑一次。
- **核心指标**（全部可由保存的 JSONL trace 与结果文件离线重算）：
  1. 确定性校验首次通过率 / 修复后通过率；
  2. 自动通过率（`auto_approved` 占比）；
  3. **自动通过准确率**（`auto_approved` 中与人工最终裁决语义一致的比例——最关键安全指标）；
  4. **复核拦截率**（起草存在实质错误时，Auditor 成功判为 `disagree` 的比例）；
  5. 返修 1 轮的净收益（修对条数 vs 改坏条数）；
  6. 消融/模型对比（单起草 vs 起草+复核；`deepseek-flash` vs `deepseek-v4-pro` 复核）与单条成本/耗时。

## 6. 分步里程碑与实施报告

1. **M1（已完成）**：实现 `fact_pipeline/`（`extractor.py`、`validator.py`、`llm.py`、`prompts.py`、`pipeline.py`、`cli.py`、`__main__.py`），12 项初始离线单测通过（覆盖隔离复核、字符串 JSON 容错、校验修复、返修 1 轮上限、新词表即时分流、断点续跑与跳过复核消融）。
2. **M2（已完成）**：实现 `compare.py`（`strict_match` + `core_match` + 字段级 `field_diffs`）、`evaluate.py`（分层切分 + 离线重算全部指标）、`lookup_fact` 目标自查禁用与评测期切分集答案脱敏；单元测试扩至 18 项全部通过；冻结 `fact_pipeline/eval/split.json`（种子 42：`dev` 24 条、`test` 60 条）。
3. **M3（已完成）**：
   - `dev` 四组受控迭代（指标留档 `fact_pipeline/eval/metrics/dev-p{1,2,3,4}.json`）：`p1`（25.0% / 拦错 5.6%）→ `p2` 强门禁与完整规范（59.1% / 拦错 18.2%）→ `p3` Flash 思考模式 Auditor（**68.2%** / 拦错 33.3% / 0 误杀 / $0.39）→ `p4` `deepseek-v4-pro` 跨模型 Auditor（66.7% / 拦错 50.0% / 0 误杀 / $0.93）；
   - `test` 60 条盲测（仅运行一次，配置 `p3`，指标留档 `fact_pipeline/eval/metrics/test-p3.json`）：过门禁 60/60（首次 57/60），自动通过 48/60（80.0%），自动通过 Core 准确率 43.8%（配对仅起草基线 33.3%，+10.4pp），复核拦错率 45.9%（17/37），0 误杀（0/20），人工分流精准率 100%（12/12），总费用 $0.824（缓存命中 76.7%）。
4. **M4（已完成）**：
   - 对 `v0.113.1` 全部 57 条未审新增实体（17 新遗物 + 40 新能力）执行实战跑批（`build/fact-pipeline/v0113-new-batch/`，汇总留档 `fact_pipeline/eval/metrics/v0113-new-batch-summary.json`）：44/57（77.2%）自动通过且 0 校验报错，13/57（22.8%）分流人工队列（12 缺词表 + 1 返修后分歧），总费用 $0.831（缓存命中 78.3%）；抽查 `AIM`/`STRETCH`/`HONING`/`YAH`/`WHITTLING_KNIFE` 全部准确；
   - 新增 [`docs/fact-pipeline.md`](../../fact-pipeline.md) 并更新根 `README.md`。
   - 未改动游戏 Mod 或已发行包，无需安装/真机验证；57 条实战产物保留在本地 `build/` 待人工抽查裁决后再合入 `feature/v0113`。

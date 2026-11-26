# 自动化程序集 IL 事实提取与对抗复核管线（Fact Pipeline）

> 说明：对应 [`TASK-030`](tasks/TASK-030-fact-pipeline/README.md)，所有评测指标保存在 [`fact_pipeline/eval/metrics/`](../fact_pipeline/eval/metrics/)，可脱离 API Key 复现核对。完整反编译 IL 仅缓存在被 Git 忽略的 `build/` 目录，不入库。

## 1. 解决什么问题

实时评分引擎（`advisor/deck_fitness.py`）依赖按类代码指纹锁定的程序集机制事实（`data/assembly/facts.json`），不信任自然语言描述正则。每次游戏版本更新（如 `0.111.0 → 0.113.1`），代码变动的条目会自动标为 `stale`，新增角色还会带来上百条未审实体。

过去这些事实需要开发者在对话中手工分批起草、另开会话复核、再人工裁决。`fact_pipeline` 将这一流程工程化为**可重复运行的 Code-to-Fact 人机协作初筛与对抗复核管线**，覆盖卡牌（`cards:`）、能力（`powers:`）与遗物（`relics:`）三类实体：自动完成初稿起草、确定性门禁校验与独立对抗复核，先把缺词表、存在 Codex 冲突或复核分歧的高风险条目精准分流进人工队列（`needs_human`），其余条目输出为带 IL 证据的初筛通过草稿（`auto_approved.json`），经人工抽查确认后再经 `merge_fact_reviews.py` 合入生产 `facts.json`（不作为无人值守直接入库系统）。

## 2. 架构

```text
输入（entity_keys + sts2-extract.json + facts.json + knowledge.json + sts2.dll / IL 缓存）
  │
  ▼
程序预取（零工具调用开销）
  ├─ extract 结构化元数据（overrides / calls / canonical_vars / card_ctor / upgrade_deltas / fingerprint）
  ├─ 目标类完整 .NET IL（Sts2FactExtractor --dump，按程序集哈希落盘缓存）
  └─ Spire Codex 目录条目（仅作交叉核对，冲突时以程序集为准）
  │
  ▼
Stage 1：起草 Agent（Drafter，最多 6 次只读工具调用）
  ├─ 按需只读工具：dump_class / read_enum / lookup_fact（禁止自查目标实体防泄露）
  └─ submit_fact → 确定性校验门禁（复用 merge_fact_reviews 词表与结构硬规则）
       └─ 校验不通过：精确错误清单退回自修（最多 2 次）
  │
  ▼
Stage 2：独立对抗复核 Agent（Auditor，全新隔离上下文，最多 4 次只读工具调用）
  ├─ 不接收 Drafter 对话历史；强制先输出独立 il_reading 再给 verdict
  └─ submit_audit(il_reading, issues, verdict, needs_new_vocabulary, notes)
  │
  ▼
Stage 3：分流与受控返修
  ├─ agree 且无新词表需求且无 codex_conflicts → auto_approved（人机协作初筛通过，人工抽查后对接 merge_fact_reviews.py）
  ├─ agree 但存在未消解 codex_conflicts → needs_human（reason=codex_conflict）
  ├─ disagree 且无新词表需求 → 带具体意见返修 1 轮（重新过门禁）→ 全新 Auditor 复审
  │    ├─ 复审 agree 且无 codex_conflicts → auto_approved（标记 revised=true）
  │    └─ 仍 disagree / 存在 codex_conflicts → needs_human
  └─ 触发 needs_new_vocabulary / 校验耗尽 / 超预算 → needs_human（输出 review-queue.md）
```

### 核心模块

| 模块 | 职责 |
|---|---|
| [`fact_pipeline/extractor.py`](../fact_pipeline/extractor.py) | 只读封装 `Sts2FactExtractor`（`--dump` / `--dump-any` / `--enum`）与磁盘缓存，支持测试夹具注入 |
| [`fact_pipeline/router.py`](../fact_pipeline/router.py) | 确定性静态 IL 复杂度评分与分级自适应路由器 `ComplexityRouter`（`simple` / `medium` / `hard` 与返修升级、高风险放行门禁及离线风险仿真） |
| [`fact_pipeline/validator.py`](../fact_pipeline/validator.py) | 复用 `merge_fact_reviews` 词表并收紧结构规则（禁止 `effects` 混入 `when`、禁用 `generate`、强制 `target`/`destination`/`offer_rewards`/trigger 键、校验 `add_card.card` 目录存在性并拦截 `when=="turn_start"` + `counter=="turn"` + `every>1` 永假组合） |
| [`fact_pipeline/llm.py`](../fact_pipeline/llm.py) | OpenAI 兼容客户端（默认 DeepSeek），精确统计含缓存命中的费用，支持独立切换 Auditor 思考模式与跨模型 |
| [`fact_pipeline/prompts.py`](../fact_pipeline/prompts.py) | 固定可缓存系统规范前缀 + 预取上下文 + 强制先独立读 IL 再给裁决的 Auditor 提示 |
| [`fact_pipeline/pipeline.py`](../fact_pipeline/pipeline.py) | 实体状态机、严格上下文隔离、保留 `initial_draft` 供配对消融、并发执行与按指纹断点续跑 |
| [`fact_pipeline/compare.py`](../fact_pipeline/compare.py) | 语义事实比较器（区分 `strict_match` 与无损归一化的 `core_match`，输出字段级 `field_diffs`） |
| [`fact_pipeline/evaluate.py`](../fact_pipeline/evaluate.py) | 固定种子分层切分 `dev`/`test`、防泄露脱敏、离线重算全部指标 |

## 3. 关键设计取舍

1. **程序预取 + 固定前缀缓存**：目标类的 `extract`、完整 IL 与 Codex 对照由程序一次性注入，Agent 仅在遇到辅助类、父类或未知枚举时才调工具；跨实体共享同一份系统规范前缀，实测盲测集 prompt cache 命中率达 **76.7%**，单条全流程均价降至 **$0.0137**。
2. **确定性校验门禁前置于模型复核**：在 `dev-p1` 中发现大量错误属于约定与结构违规（例如把抽牌触发塞进 `effects`、使用旧别名 `generate`、把 `damage.target` 写成 `enemy`）。将这些硬规则沉淀到 `validator.py` 后，`dev` 首次过门禁升至 **100%**，`auto_core_prec` 直接跃升 **+34.1pp**。
3. **解决 Auditor 橡皮图章问题**：`dev-p1` 中同模型非思考 Auditor 几乎全票放行（拦错率仅 **5.6%**，连 `VOID` 正负号写反都未拦住）。通过“完全隔离上下文 + 强制先写 `il_reading` 再给 `verdict` + 仅对 Auditor 开启思考模式”，`dev` 拦错率升至 **33.3%**（盲测集 **45.9%**），同时保持 **0% 误杀率**。
4. **严格防泄露评测**：`lookup_fact` 强制禁止查询目标实体本身；`eval run` 模式会对整个 `dev` + `test` 切分集的所有条目剥离已审字段，确保模型无法从历史事实库偷看答案。
5. **不回头用测试集调参**：所有 prompt、门禁与比较器迭代只在 `dev`（24 条）完成；60 条 `test` 盲测集（含 9 条历史上连人工双审都有分歧的 `resolved` 难题）冻结后只跑一次，真实暴露长尾短板。

## 4. 评估结果

数据切分见 [`fact_pipeline/eval/split.json`](../fact_pipeline/eval/split.json)（基于 `v0.113.1` 人工交叉复核后的已审事实，种子 42，无重叠分层抽样：`dev` 24 条、`test` 60 条）。

### 4.1 开发集迭代对比（`dev`，24 条，同口径冻结比较器）

| 配置 | 首次过门禁 | 自动通过 | 自动通过 Core 准确 | 复核拦错率 | 误杀率 | 返修修对/改坏 | 费用（均价） | 缓存命中 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p1` 基线（松门禁 + 非思考 Flash） | 23/24 | 24/24 (100%) | 25.0% (6/24) | 5.6% (1/18) | 0/6 (0%) | 0 / 0 | $0.24 ($0.010) | 57.8% |
| `p2` 强门禁 + 完整规范 + 清单复核 | 24/24 | 22/24 (91.7%) | 59.1% (13/22) | 18.2% (2/11) | 0/13 (0%) | 0 / 0 | $0.31 ($0.013) | 63.1% |
| **`p3` + Flash 思考模式 Auditor（定稿）** | **24/24** | **22/24 (91.7%)** | **68.2% (15/22)** | **33.3% (3/9)** | **0/14 (0%)** | **1 / 0** | **$0.39 ($0.016)** | **74.6%** |
| `p4` + `deepseek-v4-pro` 跨模型 Auditor | 24/24 | 21/24 (87.5%) | 66.7% (14/21) | 50.0% (5/10) | 0/14 (0%) | 0 / 0 | $0.93 ($0.039) | 73.3% |

- `p1 → p2`（强门禁 + 规范）贡献了最大单次跃升（**+34.1pp**）；
- `p2 → p3`（思考模式 Auditor）拦错率近乎翻倍，最终准确率最高且成本仅为 `v4-pro` 的 42%；
- `p4`（`v4-pro` 跨模型复核）拦错率最高（50.0%、0 误杀），但贵 2.4 倍，且非思考起草端在返修轮未能把 Pro 挑出的复杂差异全部修对。

### 4.2 盲测集正式评估与配对消融（`test`，60 条，配置 `p3`，仅运行一次）

| 方案（同一批 60 条初稿配对对比） | 过门禁 | 自动通过 | 自动通过 Core 准确 | 自动通过 Strict 准确 | 复核拦错率 | 误杀率 | 人工队列精准率 | 费用（均价） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 消融基线：仅起草端 + 确定性门禁 | 60/60 | 60/60 (100%) | 33.3% (20/60) | — | — | — | — | — |
| **全链路：起草 + 思考 Auditor + 1 轮返修** | **60/60 (首次 57)** | **48/60 (80.0%)** | **43.8% (21/48)** | **35.4% (17/48)** | **45.9% (17/37)** | **0/20 (0%)** | **100% (12/12)** | **$0.82 ($0.014)** |

按难度分层拆解（`test-p3`）：

| 分层 | 总数 | 自动通过 | 自动通过 Core 准确 | 说明 |
|---|---:|---:|---:|---|
| `relics_combat` | 6 | 3 | **100.0% (3/3)** | 3 条复杂遗物（`POCKETWATCH` 等）准确分流人工 |
| `cards_fool` | 6 | 5 | **80.0% (4/5)** | 第 5 条 `RUMMAGE` 仅差 `count_upgrade: 0` 冗余键 |
| `cards_simple` | 10 | 7 | **71.4% (5/7)** | 3 条特殊机制卡准确分流人工 |
| `cards_complex` | 12 | 10 | **50.0% (5/10)** | 含 `SPINNER`/`ANGER`/`SUMMON_FORTH` 等写法差异；`FEED` 甚至补出了人工漏标的 `can_be_generated_in_combat: false` |
| `powers_all` | 12 | 11 | **27.3% (3/11)** | 复杂多钩子能力仍需更严格字段规范与人工复核 |
| `relics_out_of_combat` | 5 | 5 | **20.0% (1/5)** | `between_fights` 细节字段表达自由度过高 |
| `cards_hard` / `relics_hard` | 9 | 7 | **0.0% (0/7)** | 历史上人工双审即存在分歧的极难条目，包含 `SELF_DEPRECATE` 升级增量误写、条件漏标等真实错误 |

### 4.3 复杂度自适应路由与离线风险权衡（`ComplexityRouter`，`--routing auto|fixed`）

针对 `test-p3` 暴露出的明显难度断层（简单卡牌 `71.4%–80.0%` vs `powers` `27.3%`、`between_fights` `20.0%`、`hard` `0.0%`，且 11 次返修仅 1 次完全转正、8 条 `revised_auditor_agreed` 几乎全错），新增纯静态 IL 复杂度路由器 [`fact_pipeline/router.py`](../fact_pipeline/router.py)（CLI 参数 `--routing auto|fixed`，默认 `fixed` 保持向后兼容）：

- **确定性静态特征评分（调 Drafter 前零 LLM 开销）**：综合扫描实体类别（`powers`/`relics`/`monsters` 先验；无钩子 `<30` 行标记型 power 保留 `simple`）、非元数据游戏钩子数（`>= 2` 显著加权）、跨战斗持久状态与局外钩子（`AfterObtained`/`AfterCombatEnd`/`SavedProperty`/实例字段写 `stfld` 等）、异步状态机 `<...>d__` / 闭包 `<>c` 数量、以及条件分支密度与调用数，分为 `simple`（`<= 2`）、`medium`（`3–5`）、`hard`（`>= 6`）三档。
- **自适应路由与门禁策略（`--routing auto`）**：
  - `simple`：保持低成本非思考 Flash 快速通道；
  - `medium`：自动开启 `drafter_thinking=True` 与 `auditor_thinking=True`，并将工具额度提升至 `8/5`；
  - `hard`：自动开启双端思考模式、升级起草模型至 `--hard-model`（默认 `deepseek-v4-pro`）、工具额度提至 `10/6`，并默认收紧自动放行门禁（即使复核 `agree` 也分流 `needs_human`，`reason="high_risk_tier_requires_human"`）；
  - **返修轮升级**：凡首轮 Auditor `disagree` 进入返修轮，一律开启 `drafter_thinking=True`、提额至 `6` 次工具调用（`medium`/`hard` 或多条 issue 同步升级模型），且默认要求返修条目进入人工确认（`reason="revised_requires_human_review"`）。

**可复现离线风险分层与门禁权衡**（完全使用领域先验规则、未对测试集调参，直接对已落盘的 `dev-p3.json` 与 `test-p3.json` 结合 `sts2-extract.json` + 本地 IL 缓存运行 `simulate_routing_policy`）：

1. **静态复杂度分档的单调风险区分度（`test-p3`，60 条盲测）**：

| 静态 IL 复杂度档位 | 条数（占比） | 原 `p3` 自动通过数 | 原 `p3` 自动通过 Core 准确率 | 原 `p3` 整体 Core 准确率 | `--routing auto` 动作 |
|---|---:|---:|---:|---:|---|
| `simple`（`score <= 2`） | 23（38.3%） | 21 | **57.1%（12/21）** | 52.2%（12/23） | 保持低成本 Flash 快速通道 |
| `medium`（`score 3–5`） | 15（25.0%） | 11 | **45.5%（5/11）** | 33.3%（5/15） | 开启起草思考模式 + 提额 |
| `hard`（`score >= 6`） | 22（36.7%） | 16 | **25.0%（4/16）** | 18.2%（4/22） | 升级模型与思考 + 收紧自动放行 |

> 注：在 24 条开发集 `dev-p3` 上，`simple` 档原 `p3` 自动通过 Core 准确率为 **100.0%（7/7）**，而 `medium` 为 `50.0%（4/8）`、`hard` 为 `57.1%（4/7）`（整体 `44.4%`）。

2. **门禁阈值权衡（离线重放 `test-p3` 60 条，仅计门禁分流下界，尚未计入 `medium`/`hard` 开启思考与返修升级带来的实网准确率提升）**：

| 放行门禁策略 | 自动通过率 | 自动通过 Core 准确率 | 相比基线多拦错条数 | 误分流正确条数 | 人工队列真实含错率 |
|---|---:|---:|---:|---:|---:|
| 基线 `fixed`（全档 + 返修均可自动通过） | 80.0%（48/60） | 43.8%（21/48） | 0 | 0 | 100.0%（12/12） |
| 仅拦截 `hard` 档（`max_tier=medium, allow_rev=True`） | 53.3%（32/60） | 53.1%（17/32，**+9.4pp**） | +12 | 4 | 85.7%（24/28） |
| **默认 `auto`（`max_tier=medium, allow_rev=False`）** | **45.0%（27/60）** | **63.0%（17/27，+19.2pp）** | **+17** | **4** | **87.9%（29/33）** |
| 最严门禁（`max_tier=simple, allow_rev=False`） | 28.3%（17/60） | **70.6%（12/17，+26.8pp）** | +22 | 9 | 79.1%（34/43） |

> 关键观察：禁止返修轮自动放行（`allow_rev=False`）在 `test-p3` 上额外拦截了 **5 条** 漏网错误（`12 → 17`），且**零额外误分流正确条目**（`4 → 4`），使自动通过 Core 准确率由 `43.8%` 跃升至 **`63.0%`（+19.2pp）**，同时人工审核队列仍有 **87.9%（29/33）** 是真正需要修正的条目。待沙箱外联网跑一次 `--routing auto` 实网评测，即可进一步验证 `medium`/`hard` 与返修轮开启思考/模型升级的净准确率增益与成本曲线。

## 5. `v0.113.1` 实战跑批（M4）

使用定稿配置 `p3`（`deepseek-flash` 起草 + `--auditor-thinking` 独立复核）对 `v0.113.1` 全部 **57 条新增未审实体**（17 件新遗物 + 40 个新增能力）执行一次真实批处理（结果目录 `build/fact-pipeline/v0113-new-batch/`，汇总留档见 [`fact_pipeline/eval/metrics/v0113-new-batch-summary.json`](../fact_pipeline/eval/metrics/v0113-new-batch-summary.json)）：

| 指标 | 结果 |
|---|---|
| 批次规模 | 57 条（17 新遗物 + 40 新能力，覆盖全部愚者核心机制能力） |
| 确定性门禁通过 | 44/44 自动通过条目 `validate_fact` 0 报错 |
| 自动通过（`auto_approved`） | **44/57（77.2%）**（遗物 13/17、能力 31/40；其中 13 条经 Auditor 挑刺后返修收敛） |
| 分流人工队列（`needs_human`） | **13/57（22.8%）**（12 条识别出需要新增词表，1 条返修后仍有分歧，已输出 `review-queue.md`） |
| 成本与缓存 | 总费用 **$0.831**（均价 **$0.0146/条**），prompt cache 命中率 **78.3%** |

关键愚者新能力与新遗物抽查结果（均与 `docs/dev/mechanic-facts.md` 愚者词表约定一致）：
- `powers:AIM`：`trigger`、`duration: until_triggered`，`modifier: damage (+1/stack, from_arrow & powered_attack)`；
- `powers:STRETCH`：`trigger`、`duration: until_triggered`，`modifier: block (+1/stack)`；
- `powers:HONING`：`trigger`、`duration: combat`，`modifier: damage (multiply_percent_per_stack: 1, from_arrow & powered_attack)`；
- `powers:YAH`：`trigger`、`duration: combat`，`when: arrow_dislodged (once_per_turn) -> energy (+1/stack)`；
- `relics:WHITTLING_KNIFE`：`trigger`，`when: shuffle -> apply_power Aim 5`。

> 注：本批次产物位于本地 `build/fact-pipeline/v0113-new-batch/`（`auto_approved.json` 与 `review-queue.md`），尚未自动合入 `feature/v0113` 的 `facts.json`，待人工抽查与裁决 13 条队列后再按 `merge_fact_reviews.py` 合入。

## 6. 已知限制与定位边界

- **明确人机协作初筛定位，严禁无人值守直合生产 `facts.json`**：60 条盲测集上 `auto_approved` 的 Core 准确率为 **43.8%（21/48）**（其中常规战斗遗物 `100%`、愚者卡 `80%`、简单卡 `71.4%`，但 `powers` 仅 `27.3%`、`between_fights` 仅 `20%`、历史分歧难题 `0%`）。本管线的真实工程定位是**高召回人机协作初筛器**：自动拦截 100% 缺词表条目、阻断未消解 `codex_conflicts` 与拼写编造卡牌 ID，并为常规条目提供带 IL 证据的结构化初稿以大幅压缩人工审核成本；`auto_approved.json` 在合入生产事实库前仍须人工抽查与裁决。
- **返修轮转化率偏低（盲测 11 次返修仅 1 次完全转正）**：当思考模式 Auditor 发现深层 IL 误读时，非思考模式的起草端在返修轮往往难以彻底重构复杂结构；后续若给返修轮也开启思考模式或交由 `v4-pro` 返修，有望显著提高挽回率。
- **门禁通过率口径**：`evaluate.py` 的 `schema_final_pass` 严格要求最终 `validation.ok` 为真（无 `validation_problems`），避免 `needs_new_vocabulary=True` 但 schema 校验未通过的草稿虚增通过率。
- **语义比较器对部分低频写法仍偏严**：如 `note_text`、`count_upgrade: 0`、`to` vs `destination` 会拉低自动指标；坚持不在测试集上回头改比较器以保评测诚实。

## 7. 运行命令

```powershell
# 离线单测（无需 API Key 或 sts2.dll）
python -m unittest tests/test_fact_pipeline.py -v

# 导出某分支数据并生成分层切分
python -m fact_pipeline eval export-branch --branch feature/v0113
python -m fact_pipeline eval split --extract build/fact-pipeline/v0113-inputs/sts2-extract.json --facts build/fact-pipeline/v0113-inputs/facts.json

# 运行离线评测（需要 DEEPSEEK_API_KEY；支持 --routing auto|fixed）
python -m fact_pipeline eval run dev --extract build/fact-pipeline/v0113-inputs/sts2-extract.json --facts build/fact-pipeline/v0113-inputs/facts.json --knowledge build/fact-pipeline/v0113-inputs/knowledge.json --auditor-thinking --routing auto --out build/fact-pipeline/eval/dev-auto

# 对未审实体实战跑批（输出 auto_approved.json 与 review-queue.md）
python -m fact_pipeline run --extract build/fact-pipeline/v0113-inputs/sts2-extract.json --facts build/fact-pipeline/v0113-inputs/facts.json --knowledge build/fact-pipeline/v0113-inputs/knowledge.json --states unreviewed stale --types relics powers --auditor-thinking --routing auto --out build/fact-pipeline/v0113-run
```

# TASK-001：固定选牌场景评测基线

- 状态：已验收
- 创建者：Codex
- 实现者：DeepSeek
- 产品依据：`docs/product-spec.md` 的 P0-3、P0-6 与“P0 不收集训练样本”
- 前置任务：无

## 目标

建立一套包含 24–30 个真实稳定卡牌 ID 的固定选牌场景，以及可重复运行的离线评测器，
用于检查推荐结果、动态跳过和关键评分因子。

评测必须由提交在仓库中的显式场景驱动，不读取历史选择、用户行为、AppData 当前局、
本机 tier 文件或网络。这个任务只建立评测基线，不修改推荐权重。

## 当前证据

- `docs/project-status.md` 明确要求回放至少 20–30 个固定场景；
- `scripts/eval_advisor_iterations.py` 当前读取
  `load_labeled_card_reward_decisions()`，不适合作为 P0 固定场景评测入口；
- `advisor/card_reward.py` 已返回推荐索引、动态跳过、候选分数和可追踪因子；
- `data/knowledge.json` 包含当前 catalog 的稳定卡牌、遗物和角色 ID；
- 现有单元测试覆盖单条规则，但没有独立的、可人工审阅的产品场景集。

## 实现范围

### 必须修改

- 新增 `data/advisor_eval_scenarios.json`；
- 新增 `advisor/evaluation.py`，实现纯离线场景加载、校验和评测；
- 新增 `scripts/eval_card_reward_scenarios.py`，作为 P0 固定场景评测入口；
- 新增 `tests/test_advisor_evaluation.py`；
- 更新 `README.md` 中的 P0 推荐器评测命令与说明，不再把历史选择权重迭代描述为
  当前 P0 流程；
- 在 `scripts/eval_advisor_iterations.py` 顶部和 CLI 帮助中明确标记为
  “非 P0 的旧实验工具”，不得让新评测器调用它。

### 场景集要求

场景总数为 24–30，使用 `data/knowledge.json` 中可验证存在的真实稳定 ID。至少覆盖：

- 4 个牌组功能缺口：输出、防御、抽牌、范围；
- 4 个 HP/章节/楼层上下文；
- 4 个遗物与卡牌协同；
- 4 个牌组协同：包括 provider/payoff、重复牌惩罚和多信号并存；
- 4 个真实图结构形式的路线压力：普通、可选精英、必经精英、Boss；
- 4 个跳过行为：强候选压低跳过、成熟牌组跳过、`can_skip=false`、未知或低信息候选。

每个场景必须包含：

- 唯一 ID、分类、中文说明和人工可读理由；
- 完整的 `state`、有稳定 ID 的 `options` 和 `can_skip`；
- 至少一种明确断言：推荐卡/跳过、候选相对顺序、跳过资格、决策状态或必须出现/
  不得出现的 factor code；
- 预期结论必须来自产品规则和场景理由，不能先运行当前算法再把输出抄成期望值；
- 对确实存在策略争议的场景，优先断言相对分数或因子，不伪造唯一“正确答案”。

### 评测器要求

- 使用临时 SQLite，并仅从 `data/knowledge.json` 同步 catalog；
- 不加载 `data/local/`、社区统计、历史决策、当前局检查点或 AppData 数据库；
- 运行前校验场景结构以及所有可校验的角色、卡牌和遗物 ID；
- 调用生产代码 `recommend_card_reward()`，不得复制一套评分实现；
- 输出总数、通过数、失败数、通过率和逐场景结果；
- 失败项至少显示期望、实际推荐、跳过分、候选分、相关 factor code 和场景理由；
- 相同输入连续运行两次时，除耗时或生成时间外，结果必须一致；
- 支持 `--strict`：存在场景断言失败时以非零状态退出；
- 默认模式允许生成诚实的基线报告；本任务不要求当前推荐器达到 100%。

### 允许修改

- 为复用代码，可对测试辅助函数做最小提取；
- 可新增一个默认被 Git 忽略的评测结果文件路径，但场景源文件必须提交；
- 可补充必要的类型注解和输入错误信息。

### 必须补充的测试

- 合法场景加载和运行；
- 缺少字段、重复场景 ID、未知卡牌/遗物 ID 被明确拒绝；
- 推荐卡、跳过、相对顺序、factor code 断言均有正反例；
- `--strict` 或其底层状态判定在失败时返回非零；
- 相同输入连续评测结果一致；
- 测试证明评测器不调用历史决策加载方法和网络。

## 禁止事项

- 不得修改 `advisor/card_reward.py`、`advisor/contextual_scoring.py` 或任何评分权重；
- 不得为了让场景通过而降低、删除或运行后改写断言；
- 不得把玩家选择当作标准答案或胜率标签；
- 不得恢复逐次决策持久化、自学习或自动调权；
- 不得读取真实 `active-run.json`、本机 tier、AppData 数据库或网络；
- 不得扩展到路线推荐、商店、篝火、药水决策、RAG 或 LLM；
- 不得删除旧数据库表或迁移；旧实验工具的彻底清理另行评审。

## 涉及文件

- `data/advisor_eval_scenarios.json`：可人工审阅的固定场景；
- `advisor/evaluation.py`：校验和评测逻辑；
- `scripts/eval_card_reward_scenarios.py`：命令行入口；
- `tests/test_advisor_evaluation.py`：评测基础设施测试；
- `README.md`：当前 P0 评测入口；
- `scripts/eval_advisor_iterations.py`：仅增加旧实验工具警示。

## 验收标准

### 自动验证

- [ ] 场景数量、分类覆盖和 ID 完整性满足任务要求；
- [ ] 评测器只调用生产推荐器并使用临时数据库；
- [ ] 默认模式生成完整、可复现的基线报告；
- [ ] `--strict` 能准确反映断言失败；
- [ ] 新增评测测试通过；
- [ ] 完整 Python 测试通过；
- [ ] `git diff` 中没有评分逻辑和权重修改。

### 人工审查

- [ ] Codex 抽查场景理由不是根据当前输出倒填；
- [ ] Codex 抽查真实 ID、地图结构与断言语义；
- [ ] DS 如实报告基线失败项，不自行调分；
- [ ] README 不再把历史行为标签迭代表述为 P0 当前能力。

本任务不包含真机验证，也不因为场景报告生成而更新
`docs/project-status.md` 中的真机完成状态。

## DeepSeek 实施报告（第三轮返修 — Codex 第二轮审查）

### 修改内容

| 文件 | 目的 |
|---|---|
| `advisor/evaluation.py` | **(1)** 修复 `check_assertions()` 与 `evaluate_scenarios()` 之间 passed/total 参数顺序颠倒（原返回 `(all_passed, passed, total, results)` 但调用处按 `(all_passed, total, passed_count, results)` 解构）；**(2)** 重写断言容器深度校验：`assert_factors_include` 等必须是非空字符串列表，`assert_card_factors_include` 必须是 `dict[str, non-empty-list[str]]`，`card_order` 至少 2 个非空字符串，空 list/dict 不算可执行断言；**(3)** `validate_scenario_ids()` 新增交叉校验：`assert_card_factors_include/exclude`、`assert_card_score_above/below`、`card_order`、`recommended_card` 引用的卡牌 ID 必须存在于场景 options 中 |
| `data/advisor_eval_scenarios.json` | **(4)** `relic_vulnerable_synergy_011`：重写语义为"弹珠袋提供开场易伤窗口应提高攻击牌价值，而非奖励重复易伤"；Pommel Strike → relic_synergy（攻击牌利用易伤窗口），Thunderclap → 不获得 relic_synergy（效果重叠）；**(5)** `multi_signal_coexistence_015`：从 block_efficiency 零增量改为牌组同时缺乏输出和防御时 Iron Wave 应同时获得 attack_coverage + defense_coverage 两个非零信号 |
| `tests/test_advisor_evaluation.py` | **(7)** 清理 `original_create_connection` 未使用变量和 `import tempfile as tmpfile` 未使用导入；新增 16 个测试（62 → 62 全覆盖）：统计顺序不变式（2 个）、容器深度校验（9 个）、卡牌 ID 交叉校验（5 个） |
| `README.md` | **(6)** 重写"运行"章节：P0 实时模式优先（`python -m realtime.host` + Godot 面板），旧 Web/RAG/CLI/文件桥接全部归入"非 P0 旧实验入口"独立章节，不再混写 |

未修改评分逻辑文件：`advisor/card_reward.py`、`advisor/contextual_scoring.py`、`storage/relational.py`。

### 验证证据

- **断言统计修复验证**：报告从 "断言总数 90、通过 92、通过率 102.2%" 修正为正常值 "断言总数 93、通过 92、通过率 98.9%"
- **容器校验**：`{"assert_factors_include": "abc"}`、`{"assert_card_factors_include": {}}`、`{"assert_factors_include": []}` 等 9 种无效输入均被明确拒绝
- **卡牌 ID 交叉校验**：`validate_scenario_ids()` 拒绝断言中引用但不在 options 中的卡牌 ID；正向测试验证合法引用通过
- **场景 011（重写）**：诚实失败 —— Pommel Strike 未从弹珠袋获得 relic_synergy（弹珠袋缺少 supports_attack 标签），Thunderclap 也未获得 relic_synergy（正确拒绝冗余易伤）。这是干净的遗物协同产品缺口
- **场景 015（重写）**：通过 —— Iron Wave 在双缺牌组中同时获得 attack_coverage 和 defense_coverage
- **默认 CLI**：25/26 场景通过，1 诚实失败，无报告文件副作用
- **`--strict`**：返回 1（1 个场景断言失败）
- **新评测测试**：62/62 OK（原 46 + 新增 16）
- **完整测试**：125/125 OK（原 109 + 新增 16）
- **评分文件 git diff**：空（未修改）
- **网络拦截**：socket patch 测试验证评测路径无连接尝试
- **无 --database**：CLI help 不含 --database，始终使用临时 SQLite

### 本轮返修的额外发现

1. `relic_vulnerable_synergy_011`：弹珠袋（BAG_OF_MARBLES）的 effect_tags 不包含 `supports_attack`，因此攻击牌 Pommel Strike 无法从弹珠袋获得 relic_synergy。这是一个双层缺口：既缺少 supports_vulnerable 检查（已发现），也缺少"开场施加 debuff → 对应攻击窗口"的间接协同推导。Codex 需决定是否在后续任务加入间接协同规则。
2. `multi_signal_coexistence_015` 现在使用 attack_coverage + defense_coverage 双信号，两者均为非零 delta，正确验证了多信号并存的产品语义。
3. 场景 011 的重写比原来更精确地反映了"不奖励重复易伤"的产品原则，且如实暴露了当前评分对间接协同（debuff 窗口 → 攻击牌价值）的缺失。

## Codex 审查

### 结论

需修复。35/35 新测试、98/98 完整测试和 26 场景 CLI 均能运行，但当前
“26/26 通过”不能证明固定场景达到产品评测目标，存在断言语义过弱、输入校验绕过和
默认写入生成物等问题。

### 证据与问题

独立验证：

- `D:\miniconda3\envs\sts2\python.exe -m unittest discover -s tests -v
  -p test_advisor_evaluation.py`：35/35 通过；
- `D:\miniconda3\envs\sts2\python.exe -m unittest discover -s tests -v`：
  98/98 通过；
- `D:\miniconda3\envs\sts2\python.exe
  scripts/eval_card_reward_scenarios.py --strict`：26/26 场景、84/84 断言通过；
- 评分文件修改时间仍早于本任务文件，未发现本轮修改评分逻辑的迹象；但当前仓库大量
  文件未纳入 Git，`git diff` 为空本身不能作为“未修改”的充分证据。

必须修复：

1. **断言必须验证目标候选，而不是在所有候选中全局找因子。**
   增加按稳定卡牌 ID 指定的候选 factor include/exclude 断言，以及专门的跳过 factor
   断言。现有 `assert_factors_include` 可以保留为明确命名的全局断言，但不能用它证明
   某张目标牌获得协同。
2. **场景通过必须与场景描述一致。** 当前 `output_gap_001` 最终推荐“耸肩无视”仍
   判定通过；`relic_block_synergy_012` 最终推荐“剑柄打击”仍判定通过；
   `strength_attack_synergy_016` 最终推荐“耸肩无视”仍判定通过。它们可以不强制唯一
   推荐，但必须验证目标候选实际得到预期因子及相对变化，不能只证明任意候选出现过
   同名因子。
3. **不得为追求全绿而回避已发现的产品缺口。**
   `relic_vulnerable_synergy_011` 不得继续用第二个通用 `supports_attack` 场景代替
   已发现的 vulnerable 关系缺口；`multi_signal_coexistence_015` 不得因为
   `block_efficiency` 为零就删除该预期。若当前评分不满足合理产品断言，应让基线诚实
   失败并记录，留给后续校准任务。
4. **固定场景集只使用真实稳定 ID。**
   删除 `UNKNOWN_*` 的 catalog 校验特例，用真实低信息卡替换
   `unknown_card_uncertain_024`；未知卡安全降级由已有推荐器单元测试或单独的基础设施
   单元测试覆盖，不混入产品固定场景。
5. **校验器必须拒绝无效评测。**
   拒绝空 `assertions`、未知断言键、错误断言类型以及没有产生任何可执行断言的场景；
   测试必须覆盖拼错断言名不能静默通过。
6. **`card_order` 必须表示严格高于。**
   当前使用 `>=`，平分也会被写成“排名高于”。改为严格 `>`；如确需允许平分，新增
   语义不同且名称明确的断言。
7. **评测入口必须始终隔离真实数据库。**
   P0 CLI 删除 `--database`，不得允许用户把 catalog 同步到任意真实数据库；测试如需
   注入 repository，应使用不暴露给 CLI 的内部接口或临时目录。
8. **默认运行不得污染仓库。**
   只有显式传入 `--output` 才写报告；删除已生成的
   `data/advisor_eval_report.json`。若未来保留默认输出，则必须先纳入 `.gitignore`，
   但当前优先采用显式输出。
9. **无网络测试必须真正设置拦截器。**
   当前测试只是正常运行后声称没有网络。至少 patch socket 连接入口并设为抛错，再完成
   评测，证明该路径没有连接尝试。
10. **README 必须与 P0 真实架构一致。**
    当前仍宣称 Vue 是实时界面、`game_state_events`/局面/候选/结果标签写入 SQLite、
    决策 outcome 被持久化，以及历史样本权重迭代属于核心能力。需要按
    `docs/product-spec.md` 和 `docs/project-status.md` 修正；RAG/Web 只能标为可选旧入口，
    不能继续写成 P0 默认运行方式。
11. 清理本任务新增代码中的未使用项，包括
    `REQUIRED_ASSERTION_FIELDS`、`ScenarioResult.card_names`、`_format_factors`、未使用
    import 和 CLI 中未使用的 `RELATIONAL_DB_FILE`。

修复后实施报告必须重新给出：

- 场景校验和候选级断言测试；
- 默认 CLI 运行前后的仓库生成物检查；
- 真实网络连接拦截测试；
- 当前基线的真实通过/失败场景，允许且鼓励如实出现产品规则失败；
- 新测试、完整测试和严格模式的实际退出码。

本次返修仍不得修改 `advisor/card_reward.py`、
`advisor/contextual_scoring.py` 或评分权重。

### 后续冗余审计（不在本次返修执行）

TASK-001 验收后，下一任务优先清理下列 P0 冲突或重复入口：

- `advisor/map_path.py`：没有运行时引用，并包含“无真实边时连接下一行”的禁止性
  fallback；
- `advisor/iteration.py`、`scripts/eval_advisor_iterations.py`、
  `tests/test_advisor_iteration.py`：仅服务历史选择权重实验，与当前 P0 不收集样本冲突；
- `scripts/watch_game_state.py`：功能已被 `python -m realtime.host` 覆盖；
- `data/advisor_iteration_report.json`、根目录 `p0-host*.log` 和当前评测报告等生成物；
- README 中旧 Vue/FastAPI 实时入口与 SQLite 逐事件历史描述。

`rag/`、`api.py`、`app.py`、`main.py` 和 `frontend/` 虽不属于 P0 实时链路，但仍构成
可选攻略解释入口，暂不直接删除；后续只做隔离和明确标记。旧 SQLite 表是否删除涉及
迁移兼容，必须另开任务，不能顺手处理。

另发现 `packaging/sts2-guide.spec` 把 spec 所在的 `packaging/` 目录当成项目根目录，
将尝试读取不存在的 `packaging/realtime/host.py` 和 `packaging/data/*`。这是后台 EXE
任务的构建阻塞，不属于本次评测返修或冗余清理。

## Codex 第二轮审查

### 结论

需再次修复。候选级断言、真实 ID、临时数据库、显式输出、网络拦截和严格排序已经按
要求落地；独立验证中新测试 46/46、完整测试 109/109 通过，默认评测得到 24/26。
但报告统计、嵌套断言校验、两个失败场景的策略语义和 README 仍有阻塞性问题。

### 必须修复

1. **修复断言统计顺序。**
   `check_assertions()` 返回 `(all_passed, passed, total, results)`，调用处却按
   `(all_passed, total, passed_count, results)` 接收，导致实际报告出现
   “断言总数 90、通过 92、通过率 102.2%”。统一返回顺序并新增测试，明确断言
   `0 <= passed <= total` 且本次报告应为通过 90/总数 92，而不是通过 92/总数 90。
2. **完整校验断言容器和内部元素类型。**
   当前以下输入仍会被 `validate_scenario_structure()` 接受：
   `{"assert_factors_include": "abc"}`、
   `{"assert_card_factors_include": []}`、
   `{"assert_factors_include": []}`。
   列表断言必须是非空字符串列表；候选级断言必须是非空
   `dict[str, non-empty list[str]]`；分数 map 必须是
   `dict[str, int|float]`；`card_order` 至少两个非空字符串。仅由空 list/dict 组成的
   assertions 不能算“有可执行断言”。
3. **候选级断言引用的卡牌必须确实在本场景 options 中。**
   至少校验 `assert_card_factors_include/exclude`、卡牌分数 map 和 `card_order` 的键；
   否则不存在候选的 exclude 断言会天然通过。补充正反测试。
4. **改正 `relic_vulnerable_synergy_011` 的策略语义。**
   弹珠袋已经在开场施加易伤，雷霆一击再次施加易伤主要是效果重叠，不能把
   “supports_vulnerable + vulnerable 卡牌”直接定义为协同。该场景应验证：
   已知会提供开场易伤的遗物提高能利用该窗口的伤害/攻击牌价值，并避免奖励重复施加
   易伤本身。可以让正确的新断言如实失败，但不得在 TASK-001 修改评分。
5. **改正 `multi_signal_coexistence_015` 的策略语义。**
   `block_efficiency` 是相对中性基准的增减因子；铁斩波恰好
   `5 block / 1 cost` 得到零增量并被省略，不足以证明产品 bug。把场景改为真正需要
   同时验证两个非零信号的上下文，例如让牌组同时缺输出和防御，断言铁斩波同时得到
   `attack_coverage` 与 `defense_coverage`。不要为了制造失败要求输出零增量因子。
6. **彻底修正 README 的实际运行说明。**
   顶部边界已修正，但下半部分仍把 Web 应用标为“推荐”、让 P0 用户启动
   `scripts/watch_game_state.py`、声称事件/决策/选择写入 SQLite，并称 Vue 显示实时
   建议。P0 快速开始必须改为 `python -m realtime.host`（开发态）和 Godot 游戏内
   面板；旧 Web/RAG/历史迭代内容必须整体置于明确的“非 P0 旧实验入口”章节，不能与
   当前能力混写。
7. 清理本轮测试中的无用变量/import，例如
   `original_create_connection` 和 `import tempfile as tmpfile`。

### 第二轮独立验证

- 新评测测试：46/46 通过；
- 完整测试：109/109 通过；
- 默认 CLI：24/26 场景，无报告文件副作用；
- 严格 CLI：按当前两项失败返回 1；
- 报告统计错误：90 total / 92 passed / 102.2%；
- 手工校验确认三种无效嵌套断言均被错误接受；
- README 仍存在旧 P0 启动与持久化说明。

修复后重新提交待审查；仍不得修改推荐评分逻辑或权重。TASK-002 暂不激活。

## Codex 最终审查

### 结论

已验收。固定场景评测基础设施、场景集和 P0 文档入口达到 TASK-001 范围要求。

### 独立验证

- 评测专项测试：62/62 通过；
- 完整 Python 测试：125/125 通过；
- 默认评测：26 个场景，25 通过、1 失败、0 运行错误；
- 断言统计：92/93，通过率 98.9%，满足 `passed <= total`；
- 严格模式：因 1 个诚实产品缺口返回 1；
- 默认运行未生成报告文件；
- CLI 不暴露 `--database`，评测使用临时 SQLite；
- 固定场景使用真实稳定 ID，候选级断言和嵌套输入校验测试通过；
- 未修改推荐评分逻辑和权重。

保留的失败场景 `relic_vulnerable_synergy_011` 是后续策略校准输入，不是本任务实现
失败。后续实现时必须把“开场易伤带来的攻击窗口收益”和“重复施加易伤的重叠代价”
分开建模，再决定净影响，不能简单把所有易伤卡或所有攻击牌统一加分。

README 已把 P0 运行命令和 Godot 游戏内面板放回主链路；完整 P0 安装说明仍需在
TASK-002 中把 `requirements-p0.txt` 与无需 API Key 放在可选 RAG 依赖之前。

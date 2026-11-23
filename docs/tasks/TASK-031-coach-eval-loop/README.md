# TASK-031：Review Coach Judge 校准与引擎反例闭环（A2）

- 状态：已完成并合入 `main`
- 创建日期：2026-11-24（A1 `TASK-030` 合入 `main` 推送后启动）
- 分支：`feature/coach-eval-loop`（基于 `main` `b199bbd`）
- 关联文档：[`docs/review-coach.md`](../../review-coach.md)、[`TASK-028`](../TASK-028-review-coach/README.md)

## 1. 背景与目标

`TASK-028` 完成后，首轮评估与人工评分如实暴露了两项核心未闭环短板：
1. **对话审查 Agent（`coach/src/critic.ts`）未校准**：20 题样本偏小，同 prompt 多次运行退回率波动 20%–45%；逐条检查 `ask-dev` 输出发现明显误杀（如 `Q04` 引用 E4 原文仍被判 `unsupported`，`Q03`/`Q05` 误杀合理趋势概括与中文并列短语），且尚无误杀率（FPR）与漏检率（1 − Recall）定量基准。
2. **复盘报告 48%（14/29）被评“引擎错了”**：高频备注指向高费牌可打出性（`埋葬` 4 费、`折射` 3 费、`恶魔形态` 3 费）。排查 `advisor/deck_mechanics.py` 确认存在真实游戏物理建模缺陷：`cost > base_energy` 的卡牌在无额外能量时仍按 `supply / demand` 获得高打出率，且整回合满费大牌按可连续拆分能量估算，没有体现单回合能量打包约束。

本任务分两段形成完整的“Eval 发现问题 → 改造 → 留出集量化复测”闭环。

## 2. A2-1：Critic LLM-as-a-Judge 校准

- **基准集**：`coach/eval/critic-bench.json`（60 例，按 20 `dev` / 40 `test` 固定分层切分），全部基于真实对局的真实 JSON 证据池构建：
  - 真实 `ask-dev` 草稿/重写样本（逐句对照证据标注真阳性 vs 误杀）；
  - 7 类可绕过程序数字/卡名检查的隐蔽幻觉反例（首选/玩家选择说反、跨层数字错配、编造战斗/输赢因果、超范围问题强答、无依据因果解释、把引擎意见夸大成定论、汉字数字/手工计数编造）；
  - 针对旧版 `b5` 误杀模式构建的合理正例（直接复述 E0/E4 verdict、趋势概括、中文并列短语、基于 `what_if` 的方向建议、诚实拒答）。
- **评测器**：`coach/src/criticeval.ts`（CLI：`coach eval critic <dev|test>`），每例重复跑 3 轮以同时测量 **Recall、False Positive Rate（误杀率）、Precision、F1、三轮一致率与退回率方差、单次成本**。
- **对照方案**：在 `critic_dev`（20 例 × 3 轮）上对比 `b5` 基线 vs `c1` 校准原子句级 Rubric（非思考 Flash）vs `c2` 思考模式 Flash（及可选 `c3` `deepseek-v4-pro`），选最优方案在冻结的 `critic_test`（40 例 × 3 轮）上验证一次并合入 `coach/src/critic.ts`，补齐离线单元测试。

## 3. A2-2：引擎高费可打出性修复与闭环复测

- **建模修复（仅基于真实游戏规则，不对测试集拟合参数）**：
  1. **超基础能量上限（`cost > base_energy`）**：单回合没有足够额外能量时无法打出费用高于基础能量的牌；按马尔可夫上界 `afford = min(1.0, extra_energy / (cost - base_energy))`（`extra_energy == 0` 时为 `0.0`）约束其打出率，且不可支付部分不挤占其他可打出牌的能量需求（仍计入牌组稀释）。
  2. **高费整块挤占（验证后决定是否纳入）**：在 `data/golden_set/card_reward.json`（79 题）与 `dev` 10 局对局上独立对比验证，只保留对 79 题金样本和 `dev` 均无退化且符合游戏物理的改法。
- **验证门禁**：
  - 全量既有 Python 单测通过 + 新增高费不可打出/有能量来源可打出回归测试；
  - 79 题标准答案集（`golden_set`）不退化；
  - 在 15 局 `test` 对局和 29 条原人工评分条目上如实报告前后变化（`useful` 不退化、多少条高费 `engine_wrong`/`not_useful` 改善、哪些属于流派意图等已知边界仍保留）。

## 4. 交付与收口

- 更新 `docs/review-coach.md`（补充 Critic Judge 校准结果与引擎闭环复测数据）与 `README.md`；
- 确保全量 Python 与 Coach TypeScript 单测通过、CI 绿灯。

## 5. 实施与验证报告

### 5.1 A2-1：Critic LLM-as-a-Judge 校准

- **新增/修改文件**：
  - `coach/eval/critic-bench.json`：60 例固定切分基准（`dev` 20 / `test` 40，正反各半），引用真实 `ask-dev` 证据池；
  - `coach/src/critic.ts`：保留 `CRITIC_PROMPT_B5` 基线，新增校准版 `CRITIC_PROMPT_C1`（7 类错误显式清单 + 反误杀规则 + 自检一致性）并支持 `CritiqueOptions` 与思考模式 fenced JSON 解析；盲测验证后将默认 `CRITIC_PROMPT` 切换为 `CRITIC_PROMPT_C1`；
  - `coach/src/criticeval.ts`、`coach/src/cli.ts`：新增 `coach eval critic <dev|test> [--variant b5|c1|c2] [--repeats N]` 命令，输出指标到 `coach/eval/metrics/critic-*.json` 与明细到 `coach/eval/results/critic/`；
  - `coach/test/critic.test.ts`：新增 4 项离线单元测试（基准平衡性、指标与方差计算、思考模式 JSON 解析、默认 prompt 切换验证）。
- **验证结果（`deepseek-flash`，每例 3 轮重复）**：
  - `dev` 20 例（60 trials）：
    - `b5` 基线：Recall `90.0%`（27/30），FPR `10.0%`（3/30），F1 `0.9000`，三轮全一致 `90.0%`，极差 `0.100`，`$0.029`；
    - `c2` 思考模式（初版 dev）：Recall `96.7%`（29/30），FPR `16.7%`（5/30），F1 `0.9062`，三轮全一致 `85.0%`，`$0.158`（3× 费用、5× 耗时且误杀更高，淘汰）；
    - `c1` 校准版：Recall **`100.0%`**（30/30），FPR **`0.0%`**（0/30），F1 **`1.0000`**，三轮全一致 **`100.0%`**，极差 `0.000`，`$0.051`。
  - **`test` 40 例盲测（120 trials，未基于 test 调参）**：
    - `b5` 基线：Recall `100.0%`（60/60），FPR `28.3%`（17/60），Precision `77.9%`，F1 `0.8759`（多数投票 `0.8889`），三轮全一致 `90.0%`（36/40），`$0.096` / `1.96s`；
    - `c1` 校准版：Recall **`100.0%`**（60/60），FPR **`11.7%`**（7/60，**−16.7pp**），Precision **`89.6%`**（+11.6pp），F1 **`0.9449`**（+6.9pp，多数投票 **`0.9524`**），三轮全一致 **`97.5%`**（39/40，+7.5pp），`$0.104` / `2.16s`。

### 5.2 A2-2：引擎高费可打出性修复与反例闭环

- **修改文件**：
  - `advisor/deck_mechanics.py`：在 `_Engine.run` 中对 `cost > base_energy` 卡牌施加零参数马尔可夫上界 `afford = min(1.0, extra_energy / (cost - base_energy))`（`extra_energy <= 0` 时为 `0.0`）；不可支付部分不占用可打出牌的能量预算（`demand`、`kept_demand`、`x_value`、`energy_spent`），同时保持牌组抽牌稀释并以 `raw_demand` 计算 `_capabilities["energy"]`；
  - `tests/test_deck_fitness.py`：新增 `HighCostAffordabilityTests`（3 项单测：4 费 `BURY` 零额外能量不可打出且增加原始能量需求；`WISP` 部分额外能量与 `SOZU` 4 费基础能量逐级解锁；5 费 `METEOR_STRIKE` 零启动能量不自举 Plasma，加 `TURBO` 后解锁）；
  - `coach/eval/metrics/engine-a2-comparison.json`：完整记录修改前后在 79 题金样本、10 局 `dev`、15 局 `test` 与 29 条原人工评分上的逐条对比。
- **方案筛选与拒收记录（仅在 `golden_set` + `dev` 上验证）**：
  - 方案 `H1_H2_afford_plus_packing`（对 2+ 费牌加打包惩罚）：`golden_set 33/79`（−1）、`dev` 选牌 `56/135`（−3）→ **拒收**；
  - 方案 `H1_fullturn`（对 `cost >= base_energy` 加满费挤占惩罚）：`golden_set 34/79`、`dev` 选牌 `58/135`（−1）→ **拒收**；
  - 方案 `H1_afford_cap`（零参数超基础能量上限 + `raw_demand`）：`golden_set 34/79`（`obvious_errors_first = 0`）、`dev 132/221`（选牌 `59/135`，tier-1 `24` 条）→ **零退化，采纳**。
- **留出集与 29 条原人工评分复测结果**：
  - **9/9 条 `useful` 结论**：100% 保持 tier-1、推荐与分差零退化；
  - **15 局 `test` tier-1 标记**：`31 → 30`，唯一消除的 tier-1 误报是用户备注 `"四费不一定可以打得出来啊"` 的 `1773853598 F15-card`（`2:BURY 63.3 → 60.0`，玩家选择的 `0:DEFILE 60.8` 升至首选，`tier 1 → tier 0`）；
  - **已知未覆盖项如实保留**：其余 19/20 条非 `useful` 评分属于未建模的流派意图（大剑流/循环流）、跨幕后期价值（缓冲）或单回合手牌组合离散打包问题，不在无参数期望解析层强行拟合。

### 5.3 自动验证证据

- Coach TypeScript：`npm run typecheck` 通过，`npm run build` 通过，`npx vitest run test/verifier.test.ts test/notes.test.ts test/ratings.test.ts test/agent.test.ts test/chat.test.ts test/ask.test.ts test/critic.test.ts` **7 个测试文件 / 38 项测试全部通过**（`test/engine.test.ts` 因本机 `.venv-coach` Python 3.12 符号链接缺失未在本地运行，将由 GitHub Actions Ubuntu CI 与 Docker 冒烟共同覆盖）。
- Python 引擎与复盘：`unittest` 运行 `test_deck_fitness`、`test_campfire_decision`、`test_merchant_decision`、`test_card_reward_guard`、`test_card_reward_coverage`、`test_character_mechanics`、`test_review`、`test_fact_pipeline`、`test_golden_set` **269/269 全部通过**。
- 前置阶段 CI 验证：`main` 最新提交 `b199bbd`（含 A1 `TASK-030`）GitHub Actions run `45212225100` 全绿（`conclusion: success`）。

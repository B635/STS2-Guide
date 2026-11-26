# 实战战损统计校准与两阶段 CRN 离散回合微模拟（`deck_fitness_4`）

> 编写时间：2026-11-26  
> 适用分支：`feature/v0113`（游戏版本 `v0.113.1` / `sts2.dll` `62ecbe91…`）  
> 相关代码：
> - 战损校准：[advisor/combat_calibration.py](../advisor/combat_calibration.py)、[scripts/calibrate_combat_model.py](../scripts/calibrate_combat_model.py)、[data/combat_calibration.json](../data/combat_calibration.json)
> - 离散微模拟与策略：[advisor/turn_micro_sim.py](../advisor/turn_micro_sim.py)、[advisor/deck_fitness.py](../advisor/deck_fitness.py)、[advisor/policies.py](../advisor/policies.py)
> - 单测：[tests/test_combat_calibration.py](../tests/test_combat_calibration.py)、[tests/test_turn_micro_sim.py](../tests/test_turn_micro_sim.py)

---

## 1. 研究动机与问题定义

既有选牌引擎 `card_reward:deck_fitness_3`（见 `advisor/deck_mechanics.py` 与 `advisor/deck_fitness.py`）基于确定性**平均场（mean-field）**假设评估牌组在 6 个能力维度上的场均供需比。这套架构在速度、可解释性与无外部模型依赖上优势显著，但也存在两类结构性盲区：

1. **经验战损权重的真实对局校验缺失**：各幕 × 战斗类型（`Monster` / `Elite` / `Boss`）的 6 维能力缺口究竟对应多少真实 HP 损失（尤其是高危尾部战损 $Q_{0.75}[\Delta\text{HP}]$），此前缺乏真实战斗日志的交叉验证拟合。
2. **离散手牌与尾部风险（Tail Risk）平滑失真**：平均场只看“每回合平均能量与平均格挡”，难以刻画：
   - 离散 5 张抽牌与抽牌链下的手牌方差；
   - 整数能量背包中多张 `2–3` 费高费卡同手互卡（clumping）；
   - `Retain`（保留）跨回合留牌等待敌方高伤害峰值回合的防御价值；
   - 不可打出状态/诅咒牌占据抽牌位的手牌阻塞；
   - 敌方高伤爆发回合导致的尾部未格挡战损 $\text{CVaR}_{0.25}(\Delta\text{HP})$。

本工作在不改动 `data/guide_mechanics.json`、`packaging/compatibility.json` 与生产基线 `deck_fitness_3` 的前提下，独立构建了**单调约束实战战损校准模型**与**两阶段解析初筛 + 共同随机数（CRN）离散回合微模拟器**，并以 `card_reward:deck_fitness_4` 注册进策略体系完成严格对照评测。

---

## 2. 实战战损数据集与零泄漏单调约束校准

### 2.1 数据提取与严格零泄漏清洗

由 `scripts/calibrate_combat_model.py` 从 `build/collected-runs/`（125 个原始 `.run` 文件）与 `coach/eval/runs/dev/`（10 个开发对局）提取单人战斗样本。

> **关键防泄漏审计发现**：核对 `coach/eval/split.json` 时发现，`build/collected-runs/` 中恰好包含全部 **15 局 `test` 留出对局**的副本。校准脚本在扫描阶段即按 `split.json["test"]` 严格剔除全部 15 局，并在代码中设置路径断言拒绝读取任何 `runs/test/` 文件，确保 **`test_split_leakage = 0`**。

清洗流水线与真实样本统计（见 `data/combat_calibration.json`）：

| 清洗步骤 | 数量 | 说明 |
|---|---:|---|
| 原始扫描文件（`collected-runs` + `dev`） | 135 | 125 + 10 |
| 剔除 `split.json["test"]` 留出对局 | −15 | 保证 0 测试集泄漏 |
| 剔除跨目录重复对局（`(seed, start_time)` 去重） | −10 | `dev` 10 局已在 `collected-runs` 中存在 |
| 剔除多人联机对局（`len(players) != 1`） | −32 | 单人原版边界 |
| 剔除牌组重放不一致对局（`replay_run` 报错） | −5 | 未记录变牌/事件入牌的对局 |
| 保留有效单人对局 | **73** | 含 28 局完局 + 45 局含有效前置战斗的放弃局（其中 54 局含 ≥1 场完整战斗） |
| 剔除放弃局末层中途退出战斗 | −29 | 放弃局最终战斗 `hp_after=0` 导致 `damage_taken` 截断 |
| 剔除未转换遭遇事实战斗 | −12 | 缺程序集遭遇事实 |
| **最终有效单人战斗样本** | **558 场** | 覆盖 **54 局**、5 角色（`REGENT 147 / DEFECT 119 / NECROBINDER 118 / IRONCLAD 101 / SILENT 73`） |

按幕 × 战斗类型分布：
- **Act 1**：`MONSTER 247` / `ELITE 32` / `BOSS 30`（合计 309）
- **Act 2**：`MONSTER 137` / `ELITE 18` / `BOSS 19`（合计 174）
- **Act 3**：`MONSTER 64` / `ELITE 4` / `BOSS 7`（合计 75）

### 2.2 带物理单调约束的分层回归与分位数校准

对每场战斗，还原进房前真实牌组（含升级与附魔）和遗物，调用 `encounter_turns` 与 `deck_capabilities` 计算 6 维凹满足度特征 $z_d = g(s_d)$（与 `DeckFitness` 同构，$d \in \{\text{frontload}, \text{scaling}, \text{aoe}, \text{defense}, \text{consistency}, \text{energy}\}$，其中 `consistency` 对应引擎 `cycle`）及遭遇标称攻击总量 $a = \text{nominal\_turns} \times \text{incoming\_attack}$。

模型形式为：
$$\widehat{\Delta\text{HP}}_{s}(z, a) = \max\!\left(0,\; b_s + \gamma_s \cdot a - \sum_{d=1}^{6} w_{s,d}\, g(s_d)\right), \quad \text{s.t. } \gamma_s \ge 0,\; w_{s,d} \ge 0$$

- **物理单调性硬约束**：因 $g(s_d)$ 严格单调递增且 $w_{s,d} \ge 0$，提升任意维度能力**绝不会**预测更高掉血；上分位尾部模型强制 $\hat{q}_{0.75} = \max(\hat{y}_{\text{mean}}, \tilde{q}_{0.75})$，同时保持单调性与分位一致性。
- **消除辛普森悖论**：全局拟合引入分层固定效应截距（避免“第三幕牌组防御更高且 Boss 伤害更高”造成防御权重被压零），并通过中心化坐标下的 L2 正则与分层先验收缩（`global -> fight_kind -> (act, fight_kind)`），使小样本层（如 `act3:ELITE n=4`、`act3:BOSS n=7`）稳健借力不致过拟合。
- **按 `run_id` 分组 5 折交叉验证（`GroupKFold(n_splits=5, group=run_id)`）**：确保同一局对局的所有战斗严格落入同一折。

### 2.3 5 折 GroupKFold 袋外（OOF）校准指标与经验权重

| 切片 | 样本数 | OOF MAE | 朴素分层基线 MAE | OOF Pinball@0.75 | 朴素基线 Pinball@0.75 | OOF Spearman $\rho$ (Mean / Q75) | Q75 覆盖率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **总体（Overall）** | **558** | **7.654** | 7.680 | 3.500 | 3.489 | **0.513 / 0.518** | **74.2%** |
| `MONSTER` | 448 | **6.386** | 6.410 | **3.029** | 3.037 | 0.276 / 0.281 | 74.6% |
| `ELITE` | 54 | **11.901** | 11.907 | 5.156 | 5.049 | 0.092 / 0.030 | 74.1% |
| `BOSS` | 56 | **13.701** | 13.768 | 5.673 | 5.605 | 0.220 / 0.224 | 71.4% |
| `Act 1` | 309 | **6.153** | 6.215 | **3.011** | 3.028 | 0.445 / 0.445 | 74.1% |
| `Act 2` | 174 | **9.097** | 9.106 | 3.898 | 3.866 | 0.421 / 0.432 | 74.7% |
| `Act 3` | 75 | 10.492 | 10.407 | 4.589 | 4.518 | 0.080 / 0.060 | 73.3% |

典型分层经验减伤权重 $w_{s,d}$（单位：每单位满足度增益节省的 HP）：

| 分层 | $n$ | `frontload` | `scaling` | `aoe` | `defense` | `consistency` | `energy` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `act1:MONSTER` | 247 | 0.357 | 0.000 | 0.598 | 0.192 | 0.019 | 0.000 |
| `act1:ELITE` | 32 | 0.144 | 0.002 | 0.236 | 0.111 | 0.179 | 0.004 |
| `act1:BOSS` | 30 | 0.110 | 0.002 | 0.464 | 0.094 | 0.024 | 0.000 |
| `act2:MONSTER` | 137 | 0.345 | 0.011 | 0.449 | 0.230 | 0.166 | 0.052 |
| `act2:ELITE` | 18 | 0.155 | 0.002 | 0.225 | 0.129 | 0.195 | 0.012 |
| `act2:BOSS` | 19 | 0.118 | 0.001 | 0.442 | 0.104 | 0.087 | 0.005 |
| `act3:MONSTER` | 64 | 0.336 | 0.021 | 0.455 | 0.181 | 0.012 | 0.000 |
| `act3:ELITE` | 4 | 0.141 | 0.001 | 0.239 | 0.120 | 0.164 | 0.007 |
| `act3:BOSS` | 7 | 0.144 | 0.000 | 0.470 | 0.094 | 0.075 | 0.000 |

**关键经验发现**：
- 在大样本走廊战（`MONSTER`, $n=448$）与第一幕（$n=309$）上，单调模型在 **OOF MAE 与 Pinball@0.75 双指标均优于朴素分层基线**，总体 `Q75` 覆盖率 **74.2%** 贴近理论 75%。
- 实战数据印证了爬塔直觉：走廊战减伤主要由 `frontload` + `aoe` + `defense` 驱动；精英战 `consistency` 权重明显抬升；Boss 战对多体/召唤物 `aoe` 与持续输出更敏感；而第三幕（$n=75$）与精英（$n=54$）样本仍偏少且受药水与特定机制影响大，单靠牌组特征的额外 MAE 增益有限。

---

## 3. 两阶段解析初筛 + CRN 离散回合微模拟（`deck_fitness_4`）

### 3.1 架构与正交化设计

`DeckFitnessV4Policy`（版本标识 `card_reward:deck_fitness_4`，注册于 `advisor/deck_fitness.py` 与 `advisor/policies.py`）采用严格两阶段结构：

1. **第一阶段（解析初筛）**：完整运行 `deck_fitness_3` 管道（含 6 维平均场能力、前瞻选牌采样与按幕校准的 `pick_margin`）。若第一名领先次优项超过 `close_gap = 4.5` 分，直接跳过第二阶段（零额外耗时）。
2. **第二阶段（前排胶着候选 CRN 微模拟 + 战损校准）**：对距第一名分差 `<= 4.5` 分的前排候选（最多 Top-3 张牌 + `skip`）执行：
   - **耦合秩键 CRN（`K = 24`）**：`advisor/turn_micro_sim.py` 为同一决策生成固定种子的共享敌方攻防序列与共享基础牌组洗牌键（`base_keys[k][cycle][slot]`）+ 独立候选牌插槽键（`cand_keys[k][cycle]`），确保加入 1 张新牌后基础牌组相对抽牌顺序完全不变，显著压低蒙特卡洛比较方差。
   - **离散回合执行**：模拟每回合抽 5 张（含 `Innate` 首回合固有、抽牌卡链式补牌、上限 10 张手牌、洗牌循环）、整数能量背包贪心出牌（优先 0 费抽牌/回能与斩杀、动态评估 8 类 `cost_rule`、`X` 费、`unplayable` 状态/诅咒卡手）、**`Retain` 跨回合留牌**（在 0 伤/已满防回合主动留住纯防御保留牌，等待敌方爆发回合打出）以及每回合未格挡掉血 $\Delta\text{HP}_t = \max(0, \text{incoming}_t - \text{block}_t)$，汇总均值战损 $\mathbb{E}[\Delta\text{HP}]$、最坏 25% 尾部风险 $\text{CVaR}_{0.25}(\Delta\text{HP})$ 与高费卡手率 `clump_rate`。
   - **正交增量合成（避免重复计分与侵蚀 `pick_margin`）**：
     - 在早期原型中，若直接把单场微模拟的绝对减伤加到卡牌候选上而 `skip` 保持 0，会系统性抵消已校准的 `pick_margin`，导致多个应跳过的局面被误推选牌（金样本一度由 33 降至 29）。
     - 最终冻结版改为仅注入**与第一阶段正交的三项信号**：
       1. 竞争卡牌间的**零和经验战损校准差**（`calibration_scale = 0.50`，保留 `skip` 门槛）；
       2. 离散执行**下行风险护栏**（`downside_scale = 0.12` 对 $\min(0, \Delta\text{risk}_{\text{sim}})$ 扣分，`clump_penalty = 0.40` 对高费卡手率增量扣分；对含 `profile_gaps` 的部分建模牌不施加负向微模拟扣分，防止误杀）；
       3. **`Retain` 峰值留牌尾部方差缩减奖励**（`retain_tail_scale = 0.08`，仅在 `retain=True` 且实际触发 `retain_saves > 0` 时奖励 $\Delta(\text{CVaR}_{0.25} - \mathbb{E}[\Delta\text{HP}])$）。
     - 总调整幅度截断在 `[-2.0, +2.0]` 分，并在推荐因子中通过 `deck_fitness:combat_calibration` 与 `deck_fitness:turn_micro_sim` 透明披露中文解释。

---

## 4. 对照评测结果（金样本 + Dev + 一次性只读 Blind Test）

所有参数在 `golden_set`（79 题）与 `dev`（10 局 / 135 题）定型冻结后，仅对留出集 `coach/eval/runs/test/`（15 局 / 174 题）执行**唯一一次只读盲测**，未在查看测试集后做任何参数回调。

### 4.1 核心指标总表

| 评测集 | 题数 | Stage-2 触发 | `deck_fitness_3` 命中 | `deck_fitness_4` 命中 | 净变化 | 明显错误排第一 (`v3`/`v4`) | McNemar 精确 $p$ | 配对 Bootstrap 95% CI | `v3` CPU 延迟 (P50 / P95 / Max) | `v4` CPU 延迟 (P50 / P95 / Max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Golden Set** | 79 | 76/79 | 33/79 (**41.77%**) | 33/79 (**41.77%**) | +0 (0.00%) | **0 / 0** | 1.000 | `[0.00%, 0.00%]` | 93.8 / 187.5 / 234.4 ms | 125.0 / **250.0** / 281.2 ms |
| **Dev (10 局)** | 135 | 132/135 | 59/135 (**43.70%**) | 57/135 (**42.22%**) | −2 (−1.48%) | **0 / 0** | 0.625 | `[-4.44%, +1.48%]` | 109.4 / 296.9 / 484.4 ms | 140.6 / **234.4** / 312.5 ms |
| **Blind Test (15 局)** | 174 | 168/174 | 78/174 (**44.83%**) | 78/174 (**44.83%**) | **+0 (0.00%)** | **0 / 0** | 1.000 | `[-2.87%, +2.87%]` | 93.8 / 203.1 / 437.5 ms | 109.4 / **187.5** / 265.6 ms |
| **三集合计** | **388** | 376/388 | 170/388 (43.81%) | 168/388 (43.30%) | −2 (−0.52%) | **0 / 0** | — | — | — | **全集 P95 ≤ 250 ms** |

> 注：以上延迟均为预热后（`warm_up()`）单次 `recommend()` CPU 时间；在 388 次真实评测调用中，`deck_fitness_4` 超过 `300 ms` 的次数仅 **1/388**，在 Golden Set（`P95 = 250.0 ms`）、Dev（`P95 = 234.4 ms`）与 Blind Test（`P95 = 187.5 ms`）上全部满足实时延迟预算。

### 4.2 典型修正案例（Dev 与 Blind Test）

1. **`history-1791823702-f6`（Blind Test，亡灵契约师，`best = GRAVE_WARDEN`）**：
   - `v3` 给 `REAVE` `58.0` 分、略高于 `GRAVE_WARDEN` `57.6` 分；
   - `v4` CRN 微模拟发现 `REAVE` 在离散回合中使场均战损上升 `+2.95 HP`、$\text{CVaR}_{0.25}$ 恶化 `+5.90 HP`（`sim_delta = -0.64`），而 `GRAVE_WARDEN` 反而减伤 `+9.12 HP` / $\text{CVaR}_{0.25}$ 改善 `+9.23 HP`；最终 `v4` 将首选修正为 **`GRAVE_WARDEN`（57.6 vs 57.3）**，命中真实玩家选择。
2. **`history-1774030566-f17`（Dev，静默猎手，`best = skip`）**：
   - `v3` 给 `ECHOING_SLASH` `57.4` 分（仅高于 `skip` `56.8` 分）；
   - `v4` CRN 微模拟捕捉到该牌离散执行下场均战损恶化 `+10.25 HP`、尾部 $\text{CVaR}_{0.25}$ 恶化 `+17.70 HP`（`sim_delta = -1.90`），将其修正至 `55.6` 分并正确推荐 **`skip`**。
3. **`history-1772819860-f5`（Blind Test，储君，`best = skip`）与 `history-1774202131-f8`（Blind Test，铁甲战士，`best = skip`）**：
   - `v3` 均以 `+0.1` 分极微弱优势推荐了 `ASTRAL_PULSE` 与自损血攻击牌 `BREAKTHROUGH`；
   - `v4` 微模拟识别出两张牌分别使 $\text{CVaR}_{0.25}$ 恶化 `+4.68 HP` 与 `+3.97 HP`，均果断翻回 **`skip`**，命中真实选择。

### 4.3 诚实边界与误差拆解

本次盲测与开发集同样暴露了 3 个值得后续重点改进的真实边界，如实记录而不掩饰：

1. **轻量微模拟器低估纯能量/过牌辅助牌的跨卡协同**：在 Blind Test 的两例 `WISP`（`history-1773853598-f14`、`history-1791474137-f11`）上，由于孤立微模拟中能量溢出时 0 费消耗回能牌仅体现为“占 1 次抽牌位”，`sim_mean_hp_saved` 出现 `-2.2 ~ -2.9 HP` 负值，将原本高出 `skip` 0.1–0.6 分的 `WISP` 压回 `skip`（构成盲测 3 例负翻转中的 2 例）。
2. **`partly_scored` 保护的非对称效应**：在 Blind Test 的 `history-1773037084-f9`（`POMMEL_STRIKE 47.9` vs `DRUM_OF_BATTLE 47.8`）中，因 `DRUM_OF_BATTLE` 带有未建模缺口被护栏跳过负向微模拟扣分（`sim_delta = 0`），而完整建模的 `POMMEL_STRIKE` 受到 `-0.43` 扣分，导致次优的未完整建模牌反超。
3. **高费爆发牌的 `clump_penalty` 未按正收益抵消**：在 Dev 的 `history-1772816067-f2`（`BLUDGEON 42.1` vs `skip 42.0`）中，3 费 `BLUDGEON` 在微模拟里实际节省了高达 `+24.5 HP` 均值战损与 `+29.4 HP` 尾部战损，却因为 `clump_rate = 0.199` 单独被扣 `-0.08` 分落到与 `skip` 同分 `42.0` 并在平分决胜输给 `skip`。后续若对正收益显著高于卡手代价的高费卡放宽 `clump_penalty`，即可消除该边界误杀。
4. **策略上线决策**：鉴于三集合计 Top-1 一致率与 `deck_fitness_3` 处于统计等价区间（`388` 题净差 `-2`，所有子集 McNemar $p \ge 0.625$），本轮**保持 `card_reward:deck_fitness_3` 为兼容清单与生产默认推荐器**，同时将 `card_reward:deck_fitness_4` 完整注册于 `advisor/deck_fitness.py` 与 `advisor/policies.py` 供对比评测与下一轮迭代使用。

---

## 5. 自动化验证证据

- `tests/test_combat_calibration.py`：**5/5 OK**（验证 `GroupKFold` 严格零跨折泄漏、对抗性反相关数据下非负权重硬约束、Pinball/Spearman 数学、`data/combat_calibration.json` 零测试泄漏与全 9 分层 6 维严格单调非增性）。
- `tests/test_turn_micro_sim.py`：**6/6 OK**（验证 CRN 确定性与零方差基线、`Retain` 峰值留牌降低 $\text{CVaR}_{0.25}$、高费互卡 `clump_rate` 上升、不可打状态牌阻塞、动态 `cost_rule` + `X` 费执行、`deck_fitness_4` 注册与延迟门禁）。
- `tests/test_deck_fitness.py` + `tests/test_golden_set.py`：**167/167 OK**（合计核心套件 **178/178 OK**，既有 `deck_fitness_3` 行为零破坏）。

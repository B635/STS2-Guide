# TASK-029：转到游戏 v0.113.1（public-beta，新角色愚者）

- 状态：进行中（第 1–4 步及第 5/6 步源码与自动验证完成，含 Wave 1A 能量/费用边界修复；尚未导出 `.pck`/打包、未安装、未真机）
- 决定：2026-11-15 用户决定数据与 Mod 以 public-beta v0.113.1 为准（主分支仍是 0.111.0，朋友继续用 0.3.15）。
- 分支：`feature/v0113`（从 `feature/review-coach` 开出）。目录、程序集、59 条新增事实、愚者共用引擎机制、0.3.16 兼容清单与自动回归均已完成；尚待沙箱外 `.pck`/安装包构建、安装及真机验收。

## 版本差异（提取器 `bin/next`，程序集 62ecbe91…）

| 类别 | 总数 | 新增（未审） | 代码有变动（stale） |
|---|---|---|---|
| 卡牌 | 716 | 121 | 69 |
| 能力 | 304 | 41 新增（未审合计 126，含此前未审的） | 21 |
| 遗物 | 315 | 17 | 79 |
| 怪物 | 117 | 2 | 37 |
| 遭遇 | 90 | 0 | 12 |
| 附魔 | 22 | 0 | 5 |
| 充能球 | 5 | 0 | 1 |

`stale` 是类的代码指纹变了，可能只是内部改写；逐条对照 IL 后才能恢复为 `converted`。清单见 `build/assembly-coverage.json`（`scripts/refresh_assembly_facts.py` 生成）。

## 已完成

1. 用新程序集重新提取（`data/assembly/sts2-extract.json`）；目录改取 Spire Codex beta 频道（`data/knowledge.json`，0.113.1，含 FOOL）；
   `refresh_assembly_facts.py` 把变动的事实标为 `stale`。

## 迁移工作清单（编号保留，当前执行顺序见下）

2. 复核 stale 事实（卡牌 69、遗物 79、怪物 37、遭遇 12、能力 21、附魔 5、充能球 1）：起草 + 独立复核。
3. 新增内容的事实：愚者卡牌与能力、17 个新遗物、2 个新怪物；引擎需要的新机制（愚者的资源／关键词）先盘点。
4. 非古神遗物的局外事实（TASK-027 票 06b）：107 个草稿按 0.111.0 写成，复核结果 d 35/36、e 36/36、f 34/35 同意
   （不同意：THE_COURIER 删牌不补货、LIZARD_TAIL 战斗外也挡致死伤害；HAPPY_FLOWER／POLLINOUS_CORE／PENDULUM 的计数跨战斗不清零）。
   其中 36 个在新版有变动，随第 2 步一起对照新 IL 重看，其余 71 个可直接合并。批次 e 的复核是在程序集更新后读的 IL，只作线索。
5. 引擎、测试、金样本（golden set）与历史检验在新数据上重跑；休息处提前一层等地图规则变化。
6. Mod 对新版游戏重新编译，加入愚者；兼容清单、发行识别、版本号；真机验收。
7. 之后回到原计划：药水、事件、选牌判断（流派意图等，见 `docs/review-coach.md` 的人工评分）。

### 当前执行顺序（2026-11-23）

- 先做无安装的编译和全量基线。引擎与 Mod 是两条可独立验证的链路，不能等事实填完才第一次编译，也不能提前更新发行清单来消除测试失败。
- 下一片从愚者基础牌组能实际抓到的射击牌入手。优先补关联能力与箭、扎穿、瞄准的评分；用同一候选比较有箭／无箭及有关联能力／无关联能力的局面。确认已知效果能改变得分、未知效果不会白送收益、跳过仍可推荐。沿用生产 SQLite 与 `card_profile`，不另建职业评分器。然后补拉伸、旋律及剩余依赖。126 个待审能力含旧版积压，不用它们全部完成来阻塞第一个评分切片。
- 同步分类现有回归。目录与程序集旧期望要重核；旧角色多段伤害、Buffer、路线与篝火规则必须按实际行为验证；清单与发行包的身份断言保留到组包时一起验证。金样本按候选排序、差值方向和缺口解释验收，不能只看事实数量。
- 最后才切换候选包兼容清单、安装及真机验证。商店、事件、复盘 Coach 不进入本次适配的关键路径。

## 进度

- 2026-11-17 第 2 步完成：代码有变动的 191 条事实（卡牌 69、遗物 79、怪物 37、遭遇 12、能力 21、附魔 5、充能球 1）全部按新 IL 重审并独立复核后合并，`stale` 归零。
  起草每批 2 个代理、复核另派；复核不同意 1 条（OMNISLICE：变化在于溅射算作有力攻击，伤害上限并非新增），开发者照复核裁决。
  有改动的：卡牌 13（Expect a Fight 重做、Memento Mori 重做、Expertise 保留整手、Forgotten Ritual 有条件、Enlightenment 不再消耗、Dowsing 4 个问号房、Phantom Blades 8、Speedster 3、Minion Dive Bomb 11、Salvo 10、Omnislice 溅射算有力攻击，另有 Byrdonis Egg 与 3 张极速模式排除）；
  遗物 9（Jeweled Mask 整场免费、Ringing Triangle 给保留手牌、Juzu Bracelet 也去掉精英（正常游戏无实际影响）、Pael's Eye 额外回合先结算、Archaic Tooth／Touch of Orobas 加愚者对照、Brilliant Scarf 第 6 张、Byrdpip 进化、Power Cell 先取非固有）；
  怪物 8（Infested Prism 循环与脉动、Thieving Hopper 不偷任务牌、Tough Egg 保留穿刺、Axebot 固定生命、Tunneler 24、Ceremonial Beast 力量有条件、Rocket 改名、Kin Follower 内部改写）；能力 8（Buffer、Draw Cards Next Turn、Imitation Learning、Juggling、Nightmare、One For All、Phantom Blades、Thorns）。遭遇、附魔、充能球无行为变化。
  恐惧（GameMode 3）与极速（GameMode 4）模式专属的代码一律不进事实，只在证据里注明；这两个模式不在支持范围。
- 同时并入（第 4 步）：非古神遗物的拾起与局外事实；复核裁决：THE_COURIER 删牌不补货，LIZARD_TAIL 战斗外也挡致死伤害（词汇新增 `lethal_damage`，估值暂报缺口），HAPPY_FLOWER／POLLINOUS_CORE／PENDULUM 计数跨战斗（`counter: run`，引擎仍按每场重数，长期触发率相同）。
- 2026-11-23 第 3 步卡牌部分完成。121 条新增事实经独立复核，104 条直接同意，17 条分歧已按 IL 裁决；当前程序集下卡牌 716/716 已审，无 stale。仍待审能力 126（含此前未审）、遗物 17、怪物 2。
- 生产评分路径抽查：临时构建新版 SQLite 后运行 `scripts/report_engine_coverage.py`，愚者颜色的 90 条卡牌记录中，仅 13 条在基础和升级档都没有报告缺口（含目录中的多人牌，不是单人池的准确率）。`EPIC_RETRIEVAL` 只计算直接攻击，松脱回能报缺；`FLORESCENCE` 的生成牌增伤、`TRICK_SHOT` 的射击和有条件回手均报缺。不能据已审数量宣称角色可用。
- 本机 0.113.1 的 `SmithRestSiteOption.OnSelect` 将参数 1 传给 `CardSelectorPrefs(UpgradeSelectionPrompt, int)`，该构造器用同一个整数设置选择上下限，再由 `FromDeckForUpgrade` 读取。替换 0.111.0 的 `SmithCount` 并对本机程序集验证调用点；Mod 编译 0 警告 0 错误，未安装，未启动游戏。
- 自动回归从 839 项（16 失败、1 个类初始化错误）到 844 项（16 失败，无错误）；逐名比较，剩余失败均在改动前基线中。新增 3 项事实合并测试及恢复运行的 2 项 Mod 行为测试都通过。下一步单列兼容清单、旧目录断言和真实评分行为回归，不将这 16 项统一视为可更新的旧测试。
- 2026-11-25（`0ff01bb`）：第 3 步新增事实（17 件遗物、40 个 v0.113.1 新增/愚者能力、2 个宠物怪物，共 59 条）完成独立审查并合入 `data/assembly/facts.json`；当前卡牌 716/716、遗物 315/315、怪物 117/117、遭遇 89+1、附魔 22/22、充能球 5/5、能力 218/304 converted（其余 86 条能力为旧版历史积压），全部 0 stale。愚者共用评分内核接入箭槽守恒 `arrow_avail`、扎穿概率 `impaled_prob`、瞄准 `Aim`、拉伸 `Stretch`、旋律 `Wah/Cho/Ki` 与歌弓 `SONG_BOW`，并修复独立审查发现的 16 处事实/数学/回退边界问题；生产覆盖率 `ALL 703/1271 (55%)`、`FOOL 54/90 (60%)`、`RELIC 223/315 (71%)`。兼容清单与协议示例同步至 `0.3.16-alpha.0` / Mod `0.3.16` / 游戏 `0.113.1`（发行指纹 `a3c6d6b9…`）；Mod C# DLL 0 错误 0 警告、`GuideBehaviorTests` 通过、全量非阻塞 Python 套件 `775/775 OK`（未打包 `.pck`/安装包、未安装、未真机）。
- 2026-11-26（Wave 1A）：完成 `advisor/deck_mechanics.py` 四项能量与费用边界修复：
  1. 跟踪单回合峰值额外能量 `max_turn_extra`（按固定回合 `fixed_turn_energy` 区分不同回合、对局内升级 `upgraded_profile` 去重，并覆盖卡牌/遗物/能力事件/Plasma 充能球）及负能量赤字 `max_turn_loss`，阻止单张 `+1` 能量源（`WISP`/`HAPPY_FLOWER`/`LANTERN`/未升级 `DOUBLE_ENERGY`）解锁 5 费 `METEOR_STRIKE` 或自举 Plasma；
  2. 为全部 8 类 `cost_rule` 引入离散 `(probability, state_cost)` 分布评估可打出度，消除 `KINGLY_KICK`、`CODA`、`MODDED`/`FRANTIC_ESCAPE` 及 `per_type_this_turn` 降费卡在 `E[cost] > base_energy` 处的 Jensen 断崖；
  3. 用与牌组尺寸无关且去重 `upgraded_profile` 的 `brick_demand` 替换 `raw_demand`，防止死卡稀释使无关 0 费卡虚增 `capabilities["energy"]`，同时保持加入不可打出高费卡必拉低 `energy`；
  4. 无非 X 基础效果的 X 费卡在 `x_value <= 0` 时 `plays = 0`（防止免费触发 `SHURIKEN`/`ORNAMENTAL_FAN` 等），保留 `TEMPEST+`/`MALAISE+` 等 `X=0` 有效卡，并将 X 费消耗计入 `self.energy_spent`（`HELIX_DRILL` 生效）。
  新增 5 项边界回归单测后，`test_deck_fitness` / `test_golden_set` / `test_campfire_decision` / `test_merchant_decision` / `test_character_mechanics` **235/235 OK**，`test_neow_decision` / `test_event_decision` / `test_assembly_facts` **28/28 OK**；79 题金样本 `33/79 (41.77%)`、`obvious_errors_first = 0`（无回退）。
- 距离 v0.113.1 可安装真机验收剩余步骤：① 合并其余 Wave 改动后在沙箱外运行完整 `scripts/build_public_beta.ps1`（含 Godot `.pck` 导出、冻结 EXE、安装包及 payload 审计）；② 关闭游戏后安装并核对 Mod 三件套与发行指纹；③ 完成含愚者第一幕及常规角色在内的真机回归验收。

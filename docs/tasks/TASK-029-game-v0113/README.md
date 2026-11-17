# TASK-029：转到游戏 v0.113.1（public-beta，新角色愚者）

- 状态：进行中（第 1、2 步完成；第 4 步的遗物局外事实已并入）
- 决定：2026-11-15 用户决定数据与 Mod 以 public-beta v0.113.1 为准（主分支仍是 0.111.0，朋友继续用 0.3.15）。
- 分支：`feature/v0113`（从 `feature/review-coach` 开出）。**本分支目前测试不保证通过**：目录与提取结果已换成新版，事实尚未复核。

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

## 待做（按顺序）

2. 复核 stale 事实（卡牌 69、遗物 79、怪物 37、遭遇 12、能力 21、附魔 5、充能球 1）：起草 + 独立复核。
3. 新增内容的事实：愚者卡牌与能力、17 个新遗物、2 个新怪物；引擎需要的新机制（愚者的资源／关键词）先盘点。
4. 非古神遗物的局外事实（TASK-027 票 06b）：107 个草稿按 0.111.0 写成，复核结果 d 35/36、e 36/36、f 34/35 同意
   （不同意：THE_COURIER 删牌不补货、LIZARD_TAIL 战斗外也挡致死伤害；HAPPY_FLOWER／POLLINOUS_CORE／PENDULUM 的计数跨战斗不清零）。
   其中 36 个在新版有变动，随第 2 步一起对照新 IL 重看，其余 71 个可直接合并。批次 e 的复核是在程序集更新后读的 IL，只作线索。
5. 引擎、测试、金样本（golden set）与历史检验在新数据上重跑；休息处提前一层等地图规则变化。
6. Mod 对新版游戏重新编译，加入愚者；兼容清单、发行识别、版本号；真机验收。
7. 之后回到原计划：药水、事件、选牌判断（流派意图等，见 `docs/review-coach.md` 的人工评分）。

## 进度

- 2026-11-17 第 2 步完成：代码有变动的 191 条事实（卡牌 69、遗物 79、怪物 37、遭遇 12、能力 21、附魔 5、充能球 1）全部按新 IL 重审并独立复核后合并，`stale` 归零。
  起草每批 2 个代理、复核另派；复核不同意 1 条（OMNISLICE：变化在于溅射算作有力攻击，伤害上限并非新增），开发者照复核裁决。
  有改动的：卡牌 13（Expect a Fight 重做、Memento Mori 重做、Expertise 保留整手、Forgotten Ritual 有条件、Enlightenment 不再消耗、Dowsing 4 个问号房、Phantom Blades 8、Speedster 3、Minion Dive Bomb 11、Salvo 10、Omnislice 溅射算有力攻击，另有 Byrdonis Egg 与 3 张极速模式排除）；
  遗物 9（Jeweled Mask 整场免费、Ringing Triangle 给保留手牌、Juzu Bracelet 也去掉精英（正常游戏无实际影响）、Pael's Eye 额外回合先结算、Archaic Tooth／Touch of Orobas 加愚者对照、Brilliant Scarf 第 6 张、Byrdpip 进化、Power Cell 先取非固有）；
  怪物 8（Infested Prism 循环与脉动、Thieving Hopper 不偷任务牌、Tough Egg 保留穿刺、Axebot 固定生命、Tunneler 24、Ceremonial Beast 力量有条件、Rocket 改名、Kin Follower 内部改写）；能力 8（Buffer、Draw Cards Next Turn、Imitation Learning、Juggling、Nightmare、One For All、Phantom Blades、Thorns）。遭遇、附魔、充能球无行为变化。
  恐惧（GameMode 3）与极速（GameMode 4）模式专属的代码一律不进事实，只在证据里注明；这两个模式不在支持范围。
- 同时并入（第 4 步）：非古神遗物的拾起与局外事实；复核裁决：THE_COURIER 删牌不补货，LIZARD_TAIL 战斗外也挡致死伤害（词汇新增 `lethal_damage`，估值暂报缺口），HAPPY_FLOWER／POLLINOUS_CORE／PENDULUM 计数跨战斗（`counter: run`，引擎仍按每场重数，长期触发率相同）。
- 剩余未审：新增卡牌 121、能力 126（含此前未审）、遗物 17、怪物 2。

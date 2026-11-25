# 程序集机制事实：审核规范

适用于 TASK-025 票 04–10。所有卡牌、遗物、能力的评分事实都按本规范写入
`data/assembly/facts.json`，只以当前游戏程序集为依据（产品规格 2.8、ADR 0001）。

## 工作流

1. 看初分与钩子：`data/assembly/facts.json` 中该条目的 `hooks`、`hook_tokens`、遗物的 `triage`。
2. 读代码：`tools/Sts2FactExtractor/bin/Release/net9.0/Sts2FactExtractor.exe <sts2.dll> --dump <类名>`
   输出该类及其状态机、匿名方法的完整 IL；异步方法的真实逻辑在 `<方法名>d__N::MoveNext`。
   数值来自 `get_CanonicalVars` 与 `OnUpgrade`（提取结果中的 `canonical_vars` / `upgrade_deltas` 已解码常见写法）。
   卡牌费用是 `CardModel::.ctor` 的第一个整数参数。
3. 写事实：按下文结构填写，`evidence` 写明方法名与关键调用；Spire Codex 目录只用来交叉核对，冲突时以程序集为准并在 `evidence` 注明。
4. 状态：能完整描述的写 `converted`；只有选牌时**真正无法得知**的部分（例如取决于敌人下一步意图）才写入 `unknowable`，
   并在 `reason` 说明为什么无法得知。能用牌组、路线、已知分布估算的都不属于无法得知。
5. 记下审核依据的指纹：`fingerprint` 保持为提取结果中的当前值，`reviewed_by` 写票号。
6. 确认（2026-10-02 用户决定）：每条起草的事实由另一个独立 Agent 对照 IL 逐条复核，结论写入 `audit`
   （`{"by": ..., "verdict": "agree"|"resolved", "notes": ...}`）；分歧由开发者依据 IL 裁决并写明。用户在真机中抽查。
7. `scripts/merge_fact_reviews.py --check <batch>` 可用于未审草稿；正式合并要求审查结论、审核人和说明齐全。每条输入都是完整复核记录而非字段补丁，不能依赖上一个版本的可选字段。合并输出只提示顶层机制；用 `scripts/report_engine_coverage.py` 检查生产评分路径的实际缺口。

## 公共结构

```json
{
  "status": "converted",
  "category": "baseline",          // 仅遗物：baseline | trigger | irrelevant
  "cost": {"base": 1, "upgraded": 0},          // 仅卡牌；X 费写 "X"，不能打出写 null
  "keywords": ["Exhaust"],                      // 仅卡牌：Exhaust / Ethereal / Innate / Retain / Unplayable
  "effects": [ EFFECT, ... ],                   // 卡牌：每次打出的效果，按执行顺序
  "mechanics": [ MECHANIC, ... ],               // 遗物与能力：触发式或静态效果
  "unknowable": [{"what": "...", "reason": "..."}],
  "evidence": ["..."],
  "reviewed_by": "TASK-025 ticket NN"
}
```

## EFFECT（一次性效果）

| kind | 字段 | 说明 |
|---|---|---|
| `damage` | `amount`, `upgrade`, `hits`, `target` | `target`: `single` / `all` / `random` |
| `block` | `amount`, `upgrade` | 自己获得格挡 |
| `draw` | `amount`, `upgrade` | 抽牌 |
| `energy` | `amount`, `upgrade` | 本回合获得能量 |
| `apply_power` | `power`, `amount`, `upgrade`, `target` | `target`: `self` / `enemy` / `all_enemies`；`power` 用能力类名去掉 `Power`（`Strength`、`Vulnerable`…） |
| `hp_loss` | `amount`, `upgrade` | 自己失去生命 |
| `heal` | `amount`, `upgrade` | 自己回复生命 |
| `exhaust_self` | — | 打出后耗竭自身（也可写在 `keywords`） |
| `exhaust_other` | `count`, `choice` | 耗竭手牌中其他牌；`choice`: `random` / `choose` / `all` |
| `discard` | `count`, `choice` | 丢弃手牌 |
| `add_card` | `card` 或 `pool`, `count`, `destination`, `free` | 生成牌；`destination`: `hand` / `draw` / `discard`；`pool` 写奖励池名称 |
| `upgrade_cards` | `scope`, `count` | `scope`: `hand` / `deck`（本场战斗） |
| `channel_orb` / `summon` / `forge` / `gain_stars` | `orb` / `amount` | 角色资源 |

`amount` 是基础值，`upgrade` 是升级后的**增量**（没有写 0）。多段伤害写 `hits`。
有条件的效果在该 EFFECT 上加 `condition`（见下）。

## MECHANIC（触发式或静态效果）

触发式：

```json
{"when": "card_played", "filter": {"card_type": "Attack"}, "every": 3, "counter": "turn",
 "limit": null, "condition": null, "effect": EFFECT}
```

- `when`：`combat_start`、`turn_start`、`turn_end`、`card_played`、`card_exhausted`、`card_discarded`、`shuffle`、
  `hand_emptied`、`damage_received`、`enemy_died`、`potion_used`、`stars_spent`、`orb_channeled`、`card_generated`、`block_broken`。
- `turns`：只在指定回合（如 `[2]`）；`every_turns`：每 N 回合一次。
- `filter`：`card_type`（`Attack`/`Skill`/`Power`/`Status`/`Curse`）、`min_cost`、`tag`（`Strike`/`Minion`/`Shiv`…）、
  `upgraded`、`enchanted`、`x_cost`、`fight_kind`（`MONSTER`/`ELITE`/`BOSS`）。
- `every`：每 N 次满足条件的事件触发一次；`counter`：计数在 `turn` 或 `combat` 内累计。
- `limit`：`once_per_turn` / `once_per_combat`。
- `condition`：`{"no_block": true}`、`{"per_card_in_hand": true}`、`{"hand_empty": true}`、`{"block_at_least": n}`、
  `{"hp_ratio_at_most": r}`、`{"no_card_type_played_this_turn": "Attack"}`、`{"cards_played_this_turn_at_most": n}`、`{"no_potions": true}`。

静态修正：

```json
{"modifier": "hand_draw", "add": 1, "turns": [1]}
```

- `modifier`：`hand_draw`、`max_energy`、`damage`（`filter` + `add`/`multiply`）、`block`（`filter` + `multiply`，可有 `limit`）、
  `energy_cost`（`filter` + `add`）、`x_value`（`add`）、`keep_block`（`max`）、`retain_hand`、`no_draw_during_turn`、
  `cards_per_turn_limit`（`max`）、`hp_loss_per_turn_cap`（`max`）、`hp_loss_reduce`（`amount`）、`card_added_to_deck`（`filter` + `upgrade`/`enchant`）。

## 遗物类别

- **irrelevant**：只有局外钩子（拾起、进房间、战斗结束、商店、篝火、地图、奖励数量），`mechanics` 为空。
  拾起时的一次性改动（升级、加牌、移除）已经体现在当前牌组状态里，不再计入。
- **baseline**：不依赖出什么牌、直接改变牌组能力（战斗/回合开始的格挡、能量、抽牌，静态修正）。
- **trigger**：由特定卡牌或出牌动作触发、让相关卡牌更有价值。
- 修改卡牌奖励或入牌组时改动卡牌的遗物（如冰冻蛋）写 `modifier: card_added_to_deck`，模型据此调整候选牌本身。

## 能力

卡牌施加的能力若不是 Strength / Dexterity / Vulnerable / Weak / Poison 等通用能力，需要为该能力类单独写事实
（`entity_key` 为 `powers:<ID>`，结构同遗物的 `mechanics`），卡牌的 `apply_power` 引用它。

## 已核对的枚举值（程序集 0861bfa1…）

用 `Sts2FactExtractor.exe <sts2.dll> --enum <枚举名>` 读取，不要从描述文字推断。

| 枚举 | 值 |
|---|---|
| `CardType` | 1 Attack，2 Skill，3 Power，4 Status，5 Curse，6 Quest |
| `PileType` | 1 Draw，2 Hand，3 Discard，4 Exhaust，5 Play，6 Deck |
| `CardKeyword` | 1 Exhaust，2 Ethereal，3 Innate，4 Unplayable，5 Retain，6 Sly，7 Eternal |
| `RoomType` | 1 Monster，2 Elite，3 Boss，4 Treasure，5 Shop，6 Event，7 RestSite，8 Map |
| `MapPointType` | 1 Unknown，2 Shop，3 Treasure，4 RestSite，5 Monster，6 Elite，7 Boss，8 Ancient |

## 已记录但暂不计分的写法（TASK-025 票 04 遗物审核扩充）

事实如实记录，评分引擎不能换算的部分会在推荐中显示为缺口，不会被猜测计分（引擎对不认识的字段、条件、过滤一律视为不支持）。

- 效果：`orb_passive`、`orb_passive_all`、`orb_slots`、`set_free_this_turn`、`extra_turn`、`cost_reduction`、`add_pet`、
  `procure_potion`、`set_enemy_hp`、`replay_card`、`move_card`。
- 时机：`lethal_damage`。计数：`counter: "run"`（跨战斗不清零；期望值同 `combat`）。
- 回合：`turns_from: n`（第 n 回合起）。数值无法写成常数时用 `amount_from` 说明来源（如“等于受到的伤害”“等于当前回合数”）。
- 条件：`turn_at_least`、`unblocked_damage`（引擎已支持）；`during_own_turn`、`cards_played_last_turn_at_most`、
  `card_types_played_this_turn_include_all`、`unspent_energy_last_turn`、`after_rest_site`、`previous_room`、`gold_at_least` 等（暂不计分）。
- 修正：`power_amount_given`、`card_play_count`、`damage_received`、`energy_carryover`、`retain_hand`、`keep_block`、
  `hp_loss_per_turn_cap`、`hp_loss_reduce`、`power_gain`、`auto_play_turn`、`no_potions`、`card_reward_*`（暂不计分）。
- 只改变“提供哪些卡牌奖励”的遗物（卡池、额外选项）不影响对已提供候选的评分，归为 `irrelevant`，行为写进证据。

## 遗物的拾起与局外效果（TASK-027 票 01，2026-10-28）

选遗物（Neow、古神、商店、宝箱、事件）时，遗物的价值往往就是拾起那一刻的改动（删牌、加金币……）和之后每场战斗外的效果。
这些写在两个新字段里；牌组引擎只读 `mechanics`，所以已拥有的遗物照旧（拾起的改动已在当前牌组状态里）。

```json
{
  "obtain": [ OBTAIN, ... ],            // AfterObtained（以及选项准备时 SetupForPlayer 决定的参数）按执行顺序
  "between_fights": [ MECHANIC, ... ],  // 战斗外的持续效果：战斗结束、进房间、篝火、卡牌奖励、商店、地图……
  "obtain_params": {"card": "...", "relic": "...", "character": "..."}   // 选项生成时才确定的参数（如海玻璃对应的角色）
}
```

### OBTAIN（拾起时的一次性效果）

| kind | 字段 | 说明 |
|---|---|---|
| `gain_gold` / `lose_gold` | `amount`（或 `amount_from`） | `PlayerCmd.GainGold` / `LoseGold` |
| `gain_max_hp` / `lose_max_hp` | `amount` | `CreatureCmd.GainMaxHp` / `LoseMaxHp` |
| `heal` | `amount` 或 `amount_from: "full"` | `CreatureCmd.Heal` |
| `hp_loss` | `amount` | 战斗外 `CreatureCmd.Damage`（没有格挡） |
| `set_hp` | `amount` 或 `amount_from` | `CreatureCmd.SetCurrentHp` |
| `remove_from_deck` | `count`、`choice`（`choose` / `random` / `all`）、`filter` | `CardSelectCmd.FromDeckForRemoval` + `CardPileCmd.RemoveFromDeck` |
| `transform_cards` | `scope: "deck"`、`count`、`choice`、`filter`、`into` | `into`：结果牌 ID，或 `{"pool": ..., "rarity": ..., "same_type": true}` 写明 `CreateRandomCardForTransform` 的取牌规则；结果升级写 `upgraded` |
| `upgrade_cards` | `scope: "deck"`、`count`（数字或 `"all"`）、`choice`（`choose` / `random` / `all`）、`filter` | `CardCmd.Upgrade`；`filter` 可用 `card_type`、`tag`、`card` |
| `add_card` | `destination: "deck"`，`card` 或 `pool`，`count`，`choose_from`，`upgraded` | `pool`：`{"source": "character" / "colorless" / "character_of"（配 `obtain_params.character`）, "rarity": ..., "card_type": ...}`；从 N 张里选 k 张写 `choose_from: N` + `count: k`；可以不选写 `skippable: true` |
| `enchant_cards` | `enchantment`、`amount`、`count`、`choice`、`filter` | `CardCmd.Enchant`；附魔 ID 去掉命名空间 |
| `gain_potion` | `potion`（ID）或 `"random"`、`count` | `PotionCmd.TryToProcure`；栏满时是否丢弃写进证据 |
| `potion_slots` | `amount` | `PlayerCmd.GainMaxPotionCount` |
| `gain_relic` | `count`、`source`（`"next_in_pool"`，或 `rarity`） | `RelicFactory.PullNextRelicFromFront` + `RelicCmd.Obtain` |
| `offer_rewards` | `rewards: [{"type": "card", "options": 3, "pool": {...}}, {"type": "relic"}, {"type": "potion"}, {"type": "gold", "amount": n}]` | `RewardsCmd.OfferCustom`：照构造的奖励逐项写 |
| `replace_relic` | `from`、`to` | `RelicCmd.Replace`（如把初始遗物换成升级版） |
| `remove_relic` | `relic` 或 `which` | `RelicCmd.Melt` 等 |
| `add_pet` | `monster` | `PlayerCmd.AddPet` |
| `apply_power` | 同 EFFECT | 拾起时施加、整局保留的能力，写明 `persist` |

- `choice` 另可写 `first`：游戏按牌组顺序取第一张符合条件的牌（`FirstOrDefault`），结果确定。
- 变化的 `into` 另可写 `"pool": "original"`（被变化那张牌自己的卡池，即它的颜色）与 `"exclude_original": true`；
  `rarity` 可为列表（如 `["Common", "Uncommon", "Rare"]`）。加牌的 `pool` 另可写 `"source": "colorless"` 与 `tag`
  （如 `{"source": "character", "rarity": "Basic", "tag": "Strike"}` 表示该角色的基础打击）。
- `filter` 键：`card_type`、`tag`、`card`、`rarity`、`transformable` / `removable`（`CardModel.IsRemovable`：没有永恒关键词；
  牌组中的牌 `IsTransformable` 与之相同）、`upgradable`。估值程序遇到其他键会报缺口，不会当作“所有牌”。
- 数值来自 `get_CanonicalVars`（`CardsVar` → `count`、`GoldVar` → `amount`……），写数值，不写变量名。
- 随机结果写清分布来源（卡池、稀有度、是否排除初始牌），不要估概率；做不到的写 `open_questions`。
- 只改变“之后会提供什么”（卡池、奖励数量）的写进 `between_fights` 的修正，不要写成 OBTAIN。

### between_fights（战斗外的持续效果）

用 MECHANIC 结构，`when` 取：`combat_end`（`filter.fight_kind` 可限定 `MONSTER` / `ELITE` / `BOSS`）、`room_entered`（`filter.room_type` 用 RoomType 名）、
`rest_site`、`card_reward`、`merchant`、`treasure`、`map_generated`、`gold_gained`、`potion_procured`、`lethal_damage`（战斗外受到致死伤害时，如蜥蜴尾巴；估值程序暂不计分，报缺口）；效果用 OBTAIN 的 kind（如每场战斗后 `heal`）。
修正（`modifier`）：`card_reward_options`（`add`）、`card_reward_rarity`、`card_reward_upgraded`、`rest_site_heal`（`add` / `multiply`）、`rest_site_option`（`option`）、
`merchant_price`（`multiply`）、`gold_gained`（`multiply`）、`potion_drop`、`no_potions`、`free_travel`、`unknown_room_types`、`map_change`（写 `amount_text` 说明）。
有次数或时限的写 `limit`（如 `{"combats": 3}`）。

## 卡牌与能力审核扩充（票 05）

- 卡牌字段：`upgrade_keywords: {"add": [...], "remove": [...]}`（升级增减关键词）、`multiplayer_only: true`、`star_cost`（星辉费用，写法同 `cost`）、`can_be_generated_in_combat: false`（战斗中不能被生成）、
  `cost_modifier`（按局面改变自身费用，如每打出一张攻击牌费用 −1）。
- 能力字段：`duration`：`{"kind": "combat" | "turns_decrement" | "this_turn" | "until_triggered", ...}`。
- `transform_cards`（2026-10-22 起计分）：`into` 结果牌、`scope` `hand` / `draw`、`choice` `choose`（`count` 为数字或 `any`）或 `all` 配 `filter: {card_type: Status}`；结果升级标记 `upgraded_if_self_upgraded` / `upgraded_result_if_upgraded`。随机结果与按类型整批变形仍不计分。
- `exhaust_other` / `discard` 可带 `filter: {card_type}`；`upgrade_cards` 的 `scope` 另支持 `discard`、`filter: {upgradable: true}`，并可用 `upgraded` 记录升级后分支（如 Armaments）。
- 效果（暂不计分）：`auto_play`、`auto_play_card`、`auto_play_from_draw`、`gain_max_hp`、
  `add_keyword`（给牌临时加关键词）、`remove_block`、`remove_power`、`trigger_power`（立即结算一次能力，如中毒）、`extra_card_reward`（局外收益）、`evoke_orb`（激发充能球，`orbs` + `times`）、`shuffle`（弃牌堆洗入抽牌堆）、`doom_kill`（斩杀生命不高于灾厄层数的敌人）、`kill_osty`、`spread_debuffs`（把目标的减益复制给其他敌人）、`add_replay`（让选中的牌多结算一次）、`end_turn`（立即结束回合）、`gain_gold`（局外收益）、`stun`（眩晕目标）、`lose_max_hp`。
- 时机（暂不计分）：`on_chosen`（状态牌被选入牌组时），局外的 `combat_end`、`rest_site`、`room_entered`、`map_generated`、`quest_complete`；效果 `remove_from_deck`、`rest_site_option`、`force_event`；修正 `play_lock`（手牌中时禁止打出其他牌）。卡牌字段 `card_type_text`：类型由事件决定时的说明（如 MAD_SCIENCE）。
- Osty 造成的伤害：`damage` 加 `source: "osty"` 和条件 `osty_alive`（暂不计分）。
- 时机（暂不计分）：`auto_post_play_phase`、`card_drawn`、`block_gained`、`power_applied`、`power_applied_self`、`creature_added`（敌人中途加入战斗）。
- 修正（暂不计分）：`exhaust_on_play`、`vulnerable_multiplier`、`weak_multiplier`、`strength` / `dexterity` / `focus`（临时力量、敏捷、集中）、`poison_trigger_count`、`negate_debuff`（人工制品）、`hp_loss_negate`（缓冲）、`card_target`（改变牌的目标）、`orb_value`（集中对充能球的加成）、`damage_cap` / `hp_loss_cap`（每次伤害或失血的上限，如无实体）、`card_play_result_location`（打出后去向）、`star_cost`、`sovereign_blade_block`、`add_keyword`（能力给牌加关键词）。时机：`stars_gained`、`damage_dealt`、`forge`、`osty_hp_lost`、`orb_evoked`、`energy_spent`、`power_removed`。效果：`kill_owner`、`kill_target`。

## 数值随局面变化（票 05 起）

数值依赖战斗状态时，不写自由文本，而用结构化来源（引擎按牌组每回合期望估算；原文可留在 `amount_text` / `hits_text`）：

- `scale: {"per": 来源, "factor": n, "factor_upgrade": m, "min": 0}`：数值 = `amount` + `factor` × 来源；升级时 `factor` 加 `factor_upgrade`。
- `hits_scale: {"per": 来源, "base": b}`：攻击次数 = `b` + 来源；条件次数用 `{"per": "condition", "condition": {...}, "base": 1, "hits_if": 2}`。
- `count_scale: {"per": 来源}`：耗竭/弃牌张数随局面。
- 来源：`exhaust_pile`（耗竭堆张数）、`block`（当前格挡）、`enemy_vulnerable`（目标易伤层数）、`strength`（力量）、`strike_cards`（打击标签牌数）、
  `hand_size`（打出时其余手牌数）、`hp_loss_events`（本场失血次数）、`non_attack_in_hand`（手牌中非攻击牌数）、`attack_draw_chain`（连续抽到攻击牌的期望长度）、
  `own_previous_plays`（本场此前打出本牌的次数）、
  `cards_drawn_this_combat`（本场已抽牌数）、`draw_pile`（抽牌堆张数）、`target_debuff_types`（目标身上的负面效果种数）、
  `discards_this_turn`（本回合已弃牌数）、`hand_space`（手牌上限 10 减当前手牌）、`enemy_count`（敌人数）、
  `orbs_channeled_this_combat`（本场已充能的球数）、`lightning_channeled_this_combat`（本场已充能的闪电球数）、`cards_played_this_combat`（本场已打出的牌数）、
  `energy_spent_this_turn`（本回合已花的能量）、`skills_in_hand`（手中技能牌数）、`energy_after_cost`（付完本牌费用后剩余的能量）、`enemies_killed_by_this_card`（本次打出击杀的敌人数，
  引擎按“每回合击杀数 × 本次伤害占全队伤害的比例”估计）。不在列表内的来源写 `amount_text` 并加 open_question。
- 效果上的 `condition`：`target_has_power`（Vulnerable/Weak）、`exhaust_pile_at_least`、`lost_hp_this_turn`、`card_exhausted_this_turn`、`target_alive`。
- 升级增加攻击次数：`hits_upgrade`。

## 充能球与附魔（票 07 / 10 扩充）

充能球（`orbs:<ID>`，如 `orbs:LIGHTNING_ORB`）：

```json
{
  "status": "converted",
  "passive": {"when": "turn_end", "effect": EFFECT, "focus": "add"},   // 被动：时机 + 效果；focus 写集中如何作用
  "evoke": {"effect": EFFECT, "focus": "add"},                          // 激发效果
  "mechanics": [ MECHANIC, ... ],                                        // 其他钩子（如进入时、能量相关）
  "evidence": ["..."], "reviewed_by": "TASK-025 ticket 07"
}
```

- `focus`：`"add"`（数值 + 集中层数）、`"none"`（不受集中影响）或写明的其他规则；数值以 IL 为准。
- 随机目标、对所有敌人等用 EFFECT 的 `target`。

附魔（`enchantments:<ID>`）：附在一张卡牌实例上，`mechanics` 用 MECHANIC 结构，
效果作用于该牌时写 `"scope": "enchanted_card"`（如打出该牌时额外造成伤害、费用变化）；数值通常随附魔层数 `amount_per_stack`。

充能球与附魔的附加字段：`state`（充能球自身累积的数值，如黑暗球的激发值、玻璃球递减的数值），
`can_enchant`（可附魔的卡牌条件）、`adds_keywords` / `removes_keywords`（附魔增减关键词）；
效果 `enchantment_amount`（附魔层数变化，`persist: "run"` 表示写回牌组）、`set_cost`（随机或固定改费）；
修正 `start_position`（开局所在牌堆位置）；时机 `orb_passive_or_evoke`。以上暂不计分。

## 怪物（2026-10-10，用户决定）

怪物事实（`monsters:<ID>`）用于计算局面需求：每回合受到多少伤害、几段、怪物给玩家塞什么牌和减益。
全部以 IL 为准；目录中的招式与模式（Spire Codex）只用于交叉核对。

```json
{
  "status": "converted",
  "hp": {"min": 42, "max": 46, "min_ascension": 44, "max_ascension": 48, "ascension_level": 8},
  "innate": [ EFFECT, ... ],                    // 进场时自带的能力等（如 Artifact 3）
  "moves": [
    {"id": "SLICE", "intents": ["Attack", "Defend"],
     "effects": [ {"kind": "damage", "amount": 6, "amount_ascension": 7, "ascension_level": 9, "hits": 1, "target": "player"},
                  {"kind": "block", "amount": 5, "target": "self"} ]}
  ],
  "pattern": {
    "initial": "INIT",
    "states": [
      {"id": "BUTT_MOVE", "type": "move", "move": "BUTT", "next": "SLICE_MOVE"},
      {"id": "RAND", "type": "random", "branches": [{"to": "A_MOVE", "weight": 1, "max_repeat": 1, "cooldown": null}]},
      {"id": "INIT", "type": "conditional", "branches": [{"to": "BUTT_MOVE", "condition": "IsAlone", "default": false}]}
    ]
  },
  "mechanics": [ MECHANIC, ... ],               // 其他钩子：死亡时、受伤时、回合开始等
  "unknowable": [], "evidence": ["..."], "open_questions": [], "reviewed_by": "..."
}
```

- 数值来自取值方法（如 `get_ButtDamage`）；`AscensionHelper.GetValueIfAscension(level, a, b)` 在有该进阶时返回 **a**（第一个数是进阶值），
  否则返回 b（已核对 `AscensionHelper` 的 IL：`HasAscension` 为真取 `ldarg.1`）。两组值都要记：`amount` 写基础值 b，
  `amount_ascension` 写进阶值 a，并写 `ascension_level`。多段攻击写 `hits`。
- EFFECT 的 `target`：`player`（对玩家）、`self`、`all_allies`（所有怪物）、`random_ally` 等；
  给玩家的减益写 `apply_power` + `target: "player"`；塞牌写 `add_card`（`card`、`count`、`destination`）。
- `intents` 按意图类名去掉 `Intent`（`SingleAttack`、`MultiAttack`、`Defend`、`Buff`、`Debuff`、`Status`…）。
- `pattern.states` 照状态机原样记录：`move`（`next` 为后续状态）、`random`（各分支 `weight`、连续上限 `max_repeat`、冷却 `cooldown`）、
  `conditional`（各分支的 `condition` 原文；在普通单人战斗第一次到达时会走的分支标 `"default": true`，判断依据写进 evidence）。
- 取决于玩家行为或随机数的条件不要猜，写清条件；真正无法得知的进 `unknowable`。

## 遭遇（2026-10-11）

遭遇事实（`encounters:<ID>`）记录一场战斗实际生成哪些怪物，用来把怪物的每回合期望组合成局面需求。

```json
{
  "status": "converted",
  "room_type": "Monster",                      // RoomType 枚举名：Monster / Elite / Boss
  "is_weak": true,                              // 前几场的弱遭遇（get_IsWeak 或同义属性），没有就写 false
  "variants": [                                 // 固定阵容只有一项；随机阵容每种一项并给出概率
    {"weight": 1.0,
     "spawns": [ {"monster": "CHOMPER", "slot": null, "set": {}, "initial_state": "CLAMP_MOVE"},
                 {"monster": "CHOMPER", "slot": null, "set": {"ScreamFirst": true}, "initial_state": "SCREECH_MOVE"} ]}
  ],
  "evidence": ["GenerateMonsters ..."], "open_questions": [], "reviewed_by": "..."
}
```

- `spawns` 一只怪一项，同种怪出现几次就写几项；`monster` 用怪物事实的 ID。
- `set`：遭遇在生成时对怪物设置的字段（如 `ScreamFirst`、`StartsWithDance`）；能由此确定开场状态时写 `initial_state`（怪物事实里的状态 ID）。
- 阵容由随机数决定时，逐一列出可能的阵容与权重（从 `GenerateMonsters` 的 IL 读出，不要估）。无法穷举时写入 `unknowable`。

## 愚者（TASK-029，游戏 v0.113.1，2026-11-17）

机制说明与完整提案见 `build/fact-reviews/task029-fool-vocabulary.md`（读 IL 得出）。写法：

- 卡牌 `tags` 写 `"Shoot"`、`"Wah"`、`"Cho"`、`"Ki"`；有的牌标签来自 `get_Tags` 覆写而不是 `CanonicalTags`，旋律判断读的是 `card.Tags`，以它为准。
- 瞄准、拉伸用 `apply_power`（`"Aim"`、`"Stretch"`）。

| Need | Proposal | Reuses |
|---|---|---|
| A shot | `damage` + **`"source": "arrow"`**. Meaning: needs an available arrow (none: no damage), the damage carries FromArrow, and the target is impaled after the attack | `source` key (Osty) |
| Extra arrow slot | effect **`arrow_slots`** `{amount, upgrade}` (this combat) | like `orb_slots` |
| Loosen arrows | effect **`dislodge_arrow`** `{count: 1 or "all", owner_only}` + `condition` (RECALL, UPROOT, WREST: `"all"`) | — |
| Arrow destroyed after impaling | QUIETUS power: modifier on the next shot `{"modifier": "damage", "multiply": 2, "filter": {"from_arrow": true}, "limit": "next_shot"}` + `"destroys_arrow": true` | `damage` modifier |
| Target impaled | effect condition **`target_impaled`** (`"any"` or `"owner"`, after the helper called) | `condition` |
| Owner has an arrow | effect condition **`has_arrow`**: true | `condition` |
| Melody held | effect or mechanic condition **`melody_played_this_turn`**: `["Cho", "Ki"]` (any of) | `condition` |
| All melodies | condition **`melodies_this_turn_at_least`**: 3 | — |
| Melody count | scale source **`melodies_this_turn`** (0–3, own card excluded) | `scale` / `hits_scale` / `cost_modifier` |
| Family cards owned | scale source **`melody_cards_in_combat`** (WAH_SYMPHONY) | `scale` |
| Sadness made | scale source **`sadness_generated_this_combat`** (ANGUISH) | `scale` |
| Melody gained | `when: "card_played"` + `filter: {"owner_card": true, "tag": "Cho", "first_of_tag_this_turn": true}`. Any melody: `"tag": ["Wah", "Cho", "Ki"]` + `"per_new_tag": true` (MOTIF draws per tag) | `first_of_tag_this_turn` (PHANTOM_BLADES) |
| Arrow events | `when`: **`arrow_shot`**, **`arrow_impaled`**, **`arrow_dislodged`** (YAH: `limit: once_per_turn`) | MECHANIC |
| Sadness generated | `when: "card_generated"`, `filter: {"card": "SADNESS"}` | documented `card_generated` |
| Shot damage bonus | damage-modifier filter key **`from_arrow`**: true. HONING: `{"modifier": "damage", "multiply_percent_per_stack": 1, "filter": {"dealer": "owner", "powered_attack": true, "from_arrow": true}}` | `damage` modifier |
| Enemy takes more from shots | DISLIKE_YOU power (Debuff, removed at turn end): `{"modifier": "damage_received", "filter": {"from_arrow": true, "powered_attack": true}, "multiply_percent_per_stack": 1}` | `damage_received` |
| More damage to impaled | SERRATED_ARROWS: `{"modifier": "damage", "filter": {"dealer": "owner", "powered_attack": true, "target_impaled": "any"}, "multiply_percent_per_stack": 1}` | — |
| Sadness / Joy cards | `add_card` `{card: "SADNESS"/"JOY", count, destination: "hand"}`. Transform: `transform_cards` `{scope: "hand", choice: "random", count: 1, filter: {"card": "SADNESS"}, into: "JOY"}` | existing |
| Tarot | `add_card` `{"pool": {"source": "tarot", "allowed_in_run": true}, count, destination: "hand", "upgraded": …}` | `pool` |
| Aim / Stretch facts | `powers:AIM` and `powers:STRETCH` with `duration: {"kind": "until_triggered", "consumed_per": …}` and modifier `damage` / `block` with `add_per_stack: 1`, `"per_hit": true` / `"per_block_instance": true` | existing power-fact shape |


`multiply_percent_per_stack: n` 表示倍率 1 + n × 层数 / 100。

起草与复核中补充的约定（批次 f1、f2）：

- 射击有两种写法：`ShootArrowAtTarget` 配攻击命令，或配一个动作（`Func<Task>`）：先射出，有箭才执行动作，最后施加扎穿；没有箭时整张牌无效（能量照扣）。
- `dislodge_arrow` 写 `count`（数字或 `"all"`）与 `owner_only`；RECALL、UPROOT、WREST 是一直松脱到目标身上没有本玩家的箭为止（`count: "all"`），目标未被本玩家扎穿时整张牌无效。
- 整张牌都取决于扎穿的（如 NAIL），在每个效果上写 `condition`。
- 重射的瞄准算两倍：伤害效果写 `aim_multiplier: 2`。
- 写清判定时机：`had_arrow_at_play_start` 是出牌前有箭，`shot_succeeded` 是射击成功，`target_impaled_at_play_start` 在松脱之前判定，不要用结算后的箭数替代。
- EPIC_RETRIEVAL 写作 `when: arrow_dislodged`，触发条件为松脱来源是本牌且扎穿来自持有者；每次匹配分别回能，致死一击松脱的箭也算。
- 尚不能估值的动态量要报告缺口。FLORESCENCE 的生成牌增伤取这一次快照中的射击牌数（`damage_from: shoot_cards_snapshotted_by_this_florescence`），不是本场累计；不能把未算增伤的生成牌当成完整效果。
- 只在多人模式出现的牌写 `multiplayer_only: true`（COVERING_FIRE、ENSEMBLE、SELF_DEPRECATE、COMPOST、ENRAPTURE、YOGA）。
- `cost_modifier` 可带 `condition`（如 CHO_LIMBS：持有哇或叽旋律时减费）；箭矢事件的 `filter` 可用 `dealer: owner`。
- 尚无写法、暂写 `amount_text`：手牌中的悲伤张数（GROWTH，建议来源名 `sadness_in_hand`）、卡牌自身数值整局成长（SHOW_OFF，能力触发写 `grow_card_var`）。
- 遗物与宠物能力补充修正（v0.113.1，暂不计分）：`attack_hit_count`（如 THUMBSCREW，对多段攻击增加攻击次数）、`damage_target_redirect`（如 WUPPYS_LOYALTY，把针对持有者的受加成攻击伤害重定向给宠物）。
- 引擎暂不计分：箭的供应与回收、扎穿条件、悲伤、塔罗牌（报缺口）；计划先支持拉伸、瞄准、旋律，箭按“每回合最多按箭位数射击”的暂定规则估计（TASK-029 第 3 步之后）。

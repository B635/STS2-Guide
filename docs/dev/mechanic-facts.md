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

## 卡牌与能力审核扩充（票 05）

- 卡牌字段：`upgrade_keywords: {"add": [...], "remove": [...]}`（升级增减关键词）、`multiplayer_only: true`、
  `cost_modifier`（按局面改变自身费用，如每打出一张攻击牌费用 −1）。
- 能力字段：`duration`：`{"kind": "combat" | "turns_decrement" | "this_turn" | "until_triggered", ...}`。
- 效果（暂不计分）：`auto_play`、`auto_play_card`、`auto_play_from_draw`、`transform_cards`、`gain_max_hp`。
- 时机（暂不计分）：`auto_post_play_phase`、`card_drawn`、`block_gained`、`power_applied`、`power_applied_self`、`creature_added`（敌人中途加入战斗）。
- 修正（暂不计分）：`exhaust_on_play`、`vulnerable_multiplier`、`strength`（临时力量）。

## 数值随局面变化（票 05 起）

数值依赖战斗状态时，不写自由文本，而用结构化来源（引擎按牌组每回合期望估算；原文可留在 `amount_text` / `hits_text`）：

- `scale: {"per": 来源, "factor": n, "factor_upgrade": m, "min": 0}`：数值 = `amount` + `factor` × 来源；升级时 `factor` 加 `factor_upgrade`。
- `hits_scale: {"per": 来源, "base": b}`：攻击次数 = `b` + 来源；条件次数用 `{"per": "condition", "condition": {...}, "base": 1, "hits_if": 2}`。
- `count_scale: {"per": 来源}`：耗竭/弃牌张数随局面。
- 来源：`exhaust_pile`（耗竭堆张数）、`block`（当前格挡）、`enemy_vulnerable`（目标易伤层数）、`strength`（力量）、`strike_cards`（打击标签牌数）、
  `hand_size`（打出时其余手牌数）、`hp_loss_events`（本场失血次数）、`non_attack_in_hand`（手牌中非攻击牌数）、`attack_draw_chain`（连续抽到攻击牌的期望长度）、
  `own_previous_plays`（本场此前打出本牌的次数）。不在列表内的来源写 `amount_text` 并加 open_question。
- 效果上的 `condition`：`target_has_power`（Vulnerable/Weak）、`exhaust_pile_at_least`、`lost_hp_this_turn`、`card_exhausted_this_turn`、`target_alive`。
- 升级增加攻击次数：`hits_upgrade`。

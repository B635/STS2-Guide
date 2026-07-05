# STS2 Guide 数据架构

> 详细说明文档；产品边界以 [`product-spec.md`](product-spec.md) 为准。

## 存储边界

- SQLite：结构化实体、卡牌/遗物效果标签、来源快照、社区聚合统计和每局最终摘要。
- `active-run.json`：唯一可替换的当前局检查点；用于保存退出后的恢复，不是历史库。
- `events/`：短生命周期事件队列，处理后删除。
- 原文快照 + FAISS：攻略正文和语义切片，只用于可选解释，不参与实时评分。
- `knowledge.json`、`community_scores.json`：可重建导入快照，不是在线查询数据库。

P0 实时链不得向 `run_states`、`decision_events`、`decision_outcomes` 或
`game_state_events` 追加历史。旧表仅保留迁移兼容；一局结束只向
`run_summaries` 写入最终摘要。

## SQLite 结构

```text
catalog_entities
  ├── cards
  ├── relics
  └── entity_effect_tags

data_sources
  └── source_snapshots
        └── entity_statistics

run_summaries
```

`entity_effect_tags` 使用实体稳定 ID 保存 `tag + magnitude + source_field`：

- 数值字段优先生成 `damage`、`block`、`draw`、`energy_gain` 等标签；
- 类型、目标、关键词和 Powers 生成范围、能力与机制标签；
- 可版本化的本地描述规则补充降费、升级、力量、易伤、耗竭等语义；
- 标签生成不调用网络、RAG 或 LLM；
- `effect_tag_version` 变化时可从 `catalog_entities.payload_json` 全量重建。
- 自身耗竭/虚无与“引用或支持该机制”使用不同标签；
- 药水状态继续进入当前局检查点，但 P0 不把药水标签用于选牌评分。

## 实时数据流

```text
只读 Mod
  └── StateEvent v3
        ├── events/ 临时队列
        └── 本地后台
              ├── active-run.json（状态 + 最近地图 + 打开的决策）
              ├── SQLite 结构化查询
              └── advice-event.json
                    └── 游戏内 Godot 面板
```

选牌事件本身不重复携带整张地图。后台从同一 Run ID 的检查点读取最近地图上下文，
与事件中的牌组、HP、遗物和候选合并后评分。

## 透明评分

`contextual_state_v3` 从 50 分开始，按以下有界因素加减：

- 卡牌基础效果和费用效率；
- 当前牌组的输出、防御、抽牌、能量、范围与成长缺口；
- 当前牌组已有机制的协同；
- 当前遗物与候选效果的协同；
- HP、章节、楼层和近期路线压力；
- 升级、附魔、负面状态和重复牌成本。

`contextual_state_plus_community_v3` 在其上叠加低权重、带来源和样本量的社区先验。
社区先验不能单独触发跳过。

跳过使用动态分数，考虑牌组大小、功能缺口、候选状态适配、重复度和近期路线压力。
它不是固定 50，也不是胜率。

路线压力按每条可选路径计算平均风险与精英/Boss 可达比例。评分器在通用威胁、
精英或 Boss 中选择一类路线因子，不能对同一节点重复计分。

## 数据质量限制

- 社区统计目前是混合版本观察性数据，只能作为弱先验；
- 没有奖励“被提供次数”，因此不能计算真实抓取率；
- 没有足够且按版本隔离的状态—选择—结果样本，不能输出校准胜率；
- 描述规则只生成可审计标签，不能把自然语言相似度直接当作实时决策分。

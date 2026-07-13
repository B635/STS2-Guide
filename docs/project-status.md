# STS2 Guide 当前实现状态

> 更新时间：2026-07-13
>
> 产品依据：[`product-spec.md`](product-spec.md)
>
> 本文只记录源码、自动验证、安装态和真机证据。代码存在不等于真机完成。

## 当前结论

P0 的开发态真实闭环已经通过：只读 Mod 能捕获普通选牌，本地 Python Host 离线计算，
游戏内左侧抽屉显示卡牌与跳过推荐，决策关闭后清理建议；保存继续保持同一局身份，放弃
后只保存最终摘要并删除临时状态。

这允许项目进入 P0.5 架构收口。当前 P0.5 只有决策内核基础和 Card Reward 适配，尚未达到
完成定义；P1 的商店、路线、篝火等建议均未实现。当前发布 EXE 早于协议 v4 和本轮改动，
完成 P0.5 后必须重新构建和回归，不能把旧 `dist/` 产物视为当前发布版本。

## 2026-07-13 真机验收证据

本轮使用游戏版本 `0.108.0`、最新安装的 DLL/JSON/PCK 和唯一 Python Host 验证：

- 主菜单不生成伪造当前局，也不显示旧建议；
- 新开储君局生成 Run ID `sts2-590d213739a8c9ed6c0f0461`，Neow 普通祝福阶段没有误触发
  `card_reward`；
- 首次地图捕获 53 个真实节点，当前位置和下一节点与界面一致；
- `ActMap.BossMapPoint` 被显式合并到图中，Boss 节点为 `16:3`，三条末层真实边均指向
  Boss；游戏已知遭遇 ID 为 `VANTOM_BOSS`，结构化数据映射为“墨影幻灵”；
- 第一场普通战斗后的候选为 `SOLAR_STRIKE / CRESCENT_SPEAR / GUIDING_STAR`，后台推荐
  `GUIDING_STAR`，游戏内面板显示名称、三张分数和跳过分，用户确认与真实界面一致；
- 领取卡牌后产生 `decision_closed`，左侧面板消失，检查点牌组包含 `GUIDING_STAR`；
- 进入商店时没有错误显示选牌面板。商店库存尚未捕获，符合“P1 未实现”的安全降级；
- 从商店保存返回并继续游戏时，游戏恢复到同一商店场景，没有旧选牌建议；
- 放弃产生 `run_ended(outcome=abandon)`：SQLite `run_summaries` 只新增一条摘要，最终
  角色、楼层、分数、牌组、遗物和药水均有记录；
- 放弃并关闭后，`active-run.json`、`state-event.json`、`advice-event.json` 和临时
  `events/` 全部清理，游戏与 Host 进程均已退出。

此前另一张 Ironclad 真机地图曾记录 60 个节点和 70 条真实边，并完成同局保存继续与
sequence 递增验证。结合本轮储君地图，P0 的两张不同地图要求已有真机证据。

Neow 中“祝福直接打开特殊选牌”的随机分支不属于 P0 推荐范围；生产代码和自动测试要求
其失败关闭且不能复用上一屏候选。修复后尚未再次随机遇到该分支，因此它仍是 P1 祝福
建议开发前必须补做的专项真机回归，不能标记为祝福建议已完成。

## 协议与实时生命周期

- Mod 当前只生产 JSON Schema v4；v4 增加真实 `boss_encounter_ids`；
- Python 消费端兼容 schema v1-v4，以便旧夹具继续回放；
- 事件使用 `run_id + event_id + sequence`，候选使用带位置的稳定实例 ID；
- 面板只有在 Run、事件和可见候选全部匹配时显示，不能回放上一组分数；
- `active-run.json` 是唯一可原子替换的当前局检查点；
- `events/` 是处理后删除的临时队列，不是历史数据库；
- 胜利、失败或放弃后只向 SQLite `run_summaries` 写一条最终摘要；
- 实时路径不向 `run_states`、`decision_events`、`decision_outcomes` 或
  `game_state_events` 追加历史；
- Host 使用交换目录单实例锁，原子替换包含有界重试。

## P0 推荐与界面

已实现的选牌输入包括：

- 真实候选、牌组、升级/附魔/负面状态；
- HP、章节、楼层、金币、遗物和药水；
- 真实地图图、下一节点和已知 Boss；
- SQLite 静态效果标签与有限权重社区先验；
- 输出、防御、抽牌、费用、范围、成长、牌组负担和协同信号；
- 动态跳过候选；推荐分是局面适配度，不是胜率。

游戏内界面是贴住左侧的半透明抽屉，支持候选数量安全降级、内容自适应宽度、展开/收起、
实际中文名、推荐高亮和关闭销毁。P0 标准验收场景仍是三张牌加跳过。

固定选牌评测包含 26 个场景、93 个断言；它是确定性规则回归，不代表胜率校准或训练
准确率。项目不保存逐决策训练历史，也不做自学习。

## P0.5 当前状态

已经存在并有自动测试覆盖：

- `advisor/decision_core.py`：`WorldState`、`DecisionCandidate`、`DecisionRequest`、
  `CandidateAssessment`、`Recommendation` 和 `PolicyRegistry`；
- `advisor/policies.py`：`CardRewardPolicy`，将现有 Card Reward 推荐器适配为第一个策略；
- `realtime/processor.py`：Card Reward 已通过统一请求和策略注册表调用，同时保留 P0
  advice payload 兼容；
- 重复实体候选使用“位置 + 实体 ID”区分，不因两张同名卡冲突；
- 未注册策略失败关闭，策略返回未知候选或错误决策身份会被拒绝。

尚未完成：

1. `WorldState` 仍包裹较多协议原始 mapping，需要形成明确的资源、牌组、地图和场景字段；
2. 决策打开、更新、关闭与 Run 生命周期还没有统一状态机；
3. `Recommendation` 的多维评价、版本和数据缺口还没有完整可执行契约；
4. 左侧 `CardRewardAdvicePanel` 尚未抽象为通用 `Context Drawer`；
5. 需要补充迁移前后固定夹具等价测试、异常降级测试和一次 P0 真机回归；
6. 完成后重新构建后台 EXE，并执行隔离 `--help`、`--once` 和真机启动验证。

在这些验收完成前不得开始复制商店、路线或篝火分支。

## 结构化数据

`scripts/fetch_knowledge.py` 已能从 Spire Codex 构建期 API 获取策略相关实体，当前中文
可重建快照包含 1541 个实体。SQLite schema v8 已规范化导入：

| 表 | 当前行数 |
|---|---:|
| `catalog_entities` | 1541 |
| `monsters` | 115 |
| `monster_moves` | 345 |
| `encounters` | 87 |
| `encounter_monsters` | 135 |
| `events` | 66 |
| `event_pages` | 289 |
| `event_options` | 363 |
| `acts` | 4 |
| `act_entity_memberships` | 140 |
| `mechanic_constants` | 8 |

所有遭遇怪物引用和 Act Boss 引用均通过完整性检查。结构化实体和机制进入 SQLite；攻略
正文不进入关系库。本机 Mobalytics tier、真实数据库、攻略快照和运行日志继续 Git 忽略。

## P1 调研证据与未实现范围

当前游戏程序集已确认商店存在公开 API：

- `NMerchantInventory.Inventory`；
- `MerchantInventory.CharacterCardEntries / ColorlessCardEntries`；
- `RelicEntries / PotionEntries / CardRemovalEntry`；
- 商品公开 `Cost / IsStocked / EnoughGold`，卡牌另有 `IsOnSale` 和实际
  `CardCreationResult`。

这些证据只说明 P1 可实现。当前没有商店 Observer、协议事件、策略、建议文件或真机库存
记录，不能宣称商店建议完成。

路线方面，P0 已有真实节点、坐标、`Children`、下一节点和 Boss。P1 计划在真实地图上
绘制独立只读 Overlay 表达主路线和备选路线，不写入游戏原生绘图数据。当前尚无路线策略
或地图推荐叠加层。

## 自动验证与构建基线

本次 Git 基线提交前已重新执行：

- 完整 Python 单元测试：137/137 通过；
- `git diff --check`：通过；
- Mod `--no-restore` 编译：0 error、1 warning；warning 仅说明审查环境没有配置 Godot
  可执行文件，因此跳过 PCK 打包，DLL 和 JSON 编译成功；
- 最近一次带 Godot 的安装构建：0 warning、0 error，DLL/JSON/PCK 与游戏目录哈希一致；
- Git 上传边界审计：`.env`、`.mcp.json`、本机 SQLite、私有 tier、Host 日志、EXE、
  Mod `bin/obj/artifacts` 和游戏安装目录文件均不在待提交集合；
- 变更差异未发现大小写敏感的常见 API Key 或私钥特征。

## 下一步

1. 完成 P0.5 决策内核契约和生命周期状态机；
2. 将左侧面板收口为统一 `Context Drawer`，保持 P0 视觉和交互兼容；
3. 完成等价回放、失败降级、完整测试、Mod 编译、EXE 重建和 P0 真机回归；
4. P0.5 验收后进入 P1，优先实现商店库存捕获与购买建议，再实现地图路线策略及 Overlay；
5. Neow 特殊选牌、篝火、Boss 遗物和事件按独立策略逐项增加，不共享猜测式分支。

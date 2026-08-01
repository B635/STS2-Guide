# STS2 Guide 当前实现状态

> 更新时间：2026-08-01
>
> 产品依据：[`product-spec.md`](product-spec.md)
>
> 本文只记录源码、自动验证、安装态和真机证据。代码存在不等于真机完成。

## 当前结论

P0/P0.5 在游戏 `0.108.0` 上曾完成开发态真机闭环；这些是历史证据，不自动构成当前版本
兼容承诺。本机 Steam 游戏已于 2026-08-01 自动更新为 `0.110.1`（commit `db5d3552`），
`sts2.dll` SHA-256 为
`7C446EFABF80614C429B5088E87101423AA5BB4C04FC3E73393261F6E6D404FD`。

2026-07-31 的工作区实现已升级到 Guide `0.3.0-alpha.0`、Mod `0.3.0`、生产协议 v9，
当前 Card + Route 基线 release fingerprint 为
`3ec6911d1288fe9c57b92a37c4f935fc8464d4eafabd6ccddddc896edb657e2f`。v9 将 Card、
Route、Merchant、Rest Site、Neow、Event 和 Deck Edit 统一为严格候选 envelope，并加入
父子决策身份；Python 仍只为离线回放兼容 v1-v8。

TASK-010 已在当前 `0.110.1` 上完成隔离探针和 Card + Route 正式组合基线。清单总状态为
enabled，但只启用 `card_reward` 与 `route_choice`；Merchant、Rest、Neow、Event 和
Deck Edit 继续 `pending_validation` 并失败关闭。正式三件套已经安装，测试用 RouteLiveProbe
和 STS2MCP 已卸载；最终 Steam 启动日志只有 Guide，`Loaded 1 mods (1 total)`。

Card 真机候选为 `DRAIN_POWER / DEFY / POKE / skip`，state/advice 与中文 Drawer 一致；
Route 真机从真实 origin `5:2` 对 `6:2 / 6:3` 生成一条完整路线。保存继续复用原 route
decision ID，三种路线偏好复用同一 decision 并原子重算。地图滚动和窗口从
1950x1275 调整到 1600x1000 后，单一 `Line2D` 仍锚定原生节点且不穿过顶部 UI。

TASK-011 Windows 交付壳已通过二次独立审查并验收：最新自动证据为 Python
`438/438 OK`、Mod/PCK `0 warning / 0 error`、完整 Public Beta 构建和 payload 审计通过。
最终安装器已完成真实升级、卸载边界、重装、重复启动、游戏 RUNNING → IDLE 和完全退出
验证；运行数据库保留既有 7 条最终摘要，Worker 停止后 advice 和锁均已清理。五角色第一幕
三层与真实 `run_ended` 仍属于 TASK-012，因此当前不能宣称 Windows Public Beta 已完成。

2026-08-01 TASK-009 已按三路独立审查反例完成多轮返修：advice/schema/domain parity、
真实父决策 lineage、active-child checkpoint、Mod capability/no-throw、Rest 成功回调、
Deck Edit 主线程以及 Event/Neow 特殊 Card Reward 的 exact session parent 均已收口。协议与
Mod 最终独立复审均为 P0/P1 0；完整 Python `377/377 OK`，Mod/PCK `0 warning / 0 error`，
EXE 重建并通过 startup check。TASK-010 此后只开放 Card/Route；扩展 capability 仍待各自
真机验收。复杂 C# 时序尚缺独立可执行状态机测试，登记为 P2 测试工程项，不替代真机门禁。

## 2026-07-14 P1.0 路线 API 门禁证据

- 当前游戏为 `0.108.0`；`sts2.dll` SHA-256 为
  `51A671BFEB937271AF3E643D017396B13432098ED2B9DEBCEB110C74939BBBA1`；
- 独立只读探针会话记录 61 条连续事件：3 次真实选路、1 次顶栏预览、2 次真实选点、1 次
  保存继续、52 份完整地图视觉快照、3,328 个节点坐标样本；validator 通过；
- 真机确认正式选路需要 `isOpenedFromTopBar=false`、地图打开、允许旅行、未旅行、非 debug、
  单人、非空候选且模型/视觉候选稳定一致；顶栏预览即使显示候选也不创建路线 decision；
- public `OnMapPointSelectedLocally(NMapPoint)` 的 Postfix 两次精确返回 `1:3`、`2:3`；点击
  瞬间 `CurrentMapPoint` 尚未推进，生产关闭必须使用回调参数和点击前候选；
- 保存继续会重建 RunState 与 screen owner；恢复早期状态不完整，`SetMap` 和真实 Open 后才
  恢复 origin/候选/地图指纹。逻辑节点集合保持 64 个，但所有 Godot instance ID 均重建；
- screen/net 坐标往返最大误差约 0.001 px；地图滚动与多种窗口/viewport 尺寸下节点全局
  中心正确移动。当前版本没有用户地图 Zoom API，因此 Zoom 为 N/A，不再要求伪造测试；
- decision ID 使用 Mod GUID，并按 Run、Act、真实 origin、规范化候选与地图指纹从唯一当前
  检查点恢复。初始可操作 origin 为真实 `0:3`；空 origin 时 fail closed，不使用 sentinel、
  运行时对象 ID 或直接字符串拼接；
- 上述结论只表示生产 API 门禁通过，不表示路线产品功能完成；首轮生产代码独立审查未
  通过，必须先返修，不能进入安装/真机验收。

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
- 当时进入商店没有错误显示选牌面板；该历史版本尚未捕获商店库存，安全降级正确；
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

- Mod 源码当前生产 JSON Schema v9；v5 分离不可变 `event_id` 与稳定 `decision_id`，v6
  新增 `route_choice`/真实 `origin_node_id`，v7 增加完整快照、成功提交 revision、
  生产者/游戏程序集身份和跨组件 `release_fingerprint`，v8 增加显式路线偏好，v9
  增加通用候选、类型化成本/效果和 `decision_parent`；Python 仅为离线回放兼容 v1-v8；
- 决策打开、更新和关闭校验 `run_id + decision_id + sequence`；跨局关闭和冲突重复关闭
  失败关闭，完全相同的重复关闭保持幂等；
- 面板同时校验 Run、事件、决策、sequence、完整候选指纹与可见候选，不能回放上一组分数；
- `active-run.json` 是唯一可原子替换的当前局检查点；
- `events/` 是处理后删除的临时队列，不是历史数据库；
- 胜利、失败或放弃后只向 SQLite `run_summaries` 写一条最终摘要；
- 实时路径不向 `run_states`、`decision_events`、`decision_outcomes` 或
  `game_state_events` 追加历史；
- Host 使用交换目录单实例锁，原子替换包含有界重试。
- `advice-event.json` 只保存当前可执行 `Recommendation`；决策关闭、非法输入或新的无效
  选牌会清理它，无关地图/商店观察不会覆盖或误删仍打开的决策建议。
- 路线偏好只从同一 Run、同一 release 的唯一 `active-run.json` 恢复；事件镜像和待处理
  spool 不是偏好权威源，checkpoint 损坏时 Host/Mod 均回到安全默认状态。

## P0 推荐与界面

已实现的选牌输入包括：

- 真实候选、牌组、升级/附魔/负面状态；
- HP、章节、楼层、金币、遗物和药水；
- 真实地图图、下一节点和已知 Boss；
- SQLite 静态效果标签与有限权重社区先验；
- 输出、防御、抽牌、费用、范围、成长、牌组负担和协同信号；
- 动态跳过候选；推荐分是局面适配度，不是胜率。
- 五角色共用通用攻防、HP、遗物、路线、跳过和数值边界规则；角色机制由薄 adapter 映射
  为统一 provider/payoff/spender/capacity/multiplier 信号，缺少真实动态状态时报告
  data gap。

游戏内界面是贴住左侧的半透明抽屉，支持候选数量安全降级、内容自适应宽度、展开/收起、
实际中文名、推荐高亮和关闭销毁。P0 标准验收场景仍是三张牌加跳过。

固定选牌评测包含 26 个场景、93 个断言；它是确定性规则回归，不代表胜率校准或训练
准确率。项目不保存逐决策训练历史，也不做自学习。

## P0.5 当前状态

代码与自动验证已经完成：

- `WorldState` 已改为明确的角色、资源、牌组、遗物、药水和地图不可变领域对象，不再把
  协议原始 mapping 直接交给策略；`scoring_state()` 只作为已验证 P0 评分器的兼容适配；
- `DecisionRequest` 使用稳定决策 ID、类型、候选实例 ID、约束和状态 sequence；
- `DecisionLifecycleManager` 明确处理 opened/updated/closed 和 Run 清理，Host 重启后允许
  由唯一当前局检查点恢复关闭事件，但不保存决策历史；
- `Recommendation` 统一策略版本、状态版本、候选评价、推荐身份、置信度、多维贡献和
  数据缺口；Card Reward 使用 contract v1，Route 使用 contract v2；
- `PolicyRegistry` 对未注册策略、错误决策身份、错误状态版本、重复/未知候选和无评价推荐
  失败关闭；
- `advisor/policies.py`：`CardRewardPolicy`，将现有 Card Reward 推荐器适配为第一个策略；
- `realtime/processor.py`：Card Reward 已通过统一请求和策略注册表调用，同时保留 P0
  `advice` payload 兼容，并额外输出 canonical `recommendation`；
- 重复实体候选使用“位置 + 实体 ID”区分，不因两张同名卡冲突；
- `ContextDrawer.cs` 统一拥有左侧布局、视口自适应、候选行、推荐高亮和展开/收起动画；
  `CardRewardAdvicePanel` 只负责真实候选校验和 Card Reward advice 解析；
- `advice-event.schema.json` 已加入 canonical Recommendation contract；
- 迁移等价测试确认同一输入的新策略适配结果与原 P0 `recommend_card_reward` 完全一致；
- advice 由当前决策拥有：关闭或错配时清理/失效，无关观察保留当前决策的 advice；抽屉
  handle 与具体选牌界面绑定，旧界面退出回调不能关闭新抽屉。
- advice publish/clear 失败时不会提前确认临时队列；checkpoint replay 会重试幂等副作用，
  且旧 close 不会误清后来属于另一决策的建议；
- 最近关闭 tombstone 会在 Host 重启后恢复，同一个稳定 decision ID 不能被更高 sequence
  重新打开，不同新 ID 仍可正常开始；
- Mod 的 Run 身份守卫不会把 Launch 时的临时未知身份当成新局，已结束的稳定 Run ID 也
  不能被新局重新接受；恢复选牌 decision ID 还会严格核对局面指纹和更晚失效事件；
- `run_ended` 只有在事件写入成功后才提交内存清理，瞬时写盘失败可由后续生命周期回调重试。

仍未完成：

1. 按 TASK-012 完成五角色第一幕三层 Card + Route 组合真机；
2. 补一次真实 `run_ended` 清理与新局身份隔离；
3. Public Beta 基线通过后，再对 Neow、商店、篝火、事件与 Deck Edit 逐项做 capability
   真机门禁；
4. Event/Neow 的结构化效果目录与正式“为什么”入口。

## P1.0 路线生产纵向切片（`0.110.1` 基线已完成）

首轮实现于 2026-07-15 因协议可执行性、identity、搜索边界、多维评分、失败关闭和测试覆盖
不足退回。本轮返修已完成以下自动层面的收口：

- schema/Pydantic 共同约束当时的生产 schema v8 与 v1-v7 离线回放，路线 v2 presentation 统一为
  `primary_path_node_ids` / `backup_path_node_ids` / 类型化 `paths`，Card v1/v2 可安全消费；
- Observer 使用真实 origin、模型/视觉 `Travelable` 候选一致性和两次稳定检查；完整 State 与
  地图指纹参与 UPDATED/恢复 identity。写入、绑定、predicate 或选择关闭失败均隐藏 route UI，
  不清除 Writer 恢复状态；
- RoutePolicy 使用有界记忆化搜索，任何候选不可达或输入缺口错误都会失败关闭；静态遭遇池、
  Boss 行动、升级/多段伤害、AOE、成长/资源、遗物、一次性药水、HP 与金币都以可解释因子参与；
- Drawer/Overlay 只消费 fingerprint、owner、完整元数据和候选全量匹配的 advice；地图只
  绘制一条推荐主路线，并逐帧重锚，对非法 topology、重复视觉节点和陈旧 owner 清线；
- 旧 checkpoint/result 的 release fingerprint 不一致时不恢复 lifecycle、不重放旧建议，
  当前策略会重新计算并原子替换；C# 的选牌和路线恢复同样要求精确 fingerprint；
- 自动回归为 Python `293/293 OK`，包含 64 节点图、end-to-end 100 次 P95、跨 release
  checkpoint 重算及严格 advice 正反例；本机 Godot 4.5.1 生成 DLL/JSON/PCK，
  `0 warning / 0 error`。

2026-07-30 复核发现的保存继续版本错配、首次写盘失败、旧地图 revision、严格类型、
版本矩阵和旧 recommendation 重标兼容现已进入自动反例并通过。2026-07-31 又加入三个
路线软偏好、同局恢复、偏离后的动态重算与单主路线硬约束。当前 `route_choice` 已随
TASK-010 在 `0.110.1` 启用并完成组合真机基线；五角色覆盖仍见 TASK-012。

## Public Beta 策略核心（待组合真机）

- `advisor/character_mechanics.py` 定义统一 `MechanicSignal`，五个角色 adapter 只负责把
  结构化卡牌事实转换为机制 family/role，不自行持有最终分数；
- effect tag 已升级至 v5，并修复 Rupture、Blade Dance、Tactician、Zap/Dualcast/
  Defragment、Comet、Venerate、Bodyguard/Unleash 与 Shroud 等已知反例；
- 路线模式进入 `WorldState`、Card/Route 策略、协议、checkpoint、advice 和 Drawer；
  同一真实路线决策切换模式复用 decision ID、生成新 event ID 并以 UPDATED 重算；
- Overlay 和正式 Route advice 拒绝非空 backup，只允许一条 `primary_path_node_ids`；
- `growth` 仍受低 HP 生存底线约束；缺少真实地图时 Card Reward 不伪造 route-fit；
- 机制 signals、data gaps 和资源维度已经进入正式 canonical Recommendation，而非只停留
  在兼容 payload。

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
| `mechanic_constants` | 9 |

所有遭遇怪物引用和 Act Boss 引用均通过完整性检查。结构化实体和机制进入 SQLite；攻略
正文不进入关系库。本机 Mobalytics tier、真实数据库、攻略快照和运行日志继续 Git 忽略。

## P1 扩展决策套件（自动审查通过，待真机）

旧 `0.109.1` 程序集已经静态确认并由 Mod 编译引用；以下成员必须在当前 `0.110.1`
重新核对后才可启用：

- Merchant：`NMerchantInventory.Inventory/IsOpen/Open/OnCardRemovalUsed/_ExitTree`，
  `MerchantInventory.AllEntries/Player`，各商品公开实际模型及
  `Cost/IsStocked/EnoughGold`；
- Rest Site：`NRestSiteRoom.Options/Create/EnableOptions/AfterSelectingOption/
  BeforeExitingRoom/_ExitTree`，`RestSiteOption.OptionId/IsEnabled/Title/OnSelect`，
  `HealRestSiteOption.GetHealAmount(Player)`；
- Event/Neow：`NEventRoom.Create/OptionButtonClicked/_ExitTree`，
  `EventModel.CurrentOptions/CanonicalInstance/Owner/IsFinished` 及精确
  `SetEventState/SetEventFinished`，`EventOption.TextKey/IsLocked/IsProceed/Chosen`；
- Deck Edit：升级/变化屏幕 `ShowScreen`、基础选择屏 `Create`、公开
  `CardSelectorPrefs` 约束、`CardsSelected()` 与 `_ExitTree`。

生产代码只注册 Harmony Postfix，不调用购买、选择或卡牌点击动作。Merchant 使用真实异构
库存、实际价格、售罄/买不起状态和离开候选；同一库存对象变化保持同一 decision 并发出
UPDATED。Rest/Smith 与商店移除通过结构化 `upgrade_card/remove_card` 效果和真实子屏建立
父子决策；多选组合尚未建模时直接隐藏。Event/Neow 使用真实 option identity；效果未知时
显式 data gap。通用 Drawer 严格核对 Run/Event/Decision/sequence/capability/候选，并把
最多三个关键原因作为候选悬浮提示。

`explain/` 提供 latest-only、内存态、过期拒绝的确定性解释契约，只复述冻结
Recommendation 的 factors/dimensions/data gaps；它不进入实时 EXE、不调用网络/LLM/
向量库、不重新评分，也尚未接入正式游戏内“为什么”按钮。攻略 RAG 和远程模型仍是后续
可选解释能力，不是当前交付。

## 自动验证与构建基线

当前工作区自动证据：

- 2026-08-01 最新完整 Python 单元测试：438/438 通过；包含 64 节点 DAG、
  event→checkpoint→advice、v9 严格通用候选、v1-v8 回放、跨组件兼容、独立 Card/Route
  恢复机会、五角色机制、五类扩展决策、解释边界和效果标签 v5 回归；
- 固定选牌场景：26/26、93/93 断言通过，迁移没有改变推荐基线；
- 新五类决策内核：250 样本，P50 2.654 ms、P95 21.615 ms、最大 26.155 ms；既有
  文件桥与路线端到端性能门禁也继续满足 P95 ≤ 300 ms；
- `git diff --check`：通过；
- 当前 `0.110.1` Mod `--no-restore` 由 Godot 4.5.1 生成 DLL/JSON/PCK，
  0 warning / 0 error，并已用安装脚本逐文件校验；
- 当前窗口化后台 EXE 与每用户安装器已在启用后的 compatibility manifest 下重新打包；
  完整测试、startup check、源码新鲜度和 payload 拒绝清单均进入同一构建门禁；
- 安装 artifacts：Guide EXE 34,836,097 bytes，SHA-256
  `E5E612168A977C6E5361253730F9500B8A4DCF4CA480729A5C9A5F2201591959`；安装器
  36,556,706 bytes，SHA-256
  `53D4512C6FE431E1DC992D0B09A1F7534C0B4D01D690652E8A194416C4414277`；DLL 230,400 bytes，
  SHA-256 `0591842CE51997005F31F3C2524EEFD8034F39EF422FA9E23CBA3AC3BD5F57C4`；PCK 692 bytes，
  SHA-256 `D51ADC0B1D499D7D9F492E2725CE5047940358E99E266A5998AAAB8B6B23C839`；
  JSON 344 bytes，SHA-256
  `56FA0D63B7EE2E9C897E422474477082C3FBA0DAC83B027B98A3259C624C8EB4`；
- Git 上传边界审计：`.env`、`.mcp.json`、本机 SQLite、私有 tier、Host 日志、EXE、
  Mod `bin/obj/artifacts` 和游戏安装目录文件均不在待提交集合；
- 变更差异未发现大小写敏感的常见 API Key 或私钥特征。

## 下一步

1. 执行 TASK-012 五角色第一幕三层真机组合门禁，并补一次真实 `run_ended` 清理；
2. 只有 Public Beta 基线稳定后，再按 Merchant → Rest/Smith → Neow → Event → Deck Edit
   逐项启用；任一项失败即维持 `pending_validation`；
3. Event/Neow 只有采集到当前版本真实 ID/页面/结构化效果后才评分；未知效果保持 `--`；
4. 代码签名/SmartScreen 信誉仍是发布限制，不能在没有证书时伪装为已解决。

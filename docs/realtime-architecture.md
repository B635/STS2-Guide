# STS2 Guide 实时架构

> 本文解释已经确认的实时架构，不改变
> [`product-spec.md`](product-spec.md) 的产品范围和优先级。

## P0 目标

P0 是一个本地、只读、多维上下文感知的选牌决策闭环：

```text
STS2
  └─ C# 只读 Mod
       ├─ 感知选牌、关闭、地图和 Run 生命周期
       ├─ 原子写入版本化 JSON
       └─ Godot 游戏内抽屉读取 advice-event.json
              ▲
              │
       Python 本地后台
         ├─ 校验协议、顺序和事件身份
         ├─ 维护唯一 active-run.json 检查点
         ├─ 查询 SQLite 静态结构化数据
         ├─ 执行本地确定性推荐器
         └─ 原子写回建议
```

实时链路不依赖 Vue、FastAPI、RAG、LLM、API Key 或网络。

## 组件职责

### 只读 Mod

- 只注册 Harmony Postfix，不改写游戏返回值；
- 读取真实游戏 API 中的角色、牌组、HP、遗物、药水、候选牌和地图；
- 使用稳定 ID 生成生产 schema v9 事件；Python 消费端只为离线回放保留 v1-v8 兼容；
  Card、Route、Merchant、Rest、Neow、Event 与 Deck Edit 共用严格候选 envelope 和
  Recommendation contract；路线另使用 contract v2 presentation；
- 不打分、不联网、不自动点击、不修改游戏或存档；
- 建议的 `run_id + event_id + 候选稳定 ID` 全部匹配后才显示面板。

### 本地后台

正式开发入口是：

```powershell
python -m realtime.host
```

后台只负责文件桥接、当前局检查点、SQLite 静态数据和本地推荐。它不启动 Web 服务，
也不加载向量模型。最终交付形态是一个无控制台后台 EXE。

### 推荐器

P0 只输出卡牌奖励与“跳过”的适配分。推荐器综合卡牌效果、牌组缺口、协同、遗物、
HP、章节、楼层和近期路线压力。分数不是胜率，每个内部因子必须可追踪。

## 文件交换

默认目录：

```text
%APPDATA%\SlayTheSpire2\STS2Guide\
├── events\
│   └── <run>-<sequence>-<type>.json
├── state-event.json
├── advice-event.json
└── active-run.json
```

- `events/` 是待处理临时队列，处理后删除；
- `state-event.json` 是最新事件镜像和兼容入口；
- `advice-event.json` 是当前建议，不能作为历史库；
- `active-run.json` 是唯一可原子替换的当前局检查点。

所有写入使用临时文件加原子替换。重复事件幂等；相同事件身份出现不同内容时明确拒绝。

## 当前局生命周期

1. Mod 产生事件并递增 sequence；
2. 后台校验后更新内存状态和 `active-run.json`；
3. 保存退出后，下一次启动从检查点恢复同一 Run ID；
4. 选择或跳过只更新当前决策状态，不写入历史训练表；
5. 胜利、失败或放弃后删除检查点与临时事件；
6. SQLite `run_summaries` 只保存一条最终摘要。

实时路径禁止向 `run_states`、`decision_events`、`decision_outcomes` 和
`game_state_events` 追加历史。

## 地图与敌人边界

P0 先完成了真实节点、坐标、类型、`MapPoint.Children` 连边、可选下一节点和 Boss 的
读取；P1.0 在这组事实之上生成一条只读主路线。读取失败时不猜测连边，也不发布路线。

当前 Act 的 Boss 是已知事实；普通怪物和精英在进战前只能按真实遭遇池、游戏规则和
已击败精英集合估计，不能预测为某个确定敌人。P0/P1 的敌人机制用于战前选牌与路线
风险建模，不做战斗内逐回合出牌建议。

## 存储边界

- SQLite：静态实体、效果标签、社区结构化先验、静态敌人机制和最终 Run 摘要；
- 当前局 JSON：只保存仍会影响后续决策的可恢复状态；
- 临时事件：处理后删除；
- 原文快照与向量索引：只服务可选攻略解释，不参与实时评分。

## 降级与性能

- 本机 tier 缺失、损坏或过期时继续使用结构化基线；
- 后台不可用、建议过期或候选不匹配时，面板隐藏或显示 `--`；
- 本地事件发现到建议写回的 P95 目标不超过 300 ms；
- 异常退出不得影响游戏进程和存档。

## 决策扩展架构

P0.5 决策内核、P1.0 路线和 TASK-009 扩展决策自动实现已经进入源码。所有新增能力仍按
capability 独立门禁，不能复制 `card_reward` 分支形成多套实时管线，也不能把自动实现
写成真机交付。

目标结构：

```text
真实游戏 API
  → Game Adapter（只负责读取与类型转换）
  → World State（当前局权威状态）
  → Decision Detector（识别当前真实决策）
  → Decision Kernel（身份、生命周期、过期与降级校验）
  → Policy Registry
       ├─ Card Reward Policy
       ├─ Route Policy
       ├─ Merchant Policy
       ├─ Rest Site / Deck Edit Policy
       ├─ Neow Policy
       └─ Event Policy
  → Recommendation
  → Context Drawer
```

公共概念的职责：

- `WorldState`：只描述当前局事实，不夹带某个策略的评分结果；
- `DecisionRequest`：描述决策类型、实际候选、约束、稳定身份和所引用的状态版本；
- `Policy`：声明支持的决策类型，并对同一请求产生确定性候选评价；
- `Recommendation`：统一携带候选身份、排序、分数、多维因子、缺失数据和策略版本；
- `Context Drawer`：只渲染已经通过 Run、决策和候选一致性校验的当前建议。

当前实现中，`advice-event.json` 只承载一个 canonical `Recommendation` 和必要兼容
payload。当前决策关闭、非法输入或新的无效决策会清理 advice；无关观察不会误删仍打开
决策的建议。`ContextDrawer.cs` 只负责通用布局和交互，场景 Controller 负责真实 owner、
身份与候选校验；后续策略不得把游戏状态解析重新塞回通用抽屉。

每个策略模块必须具备真实状态捕获、确定性基线、失败降级、单元测试、回放测试和真机
验收。RAG 只在用户展开解释时提供攻略证据，不接管实时决策。

### P1.0 路线纵向切片

路线必须区分两种事实：

- `map_choice`：地图观察，手动预览也可能触发，只更新 `WorldState.map`；
- `route_choice`：经真实 API 证明此刻可以选择下一节点的决策，才进入策略注册表。

目标数据流为：

```text
Verified Route Decision API
  → Route Event Adapter
  → WorldState(map) + DecisionRequest(route_choice)
  → Decision Kernel + RoutePolicy
  → Recommendation v2(route_paths)
  → RouteAdviceController
       ├─ Context Drawer
       └─ RouteMapOverlay
```

Route Event Adapter 只能从 `map_context.available_next_node_ids` 构建候选，候选 ID 直接使用
真实节点 ID。地图图更新与 `OPENED/UPDATED/CLOSED` 生命周期更新是两个独立状态变换，但
必须在同一次原子 checkpoint 替换中提交；同一事件同时携带地图和路线决策时，既不能通过
`if/elif` 漏掉其中一项，也不能拆成两次写盘暴露中间状态。

RoutePolicy 只消费领域 `WorldState` 和 `DecisionRequest`，不读取 Godot 节点、文件路径或
UI 坐标。它输出每个真实下一节点的适配度和经过边校验的内部候选路径，但只发布并显示
排序第一的一条当前主路线。UI 侧再次核验
`run_id + event_id + decision_id + sequence + candidate IDs`，并在 Overlay 绘制前检查每条
边都来自当前地图快照。

`RouteAdviceController` 同时拥有 Drawer handle 和 Overlay handle，并绑定真实
`NMapScreen` owner。Overlay 只根据当前游戏视觉节点换算坐标，忽略鼠标输入，不调用游戏
选择方法，也不写入原生地图绘制状态。视觉节点暂时无法解析时可以继续显示 Drawer 文本，
但必须隐藏线条并记录诊断。

2026-07-14 的程序集与独立真机探针已确认生产门禁：真实选路需同时满足非顶栏预览、地图
打开、允许旅行、未旅行、非 debug、单人、候选非空且模型/视觉候选稳定一致；真实选择由
`OnMapPointSelectedLocally(NMapPoint)` Postfix 参数与点击前候选关闭。Overlay 每次从当前
owner 的 `NMapPoint.GetGlobalRect().GetCenter()` 重新锚定；当前 0.108.0 没有用户地图 Zoom，
历史证据不能代替当前 `0.109.1`；当前版本必须重新覆盖地图滚动、Zoom（若生产 UI
可操作）、窗口/viewport/content scaling 的真实 transform 变化。

完整 API 证据、生产协议 v9、Recommendation v2 和真机验收项以
[`product-spec.md`](product-spec.md) 第 11.3 节及
[`TASK-007`](tasks/TASK-007-p1-route-vertical-slice.md) 为准。门禁已通过只代表可以开始生产；
2026-07-31 的 v9、发布指纹、路线三模式、五角色机制层和扩展决策自动实现已通过完整
自动回归，当前兼容
清单仍为 `pending_validation`，等待 `0.109.1` 隔离探针和正式安装后的组合真机验收，
尚未交付完成。

### P1 扩展决策通用链

v9 对每个候选统一携带稳定 ID、kind、entity、label、eligible、不可用原因、类型化 cost
和按 kind 校验的 payload。Merchant、Rest、Neow、Event 与 Deck Edit 只在 Adapter 中读取
真实对象，之后走同一个 Processor、Lifecycle、Checkpoint、Policy Registry、Advice 和
Drawer。父决策选择可能打开 Card Reward 或 Deck Edit 时，`decision_parent` 绑定父
decision/candidate/source；父子身份无法唯一证明就不显示子建议。

`ResourceBudget` 从同一 `WorldState` 派生 HP 安全底线、路线压力、金币保留目标、药水槽、
牌组负担和升级候选。未知或随机效果成为 data gap，不能由标题、描述或 LLM 变成确定分。
`explain/` 是隔离的 latest-only 内存边界，只解释冻结 Recommendation；它不由实时 Host
导入，不在 EXE 中打包，也不写当前局或执行游戏动作。

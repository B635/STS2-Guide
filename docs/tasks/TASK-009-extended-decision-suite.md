# TASK-009：扩展决策套件与只读解释 Agent

- 状态：待真机验收
- 创建者/主审：Codex
- 实现角色：Codex，可使用明确分工的子 Agent
- 前置：TASK-008 自动层通过；组合真机按用户决定延后统一执行
- 创建日期：2026-07-31

## 用户目标

在下一轮集中真机前，连续完成以下自动实现：

1. 继续改善五角色普通选牌与路线/资源联动；
2. Neow 祝福及其触发的嵌套选牌不复用旧候选；
3. 商店异构商品、价格、购买资格、保留金币和动态重算；
4. 篝火动作及 Smith 等二阶段目标；
5. 通用事件页面/选项决策；
6. RAG/Strategy Agent 作为用户主动展开的只读解释层。

真机完成前所有新增 capability 默认关闭，不能把自动实现写成已交付。

## 不可违反的边界

- Mod 继续只使用 Harmony Postfix，`affects_gameplay: false`；
- 不自动点击、不调用选择方法、不修改游戏状态、RNG 或存档；
- 新 Observer 的类型、公开成员、触发点和 predicate 必须先由当前 `0.109.1` 程序集确认；
  生命周期语义仍不清楚时建立隔离只读 probe，不用宽泛反射或 UI 文本猜测；
- 四类新决策共用一个协议迁移、一个 event adapter、一个生命周期、一个 checkpoint 和一个
  advice envelope；禁止在 Processor/FileBridge 中复制四套发布/清理分支；
- 实时策略只使用当前局状态与 SQLite 结构化数据，不调用网络、LLM 或向量库；
- RAG/Agent 只响应用户主动解释请求，不改变 Recommendation，不执行游戏动作；
- 当前局仍只保存唯一 `active-run.json`，不新增逐决策历史表；
- 本任务不做战斗逐回合、多玩家、自学习或胜率预测。

## 实施阶段

### A. API 证据门禁

为 Merchant、Rest Site、Neow、Event 分别登记：

- 精确类型与 public 成员；
- 打开、更新、关闭、真实选择的 Postfix 观察点；
- 可操作态 predicate；
- 候选稳定 ID、eligible、价格/成本和结构化结果；
- 保存继续与嵌套决策行为；
- 程序集确认、源码线索和必须真机探针的分级。

任何一类证据不足时只实现领域/策略/回放，不把猜测 Observer 接入生产。

### B. 统一生产协议

- 生产协议一次升级到 v9；v1-v8 只用于离线回放；
- 增加严格 `DecisionCandidateEnvelope`，受控 kind、稳定 ID、entity ID、label、eligible、
  unavailable reason、类型化 cost 和按 kind 校验的 payload；
- Card、Route 与新增决策均迁移到通用 envelope；
- 增加 parent context，表达 Neow/事件触发的嵌套奖励和篝火二阶段决策；
- manifest、fingerprint、C#/Pydantic/schema/example/checkpoint/advice/打包同步更新；
- capability 列表逐项标记 `pending_validation`，不能用全局 enabled 连带放开未验收功能。

### C. 共享资源模型与策略

建立不可变 `ResourceBudget`，从当前 WorldState 派生：

- HP 安全余量与路线近期风险；
- 当前金币、保留金币目标和购买后的余额；
- 药水空槽、牌组大小、重复/移除价值；
- 可升级牌及最佳升级增益；
- 角色机制、Boss 需求和剩余路线成长窗口。

实现并注册：

- `NeowPolicy`
- `MerchantPolicy`
- `RestSitePolicy`
- `CardUpgradePolicy`
- `EventPolicy`

所有策略输出统一 CandidateAssessment、dimensions、factors、data gaps 和置信度；未知结构化
效果不得由描述或 LLM 伪造成确定收益。

### D. Mod 与 Context Drawer

- 每类 Observer 只把真实对象转换成通用候选并发事件；
- 通用 Drawer 渲染名称、推荐度、可购买/不可用状态、成本和 2–3 个关键原因；
- 商店购买后同一界面候选变化产生 UPDATED；
- 篝火动作和升级目标使用 parent decision，不混成一个候选列表；
- Neow 嵌套 Card Reward 使用真实 reward source，不能显示上一局普通奖励；
- 事件翻页、选项变化、关闭和保存继续遵守 owner-scoped 清理；
- 任意身份、候选或 capability 错配立即隐藏/显示 `--`。

### E. 只读解释 Agent

- 新建与 realtime package 隔离的 `explain` 边界；
- `ExplainRequest` 只引用冻结的 run/decision/recommendation/candidate；
- 确定性解释器先返回已计算 factors、dimensions、data gaps 和 SQLite 精确实体来源；
- 可选 RAG 只检索带来源、语言和游戏版本的攻略切片；
- 可选 LLM/Agent 只能整理检索证据，不得改变分数或补写未知游戏事实；
- 返回前再次核对当前 decision；超时、过期、无模型或无语料均安全降级；
- 旧 Hybrid Search/Reranker/HyDE/LangGraph 默认不进入打包，只有评测证明需要才启用。

## 自动验收

1. v9 schema/Pydantic/C# 严格 parity，v1-v8 回放继续通过；
2. 通用候选 envelope 对 kind/cost/payload/identity 正反例全覆盖；
3. 每个策略至少覆盖高/低 HP、高/低金币、可用/不可用、未知数据与跨角色场景；
4. Merchant 覆盖异构商品、移除服务、售罄/买不起、购买后重算与离开；
5. Rest Site 覆盖 Rest/Smith/特殊动作与升级子决策；
6. Neow 覆盖普通祝福、嵌套 Card Reward、换局和旧 advice；
7. Event 覆盖页面身份、确定成本收益、随机/未知结果降级；
8. Card/Route 回归分数与身份不发生无依据漂移；
9. explanation 无网络基线、过期拒绝、引用、超时和不改变 Recommendation；
10. 实时评分及 event→advice P95 继续 ≤ 300 ms；
11. 完整 Python、Mod/PCK、EXE startup-check、JSON/PowerShell/Git 边界全部通过；
12. 独立代码审查无 P0/P1 阻断。

## 延后的组合真机

自动层通过后再一次性安装与测试。每项 capability 独立记录：

- API predicate 与候选集合；
- Drawer/Overlay 展示与关闭；
- 真实选择后的 CLOSED/UPDATED；
- 保存继续和换局；
- capability gate 开关；
- 游戏/Host 退出无残留。

真机未覆盖的 capability 继续保持 disabled，即使同一安装包中的其他决策已经通过。

## Codex 自动实施报告（2026-07-31）

### 实施结果

- 生产协议一次升级到 v9；schema、示例、Python 严格模型、C# 模型、checkpoint、
  advice 与兼容清单共同使用通用候选 envelope。v1-v8 只保留离线回放。
- `advisor/decision_core.py` 与 `realtime/processor.py` 继续作为唯一共享处理链；
  Merchant、Rest Site、Neow、Event、Deck Edit 通过策略注册表接入，没有复制文件发布、
  checkpoint、关闭清理或 identity 分支。
- 新增共享 `ResourceBudget`，让卡牌、路线与扩展策略读取同一份金币、HP、药水槽、
  升级/移除与路线风险派生事实；不单独持久化历史。
- Mod 新增只读 Merchant、Rest Site、Event 和 Deck Edit Observer；所有补丁均为
  Harmony Postfix。通用 `ContextDrawer` 校验完整决策 handle、capability、release
  fingerprint 和候选集合，并可在候选悬浮提示中展示前三个确定性关键因子。
- Rest/Smith 和商店移除使用显式 parent decision。对子屏先于父回调出现的真实时序，
  Writer 只允许在当前唯一 eligible 父候选与目标动作精确匹配时事务关闭父决策并建立
  child；多选 Deck Edit 未建模时失败关闭。
- `explain/` 只实现 latest-only、内存态、过期拒绝的解释契约与确定性基线，不进入实时
  EXE，不调用网络、LLM 或向量库，也不重新计算 Recommendation。

### 当前程序集 API 证据

本机 `0.109.1` 的 `sts2.dll` 通过 `tools/Sts2ApiProbe` 元数据/签名探针核对：

- Merchant：`NMerchantInventory.Inventory`、`IsOpen`、`Open()`、
  `OnCardRemovalUsed()`、private `Close()`、`_ExitTree()`；库存公开
  `AllEntries`/`Player`，条目公开 `Cost`/`EnoughGold`/`IsStocked`。
- Rest Site：`NRestSiteRoom.Options`、`Create()`、`EnableOptions()`、
  `AfterSelectingOption()`、private
  `OnBeforePlayerSelectedRestSiteOption(RestSiteOption, ulong)`、`_ExitTree()`；
  option 公开 `OptionId`/`IsEnabled`/`Title`/`OnSelect`。
- Event：`NEventRoom.Create(EventModel, RunState, bool)`、
  `OptionButtonClicked(EventOption, int)`、`_ExitTree()`；`EventModel.CurrentOptions`
  和 option 的 `TextKey`/`IsLocked`/`IsProceed`/`Title`/`Chosen` 为精确成员。
- Deck Edit：相关 grid/selection screen 的 `ShowScreen`/`Create`、公开
  `CardSelectorPrefs.MinSelect`/`MaxSelect`/`Prompt` 与 `NCardGrid.CardsSelected`/
  `_ExitTree()` 已核对。

这些证据只确认当前程序集存在精确成员，不替代真实生命周期验收。Observer 不调用
`OnSelect`、购买、旅行、卡牌选择或其他游戏动作。

### 自动验证

- 完整 Python：`360/360`；
- 固定五角色选牌评测：26 个场景、93 个断言；
- 新五类决策内核 250 样本：P50 `2.654 ms`、P95 `21.615 ms`、最大 `26.155 ms`；
- Mod/PCK：Godot 4.5.1、.NET 9，`0 warning / 0 error`；
- 后台 EXE：完整 `--startup-check` 通过，无残留实例锁；
- JSON、PowerShell、`compileall`、`git diff --check` 与 Git 忽略边界通过。

### 未完成与审查重点

- 2026-08-01 已由三路独立会话审查，结论为不通过；下列阻断返修完成并重新审查前保持
  “需修复”。
- 没有安装 Mod、没有启动 Host 或游戏；所有 capability 继续
  `pending_validation`，不能标记为真机完成。
- Event/Neow 运行时 API 没有暴露确定的结构化收益时，策略只返回 data gap/`--`；
  当前 Spire Codex 的 66 个事件快照也只有 ID、标题和描述，没有类型化效果。后续必须用
  真实 `TextKey` 建立经人工/API 证据审核的版本化 SQLite 效果目录，不能解析描述或让
  LLM 猜测。
- `explain/` 尚未连接游戏内“为什么”入口；RAG 仍是可选后续能力，不是实时依赖。
- 独立审查应重点检查 Observer 的真实打开/更新/关闭时序、父子 identity、v9 parity、
  capability 失败关闭以及旧 Card/Route 是否发生无依据回归。

## 独立审查结论（2026-08-01）

现有 360 个测试全部通过仍不能验收，独立最小反例证明存在假绿：

1. `advice-event.schema.json` 把 assessment `eligible` 固定为 true，但 Merchant 等生产结果
   合法包含 `eligible=false`；Processor 输出无法通过正式 advice schema。
2. Python 只要求 Deck Edit 携带 `decision_parent`，没有核对当前 Run 最近关闭父决策、真实
   selected candidate、source type 与 sequence；伪造 parent 也能发布建议。
3. Mod 对 Merchant/Rest/Neow/Event 任意 selected 候选留下无类型两分钟 parent marker，
   会把后续普通 Card Reward 或不相干 Deck Edit 错挂到旧父决策。
4. Event 翻页可能先删除旧 option binding，随后真实点击只能留下 `closed_unknown`；
   `IsProceed` 也未经证据被当作确定 leave。
5. Deck Edit `ConfigureAwait(false)` 后调用会触达 Godot API 的 Writer，并且确认/取消后没有
   立即 owner-scoped 隐藏 Drawer。
6. 新增 Harmony 入口缺少统一 no-throw 边界；Observer 异常可能冒回游戏调用。
7. Rest 从三个过早/重复入口提交 selected，没有使用当前程序集已确认的
   `OnAfterPlayerSelectedRestSiteOption(option, success, playerId)` 成功结果。
8. `ResourceBudget` 设置 `safety_violation` 后最终选择未消费；单一致命候选仍可能 recommend。
9. Pydantic 未拒绝重复 `target_candidate_ids`，与 schema `uniqueItems` 不一致。
10. capability 只在 Host advice 侧门禁；Mod 仍会创建未验收场景的 `--` Drawer。

### 返修顺序

1. 先修 advice schema、Pydantic parity、Processor 最近关闭父 tombstone 校验和安全底线；
2. 再给 Mod 增加与发行清单一致的 capability gate 和统一 Postfix no-throw；
3. 将 child marker 改为结构化、child-kind/operation 精确匹配且未知效果不 armed；
4. Rest 只在成功回调提交，Deck Edit 回到 Godot 主线程并立即清 Drawer；
5. Event 在真机时序确认前保留旧 binding、未知 Proceed 不伪装 leave；
6. 对每一问题加入能执行完整 schema/Processor/状态机的反例，重跑完整 Python、Mod/PCK、
   EXE startup-check 后交由独立会话复审。

## Codex 返修实施报告（2026-08-01）

### 已修复

- `protocol/advice-event.schema.json` 允许生产中的不可用候选，但严格要求其 `score/rank`
  为 `null`；v9 `ChoiceEffect` 在 schema、示例、Pydantic 与 C# 中统一显式
  `child_decision_type`，未知 follow-up 保持 `null`，不能默认猜成 Card Reward。
- `realtime/processor.py` 只接受当前 Run 最近真实关闭决策形成的父 tombstone，并核对父
  decision/candidate/source 哈希、关闭 sequence、child type 与 Deck Edit operation；该
  expectation 写入唯一 checkpoint，可在 Host 重启后恢复，打开任一后续决策即单次消费。
- `CandidatePayload` 拒绝重复目标 ID；资源安全底线违规候选退出可推荐排名，若没有安全
  候选则返回 `uncertain`，不会推荐致命收益。
- Mod 内嵌精确 `packaging/compatibility.json` 并在 Observer、Writer 与 Drawer 三层按
  capability 失败关闭；新增 Harmony Postfix 统一经过 no-throw 边界。
- Rest 仅使用已由 `0.109.1` 程序集确认的成功回调提交结果；Event 延迟到真实点击回调后
  观察新页面且不再猜 `IsProceed=leave`；Deck Edit 保持 Godot 主线程并在确认、取消、异常
  路径立即 owner-scoped 清理 Drawer。
- child marker 改为 exact effect 推导的 child-kind/operation，不使用墙钟 TTL；Mod 可从
  release/run/sequence/source-hash 全匹配的 `active-run.json` 恢复一次父上下文，已消费
  tombstone 不会在同一进程复用。
- Context Drawer 的原因标签可接收 hover；未引入 Prefix、自动选择、网络调用或历史决策
  持久化。

### 验证证据

- `D:\miniconda3\envs\sts2\python.exe -m unittest discover -s tests -v`：
  `370/370 OK`；其中 Mod 生命周期静态反例 `18/18 OK`。
- Mod/PCK：`.NET 9 + Godot 4.5.1`，`0 warning / 0 error`。
- `scripts/build_p0_exe.ps1`（显式 `STS2_PYTHON`）：PyInstaller 构建成功，打包 EXE
  `--startup-check` 返回 0，未留下 `.host.lock`。
- 项目 Python `compileall`、32 个 tracked/unignored JSON、全部 PowerShell 脚本 AST、
  `git diff --check` 均通过。

### 未执行与剩余边界

- 按任务限制未安装 Mod、未启动 Host 或游戏；因此没有新增真机证据，所有 capability
  继续 `pending_validation`。
- Event/Neow 没有真实结构化效果证据时仍只输出未知，不能建立 Card Reward parent；这是
  诚实降级，不是已完成祝福/事件策略。
- 当前交由两名未参与对应实现的 Agent 交叉复审；通过后也只能进入“待真机验收”，不能
  标记 Public Beta 完成。

## 交叉复审结论（2026-08-01，第二轮）

结论仍为不通过。自动测试与编译虽通过，独立反向构造又发现以下阻断：

1. **P0：特殊 Card Reward 绕过父 capability。** 只启用 `card_reward` 时，Neow/事件同步
   打开的 Card Reward 在没有 typed parent 的情况下仍可被当作普通奖励发布，可能重现旧候选
   或错误建议。特殊/未知来源必须要求精确 parent 与父 capability，否则不 Emit、不 Show。
2. `target_candidate_ids` 的 Pydantic 元素约束缺少 schema 已有的非空、最大 240 字符规则；
   `CandidateAssessment` 也未在领域层禁止 `eligible=false + score`、`score=null + rank`。
3. Deck Edit operation 只检查 eligible 候选；同屏不可用候选携带另一 operation 仍被接受。
4. Card Reward parent 没有核对真实 `reward_source`，Neow parent 甚至可接受 `COMBAT` child。
5. Host 重启直接信任保存的 `child_expectation`，没有从保留的父 observation/candidate effect
   重新派生；被污染的 operation 或 unknown follow-up 可被注入。
6. safety violation 虽不会成为 recommended，但 assessment 仍可显示 100 分，Drawer 会误导
   用户；必须输出 `score=null` 或明确阻断展示。
7. Event `SetEventFinished` 清理仍可能早于外层 Chosen Postfix，需要同样 deferred。
8. Deck Edit continuation 发现非 Godot 线程时只 return，不能立即隐藏 Drawer；必须明确投递
   回 owner 主线程完成。
9. 首次 child OPENED 后 expectation 被完全删除，合法同 decision UPDATED 及 Host 重启恢复
   会失败；必须将一次性 tombstone 转换为 active-child parent binding，直到 child 关闭。

### 第二轮返修顺序

1. 先修协议/domain parity、安全分数、reward source、checkpoint 重新派生和 active-child 绑定，
   加入全部端到端反例；
2. 再修 Card Reward 特殊来源门禁、Event deferred finish、Deck Edit 主线程 completion；
3. 重跑完整 Python、Mod/PCK、EXE startup-check，并继续由未参与实现者交叉复审。

## Codex 第二轮返修实施报告（2026-08-01）

- Pydantic 的目标候选 ID 已与 schema 同步非空、非空白、最大 240 字符和唯一约束；
  `CandidateAssessment` 在领域层同步不可用/无分 assessment 的 score/rank 不变量。
- safety violation 现在对外输出 `score=null, rank=null`，不会再让 Drawer 显示危险 100 分；
  Deck Edit 对全部候选（包括不可用）要求统一 operation 并与 parent 精确匹配。
- Neow/Event Card Reward 只能从 exact typed follow-up 派生；Mod 在父身份与父 capability 均
  通过时将语义来源分别写为 `NEOW/EVENT`，普通 Card Reward 保持真实 `CARD`。
- checkpoint 在 closed tombstone 中保存父 observation，在 active child 中保存 parent close；
  Host 重启会核对内容哈希并从父候选 effect 重新派生 expectation，不信任自报字段。
- child OPENED 后将一次性 tombstone 转换为 active-child binding；同 decision UPDATED、Host
  重启继续和 child close 均有独立反例。
- Event 完成、Card Reward 特殊观察和面板重试使用有界 deferred 顺序，保证外层 Chosen 先
  提交；特殊来源缺父级或父 capability 时不 Emit、不 Show。Deck Edit 结果通过 Godot
  `CallDeferred` 回主线程后再 owner-scoped 提交与隐藏。

验证：完整 Python `376/376 OK`；Mod 生命周期静态反例 `19/19 OK`；Mod/PCK
`0 warning / 0 error`。第二轮实现后尚未重跑最终 EXE 打包（第三轮复审前补跑）；未安装、
未启动 Host 或游戏。

## 第三轮复审结论（2026-08-01）

Python/协议上轮反例已封住；同一 Windows 用户同时重写 payload 与无密钥 SHA-256 不属于
本地只读辅助工具的认证威胁模型，content hash 只用于意外损坏与记录一致性，不是 MAC。

Mod 仍有两个 P0：

1. 把所有 CardReward Populate 延迟后，普通战斗 ShowScreen 可能先读到空 pending 且不再
   重试，直接破坏 P0 基线；普通奖励必须保持同步，只延迟已捕获的特殊来源。
2. 特殊 Populate 排队后 Event `_ExitTree` 立即删除 session，延迟回调可能把它重新当普通
   `CARD`；必须在 Populate 时同步冻结来源，并让 Exit 使用固定次数、有界的延迟清理。

返修后必须覆盖普通 `Populate→Show`、特殊 disabled `Populate→Exit→deferred`、
`Finished→Chosen→Populate→deferred` 和重试次数有界四种时序。

## Codex 第三轮返修与最终独立审查（2026-08-01）

- v9 普通 Card Reward 现在只接受 `reward_source=CARD`；Neow/Event 子奖励分别只接受
  `NEOW/EVENT`。Schema、Pydantic、Processor、示例和生产门禁反例保持一致。
- 普通 `CardReward.Populate` 恢复同步 Observe，避免 `ShowScreen` 先于 deferred observation
  而永久错过面板；只有已经捕获的特殊来源使用一次 deferred，让外层 `Chosen` Postfix 先
  建立父身份。
- Event/Neow session 改为语义 tombstone，不再按固定帧数删除。每个 CardReward 对象用弱表
  冻结具体 EventSession；`OnChosen` 成功后将完整 `DecisionParentContext` 写回 session。
- 特殊奖励在 Observer 阶段必须取得非空 exact parent；Writer 在关闭任何现有决策前比较
  `decision_id + candidate_id + source_type + source_id`。同 reward 重复 Populate 可复用当前
  parent，A/B 交错、父 capability 禁用、缺父或错父均失败关闭。
- Event Exit/Finished 只 deferred presentation close，不再先删除 binding；旧 parent 在真实
  地图选点时先记录统一 `ChildParentKey` 为已消费再清空，run 启动/结束/放弃会重置 session
  与弱表。普通新 reward 明确不恢复或继承 parent。

最终证据：完整 Python `377/377 OK`；Mod/PCK `.NET 9 + Godot 4.5.1` 编译
`0 warning / 0 error`；PyInstaller EXE 重建成功，`--startup-check` 返回 0 且无残留锁；
32 个 tracked/unignored JSON 严格解析、6 个 PowerShell AST、Python `compileall` 与
`git diff --check` 通过。协议和 Mod 两名独立审查者最终均报告 P0/P1 为 0。

未执行：按任务限制没有安装 Mod、启动 Host 或游戏；所有 capability 与 manifest 总状态
继续 `pending_validation`。C# 复杂时序目前由编译、静态契约与独立重放审查覆盖，尚缺独立
可执行状态机测试，登记为 P2 测试工程项，必须在启用 Event/Neow capability 前补强，但不
替代对应真机门禁。

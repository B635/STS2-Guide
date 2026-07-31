# TASK-007：P1.0 路线建议完整纵向切片

- 状态：待 `0.109.1` 真机复核（自动返修通过；兼容清单仍为 pending）
- 生产实现门禁：`0.108.0` 历史门禁已通过；当前 `0.109.1` 必须重新验证
- 创建者：Codex
- 实现角色：5.6 Luna 极高（由用户/客户端显式选择；无法验证时不得冒充）
- 独立审查角色：5.6 Sol 极高
- 产品依据：`docs/product-spec.md` 第 11.2、11.3 节
- 前置任务：`TASK-006-p05-contract-hardening.md`（已验收，2026-07-14）

## 目标

完成第一个非选牌策略闭环：只在真实可选择下一地图节点时，使用当前局多维状态和真实图
生成下一节点及一条当前主路线建议，通过统一 Drawer 和只读地图 Overlay 展示，用户自己
点击。其他候选路径只作为内部排序事实，不绘制第二条路线。

本任务一次只交付路线。Neow、Boss 遗物、商店、篝火、药水和事件不得顺带实现。

## 启动门禁

满足全部条件前，本任务不可进入生产代码实现：

- TASK-006 最新 DLL/JSON/PCK 已安装；
- 普通选牌、关闭清理、保存继续和放弃后新局已完成真机回归；
- 当前任务由 Codex 把状态改为“待实现”，并让 `current.md` 指向本文件；
- 实现会话由用户/客户端选择 5.6 Luna 极高；模型身份无法由 Agent 工具证明时必须如实说明；
- API 调研先通过当前本机程序集和真机日志确认下面六项硬事实。

## API 调研硬门禁

必须先提交调研报告，再写生产 Observer：

1. 如何区分真实可操作路线选择与仅手动预览地图；
2. 哪个真实方法或回调能用 Harmony Postfix 观察玩家最终选择的 `MapPoint`；
3. 保存继续、切换场景和无选择关闭时，路线 decision 与候选集合的真实时序；
4. 如何把领域 `MapPoint` 稳定映射到实际 `NMapPoint`；
5. 地图滚动以及视口、分辨率与窗口变化后，应从哪个真实 transform 取得 Overlay 坐标；
6. 同一路线机会的稳定 identity 应来自哪个真实 origin/决策对象，以及同 Act 是否可能重访
   origin；未证明严格前向时不得直接拼接 `{run_id}:route:{act}:{origin}`。

可使用 `docs/api-reference-sources.md` 登记的 STS2MCP/STS2SourceCode 作为线索，但每个成员
必须再次用当前本机程序集或真实对象日志验证。无法确认时停止任务，不得猜字段、文本或
截图坐标。

### Phase A 唯一允许范围

在“生产实现门禁”由 Codex 改为“已通过”前，实现角色只能：

- 扩展项目内 `tools/Sts2ApiProbe`，或增加与正式 Mod/PCK 隔离的开发期只读探针；
- 记录六项事实所需的结构化日志和最小脱敏夹具；
- 更新本任务的 API 调研报告与 `docs/api-reference-sources.md`；
- 编译探针并给出明确真机步骤。

开发探针不得进入正式 PCK/EXE，不得调用选点方法，不得自动操作游戏；生产提交前必须移除
临时代码或证明它完全位于不打包的开发工具中。Phase A 不得修改 schema、生产 Observer、
RoutePolicy、Drawer 或 Overlay。

## 当前证据

- P0 `MapNodeReader` 已读取 `MapPoint.Children`、真实节点、当前节点、可选下一节点和 Boss；
- `NMapScreen.Open` 只证明地图被打开，不能证明此刻存在路线决策；
- `NMapPoint.State == MapPointState.Travelable` 与
  `NMapScreen.OnMapPointSelectedLocally(NMapPoint)` 已由当前程序集和两次真实选点确认；
- P0.5 已提供 `WorldState`、`DecisionRequest`、生命周期、策略注册、Recommendation 和
  `ContextDrawer`，Route 必须复用这些契约。
- 2026-07-14 对本机 0.108.0 `sts2.dll` 的静态核验确认：`NMapScreen.Instance`、
  `IsOpen`、`IsTravelEnabled`、`IsTraveling`、`OnMapPointSelectedLocally(NMapPoint)`、
  `GetScreenPositionFromNetPosition(Vector2)`；`NMapPoint.Point/State/IsTravelable`；
  `MapPoint.coord/Children` 均为可定位成员；程序集 SHA-256 为
  `51A671BFEB937271AF3E643D017396B13432098ED2B9DEBCEB110C74939BBBA1`；
- 独立真机探针采集 61 条连续记录：3 次真实选路、1 次顶栏预览、2 次真实选点、1 次保存
  继续、52 份完整视觉快照和 3,328 个坐标样本。六项事实与失败关闭规则已登记到
  `docs/api-reference-sources.md`；开发探针 artifacts 已从游戏目录卸载，正式 Mod 未替换；
- 当前版本不存在用户可操作的地图级 Zoom API；地图滚动和窗口/viewport/content scaling
  已真机覆盖，Zoom 项为 N/A。生产路线代码门禁现已通过；本轮生产代码已实现但尚未安装
  和真机验收。

## 实现范围

- 必须修改：
  - 协议升级至 schema v6，新增 `route_choice`，保留 `map_choice` 观察兼容；
  - Recommendation contract v2 新增
    `recommendation.presentation = {kind: "route_paths", ...}`，schema 必须按
    `contract_version` 和 `decision_type` 条件约束该层级；
  - Route event adapter 把真实下一节点构建为 canonical candidates；
  - checkpoint 在同一次原子替换中分别计算地图上下文和决策生命周期更新；不得使用会让
    `route_choice` 丢失地图的互斥 `if/elif`，也不得拆成两次写盘；
  - 新增确定性 `RoutePolicy`，输出每个下一节点评价和合法候选路径；公开 Overlay 只绘制
    当前推荐路径；
  - 新增 owner-scoped `RouteAdviceController` 与只读 `RouteMapOverlay`；
  - 真实选择后用同一 decision ID 关闭并清理 Drawer、Overlay 和 advice。
- 允许修改：
  - Route adapter/Policy 专用严格图校验和路线所需领域字段；不得全局收紧 `WorldState.map`
    而破坏旧 `map_choice` 或 Card 夹具；
  - Card Reward 消费端接受 Recommendation v1/v2，但已真机稳定的 Card 生产和 UI 保持 v1，
    不随路线重写；
  - 版本化地图夹具、回放脚本、性能与协议审查脚本。
- 必须补充的测试：
  - v1-v5 兼容与 v6 schema 正反例；
  - 预览地图不创建决策、不清除当前 Card advice；
  - Route opened/updated/closed、保存恢复、跨 Run、过期 parent、非法候选；
  - 图重复节点、未知边、环、Boss 不可达和真实边路径校验；
  - 只绘制当前推荐路径，只有一个候选时也不伪造第二条线；
  - owner/handle 旧界面隔离、错配 advice 清线、坐标重定位源码契约；
  - 固定多维局面可复现、无网络、无决策历史、P95 ≤ 300 ms。

### 生产实施顺序

实现角色必须按下面顺序推进，每一阶段先让对应自动测试通过，再进入下一阶段；不得从 UI
倒推协议或为赶进度复制平行生命周期：

1. schema v6、Recommendation v2、Route adapter 与一次原子 checkpoint；
2. 确定性 RoutePolicy、严格图校验、SQLite 风险 profile 与固定场景；
3. 通用 Mod pending decision、真实 Route Observer、owner-scoped Drawer 控制；
4. `RouteMapOverlay`、当前视觉节点重绑定、滚动/窗口 transform 重定位和清理；
5. 完整 Python、协议、性能、Mod 编译和差异审计。安装与真机验收仍由 Codex/用户执行。

### 通用生命周期硬约束

- Route 必须通过统一 event adapter/registry 进入现有 `_process_decision`、PolicyRegistry、
  lifecycle 与 advice disposition；不得新增 route 专属 processor 或 file bridge 主流程；
- Mod 侧只允许一个通用 pending decision handle。必须把现有 Card 专用 pending/recovery
  收口为通用结构，禁止新增平行 `_pendingRouteDecision`；
- Python 把 Mod 提供的 `decision_id` 当作不透明稳定身份，不自行按 Run/Act/origin 猜 ID；
- `map_choice` 永远只做观察，不注册 RoutePolicy，也不清理仍有效的 Card advice；
- `route_choice` 的候选只来自 `available_next_node_ids`，`candidate_id = node_id`，Card
  `options` 必须为空。

## 路线评分基线

先实现透明的有界图搜索，不引入学习模型或 LLM。输入至少包括 HP/最大 HP、牌组即时输出、
防御、AOE、成长与资源能力、遗物、药水、金币、真实节点类型、已知 Boss 和一个版本化
默认风险 profile；P1.0 不实现用户设置 UI。

普通怪物和精英只能按房间类型与真实遭遇池计算期望风险；未确认已击败精英集合 API 前
不得伪造精确敌人。图、真实候选、Boss 可达性或版本化默认风险 profile 缺失属于关键缺口：
仍返回请求中的全部 candidate assessments，但分数/rank 为 null、
`recommended_candidate_id = null`、主备路径为空并列出 `data_gaps`。未知单张牌/遗物、
已击败精英 API 缺失属于可降级缺口：允许保守评分但必须记录 `data_gaps`，不能输出假精度。

存储继续遵守既有边界：敌人机制、遭遇池和风险常量查询 SQLite；当前图和决策只更新单一
检查点与 advice；不保存路线选择历史，不调用向量库或 RAG。

## 禁止事项

- 不调用游戏选择方法、不自动点击、不修改存档或原生地图绘制；
- 不把 `map_choice` 直接注册为 RoutePolicy；
- 不复制 Card Reward 的 processor/file bridge 生命周期分支；
- 不把路径塞进旧 `advice`、`factors` 或屏幕坐标字段；
- 不引入 RAG、LLM、HyDE、Reranker、自学习、战斗逐回合建议或多人模式；
- 不复制 STS2MCP、BoberInSpire 或游戏源码实现。

路线身份语义必须保持：同一 Run、Act 与 origin 的同一次选择机会复用 decision ID，候选或
状态更新只产生新 event ID 和 `UPDATED`；到达新 origin 才创建新 decision ID。当前尚未
证明同 Act origin 不会重访，因此使用 Mod 生成 GUID，并用 Run、Act、真实 origin、规范化
候选集合和地图指纹从单一当前检查点严格恢复。Act 初始可操作地图已有真实 origin `0:3`，
必须使用真实节点；稳定可操作界面仍无 origin 时不发 `route_choice`，不得使用 sentinel、
运行时对象 ID 或 `{run_id}:route:{act}:{origin}` 拼接伪造身份。

## 预计涉及文件

- `docs/product-spec.md`、`docs/realtime-architecture.md`、
  `docs/api-reference-sources.md`：已确认设计与 API 证据；
- `protocol/state-event.schema.json`、示例、`protocol/advice-event.schema.json`、示例：v6/v2；
- `advisor/decision_core.py`、`advisor/policies.py`、新增 `advisor/route.py`：通用契约与策略；
- `realtime/protocol.py`、`processor.py`、`checkpoint.py`、`file_bridge.py`：adapter 与生命周期；
- `ProtocolModels.cs`、`StateEventWriter.cs`、`MapObserver.cs`：生产协议与地图观察；
- 新增 `RouteChoiceObserver.cs`、`RouteAdviceController.cs`、`RouteMapOverlay.cs`；
- `tests/` 与版本化地图夹具：协议、策略、生命周期、UI 契约和回放。

## 验收标准

### 自动验证

- [x] API 调研报告逐项写出类型、成员、来源、当前程序集验证和失败降级；
- [ ] schema v6、Recommendation v2、生产者、消费者、示例和测试一致；
- [ ] v1-v5 全部旧夹具兼容；
- [ ] 完整 Python 测试、固定路线场景和性能门禁通过；
- [ ] Mod 0 error 编译并生成最新 DLL/JSON/PCK；
- [ ] Git 差异无本机路径、日志、数据库、私有 tier 或第三方代码。

### 真机验证

- [ ] Act 初始真实路线选择产生 Drawer 和唯一主路线 Overlay；
- [ ] 完成一个节点后的下一次路线决策使用新 decision ID；
- [ ] 仅手动预览地图不产生伪决策；
- [ ] 点击真实节点后建议、线条和 advice 全部清理；
- [ ] 滚动和窗口/viewport/content scaling 变化后线条仍锚定真实节点；用户地图 Zoom 在
      当前 0.108.0 为 N/A；
- [ ] 收起/展开/关闭/重开不残留旧线，原生绘制内容不变；
- [ ] Route → 战斗 → Card Reward 不显示旧路线建议；
- [ ] 保存退出再继续保持同一 Run、正确 decision identity 和递增 sequence。

## Luna 实施报告

### API 调研结论

Phase A 独立开发探针已完成构建、安装、真机采集、独立审查和卸载。以下结论同时来自当前
0.108.0 程序集与会话 `d2d4c4cc4df74048bf6cb47fbeaa3b76` 的真实对象日志：

1. **真实选路与预览区分**
   - 类型/成员：`NMapScreen.Open(bool isOpenedFromTopBar)`、`IsOpen`、
     `IsTravelEnabled`、`IsTraveling`，以及视觉 `NMapPoint.State`；
   - 触发：Open/SetTravelEnabled Postfix 与 500ms 限速、仅变化时写入的 `_Process`
     Postfix；
   - 真机结论：真实选路样本为 `isOpenedFromTopBar=false`、`IsOpen=true`、
     `IsTravelEnabled=true`、`IsTraveling=false`；顶栏预览为
     `isOpenedFromTopBar=true`、`IsTravelEnabled=false`，即使候选集合仍相等也不能创建决策；
   - 生产门禁：同一 owner 上还必须满足非 debug、单人、候选非空且模型/视觉候选稳定一致；
     任一条件缺失均 fail closed，只保留地图观察。
2. **真实选择观察点**
   - 类型/成员：public `NMapScreen.OnMapPointSelectedLocally(NMapPoint point)`；
   - 触发：只读 Harmony Postfix，记录 `point.Point.coord`、选择前最近采样的 origin、Postfix
     当下 `CurrentMapPoint`，随后继续采样旅行变化；
   - 真机结论：两次点击分别返回 `1:3`、`2:3`；Postfix 当下所选视觉节点已经变为
     `Traveled`，`CurrentMapPoint` 仍为旧 origin，约 0.6–1.1 秒后的 Close 才推进；
   - 生产约束：使用回调参数与点击前缓存候选关闭当前 decision，不能点击后重算候选或等待
     新 `CurrentMapPoint`。探针和生产代码都不得调用该方法执行选择。
3. **保存、恢复和关闭时序**
   - 类型/成员：`RunManager.SetUpNewSingleplayer(RunState, ...)`、
     `SetUpSavedSingleplayer(RunState, SerializableRun)`、`Launch()`、`CleanUp(bool)`；
     `NMapScreen.Initialize/SetMap/Open/Close/CleanUp/_ExitTree`；
   - 真机结论：顶栏预览无选择关闭不改变 origin；保存退出时 map cleanup 仍有完整状态，
     随后才 `CleanUp(graceful=true)`/ExitTree；恢复早期当前位置为空、楼层为 0，`SetMap`
     后恢复 origin、候选和地图指纹，真实 Open 后才可恢复决策；
   - 生产约束：cleanup/ExitTree 撤 UI 和 owner，但 graceful 保存不结束 Run；初始化暂态不发
     决策，同一机会从唯一当前 checkpoint 严格恢复。
4. **领域节点到视觉节点映射**
   - 类型/成员：public `NMapPoint.Point` → public field `MapPoint.coord`；探针通过 Godot
     子树公开枚举取得所有实际 `NMapPoint`，不读取 `_mapPointDictionary` 私有字段；
   - 真机结论：52 份完整快照均有 64 个唯一逻辑节点，覆盖起点 `0:3`、当前节点、候选和
     Boss `16:3`；保存继续后逻辑集合不变，但全部 Godot instance ID 重建；
   - 生产约束：按当前 screen owner 的公开 `Point.coord` 每次重建映射；重复、缺失或路径节点
     无视觉映射时隐藏 Overlay，不缓存旧对象。
5. **Overlay 坐标来源**
   - 类型/成员：Godot `Control.Position/GlobalPosition/Size/Scale/GetGlobalTransform()/
     GetGlobalRect()`；public `NMapScreen.GetNetPositionFromScreenPosition` 和
     `GetScreenPositionFromNetPosition`；
   - 真机结论：3,328 个坐标样本的 screen/net 往返最大误差约 0.001 px；实际地图滚动和
     `1680x1260`、`1680x1166`、`1910x1080`、`2099x1080`、`2204x1080` viewport 变化中，
     net 坐标稳定而 `GetGlobalRect().GetCenter()` 正确移动；
   - 当前版本没有用户可操作的地图级 Zoom API，只有拖动/纵向滚动及窗口宽高比处理；Zoom
     项为 N/A。生产 Overlay 每次 redraw 从当前 owner 的视觉节点重算全局中心，不设置
     `Scale`、不缓存屏幕坐标。
6. **稳定路线机会 identity**
   - 类型/成员：public `RunState.CurrentActIndex/CurrentMapPoint/CurrentMapCoord/
     VisitedMapCoords/Map`、`MapPoint.Children`；
   - 真机结论：保存继续后 RunState runtime object 与 screen owner 均重建，但 origin `1:3`、
     候选 `[2:3]` 和地图指纹保持一致；选择后 origin 推进为 `2:3`、候选变为
     `[3:2,3:3]`；Act 初始真实 origin 为 `0:3`；
   - 生产约束：同 Act 不重访仍未证明，使用 Mod GUID，并以 Run、Act、真实 origin、规范化
     候选和地图指纹严格恢复。运行时对象 ID、screen owner、界面打开次数、sentinel 或直接
     拼接 `{run_id}:route:{act}:{origin}` 均不能作为 decision ID。

参考来源为本机 0.108.0 `sts2.dll`、项目 `Sts2ApiProbe` 和本文登记的外部线索；没有复制
第三方代码。所有 Patch 都是 Postfix，探针只读且 `affects_gameplay=false`。

### 修改内容

- 新增 `tools/RouteLiveProbe/` 独立开发 Mod：csproj、Mod initializer、Postfix patches、
  JSONL recorder、独立 manifest/PCK 工程；没有 ProjectReference 指向正式 Mod，也没有被
  正式 csproj/PCK 引用；
- JSONL 记录地图 screen/run/visual point 快照、候选差异、选择回调、坐标 transform 和
  保存恢复生命周期；写盘异常会立即置空日志目标并禁用后续记录，安全失败；
- 新增 `build_probe.ps1`、`install_probe.ps1`、`uninstall_probe.ps1`，安装/卸载只处理
  `STS2GuideRouteLiveProbe.{dll,json,pck}` 三个精确文件；真机完成后已执行卸载并核对三文件
  均不存在，正式 `STS2GuideReadOnlyExporter.{dll,json,pck}` 保留；
- 新增标准库 `validate_probe_log.py` 和 `tests/test_route_live_probe_tool.py`，验证日志结构、
  sequence、只读 Postfix 和生产打包隔离；
- 更新 `docs/api-reference-sources.md`，纠正当前版本 `IsTravelable` 并非 public getter，并
  登记探针边界与六项真机结论；
- `.gitignore` 增加开发工具 artifacts 与 Godot 导入缓存，不提交构建产物。
- 升级 `realtime/protocol.py` 与两个 JSON Schema 至 v6；新增 `route_choice`、真实
  `origin_node_id`，并保留 v1-v5 消费兼容；新增 `protocol/route-event.example.json`。
- `advisor/decision_core.py` 增加路线契约常量、WorldMap origin/node_count 和
  Recommendation contract v2；`advisor/route.py` 实现严格图校验、SQLite
  `route_risk_profile_v1`、有界确定性候选路径评分和关键缺口 fail-closed。
- `realtime/processor.py` 将 `route_choice` 通过统一 `_process_decision`、PolicyRegistry、
  lifecycle 与 advice disposition 处理；`realtime/checkpoint.py` 在同一次原子替换中
  独立更新 map_context 与 current_decision，避免 route event 丢地图。
- `storage/relational.py` 增加结构化 mechanic constant 查询；`data/knowledge.json` 增加
  版本化 `route_risk_profile_v1`，不把当前局图或决策写入历史表。
- Mod 新增 `RouteChoiceObserver.cs`、`RouteAdviceController.cs`、`RouteMapOverlay.cs`；
  只用已验证的 `NMapScreen` Postfix、`NMapPoint.Point.coord/State` 和当前 owner 视觉
  重绑定，复用唯一 `_pendingDecision`，不调用选点方法、不修改原生绘制。
- 新增 `tests/test_route_decision.py`，覆盖合法/非法图、单候选无伪造备选、v6 schema、
  v2 advice 和一次原子 checkpoint；既有协议断言同步接受 contract v1/v2。

### 验证证据

- 当前程序集签名：
  `dotnet run --project tools/Sts2ApiProbe/Sts2ApiProbe.csproj --no-restore -- <sts2.dll> ...`，
  确认上述精确方法、参数、可见性和字段类型；
- 独立 DLL 编译：
  `dotnet build tools/RouteLiveProbe/RouteLiveProbe.csproj --no-restore
  -p:STS2GamePath=<game>`，结果 0 error；未传 Godot 路径时只有“未打 PCK”的预期 warning；
- 独立 PCK 构建：`tools/RouteLiveProbe/build_probe.ps1 -Package`，Godot 4.5.1 输出
  0 warning、0 error，并明确输出“Nothing was installed”；
- 隔离/只读/validator 测试：
  `D:\miniconda3\envs\sts2\python.exe -m unittest tests.test_route_live_probe_tool -v`，
  3/3 通过；
- 三个 PowerShell 脚本通过 PowerShell AST 语法解析；
- 真机 JSONL 共 61 条连续记录；validator 汇总：4 次 Open（3 次真实、1 次顶栏）、2 次
  selected、1 次 saved setup、3,328 个坐标样本，`opened/selected/saved/closed/coordinates`
  五类证据均为 true；14 条模型/视觉候选不一致全部位于初始化、点击旅行中或关闭后的非
  actionable 状态，反向证明生产一致性校验必须置于完整 actionable predicate 内；
- 本机 `sts2.dll` 的地图命名空间静态扫描确认没有用户地图 Zoom/Magnify/Scale API；
  `NMapScreen` 只有 scroll/drag 输入，程序集 SHA-256 为
  `51A671BFEB937271AF3E643D017396B13432098ED2B9DEBCEB110C74939BBBA1`；
- 六项 API 门禁现已通过；实施报告提交时生产代码已运行既有自动验证和 Mod 编译，但未安装
  新 artifacts、未启动游戏、未执行路线真机验收，故当时只提交“待审查”，未宣称路线验收；
  后续生产独立审查结论见本文末尾。
- 完整 Python：`191/191 OK`（含新增路线测试）；Mod：`dotnet build ... --no-restore`
  `0 error / 1 个预期 warning`（未配置 Godot，因此未打 PCK）；`git diff --check` 通过。
- 本轮明确未执行 `scripts/install_p0_mod.ps1`、任何安装脚本、Host 启动或游戏启动；生产
  artifacts 未替换游戏目录。

### 范围外发现

- 当前 `NMapPoint.IsTravelable` getter 实为 protected，与此前文档“全部 public”不一致；
  Phase A 仅用精确 PropertyInfo 对照；真机已证明公开 `State` 足以参与稳定候选门禁，生产
  实现必须使用公开 `State`，不得带入 protected 反射；
- Harmony Postfix 无法取得方法入口瞬间的严格 `CurrentMapPoint`。探针如实记录“选择前最近
  采样”和“Postfix 当下/后续采样”，没有为满足报告而加入 Prefix；
- Phase A 探针本身未修改 schema、`realtime/`、`advisor/`、生产 `MapObserver`、Drawer 或
  Overlay，也未调用任何选点/旅行方法；本轮生产实现使用独立文件和已验证 Postfix API。

## Sol 独立审查

### 结论

Phase A 探针代码与真机证据独立审查通过；六项 API 门禁已按保守身份方案收口，TASK-007
可以进入生产纵向切片。该结论只授权本任务范围内的路线实现，不代表路线功能已完成。

### 证据与问题

- 主审查会话独立检查了完整探针源码、manifest、构建/安装/卸载脚本、validator、测试和
  正式打包引用；正式 Mod/PCK/EXE 不包含 `RouteLiveProbe`；
- 所有 Harmony patch 都是 Postfix；源码没有调用 `TravelToMapCoord`、`EnterMapCoord`、
  `OnMapPointSelectedLocally` 或 `NMapPoint.OnSelected`，manifest 为
  `affects_gameplay=false`；
- 发现并修复一项安全失败问题：首次 JSONL 写入失败后原实现仍会周期性重试；现已立即将
  `_logPath` 置空，并增加静态回归断言；
- `tests.test_route_live_probe_tool`：3/3；完整 Python：183/183；PowerShell AST：通过；
  `git diff --check`：通过；
- 主审查会话重新运行 `build_probe.ps1 -Package`：Godot/.NET 0 warning、0 error，日志明确
  `Nothing was installed`；
- 主审查后在游戏关闭状态使用 `install_probe.ps1` 安装并逐项核对哈希；只新增三个精确
  probe artifacts，正式 Mod 未被替换；采集后已使用精确卸载脚本移除探针 artifacts；
- 独立审查复算 61 条日志：真实/预览状态、两次选点先后、保存恢复暂态、64 节点映射、
  滚动/窗口 transform 与保守 GUID identity 均有直接证据；
- 审查指出初始 origin sentinel 与真机 `0:3` 冲突，现已改为始终使用真实 origin，空值时
  fail closed；同时确认当前版本无用户地图 Zoom，故该项为 N/A；
- 生产实现必须保留单一通用 pending decision、一次原子 checkpoint、公开 `State`、当前
  owner 视觉映射和点击前候选校验，不得把探针反射或运行时对象 ID 带入生产。

## Sol 生产实现独立审查（2026-07-15）

### 结论

不通过，任务退回“需修复”。六项 API 门禁仍有效，但首轮生产实现没有忠实落实门禁和
P1.0 验收标准；在返修审查通过前不得生成/安装生产 artifacts，也不得开始路线真机验收。

### 阻断问题

1. **v6 协议不是可执行事实**：`state-event.schema.json` 会接受 v1-v5 的
   `route_choice`，也会接受空 `map_context`、空候选和缺失 `options`；route advice 还没有
   v2 示例。`advice-event.schema.json` 未关联顶层 `event_type` 与 recommendation 类型，且
   `paths` 只是任意 object。产品规格使用 `primary_path_node_ids` /
   `backup_path_node_ids`，生产者、schema 和 Mod 却使用 `primary_path` / `backup_path`。
2. **Card v2 消费兼容未实现**：Python `Recommendation`、advice schema 与
   `CardRewardAdvicePanel` 都拒绝 Card Reward contract v2，和本任务“消费者接受 v1/v2、
   生产仍发 v1”的迁移边界冲突。
3. **搜索无界且已超过性能门禁**：`RoutePolicy` 枚举并保存所有到 Boss 的 DFS 路径。独立
   使用每节点最多两个真实边的 34 节点合法 DAG 复测三次为 435.8/415.8/432.4 ms，单次已经
   超过 300 ms；现有测试只在 3 节点单路径上执行 5 次并取最大值，不是 P95。
4. **多维评分基线未完成**：当前只读取卡牌静态 damage/block、HP、金币和节点类型；AOE、
   成长/抽牌/能量资源、遗物、药水、Boss 身份以及普通怪/精英遭遇池均未进入评分，未知
   遗物/药水也不记录缺口。伤害能力基本是所有候选的相同常量，不能实质改变排序。
5. **关键缺口没有整体 fail closed**：单个真实候选无法到达 Boss 时仍会推荐另一个候选；
   只有 `version` 的畸形风险 profile 仍会借隐式默认值输出分数。三候选时备选路径取请求顺序
   中首个非主候选，而不是第二优合法候选。
6. **生产可行动谓词缺失**：`RouteChoiceObserver` 没有保存并校验
   `isOpenedFromTopBar=false`，也没有枚举当前 owner 的视觉
   `NMapPoint.State == Travelable` 与模型候选做稳定集合比对。`MapNodeReader` 在真实 origin
   为空时仍用首个 `startMapPoints` 伪造 origin，并把同一集合同时当候选。
7. **稳定 identity/UPDATED/重试不成立**：观察指纹忽略完整地图与 WorldState，writer 的
   same-decision 只比较 owner 和候选，checkpoint 恢复不校验地图指纹。Observer 还在事件
   写入成功前记录 fingerprint，瞬时写盘失败后同一机会不会重试。
8. **错配 advice 未失败关闭**：`RouteAdviceController` 不校验顶层 sequence/status、
   recommendation `world_sequence`、候选数量、全量匹配和 eligible；非法候选只被跳过，默认
   空行与 Overlay 仍可继续渲染。
9. **Overlay 会绘制非法路径**：当前只把 advice 中的节点 ID 连成线，不验证首节点、真实边、
   Boss 终点或主备首节点差异；重复视觉节点被静默取第一个而不是隐藏。重锚由 100 ms timer
   驱动，也没有达到规格要求的逐帧 transform 重算。
10. **Mod 生命周期健壮性不足**：旧 owner 的迟到选择即使被拒绝也会清全局 fingerprint；新增
    Route Postfix 没有统一异常隔离，捕获、文件或 UI 异常可能从 Postfix 冒泡到游戏。
11. **自动验收覆盖不足**：新增路线测试只有 8 项，未覆盖预览保留 Card advice、
    OPENED/UPDATED/CLOSED、保存恢复、跨 Run、过期 parent、非法候选、完整图反例、旧 owner、
    错配清线、坐标重绑、无网络/无历史和真实规模 P95；v1/v4 state 夹具证据也缺失。

### 独立验证结果

- 完整 Python：191/191 通过，但只表示现有断言通过，不能关闭上述漏测和反例；
- Mod：0 error、1 warning；DLL/JSON 已编译，因 Godot 未配置跳过 PCK，自动验收未完成；
- `git diff --check` 通过；`.env`、数据库、日志、私有 tier、EXE 和构建目录均保持 Git 忽略，
  未发现常见 API Key/私钥或本机游戏路径进入待提交文件；
- 本轮没有安装 Mod、没有启动 Host，也没有启动游戏。

### 返修顺序

1. 先收紧 state/advice schema、统一 v2 字段名和 Card v1/v2 消费契约，并补齐正反示例；
2. 再实现完整 actionable predicate、真实 origin、包含地图指纹的 route opportunity identity、
   commit 后指纹与失败重试；
3. 将 RoutePolicy 改为有明确扩展上限的搜索，补齐多维能力、遭遇池与严格风险 profile；
4. 对 advice 和 Overlay 做全量身份/候选/真实边/Boss/视觉映射失败关闭及逐帧重锚；
5. 补齐任务单列出的自动场景，以真实规模至少 100 次计算 P95；完整测试、最新 PCK 和 Git
   审计全部通过后，才能再次提交独立审查。

## 返修实施报告（2026-07-16）

### 实施范围与结果

本次仅落实上述生产审查的返修顺序，未增加 Neow、商店、篝火或其他决策类型，也没有安装
Mod 或启动 Host/游戏。

1. **协议与契约**：将 schema v6 的 `route_choice` 收紧为必须显式携带
   `decision_id`、`options=[]`、完整 MAP decision、完整地图上下文和
   `boss_encounter_ids`；Pydantic 同样用字段出现性拒绝默认值绕过。Advice schema 将顶层
   `event_type` 与推荐类型绑定，路线 v2 使用
   `primary_path_node_ids` / `backup_path_node_ids` / 类型化 `paths`，Card Reward 保持
   v1/v2 消费兼容。补充 v1-v5 夹具、v6 反例和 route advice v2 示例。
2. **Observer 与 identity**：路线只在两次稳定的真实可行动 predicate 后生成，核对真实
   origin、模型候选与视觉 `Travelable` 集合；观察指纹覆盖完整导出 State、地图上下文和
   节点语义指纹。写入或绑定失败、predicate 失效、旧 owner 回调、选择关闭写入失败和异常
   都会立即隐藏 Route Drawer/Overlay，但不破坏 Writer 的恢复/重试状态；恢复同时核对
   sequence、Run、地图和机会身份。
3. **RoutePolicy**：DFS 改为带扩展上限的记忆化 DAG 搜索；任何候选不可达、非法图或坏
   profile 都返回空推荐。评分纳入真实升级/多段伤害、AOE、成长/资源、遗物、一次性药水、
   HP、金币、Boss 机制和普通/精英遭遇池；A8 HP、A9 伤害阈值分别读取。HP 缺失失败关闭，
   金币和未知静态实体明确降级，不伪造精度。
4. **Advice 与 Overlay**：全量校验 envelope、sequence、候选集合、eligible、推荐 ID 和路线
   拓扑；同 owner 的新 decision 或候选形状改变会先销毁旧 Drawer/Overlay，只有同一 decision
   和相同行形状才复用。Overlay 逐帧重新锚定，并在非法边、Boss 终点、视觉映射重复或 advice
   错配时清线。
5. **自动反例**：路线专项现在覆盖 v1-v5 兼容、v6 正反契约、OPENED/UPDATED/CLOSED、保存
   恢复、跨 Run 迟到关闭、原子 checkpoint、图/Boss 反例、多维因素、无网络/无历史，以及
   64 节点图和事件到 advice 写回各 100 次的 P95 ≤ 300 ms 门禁。当前专项为 24 项。

### 已执行验证

- `D:\miniconda3\envs\sts2\python.exe -m unittest discover -s tests -v`：
  `207/207 OK`，44.103 s；`jieba` 的 `pkg_resources` 弃用警告为既有第三方警告，不影响
  测试结果。
- 使用本机 Godot 4.5.1 执行
  `dotnet build mod/STS2Guide.ReadOnlyExporter/STS2Guide.ReadOnlyExporter.csproj --no-restore`：
  DLL、JSON 和工作区 PCK 均生成，`0 warning / 0 error`。
- `git diff --check` 通过；6 个 `protocol/*.json` 文件均已成功解析，工作区 DLL、JSON、PCK
  三件套存在。Git 状态未列出 artifacts；行尾转换与用户全局 ignore 目录无读取权限的提示不
  影响 diff 检查结果。

### 未宣称完成的事项

- 没有运行 `scripts/install_p0_mod.ps1` 或任何安装命令；新 PCK 没有复制进游戏目录。
- 没有启动 `python -m realtime.host`、游戏或 MCP/探针工具。
- 本实施报告写入时尚未完成独立代码审查；后续审查结论见下文。无论自动审查结果如何，路线
  真机回归仍未完成，P1.0 也没有交付完成。

## Sol 返修独立自动审查（2026-07-16）

### 结论

通过自动层门禁：2026-07-15 审查列出的 11 项阻断均已针对实际代码、协议、反例和构建结果
复核关闭。TASK-007 改为“待真机验收”，**不是**“已验收”；新 artifacts 仍未安装，游戏与
Host 均未启动。

### 复核证据

- 重新执行完整 Python 回归：`207/207 OK`；路线专项 `24/24 OK`。P95 测试是 64 节点
  synthetic DAG 和预写事件的本地文件桥，不表示游戏真机端到端延迟。
- v1-v5 夹具、schema v6 正反例、route/Card advice v1/v2 示例以及 Pydantic runtime parity
  均通过；直接构造 `Recommendation` 也不能绕过路线 typed-path 契约。
- 最新 Mod/PCK 在 owner 隔离、失败关闭修复后重建，DLL/JSON/PCK 三件套存在，
  `0 warning / 0 error`；只写入工作区 artifacts。
- `git diff --check`、协议 JSON 解析和 Git 上传边界检查通过；无 Harmony Prefix、自动选择
  调用、探针代码或本机私密数据进入正式 Mod 路线。

### 真机剩余门禁

安装后仍需逐项完成：Act 初始可选路线的 Drawer/唯一主路线、新 origin 生成新 decision、仅预览
不创建决策、点击后清理、滚动及窗口/viewport 重锚、重开无残线、Route → Card Reward 不残留、
保存继续后 sequence/identity 恢复。任何一项失败均退回“需修复”。

## 2026-07-30 运行时返修审查

### 结论

2026-07-16 的自动审查结论不再足以进入真机验收，任务退回“需修复”。本机游戏已从
`0.108.0` 更新为 `0.109.1`（commit `c8c577f6`），当前 `sts2.dll` SHA-256 为
`016C6DF717D997FCBD8F2A55102CA63CB4A89C1FA8E4DB8DACFDDF803B6B70E1`。旧版本六项
API 真机证据不能自动继承；生产 Host 当前也不会拒绝未验证游戏版本。

独立复核重新运行完整 Python `207/207 OK`、实时与路线专项 `82/82 OK`，Mod/PCK
`0 warning / 0 error`。这些绿灯只证明现有断言通过；下面反例均不在测试覆盖内。

### 阻断问题

1. **没有版本失败关闭**：Mod 只把 `game_version` 写入事件，Host 接受缺失或任意版本；
   `0.109.1` 尚未重新完成 API/真机门禁却仍会产出建议。缺少统一的 Guide/Game/Mod/
   protocol/SQLite/policy 兼容清单。
2. **v6 Card 保存继续 identity 失效**：生产事件默认 schema v6，但 Mod 的
   `ReadResumableDecisionId` 仍硬性要求 schema v5。保存继续会生成新 decision ID，而 Host
   可能仍恢复旧 active decision。
3. **新 Run 首写失败不可恢复**：Mod 在原子写成功前递增 sequence；新局 sequence 1
   瞬时写盘失败后，下一成功事件从 2 开始，而 Python 只允许 sequence 1 接管不同 Run
   的旧 checkpoint。
4. **权威快照语义不成立**：当前 event state 大量字段可默认缺失；Card 事件没有地图时，
   Processor 会把新 state 与 checkpoint 中旧 map_context 混合。玩家选路后下一次 Card
   Reward 可能仍按上一个 origin 和候选分支计算路线压力。
5. **JSON Schema/Pydantic 类型不一致**：`sequence: "1"`、`decision.can_skip: 0`、
   `state.hp: "70"`、`map_context.node_count: "64"` 均为 Schema 拒绝而 Pydantic 接受。
6. **BeginRun 合成 abandon 不事务**：旧局 `run_ended` 写入失败后仍无条件切换 Run identity
   并清理旧 state/pending，最终摘要和清理事件无法重试。
7. **Run identity 仍有假稳定风险**：BeginRun 优先使用可能陈旧的 History seed 并直接标记
   stable，后续 current-player seed 无法纠正。
8. **Card advice 和 advice schema 仍偏松**：Card 消费端未完整校验 publish disposition、
   推荐候选、分数范围；advice schema 的部分跨字段关系仍只靠消费者。

### 返修顺序

1. 新增唯一兼容清单和生产版本门禁；默认拒绝 `0.109.1`，直到新程序集静态探针、只读
   真机探针和选牌/路线闭环全部通过；
2. 统一 v6 Card resume 读取，增加可执行的保存继续夹具/测试，保证同一 decision ID、
   新 event ID 和 `UPDATED`；
3. 将 sequence 作为成功提交后的事实；注入首次原子写失败后，下一成功事件仍能成为新
   Run 的第一个可接管事件；
4. 把完整快照与通知语义写入下一协议修订，引入 `state_revision` 或等价严格事实；禁止
   新 state 与旧地图静默合并。路线选择后下一 Card 要么携当前地图，要么明确地图缺失并
   降级，不能沿用旧 origin；
5. 让 Pydantic 使用与 JSON Schema 等价的严格类型，并为字符串数字、`0/1` 布尔、
   NaN/Infinity 和关键字段缺失补 differential tests；
6. 事务化 BeginRun：旧局结束写入失败时保留旧 identity/state/pending 并可重试；
7. 交叉确认 current RunState seed 后再提交稳定 identity；补强 Card consumer 与 advice
   schema 的失败关闭；
8. 完整测试、真实 C#/Mod 夹具、PCK 构建和差异审查全部通过后，重新安装并按当前
   `0.109.1` 完成路线真机验收。开发探针验收后必须卸载。

本轮不得增加 Neow、商店或篝火。产品规格已经记录后续 Public Beta 的单路线和路线软
偏好，但应在上述运行时基础稳定后实施，不得用 UI 扩展掩盖协议阻断。

## 2026-07-30 运行时返修实施报告

### 实施结果

本轮严格按照上一节返修顺序完成运行时基础修复，没有增加 Neow、商店、篝火或其他决策
类型，也没有安装 Mod、启动 Host 或游戏：

1. **唯一兼容事实**：`packaging/compatibility.json` 统一声明 Guide、Game、Mod、生产
   protocol、SQLite 和 policy 版本，并由规范化内容计算 `release_fingerprint`。该指纹
   已贯穿 v7 状态事件、checkpoint、advice 和 C# 显示端；缺失、错误或跨发布混装都失败
   关闭。清单状态不参与指纹，因此同一套已验证 artifacts 可以从 pending 提升为 enabled。
2. **协议和提交语义**：生产 schema 升级为 v7 完整快照，Python 继续只为离线夹具兼容
   v1-v6；事件携带成功提交 revision、生产者和程序集身份。sequence 只在原子写成功后
   提交；BeginRun 切换、旧局结束和新局首写按事务处理，失败时保留可重试事实。
3. **身份与恢复**：Card resume 与 Route opportunity 使用当前 RunState、真实 origin、
   规范化候选集合和地图指纹。旧 release checkpoint/result 不恢复生命周期，也不能被
   当前 Host 重新贴兼容标记；同身份当前事件会重新计算并原子替换。
4. **策略与发布契约**：RoutePolicy 使用有界记忆化搜索并只发布一条当前主路线；Card 与
   Route recommendation 均校验状态/推荐 ID、候选 rank/eligibility、多维因子、缺口和
   有限数值。C# Drawer/Overlay 在显示前再次核对发布指纹、envelope、候选和完整
   recommendation 元数据。
5. **当前程序集静态复核**：在本机 `0.109.1` 程序集上重新确认
   `RunManager.DebugOnlyGetState()`、`RunState.CurrentMapPoint/CurrentMapCoord/Map`、
   `NMapScreen` 的公开状态/选择/坐标 API、`NMapPoint.Point/State` 以及
   `MapPoint.coord/Children` 仍存在。静态存在不等于真机时序已经通过。

### 验证证据

- 完整 Python：`245/245 OK`；
- Mod 与 PCK：Godot 4.5.1，`0 warning / 0 error`；
- 协议 JSON、PowerShell AST、`git diff --check` 和上传边界审计通过；
- 最新 v7/fingerprint 正式三件套只在工作区生成，未安装；
- 游戏目录中仍安装隔离 RouteLiveProbe，游戏与 Host 当前未启动；
- 兼容清单保持 `pending_validation`，没有用自动绿灯冒充真机验收。

### 剩余门禁

1. 在 `0.109.1` 上运行隔离 RouteLiveProbe，逐项复核真实选路、顶栏预览、选点、保存继续、
   地图滚动、Zoom（若可操作）和窗口/viewport 变化；
2. 探针证据通过后卸载三个 probe artifacts，再将唯一兼容清单提升为 `enabled` 并重建；
3. 游戏关闭时安装正式三件套，启动唯一 Host，完成 Card/Route、保存继续、放弃与新局的
   端到端回归；
4. 任一真机项失败即退回“需修复”，不得进入 Neow、商店或篝火。

## 2026-07-30 运行时返修独立自动审查

### 结论

通过自动层门禁，可以进入 `0.109.1` 隔离真机探针；这不是正式功能验收。独立审查未发现
新的 P0/P1 代码阻断，唯一阻止当前后台发布建议的是预期中的
`compatibility.status=pending_validation`。

### 复核事实

- 本机 `0.109.1` 的 `sts2.dll` 哈希与兼容清单完全一致；
- v7 `release_fingerprint` 已贯穿 C# 生产、恢复、Python checkpoint/replay/processor、
  advice marker 以及 Card/Route 消费端，未发现旧结果重新贴标或跨版本混装绕过；
- sequence 仍只在队列原子写成功后提交；BeginRun 合成 abandon 失败时保留旧身份、状态
  和待处理决策以便重试；
- advice schema 和共享 C# contract reader 在影响生产建议正确性的字段上保持一致；
- 定向兼容性、协议 parity 与运行时提交契约 `37/37 OK`；完整 Python `245/245 OK`；
  Mod/PCK `0 warning / 0 error`。

不得在探针证据通过前手工绕过门禁。探针通过后的最小变更是提升唯一清单状态、重新构建，
再安装完全一致的正式 artifacts。

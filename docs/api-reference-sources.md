# STS2 API 参考源与使用边界

> 更新时间：2026-07-31
>
> 目的：为 Mod 侧 API 调研提供固定入口，减少对 STS2 内部类型、字段和生命周期的猜测。

本文件登记可用于确认 STS2 真实 API 和事件时机的外部参考源。它们只能作为
“API 事实”和“真机观测方法”的参考，不能覆盖 [`product-spec.md`](product-spec.md)
定义的只读产品边界。

## 使用原则

1. 优先从当前本机游戏程序集、真机日志和真实对象验证 API。
2. 外部仓库只用于辅助确认：
   - 类型名称；
   - 成员名称；
   - 生命周期和触发顺序；
   - 其他 Mod 已验证过的读取点。
3. 不复制游戏源码、第三方 Mod 代码、权重、数据文件或动作策略。
4. 不把自动点击、代打、存档修改、联网控制能力引入 P0。
5. 实施报告必须写明每个新增读取点：
   - 类型；
   - 成员；
   - 触发方法；
   - 来源文件或日志；
   - 真机验证方式；
   - 失败时的安全降级。

## 参考源

### Gennadiyev/STS2MCP（开发期真机探针）

- 地址：<https://github.com/Gennadiyev/STS2MCP>
- 角色：**开发期真机 API 探针**，仅用于观察 `get_game_state` 返回值以定位公开 API。
  不作为产品依赖，不进入打包产物。
- 已验证的 API（2026-07-08，通过 STS2MCP 调用链确认）：
  - `RunManager.Instance.DebugOnlyGetState()` → `RunState` — **公开方法**，本项目直接调用
  - 当前本机程序集中的 `RunState.Players` → `IReadOnlyList<Player>` — **公开属性**；P0
    仅支持单人模式，直接读取第一个玩家，不需要 `LocalContext` 或反射
  - `LocalContext` 的当前真实命名空间为 `MegaCrit.Sts2.Core.Context`；STS2MCP 中的
    `GetMe(RunState)` 只保留为多人模式调研线索，不进入 P0 单人生产路径
  - 参考调用点：`McpMod.MultiplayerActions.cs` / `McpMod.StateBuilder.cs`
- 可参考内容：
  - 如何判断当前游戏状态（主菜单、地图、战斗、奖励、商店、事件等）；
  - 如何读取当前 run、玩家、选牌候选、地图、奖励、profile/compendium；
  - 如何设计面向 Agent 的状态摘要；
  - 哪些游戏阶段有稳定数据。
- 禁止内容：
  - 不引入自动动作执行；
  - 不复制动作实现；
  - 不把 localhost 控制 API 作为 P0 运行依赖；
  - **不把 STS2MCP 的 DLL、JSON、MCP server、uv 配置提交进正式产品**。
- 真机观察流程：见 [`docs/dev/sts2mcp-live-probe.md`](../dev/sts2mcp-live-probe.md)

### hongyipan152/STS2SourceCode

- 地址：<https://github.com/hongyipan152/STS2SourceCode>
- 用途：辅助确认 STS2 Godot/C# 内部类型、成员和生命周期。
- 可参考内容：
  - `Player`、`RunState`、`RunManager`、`NRun` 等当前局对象关系；
  - `NMapScreen`、地图节点、真实边和当前节点读取时机；
  - `CardReward`、`NCardRewardSelectionScreen`、card row / card holder / `CardModel`
    的真实结构；
  - Neow、普通奖励、界面关闭等流程差异。
- 禁止内容：
  - 不复制游戏实现；
  - 不提交导出的游戏源码；
  - 不把未在当前本机版本验证过的源码结论直接当成生产事实。
- 注意：仓库来源非官方。使用时必须再通过当前游戏程序集或真机日志确认，尤其是
  当前本机游戏版本为 `0.108.0` 时。

#### 已确认的 Boss/地图 API（2026-07-13）

- `ActMap.GetAllMapPoints()` 只枚举普通 `Grid`，不包含特殊的 Boss 与起点节点；
- Boss 地图节点：`ActMap.BossMapPoint`、`ActMap.SecondBossMapPoint`；
- Boss 遭遇身份：`RunState.Act.BossEncounter.Id`、`SecondBossEncounter?.Id`；
- 地图节点 ID 与遭遇 ID 是两类不同事实，协议不得混用；
- 上述成员已在本机当前 `sts2.dll` 确认为公开成员，仍需安装后真机验证实际值和时机。

#### P1.0 路线 API 线索与当前验证

STS2MCP 的公开源码给出以下精确线索：

- `NMapScreen.Instance` 可取得当前地图界面；
- 实际可视 `NMapPoint` 的 `State == MapPointState.Travelable` 被用于识别当前可走节点；
- `NMapScreen.OnMapPointSelectedLocally(NMapPoint)` 是本地选点动作入口，可用于反向定位选择
  完成后的安全 Postfix；
- 地图模型仍以 `MapPoint.Children` 表达真实边。

参考文件：

- <https://github.com/Gennadiyev/STS2MCP/blob/main/McpMod.Actions.cs>
- <https://github.com/Gennadiyev/STS2MCP/blob/main/McpMod.StateBuilder.cs>

这些外部内容只作为最初线索，不能直接进入产品。下面六项已进一步通过本机当前程序集与
2026-07-14 独立只读真机探针核验：

1. `Travelable` 是否只在真正需要选择路线时出现，还是手动预览地图也会出现；
2. 哪个公开方法/回调可以用 Postfix 只读观察最终选中的真实 `MapPoint`；
3. `NMapPoint` 与领域 `MapPoint` 的稳定映射键是什么；
4. 地图滚动、viewport、分辨率和窗口变化时应从哪个真实 Godot transform 计算 Overlay
   坐标；
5. 保存继续、切换场景和无选择关闭时，路线决策对象与候选集合如何变化；
6. 同一次 origin 选择机会的稳定 identity 来自哪个真实对象，以及同 Act 是否允许重访
   origin。

生产代码不得调用 `OnMapPointSelectedLocally` 执行动作；它只能作为玩家已完成真实选择后的
Harmony Postfix 观察点。上述六项的当前版本结论与失败关闭规则见本节末尾。

#### 当前程序集静态核验（2026-07-14）

针对本机 STS2 `0.108.0` 的 `sts2.dll`（9,571,328 bytes，SHA-256
`51A671BFEB937271AF3E643D017396B13432098ED2B9DEBCEB110C74939BBBA1`）运行项目内
`tools/Sts2ApiProbe`，确认以下签名；这一步没有修改游戏或生产 Mod：

| 读取目标 | 当前程序集事实 | 结论 |
|---|---|---|
| 地图屏幕实例/状态 | `NMapScreen.Instance`、`IsOpen`、`IsTravelEnabled`、`IsTraveling` 均为 public property | 与 `Open(bool isOpenedFromTopBar)` 参数及稳定候选集合共同构成可操作门禁，不能单独使用 |
| 真实选点回调 | `NMapScreen.OnMapPointSelectedLocally(NMapPoint)` 为 public method，返回 `void` | 可作为只读 Harmony Postfix 观察点；生产代码不得调用 |
| 节点状态 | `NMapPoint.Point`、`State` 为 public property；`IsTravelable` 的 getter 在当前程序集为 protected，`State` 使用 `MapPointState.Travelable/Traveled/Untravelable/None` | 生产候选应优先依赖公开 `State`；Phase A 探针仅用单一精确 `PropertyInfo` 对照 protected 值，失败返回 null，不得扩展成宽泛反射 |
| 领域图与稳定键 | `MapPoint.coord` 为 public field，`Children` 为 public property，类型为 `HashSet<MapPoint>` | `coord` 是领域/视觉映射键；同 Act 不重访仍未证明，因此不能直接拼接 decision ID |
| Overlay 坐标 | `NMapScreen.GetScreenPositionFromNetPosition(Vector2)` 与反向方法均为 public | 可复用游戏变换计算，生产 Overlay 从当前 owner 的 `NMapPoint.GetGlobalRect().GetCenter()` 重算，不使用截图坐标 |

本次核验命令：

```powershell
dotnet run --project tools/Sts2ApiProbe/Sts2ApiProbe.csproj --no-restore -- <sts2.dll> MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapScreen MegaCrit.Sts2.Core.Nodes.Screens.Map.NMapPoint MegaCrit.Sts2.Core.Map.MapPoint MegaCrit.Sts2.Core.Map.MapPointState
```

静态核验本身不代表路线功能完成；首轮协议、策略、生产 Mod、Drawer 和 Overlay 已提交，
但 2026-07-15 独立审查未通过，必须先返修，之后才可进入安装和真机验收。

#### `0.109.1` 兼容性重新门禁（2026-07-30，进行中）

本机 Steam 游戏已自动更新：

- `release_info.json`：version `v0.109.1`、commit `c8c577f6`、发布日期
  `2026-07-20T16:59:48-07:00`；
- 当前 `sts2.dll` SHA-256：
  `016C6DF717D997FCBD8F2A55102CA63CB4A89C1FA8E4DB8DACFDDF803B6B70E1`；
- 旧 `0.108.0` 哈希和真机日志只保留为历史证据，不能用于宣称新版本兼容。

项目内 `tools/Sts2ApiProbe` 已针对新程序集重新运行，静态确认生产路线仍引用的关键成员
存在：public `RunManager.DebugOnlyGetState()`、`RunState.CurrentMapPoint/
CurrentMapCoord/Map`、`NMapScreen.Open(bool)`、`IsOpen/IsTravelEnabled/IsTraveling`、
`OnMapPointSelectedLocally(NMapPoint)`、screen/net 坐标转换、public
`NMapPoint.Point/State`、public `MapPoint.coord/Children`。正式 Mod 与隔离 RouteLiveProbe
也都针对该程序集和 Godot 4.5.1 完成 `0 warning / 0 error` 构建。

这些结果只证明类型和签名仍存在，不证明触发时序、可行动 predicate、保存恢复、地图
视觉映射或坐标行为不变。隔离 RouteLiveProbe 已在游戏关闭时安装，下一步必须用
`0.109.1` 真机重新覆盖真实选路、顶栏预览、选点、保存继续、滚动和窗口变化；完成后
卸载探针。生产兼容清单在真机门禁完成前必须把 `0.109.1` 标为 pending/disabled，Host
不得输出正式建议。

#### `0.110.1` 兼容性重新门禁（2026-08-01，进行中）

启动隔离路线测试时 Steam 自动升级了本机游戏：

- `release_info.json`：version `v0.110.1`、commit `db5d3552`、发布日期
  `2026-07-31T01:18:29-07:00`；
- `sts2.dll` SHA-256：
  `7C446EFABF80614C429B5088E87101423AA5BB4C04FC3E73393261F6E6D404FD`；
- Loader 真机日志确认旧 RouteLiveProbe 无法解析 Harmony target
  `NMapScreen._ExitTree`；测试用 STS2MCP 无法加载已经移除的
  `MegaCrit.Sts2.Core.Entities.Multiplayer.LobbyPlayer`。

本机程序集元数据进一步确认 `NMapScreen` 当前不再声明 `_ExitTree()`，但仍声明公开
`_Notification(int)`、`Close(bool)`、`CleanUp()`、`Open(bool)`、`SetTravelEnabled(bool)`、
`OnMapPointSelectedLocally(NMapPoint)` 和 `_Process(double)`。其他当前使用的 Card Reward、
Merchant、Rest、Event 与 Card Grid owner 仍各自声明 `_ExitTree()`；因此只允许修复被精确
证据确认失效的 Map owner target，不得批量猜测生命周期变化。

`STS2SourceCode` 仅辅助理解旧实现语义；其当前仓库仍含旧 `LobbyPlayer` 和
`NMapScreen._ExitTree()`，不能当作 `0.110.1` 二进制事实。下一步必须先让隔离探针针对当前
程序集成功加载，再完成真实路线时序与视觉验证。旧探针和 STS2MCP 测试文件均已卸载，
正式 Guide 未安装，Host 未启动；兼容清单继续失败关闭。

#### Phase A 独立真机探针（2026-07-14，已完成并卸载）

`tools/RouteLiveProbe` 已提供与正式 Mod/PCK 完全隔离的开发期只读 Mod：

- 独立程序集、manifest 和 PCK 名称均为 `STS2GuideRouteLiveProbe`；正式
  `STS2Guide.ReadOnlyExporter.csproj` 和 PCK 不引用它；
- 只注册 Harmony Postfix，观察当前程序集公开的 `NMapScreen.Initialize/SetMap/Open/
  SetTravelEnabled/OnMapPointSelectedLocally/Close/CleanUp/_ExitTree/_Process`，以及
  `RunManager.SetUpNewSingleplayer/SetUpSavedSingleplayer/Launch/CleanUp`；
- 通过 `NMapPoint.Point.coord` 把视觉节点映射为领域节点 ID；记录公开 `State`，并仅用精确
  定位的 protected `IsTravelable` getter 做调研对照；
- 通过 Godot 继承的 `Position`、`GlobalPosition`、`Size`、`Scale`、
  `GetGlobalTransform()`、`GetGlobalRect()` 及地图 screen/net 转换方法记录节点中心和坐标
  往返；这些成员已经通过当前游戏引用编译，正确 Overlay 锚点由下方真机证据确认；
- JSONL 写入 `%LOCALAPPDATA%/STS2Guide/dev/route-probe/`，不记录 seed、玩家名、存档正文、
  牌组或绝对游戏路径；失败时停止探针记录，不影响游戏；
- `validate_probe_log.py` 只校验 JSONL、连续 sequence 和汇总证据，不会自动宣告六项门禁
  通过。

探针已用本机 0.108.0 引用编译并由 Godot 4.5.1 生成独立 PCK；主审查后使用独立脚本安装
三个精确命名的 probe artifacts，正式 Mod 未被替换。真机采集完成后，三个 probe artifacts
已从游戏 `mods` 目录卸载，正式 Mod 仍保留；探针源码继续位于 `tools/`，不进入正式
PCK/EXE。具体动作和验证命令见 `tools/RouteLiveProbe/README.md`。

#### 六项真机门禁结论（2026-07-14）

证据来自单人新局会话 `d2d4c4cc4df74048bf6cb47fbeaa3b76` 的 61 条连续 JSONL 记录。
`validate_probe_log.py` 通过：4 次地图 Open（3 次真实选路、1 次顶栏预览）、2 次真实选点、
1 次保存继续、52 份完整视觉快照和 3,328 个节点坐标样本。原始日志只保留在本机开发
目录，不提交、不打包，也不成为运行依赖。

1. **真实选路与预览**：真实选路必须在同一 screen owner 上同时满足
   `isOpenedFromTopBar=false`、`IsOpen=true`、`IsTravelEnabled=true`、
   `IsTraveling=false`、`IsDebugTravelEnabled=false`、单人局、候选非空且模型候选与视觉
   `State == Travelable` 集合稳定一致。顶栏预览样本虽然仍显示模型/视觉下一节点，但
   `isOpenedFromTopBar=true` 且 `IsTravelEnabled=false`，因此候选存在或集合相等都不能单独
   创建决策。
2. **真实选择回调**：两次点击分别精确触发 public
   `NMapScreen.OnMapPointSelectedLocally(NMapPoint)` Postfix，参数节点为 `1:3`、`2:3`。
   Postfix 当下所选视觉节点已经变为 `Traveled`，但 `CurrentMapPoint` 仍是旧 origin；约
   0.6–1.1 秒后的 `Close` 才推进当前位置。生产关闭必须使用回调参数与点击前缓存候选校验，
   不能在点击后重算候选或等待新 `CurrentMapPoint` 才识别选择。
3. **保存、恢复与无选择关闭**：顶栏预览关闭未产生选点且 origin 不变。保存退出先出现仍
   带完整 `1:3 -> 2:3` 状态的 map cleanup，随后才是 `CleanUp(graceful=true)` 与 ExitTree；
   恢复早期 `SetUpSavedSingleplayer` 的当前位置暂为空、楼层为 0，`SetMap` 后才恢复 origin、
   候选与原地图指纹，真实 Open 后才可重新创建/恢复决策。清理回调只撤 UI/owner，不把
   graceful 保存误判为本局结束。
4. **领域/视觉映射**：稳定快照均包含 64 个唯一 `NMapPoint.Point.coord`，覆盖起点 `0:3`、
   当前节点、候选和 Boss `16:3`；保存前后逻辑节点集合相同，但 64 个 Godot instance ID
   全部重建。生产必须按当前 owner 的公开 `Point.coord` 重新建立映射，不缓存旧场景对象；
   重复、缺失或路径节点无法映射时隐藏 Overlay。
5. **Overlay 坐标**：对节点中心执行 screen/net 往返的最大误差约 0.001 px；实际覆盖地图
   滚动以及 `1680x1260`、`1680x1166`、`1910x1080`、`2099x1080`、`2204x1080` 等 viewport
   变化。同一节点逻辑 net 坐标稳定，`GetGlobalRect().GetCenter()` 随滚动和窗口变化移动，
   因此 Overlay 每次 redraw 从当前视觉节点重算全局中心，不能缓存屏幕坐标。
6. **稳定 identity**：保存继续后 RunState runtime object 与 screen owner 均重建，但真实
   origin `1:3`、候选 `[2:3]` 和地图指纹保持一致；选择后 origin 推进为 `2:3`，候选变为
   `[3:2,3:3]`。同 Act 不重访未被证明，因此使用 Mod 生成 GUID，并以 Run、Act、真实
   origin、规范化候选和地图指纹从唯一当前检查点严格恢复；禁止使用运行时对象 ID 或
   `{run_id}:route:{act}:{origin}` 直接拼接。Act 初始可操作地图已有真实 origin `0:3`；稳定
   可操作界面仍无 origin 时必须不发决策，不能用 sentinel 伪造。

当前程序集还确认 `NMapScreen` 只有拖动/纵向滚动输入与滚动状态，`NMapBg.OnWindowChange()`
只处理窗口宽高比；整个 Map 命名空间不存在用户可操作的地图级 Zoom/Magnify/Scale API。
节点 `HoverScale`/`DownScale` 仅用于悬停和按压动画。因此 STS2 `0.108.0` 的“用户地图
Zoom”验收项为 N/A；需覆盖的是地图滚动及 viewport、分辨率、窗口/content scaling 造成的
真实全局变换。不得为满足测试主动设置游戏节点 `Scale`。

以上证据通过六项生产 API 门禁，只授权开始 P1.0 路线生产纵向切片；不代表路线功能已经
实现或完成真机验收。

## 0.109.1 扩展决策程序集证据（2026-07-31）

本节只登记当前本机 `sts2.dll` 的精确类型与成员。验证方式为
`tools/Sts2ApiProbe` 加载当前游戏程序集，枚举类型、成员、访问级别与方法签名，并由
Mod 针对同一程序集完成 `0 warning / 0 error` 编译。它能够排除成员名猜测，但不能证明
界面生命周期时序；各 capability 仍须分别真机验证。

### Merchant

- 读取目标：当前商店库存、实际成本、库存/金币资格与移除服务；
- 类型：`MegaCrit.Sts2.Core.Nodes.Screens.Merchant.NMerchantInventory` 及其公开
  Inventory/entry 模型；
- 成员/方法：`Inventory`、`IsOpen`、`Open()`、`OnCardRemovalUsed()`、private
  `Close()`、`_ExitTree()`；Inventory 的 `AllEntries`、`Player`；entry 的 `Cost`、
  `EnoughGold`、`IsStocked`、`OnMerchantInventoryUpdated`、
  `InvokePurchaseCompleted(entry)`；
- 生产触发：上述生命周期的精确 Harmony Postfix，只观察回调后状态；
- 失败降级：Inventory 为空、条目无法稳定识别、资格/价格不完整或 capability 未启用时
  不创建/更新建议，关闭时清理 owner；
- 只读边界：不调用购买、移除、关闭或 UI 操作方法。

### Rest Site 与 Deck Edit

- 读取目标：当前真实可用篝火动作，以及 Smith/移除/变形的单目标子决策；
- 类型：`MegaCrit.Sts2.Core.Nodes.Rooms.NRestSiteRoom`、Rest Site option 模型、
  对应 card grid/selection screen 与 `CardSelectorPrefs`；
- 成员/方法：Room 的 `Options`、`Create()`、`EnableOptions()`、
  `AfterSelectingOption()`、private
  `OnBeforePlayerSelectedRestSiteOption(option, playerId)`、`_ExitTree()`；option 的
  `OptionId`、`IsEnabled`、`Title`、`OnSelect`；selection screen 的
  `ShowScreen`/`Create`，prefs 的 `MinSelect`、`MaxSelect`、`Prompt`，card grid 的
  `CardsSelected`、`_ExitTree()`；
- 生产触发：精确 Postfix。Deck Edit 只接受 `MinSelect == MaxSelect == 1`，并且必须从
  当前唯一、目标效果精确匹配的 eligible 父候选建立 `decision_parent`；
- 失败降级：多选、未知 prompt、父身份歧义、候选为空或 capability 未启用时不显示；
- 只读边界：不调用 `OnSelect`、选牌、升级、移除或变形动作。

### Event 与 Neow

- 读取目标：真实 event/page identity 与当前可用 option identity；
- 类型：`MegaCrit.Sts2.Core.Nodes.Rooms.NEventRoom`、
  `MegaCrit.Sts2.Core.Models.EventModel` 和 Event option 模型；
- 成员/方法：Room 的 `Create(EventModel, RunState, bool)`、
  `OptionButtonClicked(option, index)`、`_ExitTree()`；model 的 `CurrentOptions`、
  `CanonicalInstance`、`Owner`、`IsFinished`；option 的 `TextKey`、`IsLocked`、
  `IsProceed`、`Title`、`Chosen`；
- 生产触发：精确 Postfix，并以稳定 `TextKey`/page identity 建立候选；空白或超长 key
  不得作为生产身份；
- 失败降级：当前公开 API 没有提供确定结构化效果时只输出未知效果/data gap，推荐显示
  `--`；不得解析本地化描述或让 LLM 猜隐藏结果；
- 只读边界：不调用 OptionButton、选项选择、SetEventState 或 SetEventFinished。

真实打开/翻页/选择/关闭顺序、保存继续、Neow 嵌套奖励和各 screen owner 的重建行为尚未
在 `0.109.1` 组合真机中确认。上述 capability 在确认前必须保持
`pending_validation`。

### Spire Codex（构建期结构化数据源）

- 开发者页：<https://spire-codex.com/developers>
- OpenAPI：<https://spire-codex.com/openapi.json>
- API 条款：<https://github.com/ptrlrd/spire-codex/blob/main/API_TERMS.md>
- 源码许可证：<https://github.com/ptrlrd/spire-codex/blob/main/LICENSE.md>
- 用途：离线取得卡牌、遗物、药水、角色、敌人、遭遇、事件、章节、机制和社区聚合统计；
  不用于确认 Mod 的运行时对象生命周期，也不成为最终后台的联网依赖。
- 条款边界：托管 API 允许限流内社区使用，实体接口为 60 次/分钟；源码采用 PolyForm
  Noncommercial 1.0.0。两者不能混为同一授权，也不代表 Mega Crit 游戏资源可任意重打包。
- 2026-07-13 中文导出抽样：16 类共 1,616 条；事件 66、遭遇 87、怪物 115、卡牌 577、
  遗物 296。事件包含选项/页面/部分前置条件，怪物包含行动与大部分行动模式。
- 2026-07-31 再核对当前 66 个事件快照：event/page/option 提供稳定 ID、标题和描述，
  但没有类型化 `costs` / `effects`。这些描述可以用于以后带来源的 RAG 解释，不能在实时
  推荐中自动正则解析成 HP、金币、牌组或随机效果；确定性评分必须等待当前版本代码/API
  证据或人工审核的结构化效果目录。
- 数据无 SLA、完整性或 schema 稳定保证；每次更新必须在暂存区校验后再事务导入 SQLite。

## 已知选牌专项问题

### 已确认的 Card Reward 跳过 API（2026-07-14）

- 当前本机版本：STS2 `0.108.0`；
- 普通卡牌奖励界面内的“跳过”属于 `NCardRewardSelectionScreen` 的 alternate reward；
- 本机程序集方法签名：私有
  `NCardRewardSelectionScreen.OnAlternateRewardSelected(int index)`；精确字段
  `_extraOptions` 为 `IReadOnlyList<CardRewardAlternative>`；
- `CardRewardAlternative.OptionId` 是公开稳定身份；本机 IL 确认标准跳过由
  `OptionId="Skip"` 生成。生产代码按回调索引取得 exact alternative，只把这个 OptionId
  映射为 `decision_closed/outcome.kind=skipped`，不得把 reroll 或 Mod 提供的替代收益误标为
  跳过；
- 本机 `PostAlternateCardRewardAction` 当前枚举为 `None`、
  `EndSelectionAndDoNotCompleteReward`、`EndSelectionAndCompleteReward`、`DoNothing`；标准
  Skip 当前使用 `EndSelectionAndDoNotCompleteReward`，但决策身份以公开 `OptionId` 为准；
- `CardReward.OnSkipped()` 是在奖励列表直接跳过整项 Card Reward 的另一条路径，不能代替
  界面内 alternate reward 观察点；
- 证据：2026-07-14 真机保存继续后，点击界面“跳过”只打开地图，未触发
  `CardReward.OnSkipped`；旧参考源码说明 alternate reward 的概念调用链，但其方法签名与枚举
  名称已落后于 0.108.0。本机 .NET 9 元数据与 IL 探针确认了上述当前签名、字段、OptionId
  和枚举；生产 Patch 仍须通过当前 `sts2.dll` 编译及真机复测；
- 生产约束：只添加 Harmony Postfix，并以绑定的 screen owner 校验当前决策；不点击按钮、
  不根据按钮文字或节点路径猜测用户行为。

1. Neow 祝福结束后首次进入地图时，为什么 `RunStateReader` 没有 observed player。
2. 首战前是否存在稳定的当前 `Player` / `RunState` 读取点。
3. 普通战斗奖励中，真实可见候选到底位于哪个对象：
   - `NCardRewardSelectionScreen`；
   - `_cardRow`；
   - card holder；
   - `CardModel`；
   - 或其他屏幕子节点。
4. 如何稳定区分普通奖励选牌和 Neow 祝福选牌。
5. `ShowScreen` Postfix 是否早于可见候选填充；若是，应该延迟到哪一个安全时机。

上述问题中的普通 Card Reward 已在 P0 真机闭环中验证。Neow 特殊选牌、同一奖励界面内
候选 `UPDATED` 和 reroll 仍属于 P1 专项，不得从普通三选一结果外推。

## 给实现 Agent 的要求

返修涉及 STS2 内部 API 时，实施报告必须先给出“API 调研结论”，再给出代码变更。
结论至少包含：

```text
读取目标：
类型：
成员/方法：
触发时机：
参考来源：
本机验证方式：
失败降级：
是否影响只读边界：
```

没有 API 证据时，不得用字段名猜测、UI 文本、截图位置或宽泛反射来扩大生产逻辑。

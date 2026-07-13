# STS2 API 参考源与使用边界

> 更新时间：2026-07-13
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
- 数据无 SLA、完整性或 schema 稳定保证；每次更新必须在暂存区校验后再事务导入 SQLite。

## 当前 TASK-004 必查问题

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

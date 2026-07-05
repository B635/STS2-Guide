# STS2 Guide Read-Only Exporter

这个 Mod 是 STS2-Guide 的游戏适配层，只观察状态并写入本机 JSON：

- 只注册 Harmony `Postfix`；
- 不替换方法返回值，不修改玩家、卡牌、地图或 RNG；
- 不执行自动点击；
- 不访问网络，也不持有模型 API Key；
- Loader 清单声明 `affects_gameplay: false`。

当前 P0 观察完整的选牌闭环：

1. `Player.PopulateCombatState` 后缓存本地玩家引用；
2. `CardReward.Populate` 后读取候选、牌组附加状态、药水槽、遗物计数和 Modifiers；
3. `NCardRewardSelectionScreen.SelectCard` 与 `CardReward.OnSkipped` 后只读记录实际选择；
4. 事件原子写入 `user://STS2Guide/events/`，并更新 `state-event.json` 最新镜像；
5. 外部 Python 桥接按序校验、去重、生成建议并关联实际结果；
6. `NRewardsScreen.AfterOverlayClosed` 只作为无法确认选择时的保守兜底，不会误标为跳过。

事件使用内部类型转换出的稳定大写下划线 ID，不依赖当前游戏语言。
`act`、`floor`、`ascension`、药水槽和卡牌附加状态使用 v0.107.1 的直接
只读 API；游戏版本从安装目录的 `release_info.json` 读取，找不到时才回退
程序集版本。

## 构建

需要：

- Slay the Spire 2；
- 与当前游戏匹配的 .NET SDK（当前模板为 `net9.0`）；
- Godot 4.5.1 .NET，用来生成 PCK。

复制 `local.props.example` 为 `local.props`，设置本机路径后运行：

```powershell
dotnet build .\STS2Guide.ReadOnlyExporter.csproj -c Debug
```

默认只输出到 `artifacts/mod`，不会写入游戏目录。真机验证后才把
`InstallModOnBuild` 改为 `true`，或手动复制 DLL、JSON、PCK 到游戏
`mods` 目录。

当前源码已针对 STS2 public-beta `v0.107.1` 编译，0 warning / 0 error。
更新 artifacts 不代表已覆盖游戏目录；安装或升级 Mod 仍是单独操作。

本目录的接口选择参考了 STS2 社区 Mod 模板和 BoberInSpire 的公开实现，
但协议、状态模型、原子写入、幂等处理与推荐链路均为本项目独立实现。

# STS2 Guide Read-Only Exporter

这个 Mod 是 STS2-Guide 的游戏适配层，只观察状态并写入本机 JSON：

- 只注册 Harmony `Postfix`；
- 不替换方法返回值，不修改玩家、卡牌、地图或 RNG；
- 不执行自动点击；
- 不访问网络，也不持有模型 API Key；
- Loader 清单声明 `affects_gameplay: false`。

当前生产适配层观察选牌、路线和扩展决策，但未通过当前版本真机门禁的能力默认关闭：

1. `Player.PopulateCombatState` 后缓存本地玩家引用；
2. `CardReward.Populate` 后读取候选、牌组附加状态、药水槽、遗物计数和 Modifiers；
3. `NCardRewardSelectionScreen.SelectCard`、界面内
   `OnAlternateRewardSelected(index)` 中公开 `OptionId=Skip` 的 alternate，以及奖励列表
   `CardReward.OnSkipped` 后只读记录实际选择；
4. 事件原子写入 `user://STS2Guide/events/`，并更新 `state-event.json` 最新镜像；
5. 外部 Python 桥接按序校验、去重、生成建议并关联实际结果；
6. 选牌界面退出只解绑显示 owner，不猜测选择结果；无法确认时保持失败关闭。

事件使用内部类型转换出的稳定大写下划线 ID，不依赖当前游戏语言。
`act`、`floor`、`ascension`、药水槽、卡牌附加状态和地图图使用当前程序集已核对的
直接只读 API；游戏版本从安装目录的 `release_info.json` 读取，找不到时才回退程序集
版本。生产事件使用 schema v9 完整快照，并携带游戏程序集哈希、成功提交 revision、
跨组件 `release_fingerprint`、显式 `guide_preferences.route_mode`、严格通用候选和父子
决策身份；显示端也会独立校验 fingerprint、偏好与 advice 契约。

路线只在真实可选择态创建决策，候选来自 `MapPoint.Children`，公开 Overlay 只绘制一条
当前主路线。Context Drawer 提供智能均衡、稳健生存、激进成长三个互斥软偏好；模式只
从同一 Run、同一发布的 `active-run.json` 恢复，损坏或错配时重置为智能均衡。当前
`0.109.1` 的自动回归与编译已通过，但兼容清单仍为
`pending_validation`，正式三件套尚未安装，路线及 Merchant、Rest/Smith、Neow、Event、
Deck Edit 均不得宣称真机完成。

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

当前源码已针对 STS2 `v0.109.1` 编译，0 warning / 0 error。
更新 artifacts 不代表已覆盖游戏目录；安装或升级 Mod 仍是单独操作。

本目录的接口选择参考了 STS2 社区 Mod 模板和 BoberInSpire 的公开实现，
但协议、状态模型、原子写入、幂等处理与推荐链路均为本项目独立实现。

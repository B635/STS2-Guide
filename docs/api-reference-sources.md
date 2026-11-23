# STS2 API 参考源与使用边界

> 更新时间：2026-09-25
>
> 目的：为 Mod 侧 API 调研提供固定入口，减少对 STS2 内部类型、字段和生命周期的猜测。

本文件登记可用于确认 STS2 真实 API 和事件时机的外部参考源。它们只能作为
“API 事实”和“真机观测方法”的参考，不能覆盖 [`product-spec.md`](product-spec.md)
定义的只读产品边界。

## 使用原则

**0.113.1 篝火锻造（2026-11-23）**：本机 `sts2.dll`（`62ecbe91…`）经提取器 `--dump-any SmithRestSiteOption`、`CardSelectorPrefs`、`CardSelectCmd` 核对。`OnSelect` 状态机使用 `CardSelectorPrefs(UpgradeSelectionPrompt, 1)`，构造器将同一数值传给选择上下限，再调用 `FromDeckForUpgrade(owner, prefs)`。可取消不等于可同时选多张。Mod 仅读取牌组和 `IsUpgradable`，不调用任何选择命令。已通过程序集调用点检查和编译，未真机验证。

2026-09-25 STS2MCP 当前后台控制适用性、四处编译不兼容、GET 隐式动作及本机隔离适配
证据见 [`dev/sts2mcp-background-compatibility.md`](dev/sts2mcp-background-compatibility.md)。
该工具仅作用户授权的开发期测试驱动，不进入只读产品；旧 `/api/v1/state` 示例不能作为
当前上游 API 事实。隔离编译通过不等于失焦真机通过。

### 2026-10-01 TASK-025 票 02：程序集事实管线

本机0.111.0程序集SHA256 `0861bfa1…dcbe9`。开发工具 `tools/Sts2FactExtractor` 只用
`System.Reflection.Metadata` 解码元数据与IL，不加载执行程序集、不实例化模型、不调用游戏方法，
不进入产品或发行包。它枚举 `Models.Relics` 与 `Models.Cards` 命名空间下的全部顶层类（300/596），
输出重写的钩子、引用的游戏成员（含泛型实参，如 `PowerCmd.Apply<FreeAttackPower>`）、`get_Type`
后的 `CardType` 常量比较，以及按类代码指纹（含嵌套状态机与匿名方法，操作数解析为名称而非token，
同一程序集重复提取结果一致）。类名按 PascalCase→UPPER_SNAKE 与目录ID一一对应（298遗物、595卡），
仅 `DeprecatedRelic`、`VakuuCardSelector`、`DeprecatedCard` 无目录条目。

事实文件每条锁定该类指纹；游戏更新后重新提取，只有代码改动过的条目变为 stale。遗物初分只看钩子
结构（触发/基线/局外/奖励），奖励类钩子与未归类钩子一律人工审核；战纹涂料仅 `AfterObtained`，
斗篷扣仅 `BeforeSideTurnEnd`。

### 2026-09-25 TASK-024：主动耗竭、下一攻击免费与条件遗物

验证目标为本机0.111.0 / 41cef1ea、程序集SHA256 `0861bfa1df347538d932f22d580e75420f08082792eb914e53b4882764acdbe9`。
沿用只读元数据/IL探针，不实例化模型、不执行游戏方法；真机奖励屏只负责复现遗漏，
不能用界面文字替代以下程序事实。所有变化发生在本地静态标签与估值，不增加Mod读取点。

- `TrueGrit.get_CanonicalVars` 060046A1：Block=7；`OnUpgrade` 060046A4：Block+2。
  `<OnPlay>d__7.MoveNext` 0600AD68：先GainBlock，未升级从Hand随机选一张，非空时
  `CardCmd.Exhaust`；升级后`CardSelectCmd.FromHand`选择一张并Exhaust。它继承空
  CanonicalKeywords，ExtraHoverTips的Exhaust不表示自身耗竭。
- `DarkEmbracePower.<AfterCardExhausted>d__8.MoveNext` 06009EC2：同Owner、非Ethereal
  耗竭按Power.Amount抽牌。拥有该牌是潜在协同，不是能力已激活或固定每回合抽牌。
- `Unrelenting.get_CanonicalVars` 060046D8：Damage=14；`OnUpgrade` 060046DA：+6。
  `<OnPlay>d__3.MoveNext` 0600AD81：攻击后Apply FreeAttackPower(1)；后者
  `TryModifyEnergyCostInCombatLate` 060025D0仅匹配自己Attack、Hand/Play牌堆，费用0；
  `<BeforeCardPlayed>d__5.MoveNext` 06009EFF按同样条件消耗一层。不能给高费技能/能力
  套通用降费收益；本轮只披露未建模。
- `ToughBandages.get_CanonicalVars` 060022EB：Block=3；
  `<AfterCardDiscarded>.MoveNext` 06009DDA校验弃牌Owner等于遗物Owner、其Side等于
  CombatState.CurrentSide才GainBlock。本轮只依赖已识别主动弃牌来源评估有限配合，
  不用3乘假设的弃牌频率，也不声称低生命本身能触发遗物。
- 失败降级：精确实体不符不套规则；来源/牌组不完整明确缺口；未估计的目标价值、
  触发频率和下一张攻击免费通过已有零贡献说明披露。新候选须独立审查和真机复测。

### 2026-09-25 TASK-023：完美打击静态机制核对

本机0.111.0 / 41cef1ea的程序集SHA256仍与发行清单一致。只读
`System.Reflection.Metadata`解码成员IL，没有实例化游戏模型或执行游戏方法：

- `PerfectedStrike.CanonicalVars`：CalculationBase=6、ExtraDamage=2；`OnUpgrade`
  只将ExtraDamage增加1。真实计算计数谓词为`CardModel.Tags.Contains(CardTag.Strike)`。
- 计数源`Owner.PlayerCombatState.AllCards`来自Hand/Draw/Discard/Exhaust/Play五堆，
  耗竭牌仍计入，自身也带Strike；不使用卡名、语言、费用或卡牌类型猜分类。
- 全程序集`CanonicalTags`与静态目录的22个Strike ID逐一一致，包含五角色基础打击和
  同时带Minion标签的卡。目录已提供tags、vars及upgrade.extradamage，但旧发行精简库
  未保留前两字段，需要同步发行数据，而不是修改实时协议。
- `CalculatedVar.Calculate`非战斗时乘数为0，所以奖励屏CalculatedDamage可能只显示
  基础6；不能据此推断该机制没有额外伤害。TASK-023使用永久牌组加新候选一张作战前
  基准估计，不把它冒充临时生成/移除、力量、敌人状态后的当回合伤害。
- 失败边界：未知卡牌分类、读取警告、缺少/异常参数、额外实例修改或重复升级保持
  明确缺口。实际新评分尚须完整构建、安装与正常窗口回归，IL证据不替代真机。

### 2026-09-25 TASK-023：血液药水的商店恢复机会

- `BloodPotion.CanonicalVars`（token06002AC7）创建普通DynamicVar HealPercent=20；
  `Usage`（06002AC5）为AnyTime；`OnUse`的MoveNext（0600A102）仅将目标MaxHp乘
  HealPercent.BaseValue除以100传给`CreatureCmd.Heal`，没有自身附加效果。
- `PotionModel.DynamicVars`由CanonicalVars初始化，普通DynamicVar.SetOwner只保存owner。
  使用wrapper调用一次OnUse。当前程序集枚举的BeforePotionUsed覆写仅SurroundedPower
  战斗转向；AfterPotionProcured/AfterPotionUsed的BeltBuckle、ReptileTrinket效果限战斗。
- `CreatureCmd.Heal`（0600B307）没有治疗禁止/倍率Hook；`HealInternal`（060051D9）进入
  SetCurrentHpInternal，先与MaxHp取最小值再将decimal显式转int；正值截断。
  AfterCurrentHpChanged仅CombatState非null时执行。商店非战斗可恢复量因此是
  `min(max_hp-hp, floor(max_hp*20/100))`。
- 实施只对BLOOD_POTION写入带程序集来源的静态效果标签，并通过现有SQLite效果表保留。
  商店估值只表示现在使用可获得的恢复机会；购买不会自动饮用，不估计保留到未来的价值、
  战斗用药联动或所有药水。缺资料/非法数值、槽位未知仍降级；公开商店能力仍未开启。
- 本轮诊断脚本在本机忽略目录`build/task023-api/ReadOnlyMetadataIl.ps1`；仅解码元数据，
  不调用游戏方法、不成为产品或发行依赖。

### 长期调研纪律

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

## 2026-09-25：商店/篝火目标资格（TASK-020）

- 当前本机：`0.111.0 / 41cef1ea`，程序集 SHA256 与发行清单一致。新读取函数为
  `DeckEditTargetReader.Read`；本轮未启动游戏，也未宣称扩展能力真机通过。
- `Player.Deck.Cards` 提供实际牌组实例；`CardModel.IsRemovable`、`IsUpgradable` 为公开
  bool getter，`CurrentUpgradeLevel` 为实例升级次数。升级资格不得用“升级次数等于零”猜测。
- 调研 Agent 对当前 IL 的读取结论：商店 `OneOffSynchronizer.DoMerchantCardRemoval`
  使用 null filter 调用 `CardSelectCmd.FromDeckForRemoval`，后者使用 `IsRemovable` 加可选
  filter；`FromDeckForUpgrade` 使用 `IsUpgradable`。这些是本轮父目标集合的依据，不调用
  任何购买、选择、升级或移除方法。
- `SmithRestSiteOption.SmithCount` 为公开 int；只对一次选择一张牌导出完整目标集，其他
  数量保持 null（未知/未建模），不能把多选升级按单牌估值。IL 调研确认 Smith 使用该数量、
  可取消及手动确认偏好；取消返回 false，成功升级后返回 true。实际 UI 时序仍需真机。
- 当前 Merchant 类型命名空间为 `MegaCrit.Sts2.Core.Nodes.Screens.Shops`。已核对
  `NMerchantInventory.Open/Close/OnCardRemovalUsed/_ExitTree` 和
  `MerchantEntry.InvokePurchaseCompleted(MerchantEntry entry)`，无需改动既有 Postfix 目标。
  Rest 对应私有回调为 `OnAfterPlayerSelectedRestSiteOption(RestSiteOption, bool, ulong)`。
- 证据：`build/task020-api-members.log` 为本机探针输出；`tools/GuideBehaviorTests` 对当前
  程序集执行成员签名检查，并直接链接生产目标投影函数测试过滤、重复卡牌实例、空集合与
  非法/超限数据。托管测试与 IL 阅读不证明游戏内回调一定被触发。
- 失败降级：目标读取任何一项异常、非法或超过协议上限都返回 null，不导出部分集合。
  无合法目标导出空列表，两者不得混淆；目标身份只对本次快照有效。

## 2026-09-11：静态数据分支核查（TASK-013，历史记录）

- 本机目标为 `0.111.0`。现有 `data/knowledge.json` 未记录游戏版本或 API 分支，
  与当前效果有差异；不能仅升级兼容清单而沿用旧数据。
- [开发者页面](https://spire-codex.com/developers) 明确实体端点可使用
  `channel=beta`；[版本端点](https://spire-codex.com/api/beta/version) 本轮实际返回
  `beta_version=v0.111.0`。`/api/versions` 实际为空，不据此猜历史版本参数。
- 已为 `scripts/fetch_knowledge.py` 增加显式分支、预期版本和隔离输出路径。Beta 下载
  前后分别查询版本，若不符合预期或途中变化则不写出目录数据；这只是抓取一致性检查，
  不是当前游戏所有字段已验证的声明。
- 配置端点 `mechanics/constants`、`merchant/config` 没有承诺 Beta 分支版本，来源记录
  明确列为 `unversioned_endpoints`。公开 Card/Route SQLite 不收录这些配置；扩展仍暂停。
- 已下载到 Git 忽略的 `build/catalog-review/knowledge-0.111.0.json`，尚未覆盖正式
  `data/knowledge.json`。新增卡牌 ID 对应的 19 个类型均通过本机程序集元数据查找到；
  这只证明类型存在，不替代逐项数值/机制验证。
- 抽查 Beta 的暴走、跃跃欲试、引导之星、超能光束与君王矿石，已能看到新效果；正式
  采纳前还需核对升级表示、效果标签和评分。网站抽取数据不作为无误保证。
- 使用边界仍按 [API 条款](https://github.com/ptrlrd/spire-codex/blob/main/API_TERMS.md)：
  不引入第三方实现或资源图像，不抓取受限社区 tier，不将网站查询加入实时主链。

## 参考源

### 2026-09-24：保存回菜单与选牌本地化（TASK-018）

- 当前本机 `0.111.0 / 41cef1ea` 程序集验证：`CardModel.Title` 为公开 `string` 属性，
  从已与 pending 稳定身份逐项匹配的 `NGridCardHolder.CardModel` 读取，不解析界面文本。
  只供占位行显示；空标题显示“卡牌 N”，不改变候选 ID。失败沿用停止显示，不猜名称。
- `RunManager.CleanUp(bool graceful)` 为同步公开 `void`，`NMainMenu._Ready()` 为公开
  `void`。通过本地 Sts2ApiProbe 检查成员；新增 Harmony Postfix 在 cleanup 关闭当前
  观察决策，主菜单 Ready 幂等重试。读取缓存的最后完整状态，不在已清理的游戏对象上猜字段。
- 不根据 `graceful` 推断胜负/放弃；沿用 `decision_closed/closed_unknown`，不新增协议，
  不结束整局。写入失败保留 pending，UI 清理独立执行。安装后的实际回调验证见 TASK-018。

### 2026-09-23：生产稳定实体 ID（TASK-014）

- 本机 `0.111.0 / 41cef1ea` 的 `MegaCrit.Sts2.Core.Models.ModelDb.GetEntry(Type)` 为公开静态
  方法；托管验证程序直接链接生产 `StableIds` 并调用当前游戏程序集，未启动游戏或 Godot。
- 真实返回：SpeedPotion → SPEED_POTION、FairyInABottle → FAIRY_IN_A_BOTTLE、
  ExpectAFight → EXPECT_A_FIGHT、IAmInvincible → I_AM_INVINCIBLE。原剥后缀正则已复现
  SPEED 错误后修复。范围对照覆盖发行目录的卡牌、药水、遗物、五角色，排除嵌套生成类型。
- 触发点仍为既有状态快照与候选读取；失败沿用观察器的异常隔离/停止发布，不猜测备用 ID。
  此方法也被既有关闭的观察器共用，但不因此启用商店等能力。
- 验证只证明托管 API 和目录映射；实际快照/面板以及旧当前局缓存恢复仍待 TASK-018 真机。

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

## 2026-09-25 TASK-022：商店升级与精确事件效果

验证对象为本机 STS2 `0.111.0 / 41cef1ea`；`sts2.dll` SHA256：
`0861bfa1df347538d932f22d580e75420f08082792eb914e53b4882764acdbe9`。
证据采用现有 Sts2ApiProbe、只读 `System.Reflection.Metadata` IL 解码和生产编译/托管签名
测试，不调用获得奖励或选择方法。下述效果不是从描述文本推断，尚无本轮真机时序证据。

- 商店真实卡牌：`MerchantCardEntry.CreationResult.Card`（现有读取链）→
  `CardModel.CurrentUpgradeLevel : int`；只为卡牌商品投影升级次数。读取失败/缺失不可
  当作 0，消费者保留未知；非卡牌商品不可携带此字段。
- `EventModel.InitialOptionKey → OptionKey` 构造完整
  `EVENT.pages.PAGE.options.OPTION`；`ByrdonisNest.GenerateInitialOptions` 的 IL 明确包含
  `BYRDONIS_NEST.pages.INITIAL.options.EAT/TAKE`。目录存短 option ID，消费侧只能在
  当前 event 的明确 page 内解析查找，原始候选身份不变，不采用任意后缀匹配。
- `EventOption.Relic : RelicModel`、`EventModel.DynamicVars`、`RelicModel.DynamicVars`
  均为公开 getter；`DynamicVarSet.Gold/Heal`、`DynamicVar.BaseValue : decimal` 与
  `IntValue : int` 由托管签名测试核对。
- Neow 的 `NEOW.pages.INITIAL.options.GOLDEN_PEARL`：
  `Neow.PositiveOptions → AncientEventModel.RelicOption<GoldenPearl> → RelicCmd.Obtain`；
  `GoldenPearl.AfterObtained` 读取实际 `DynamicVars.Gold.BaseValue` 并调用 GainGold。
  生产读取精确 Neow/GoldenPearl 类型与完整键，输出实际金币数，不写死当前基础值 150。
- `AbyssalBaths` 的 `ABYSSAL_BATHS.pages.INITIAL.options.ABSTAIN`：
  `Abstain` 状态机读取 `DynamicVars.Heal.IntValue → CreatureCmd.Heal → SetEventFinished`；
  仅此明确组合输出治疗。`ABYSSAL_BATHS.pages.ALL.options.EXIT_BATHS` 仅结束事件，输出
  `no_op`；不把所有 IsProceed/离开文字都猜成无成本。
- 不支持项继续输出 unknown：营养牡蛎/异鸟巢吃蛋的 GainMaxHp 还会 Heal，不能省略耦合
  效果；SapphireSeed EAT 含治疗加升级子选择；浴场 IMMERSE 含增血、受伤、递增与下一页。
  不能把任意非空 Relic 泛化成单个 add_relic。

生产入口是现有 EventDecisionObserver 的选项快照，由 `VerifiedEventEffectReader` 只读
投影；类型/键/Relic 不一致、数值异常或 getter 抛出时整项回 unknown，不发布部分收益。
`VerifiedEventEffectTests` 覆盖纯投影及 API 签名，Python 测试覆盖真实完整键、当前金额变化、
检查点后读失败撤分与新局隔离。只读边界未改变，没有新增游戏调用或网络行为。

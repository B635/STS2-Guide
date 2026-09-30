# TASK-018 — 当前版本真机、演示与独立试装（后续）

- 状态：待真机验收
- 父任务：TASK-013；产品规格第 2.1–2.2 节
- 阻塞依赖：TASK-017 代码审查通过，并由用户启动后续真机任务

## 交付行为与验收

- [ ] 最终安装向导、五角色 Card/Route、保存继续、窗口变化和进程生命周期真机通过。
- [ ] 真实演示可播放，与最终包和 README 一致，用户验收通过。
- [ ] 2–3 位受支持版本玩家独立试装通过，否则明确待试装。

2026-09-23：前轮已按要求在 code-review 停止；用户本轮明确授权 Agent 操控游戏进行本机验收。
先检查并安装 TASK-017 候选，保留已有存档。外部试装仍待其他玩家参与，不冒充完成。

## 本轮前置检查

游戏当前运行 v0.111.0 / 41cef1ea；安装的 Mod 为 0.3.1，安装收据为 Guide 0.3.1-alpha.0。
尚未加载 0.3.2 候选，不能用当前进程验收最新修复。未发现运行的 Guide/开发 Host。

已通过桌面画面确认主菜单显示“继续游戏”，未放弃存档；随后正常点击退出并确认游戏进程
结束。按 computer-use 安装操作的即时确认要求，暂停在更新候选前，等待用户确认安装。
本轮尚无新候选真机通过项。

## 2026-09-23 本机验收进展（非完整验收）

用户后续明确授权修复并主动寻找范围内问题。当前返修先处理抽屉底部裁切，增加真实布局
回归后再安装复测；继续选牌/路线/退出恢复/保存继续，不扩大新功能。

- 用户确认安装后，安装日志 `Setup Log 2026-09-23 #001.txt` 记录成功；实际安装 EXE 与
  Mod DLL/JSON/PCK 均与 0.3.2-alpha.0 release-manifest 一致。未独立观察完整向导，不能
  据日志宣称向导 UI 已通过。游戏从 Steam 进程启动，继续现有 IRONCLAD 存档。
- 当前游戏 0.111.0、Mod 0.3.2、Guide 0.3.2-alpha.0，runtime-status v2 RUNNING，兼容通过。
- 战斗中打开地图产生 map_choice（sequence 23），未错误出现选路建议；55 个地图节点，
  已持有速度药水真实导出 SPEED_POTION，不再是旧 SPEED。
- 战斗操控曾被用户 Esc 中止；随后用户自行完成战斗并通知已到选牌，本轮只观察、不替其选择。
- 选牌 sequence 25：state/advice event_id 和 decision_id 一致；画面与候选顺序一致：
  完美打击 47（46.72）、无情猛攻 52（51.90）、剑柄打击 61（61.41）、跳过 32。
  剑柄打击高亮；HP 66/80、金币129、SPEED_POTION 与 BLESSING_OF_THE_FORGE 与画面相符。
  recommendation.data_gaps 为空，当前结果没有串用上一屏。
- **发现显示问题**：当前窗口的抽屉出现滚动条，底部“跳过 32”被裁切，无法四项完整直接展示。
  记录为布局待修/复测；本轮未修改代码，不能将该屏记为完全通过。
- 仍未验证：真正选路及三种偏好、选牌关闭、后台退出/异常/恢复、保存继续对应决策、
  窗口适配、其他角色；未录制正式演示，未外部试装。游戏及 Guide 保持运行。

## 2026-09-23 后续返修与 09-24 续修

- 裁切根因：固定总高度 216px，真实四行 Godot 布局需要 227px；状态栏与 VBox 间距
  漏算。ContextDrawer 改用真实内容最小高度（含字体、样式边距与容器间距），仅在视口
  不足时滚动。新增 `tools/DrawerLayoutTests` 链接生产类并运行 headless Godot，旧实现
  实测裁切 11px，修复后 0px；包含 4/6 行、路线按钮、大字体、小视口及重新放大回归。
- 0.3.3-alpha.0 完整 458 项测试、Mod/PCK 编译、EXE 自检、安装包审计通过；静默更新
  成功且安装的 EXE/Mod 三件套逐项一致。静默安装不代表向导 UI 验收。
- 从 Steam 继续原战士存档，重新领取金币/药水并打开选牌：sequence 27，HP66/80、
  金币129、两瓶药水及三张候选正确；47/52/61/跳过32 全部显示，无滚动条、无裁切。
- 异常终止仅 Guide 父子进程后，抽屉能撤下分数和高亮并显示“建议暂不可用”。
  重新启动已安装 Guide 后，心跳 RUNNING/兼容通过，却一直“等待当前建议”，advice
  文件不存在。**这是新的恢复阻断，不能把 TASK-016 自动测试通过当作真机恢复通过。**
- 09-24 续修：真实桥接回归先失败（空 events 目录返回 None）；修改为空队列时继续
  读取完整快照，仍经过原 schema、版本、身份、checkpoint/lifecycle 校验。快照指纹
  改为 advice 副作用成功后才确认，避免写入失败后无法重试。补 v9 选牌/路线恢复、
  待处理 close 优先、旧决策不能复活、碰撞/版本拒绝、写入失败重试测试。
- 独立审查发现并关闭两项：落后 mirror 曾会删除新 queue advice；将“不能接管”误当
  “跨局镜像较旧”曾会复活上一局 advice。现仅在已提交 latest payload 内容完整、同局
  sequence 严格较新，或跨局 emitted_at 严格较新时优先重放它，仍完整经过 processor。
  相同 sequence 的内容冲突、新局缺失合法接管依据与不兼容输入不放宽。90 项兼容/事件
  测试独立复核通过。本条不宣称恢复真机已通过。
- 另有非阻断显示问题：后台不可用时牌名退为英文稳定 ID；正常评分恢复后有中文名。
  当前未调整评分权重或扩展新能力。恢复问题优先，其他验收项仍未完成。

## 2026-09-24 0.3.4 恢复修复交付与真机复测

### 修改与自动证据

- `realtime/file_bridge.py`：空 spool 读取镜像；较旧镜像优先使用已提交完整 checkpoint
  经正常 processor 重放；副作用成功后才确认镜像指纹。不放宽协议、兼容或决策身份校验。
- `tests/test_realtime_events.py`、`tests/test_compatibility.py`：补空目录重启、发布失败重试、
  close 优先、v9 Card/Route、落后镜像、跨局非法接管和旧建议不可复活回归。
- 版本/清单/协议示例/README 同步 Guide `0.3.4-alpha.0`、Mod `0.3.4`。协议仍为 v9。
- 完整命令 `python -m unittest discover -s tests -q`：463 项，43.482 秒，OK；日志
  `build/task018-recovery-final-tests.log`。独立审查复核兼容/事件 90 项通过，两项阻断关闭。
- `scripts/build_public_beta.ps1` 完整通过：Mod/PCK 0 warning / 0 error、EXE 隔离自检、
  发行 SQLite、安装包与 payload 审计；日志 `build/task018-recovery-release.log`。
- 产物 `dist/installer/STS2-Guide-0.3.4-alpha.0-win-x64-setup.exe`，发布指纹
  `fd851a427cbda251804a15f4874eaf2b357d65e38484cdcc82923dca7c46af34`。
  静默安装 exit 0，安装日志 `build/task018-recovery-install.log`，收据及实际 EXE/Mod
  三件套一致。仍不代替安装向导 UI 验收。

### 真机结果（仅覆盖下面场景）

游戏从 Steam 启动，继续既有战士局 `sts2-5a7aad6d31e1f72ca684c183`，未新开/放弃存档。

1. 继续到奖励领取前，重新领取 17 金币和熔炉的祝福后打开选牌：sequence 29、HP66/80、
   金币129、两瓶药水。完美打击47、无情猛攻52、剑柄打击61、跳过32 全部显示，无裁切。
2. 空 events 目录时异常终止仅 Guide 父子进程：面板变“建议暂不可用”，分数和高亮撤下。
   不操作游戏，启动安装版 EXE：新 runtime session 恢复 RUNNING，同 event `:29` 与
   decision `card-reward:8499dd736a884eb78559aff44fd08566` 建议重建，画面四项自动恢复。
3. 实际选取剑柄打击：牌组 12→13，选牌面板撤下；点击前进进入真实路线选择。
   均衡为普通战斗90/商店85/问号91，稳健91/88/92，成长92/88/92；偏好切换生效，
   本局都推荐问号分支。这里只验证更新和身份，不将相同路线判为算法错误或质量通过。
4. 成长模式 sequence 34、decision `720e9b1bee114e929d171a6d03767ab4` 再次异常终止
   Guide：路线黄线和全部分数撤下；重启后不操作游戏，同事件、同决策及路线自动恢复。
5. 恢复均衡偏好，正常保存回主菜单：抽屉不可见。此时仍有最近 route advice 文件，
   仅 UI 撤下已证实，主菜单文件清理/继续时生命周期仍需专项复核，不能标记全部通过。
6. 正常退出游戏：runtime 为 IDLE、game_running=false、worker_running=false。
   随后使用安装 EXE `--shutdown-existing` 正常退出托盘；确认游戏、Guide 父子进程、
   开发 Host 全部结束。测试没有进入下一个地图节点，继续游戏可能按游戏机制回到奖励前。

仍需修复/复核：离线牌名显示英文 ID；主菜单 advice 生命周期。五角色完整验收、路线
偏离、小窗口真机、最终演示与外部试装不在此次已通过项内。任务维持“需修复”，不得据此
扩大功能或宣称公开交付。未 commit/push，未使用重置卡。

## 2026-09-24 0.3.5 离线牌名与主菜单生命周期收口

本节替代上节末尾两项“仍需修复”的当前结论，历史证据不删除。本轮未扩大 capability。

### 实施与自动证据

- `CardRewardAdvicePanel.cs`：校验实际屏幕候选身份时读取 `CardModel.Title`，按当前屏幕
  保存显示名；等待/不可用状态沿用这些中文名，分数为空且无高亮。隐藏时清空名字，
  不回退英文稳定 ID，不复用上一屏名称，也不缓存失效建议分数。
- `RunLifecycleObserver.cs`：使用已核对本机程序集的公开 `RunManager.CleanUp(bool)`
  Postfix 关闭当前观察会话并隐藏 Card/Route；`NMainMenu._Ready()` 提供幂等重试。
  两个新增观察器都通过 ObserverSafety 隔离，不修改游戏行为。
- `StateEventWriter.cs`：`EmitRunObservationClosed` 只提交当前 pending 的
  `decision_closed/closed_unknown`；提交成功后才清空 pending，失败保留供重试。
  使用已捕获快照，不在游戏对象清理后重新猜测数据，不结束 run 或删除恢复检查点。
- `CardRewardObserver.cs`：仅更新 TreeExit 注释，区分隐藏 UI 与显式结束观察会话。
- `tests/test_mod_runtime_commit_contracts.py` 补当前候选本地化和真实生命周期入口回归；
  `tests/test_compatibility.py` 补生产 v9 Card/Route 保存关闭、偏好保留、后台重启拒绝
  旧 advice、同局新决策恢复的真实桥接回归。协议字段和 schema v9 不变。
- 先更新产品规格的保存/恢复与不可用显示边界，并登记实际 API 证据；版本、兼容清单、
  Mod 清单、协议示例及 README 同步 Guide `0.3.5-alpha.0` / Mod `0.3.5`。
- 独立审查未发现新阻断，52 项针对性测试通过。完整独立测试
  `python -m unittest discover -s tests -q` 为 **466 项 / 46.868 秒 / OK**。
  首次曾并行运行两个全量套件，因共享 Godot 测试 DLL 编译输出被占用而失败；之后串行
  重跑通过，失败记录不计为验收成功。测试中的 locked/boom 日志为预期异常用例。
- 完整 `scripts/build_public_beta.ps1 -Python D:/miniconda3/envs/sts2/python.exe`
  成功：内置测试 **466 项 / 47.489 秒 / OK**，Mod/PCK **0 warning / 0 error**，
  EXE 自检、发行 DB、安装包和 payload 审计通过；日志 `build/task018-menu-release.log`。
- 新包 `dist/installer/STS2-Guide-0.3.5-alpha.0-win-x64-setup.exe`；发布指纹
  `c9087605b7bef57a90c640c782aa902226e754d80c795909447f0394ab4f3d50`。
  静默安装 exit 0，`build/task018-menu-install.log`；实际 EXE 和 Mod 三件套逐一匹配
  发行收据。静默安装仍不代表安装向导 UI 验收。

### 安装态真机证据

游戏经 Steam 启动，继续同一战士存档 `sts2-5a7aad6d31e1f72ca684c183`，本轮不放弃存档。

1. Guide 未启动，打开奖励选牌（sequence 37）：完美打击、无情猛攻、剑柄打击、跳过
   四项中文名字正确，显示“建议暂不可用”和 `--`，没有英文 ID 或裁切；advice 不存在。
   启动已安装 Guide，不操作游戏，恢复 47/52/61/跳过32 与剑柄打击高亮。
2. 在选牌中保存退出：sequence 38 为 `decision_closed/closed_unknown`，关闭决策
   `card-reward:17a68d185a254f80b5c5182c195a5830`；主菜单无抽屉，advice 文件删除，
   `active-run.json` 同局保留、current_decision=null，没有 selected/skipped/run_ended。
3. 重启 Guide 后继续原局；游戏回到奖励领取前。新的 sequence 39、decision
   `card-reward:cb9fbd592e444752ae24b9f3f1fd0ae8` 与已关闭 ID 不同，打开选牌后
   四项建议恢复正确。游戏恢复奖励时可预先生成候选；抽屉仅在实际选牌屏出现。
4. 选取剑柄打击，牌组 12→13，面板关闭；领取金币/药水后打开地图，路线可见。
   切至“稳健生存”后评分为普通战斗91/商店88/问号92，路线偏好生效。
5. 从地图保存退出：sequence 45，decision `3c057be61fa646d5903e09851853638d` 的
   `decision_closed/closed_unknown`；advice 删除、current_decision=null，同局检查点
   及 survival 偏好保留。在主菜单正常重启 Guide 后再次核对，仍无 advice，不复活旧路线。
6. 正常退出游戏，Controller 从 DRAINING 到 IDLE，game_running=false、
   worker_running=false；再用安装 EXE `--shutdown-existing` 正常退出，确认游戏、
   Guide 父子进程与开发 Host 全部为空。没有进入下一地图节点，继续时仍可能回到奖励前。

结论：本轮两项遗留缺陷已修复并完成对应真机回归，任务转“待真机验收”，不是整票验收。
剩余五角色完整流程、实际偏离重规划、窗口变化、其他保存断点、向导、正式演示和外部
试装仍待完成；本轮未证明策略质量或胜率提升。未 commit/push，未使用额度重置卡。

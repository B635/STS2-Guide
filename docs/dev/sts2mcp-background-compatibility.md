# STS2MCP 后台真机测试兼容性与副作用审计

日期：2026-09-25。范围：开发期独立测试工具，不是 STS2 Guide 产品依赖。
本报告只包含源码阅读和隔离编译证据；未安装、重启游戏、发送游戏动作或操作前台窗口。

## 结论

STS2MCP 的动作通道可以绕过操作系统鼠标键盘，具有后台测试的实现基础；但**当前上游
不能直接视为本机可用，也不能把它的 GET 状态接口当成只读探针**。本机原版编译已有
4 处兼容错误，且 GET 会自动打开商店和宝箱。必须先处理这些问题，再分别验证失焦读取、
单个动作、状态回执和 Guide 观察结果；HTTP 根路径可达不代表游戏帧循环仍在工作。

## 固定来源与版本

- 本轮 GitHub `main` 为
  [`55e064850a68f3b4cde7e5fd525bf9b2dec4e885`](https://github.com/Gennadiyev/STS2MCP/commit/55e064850a68f3b4cde7e5fd525bf9b2dec4e885)，
  提交日期 2026-07-29；提交说明只涉及 STS2 `0.107.1` 的 API 兼容。
- [该提交 README](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/README.md)
  仍把测试版本写为 `0.103.2`。最新发布标签
  [`0.4.0`](https://github.com/Gennadiyev/STS2MCP/releases/tag/0.4.0)
  发布于 2026-05-05，对应 `e23393d64706427e23039d04ad8e3379d63835ba`，与当前 main 不同。
  这些都不能证明本机 `0.111.0 / 41cef1ea` 兼容。
- 本机目标 `sts2.dll` SHA256：
  `0861bfa1df347538d932f22d580e75420f08082792eb914e53b4882764acdbe9`。
- root Agent 在隔离目录执行上游原版构建，实际结果为 0 警告、4 错误：
  `McpMod.StateBuilder.cs:783` 的 `StartRunLobby.MaxPlayers`，以及 973、982、1000 行的
  `LoadRunLobby.ConnectedPlayerIds` 在本机引用中不存在。证据：本地忽略文件
  `build/task022-live/mcp-compatibility-build.log`。这是编译失败，不是运行失败；不能凭此
  推断所有单人 API 均已失效，也不能用猜测的新多人字段替换。

## 输入方式：不是操作系统点击

HTTP 工作线程把闭包排入队列，游戏 `SceneTree.ProcessFrame` 回调在游戏主线程执行。
选牌是发出 Godot holder 信号，事件/篝火使用游戏控件 `ForceClick()`，出牌则提交
`PlayCardAction` 到游戏动作队列；没有在这些调用链中看到 `SendInput`、`SendKeys`、
`SetForegroundWindow`、鼠标坐标移动或窗口激活 API。
这是“不需要操纵前台鼠标”的源码证据，而不是“当前版本完全失焦可玩”的真机证据。
参见 [队列与 ProcessFrame](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L81-L157)、
[出牌](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.Actions.cs#L94-L169)、
[事件和篝火](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.Actions.cs#L270-L353)、
[奖励选牌](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.Actions.cs#L495-L538)。

上游没有专门保证失焦/最小化后继续处理的逻辑，也没有为测试而取消游戏暂停。根路径
`GET /` 在 HTTP 线程立即回答；`GET /api/v1/singleplayer` 才经过游戏主线程。
所以后续应先在用户保持其他窗口前台时，以有界超时取得真实状态，再决定是否发送动作。
不能通过切到游戏、修改暂停/窗口设置来悄悄“修复”后台条件。
参见 [请求分派](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L175-L215)、
[真实状态读取](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L385-L425)。

## 重要副作用与安全边界

### GET 会触发真实界面操作

| 读取分支 | 上游副作用 | 原始证据 |
|---|---|---|
| 普通商店 | 自动 `OpenInventory()` | [StateBuilder 545–557](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.StateBuilder.cs#L545-L557) |
| 假商店事件 | 对商人按钮 `ForceClick()` | [StateBuilder 1465–1477](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.StateBuilder.cs#L1465-L1477) |
| 宝箱 | 对箱子 `ForceClick()` 后返回正在开启 | [StateBuilder 2302–2309](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.StateBuilder.cs#L2302-L2309) |

因此定时 GET 也可能改变玩家所处 UI/状态，并触发我们的观察器。测试构建应移除上述
隐式动作；若后来确实需要开店/开箱，应是独立、明确授权和记录的操作。不能把它们产生的
Guide 事件误称为“玩家自然触发”。

### 加载与设置

初始化会调用 `Harmony.PatchAll()`、连接 ProcessFrame、启动 HTTP 服务，并在没有配置时
尝试创建 Mod 目录下的 `STS2_MCP.conf`；没有看到初始化直接开始新局或自动出牌。
但 SettingsUI 补丁会插入 Instant Mode 项，并含修改 `PrefsSave.FastMode` 的 Prefix。
所以 `affects_gameplay: false` 的 manifest 不等于没有设置修改能力。
本次测试不需要这些补丁，建议隔离构建禁用，保留原动画与游戏设置。
参见 [初始化/配置](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L37-L125)、
[SettingsUI 补丁](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.SettingsUI.cs#L81-L137)。

### HTTP 暴露与危险端点

上游只注册 `localhost`、`127.0.0.1` 前缀，没有 `+`/`*` 主机前缀；但未检查请求来源的
`RemoteEndPoint`、未鉴权，并返回 `Access-Control-Allow-Origin: *`。仅凭前缀不能完成
Windows HTTP.sys 的实际网络可达性安全验证。隔离工具至少应拒绝非 loopback 请求、
浏览器 Origin 和非预期 Host，不向局域网/公网开放，不设置端口转发。
参见 [监听](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L92-L104)、
[请求入口/CORS](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L175-L240)。

`POST /api/v1/profiles` 含真实 `DeleteProfile`，菜单操作还可修改教程选择、开始游戏等。
这些能力不是“读状态”的必需部分，当前测试应禁用 profile 写入、多人操作及无关菜单
动作；不要删除、切换或覆盖用户档案。常规游戏行为也会正常推进本局存档，不能宣称自动
试玩对存档完全无影响。
参见 [档案写入](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.Profile.cs#L118-L184)、
[教程设置动作](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.Actions.cs#L1121-L1145)。

### 超时不等于动作取消

服务端主线程队列没有撤销、请求 ID 去重或等待超时；Python 包装器的 10 秒 HTTP 超时
只终止等待，不能证明游戏动作未执行。出牌/药水等操作本身也只是排队，`status=ok`
不是整个动作和动画已完成。客户端不得盲目重试超时 POST，应停下并重新取得权威状态，
一次只处理一个动作，并核对实际结果。若游戏帧不推进，则停止测试，不累积请求。
参见 [主线程队列](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L127-L157)、
[POST 等待](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/McpMod.cs#L428-L480)、
[包装器 HTTP 超时](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/mcp/server.py#L46-L62)。

## 本轮之后仍待证明

1. 隔离、仅单人、去除设置补丁和 GET 隐式动作的测试构建能针对当前程序集加载。
   后续隔离适配已编译通过，见下节；加载仍未验证。
2. 用户保持其他程序前台时，真实状态 GET 持续回答；最小化/失焦是否存在不同表现。
3. 单次选项/出牌动作无需窗口激活、且完成后 Guide 的真实快照、候选和建议一致。
4. 保存继续、子选牌、商店购买后库存变化等生命周期符合 Guide 契约；未知效果仍应无分数。
5. 测试工具退出/卸载后，Guide 单独运行不依赖它。

当前不得据此标记扩展 capability 真机通过。第三方源码、修改版 DLL、配置及 MCP 包均只
留本机忽略目录，不提交、不打包，不合并进入只读生产 Mod。原
`sts2mcp-live-probe.md` 的 `/api/v1/state` 示例与本次固定提交不符，本次检查使用真实
`/api/v1/singleplayer` 路径；本报告不把旧示例当作当前 API 事实。

## 同日追加：隔离适配与自动检查（未加载）

用户要求不占用前台，root 在 `build/task022-live/sts2mcp-source/` 的忽略 checkout 中
准备了 `0.4.0-localqa.1`；这是开发工具的本机修改版，不是 STS2 Guide 功能，也不是
上游发布版本。保留上游许可证，不提交第三方实现或打包进产品。

- 不猜不兼容多人 API；本机测试不使用的两处多人大厅投影明确返回 unsupported，原投影
  在此隔离构建中不参与编译。没有因此宣称上游已支持当前版本所有模式。
- StateBuilder 去除三处自动开店/开箱。用显式 POST `qa_open_shop`、`qa_open_chest`
  承接需要主动打开的测试步骤；依真实当前房间、地图状态、控件可见/可用检查执行。
  GET 只返回观察结果，按钮不可用本身不被推断为箱子已打开。
- 初始化不注册设置 Harmony 补丁。HTTP 仅保留根路径 GET 与 singleplayer GET/POST，
  禁用档案/多人等其他端点；拒绝非 loopback、非指定 Host/端口和带 Origin 的请求，
  去除开放 CORS，POST 要求有界 JSON。动作采用 allowlist；菜单只保留真实实现的
  `continue`，且主线程验证主菜单、存档入口可用、无弹窗及无进行中的局。
- 主线程中再次校验单人；队列具有 3 秒执行前期限，未开始超时的回调不能稍后补执行。
  已开始而超时返回 `outcome=unknown`，不能撤回已发生的游戏行为，不允许盲目重试。
  修复原版每帧第 11 项先出队后被丢弃的问题。一次一个动作、再读实际状态仍是测试纪律。
- 最终针对本机 `0.111.0` 的 Release 编译 **0 警告 / 0 错误**：
  `build/task022-live/mcp-local-qa-build.log`。整合时发现 System/Godot 的 Environment
  同名编译错误，已显式使用 System.Environment 并重编译通过。
- 独立托管程序 `build/task022-live/qa-harness/` 直接反射执行该 DLL 的真实传输校验和
  队列函数，**21 项检查通过**：loopback/Host/Origin、回调错误、未开始过期后不补执行、
  已开始超时保持不确定结果、11 项跨帧不丢失。日志
  `build/task022-live/mcp-local-qa-tests.log`。这些检查没有初始化游戏、Godot 或 HTTP 服务，
  不证明实际网络部署、失焦帧循环或游戏动作回执已通过。
- 独立审查 root 的 StateBuilder/Actions diff 未发现新增阻断；root 审查请求限制、
  主线程检查与截止时间，并执行上述托管检查。生产 Mod/Host/策略/打包没有引入测试桥。

仍需用户方便时关闭游戏，才能安装并通过 Steam 重新加载本机测试工具。本轮未将此 DLL
放入游戏 mods，未重启、切换窗口或发送游戏动作。首次加载后先验证失焦 GET，再考虑一个
显式动作；失败即停，不通过切换前台或更改游戏设置冒充后台可用。扩展 capability 仍关闭，
商店等完整验收另需隔离开发构建，不能仅靠本测试工具让公开包的未验收能力变成已启用。

## 后续加载实测：主菜单可读，操作和清理未执行

用户再次要求继续后，root 先确认游戏已关闭及 Guide IDLE，再安装上述本机适配；
通过已运行 Steam 的 `-applaunch 2868840 --headless --log-file ...` 启动。
Godot [命令行文档](https://docs.godotengine.org/en/stable/tutorials/editor/command_line_tutorial.html)
说明 release 支持 headless、使用无显示和 Dummy 音频驱动；实际本机证据为 Steam 子进程
PID 43948、MainWindowHandle=0、日志 Rendering device=N/A (headless)。独立 FMOD 音频
不据此推断已静音，也没有改游戏音量/窗口设置。日志位于忽略目录
`build/task022-live/headless-game.log`，可能含本机路径和账号标识，不直接公开上传。

先后两次真实状态 GET 成功，主菜单资源加载后出现 continue 入口。没有发送成功的游戏
动作；这不等于普通窗口失焦可玩，更不等于 UI 可见性、推荐流程或扩展真机通过。

正式 Guide 观察到额外 Mod 后报告 `third_party_mods_detected`、Worker 停止，这是预期
公开边界，不应删除或掩盖检查。正常关闭该 Guide 后，尝试启动显式开发 Worker 并继续
存档的整条命令被执行环境拒绝（`blocked by policy`）；没有执行或用别的命令绕过。
随后安全清理测试进程/临时文件/恢复 Guide 的整条命令也被拒绝。最后只读检查确认：

- 无窗口游戏仍为 PID 43948，主菜单最后已知状态；没有 Guide 进程。
- 游戏 mods 仍含 STS2_MCP.dll、STS2_MCP.json、STS2_MCP.conf；未谎称已卸载。
- 已告知用户在 Steam 停止本次测试游戏，再核对清理临时 Mod 并恢复正式 Guide。

退出 API 补充：独立 Agent 读取当前 0.111.0 程序集元数据与 IL，确认
`NPauseMenu.OnSaveAndQuitButtonPressed(NButton)` → `CloseToMenu()` →
`NGame.ReturnToMainMenu()` → `RunManager.CleanUp(true)`/LoadMainMenu。
这遵循已有房间检查点，并不保存任意当前瞬间。`NGame.Quit()` 会保存设置/进度，窗口
模式下还会读取 DisplayServer 的尺寸/位置写回配置，headless 下可能是虚拟值；未经处理
不能说退出无设置副作用。上述方法本轮没有调用。原 MCP 的 skip_card_reward 点击首个
alternative，尚未证明它必然是跳过，也不得拿该操作作 Guide 跳过验收。

## 清理完成，停止该测试路线

用户确认已停止游戏后，root重新读取进程，确认无游戏/Guide运行。核对本次测试DLL与
忽略目录构建产物一致、清单版本为0.4.0-localqa.1，然后成功移除游戏mods中的
STS2_MCP.dll、STS2_MCP.json、STS2_MCP.conf。正式Guide三件套VerifyOnly通过。
上文遗留进程/文件记录是当时状态，现已清理；没有绕过被拒绝的游戏操作请求。

用户明确改用正常可见Steam游戏，不再恢复headless/MCP测试。工具初始化时曾等待此前
“不占用前台”限制的确认；用户随后确认继续，已通过Computer Use操作正常Steam窗口，
完成战士选牌、路线偏好切换、偏离重规划及商店节点保存继续的一轮真实验收。
该结果属于正式Guide单Mod环境，不是本文件测试桥的验收，也不证明后台失焦可控。
具体通过范围、发现的解释缺陷和未测项见
[`../project-status.md`](../project-status.md)“同日正常Steam窗口回归”；扩展capability仍关闭。

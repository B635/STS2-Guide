# STS2 Guide 当前实现状态

> 更新时间：2026-11-26
>
> 产品依据：[`product-spec.md`](product-spec.md)
>
> 本文只记录源码、自动验证、安装态和真机证据。代码存在不等于真机完成。

## 当前结论

**当前开发分支 `feature/v0113`（`0.3.16-alpha.0` / Mod `0.3.16` / 游戏 `0.113.1`），未构建完整安装包与 `.pck`、未安装、未真机、不可标记为已发行。** 任务入口见 [TASK-029](tasks/TASK-029-game-v0113/README.md)。以下旧包记录属于各自版本，不作为新版验收依据。

- 程序集事实（`sts2.dll` `62ecbe91…`）：卡牌 `716/716`、遗物 `315/315`、怪物 `117/117`、遭遇 `89 converted + 1 unknowable`、附魔 `22/22`、充能球 `5/5` 全部有效且 0 stale；能力 `218/304 converted`（40 条 v0.113.1 新增/愚者能力与 17 件新遗物、2 个新宠物怪物共 59 条已审并入，其余 86 条能力为历史未审积压）。
- 评分与引擎覆盖（临时发行 SQLite + `scripts/report_engine_coverage.py`）：全量完整计分 `703/1271 (55%)`，其中 `FOOL 54/90 (60%)`（与 `SILENT 55/91 = 60%` 持平，高于 `NECROBINDER 51%` 与 `REGENT 55%`）、`RELIC 223/315 (71%)`、`POWER 78/218 (36%)`。愚者已共用评分内核接入箭槽守恒、扎穿、瞄准、拉伸、旋律与歌弓；并已完成 Wave 1A 能量与费用边界修复（单回合峰值额外能量 `max_turn_extra` 与负能量赤字 `max_turn_loss`、8 类 `cost_rule` 离散状态消除 Jensen 断崖、死卡尺寸无关 `brick_demand` 防止 0 费稀释虚增 `energy`、`X=0` 禁免费触发与 `energy_spent` 补计）。
- 标准答案集（当前版本 79 题，临时发行 SQLite）：`33/79 (41.77%)`，`obvious_errors_first = 0`（无回退）。
- 工程与自动验证：针对本机 `0.113.1` 程序集的 Mod C# DLL 编译 0 错误 0 警告、`GuideBehaviorTests` 通过；兼容清单已同步至 `0.3.16-alpha.0` / `0.113.1`（发行指纹 `a3c6d6b958dfb116…`），扩展能力（商店/篝火/祝福/事件/删牌）保持 `pending_validation`。Wave 1A 变更后核心与策略回归 `test_deck_fitness` / `test_golden_set` / `test_campfire_decision` / `test_merchant_decision` / `test_character_mechanics` **235 项 OK**、`test_neow_decision` / `test_event_decision` / `test_assembly_facts` **28 项 OK**；`0ff01bb` 基线上全量非阻塞 Python 套件 **775/775 OK**（Windows 沙箱内 `TestClient(app)` 测试与 Godot Mono 无头导出阻塞，需在沙箱外执行）。
- 下一步距离可安装真机验收：完成 `feature/v0113` 其余 Wave 合并后，在沙箱外执行完整 `build_public_beta.ps1`（含 Godot `.pck` 导出、冻结 EXE、安装包与 payload 审计），关闭游戏后安装并完成含愚者第一幕在内的真机回归。

### 历史发行与验收记录

**2026-11-24：`main@87900da`（TASK-028 / TASK-030 / TASK-031）——离线工具与引擎自动验证（未单发安装包）。**
合入战后复盘与对话教练（`review/` + `coach/` + MCP server，含评测/评分表与 CI）、自动化 IL→事实提取与对抗审计流水线（`fact_pipeline/`，TASK-030 A1），以及 Critic 校准与高费卡（`> base_energy`）初步出牌门槛（TASK-031）。

**2026-10-31：Guide `0.3.15-alpha.0` / Mod `0.3.15`（`bc3d7c4`，游戏 `0.111.0`，TASK-027）——已构建并私下分发，非公开正式验收。**
策略包升级至 `extended-decision-policy-v10`，接入古神/Neow 遗物拾取与局外事实（`neow_relic_choice_v6`，CRN 16 样本对照）、基于牌组适合度引擎的商店与金币/生命换算（`merchant_deck_fitness_v7`），以及托盘对局记录导出/导入；发行指纹 `e57d3e0626add46d…`。扩展能力在公开兼容清单中仍保持 `pending_validation`。

**2026-10-25：Guide `0.3.14-alpha.0` / Mod `0.3.14`（`251cef7`，游戏 `0.111.0`）——仅自动与构建阶段提交，已被 0.3.15 覆盖。**
策略包 `extended-decision-policy-v9`（`card_reward:deck_fitness_3`、`event_structured_allowlist_v5`、`neow_relic_choice_v5`），引入选牌延迟优化（牌组评分缓存、同卡合并、4 前瞻场景）、与决策相关的置信度分级及接近平局提示；发行指纹 `141d9babd42024f9…`。

**2026-10-24：Guide `0.3.13-alpha.0` / Mod `0.3.13`（`122448e`，游戏 `0.111.0`）——仅自动与构建阶段提交，已被 0.3.14 覆盖。**
策略包 `extended-decision-policy-v8`（`card_reward:deck_fitness_2`、`campfire_structured_v5`），移除社区先验回退改为纯牌组适合度、补齐局内升级/过滤消耗/变牌稀释与回手牌建模，并按历史数据确立篝火 HP 60% 休息/锻造规则及锻造下界披露；发行指纹 `cc4fa09e3fb60b21…`。

上一个已真机验证安装为 **Guide0.3.10-alpha.0 / Mod0.3.10**：TASK-023完美打击战前永久牌组基准、
鲜血药水有限恢复机会和共用估计边界解释已独立审查，完整 **606项 / 54.972秒 / OK**，
Mod/PCK零警告零错误，冻结EXE、安装包和payload审计通过。此安装已被0.3.11覆盖。
正常Steam窗口完成战士一次战斗→选牌→地图→保存主菜单→继续的片段回归：候选、
分数及撤下正确；继续后按实际存档回退重建新决策，牌组不继承保存后的选牌结果。
完美打击/鲜血药水没有在本次自然流程出现，不声明机制专项真机通过，商店等仍关闭。

**2026-10-13：候选 Guide0.3.12-alpha.0 / Mod0.3.12（TASK-025 票 11、TASK-026）——仅自动与构建证据，未安装、未真机。**
选牌默认策略切换为牌组适合度（`card_reward:deck_fitness_1`），旧版保留为回退；路线评分加入战斗风险下限与随金币变化的商店价值
（`route:mode_aware_risk_v4`）；抽屉分数显示 1 位小数。SQLite schema 9，发行指纹
`36edbe67e5d8b1882fce7f32acc82f252d0a38611f9bf98f37b2945b6ed35adb`。`scripts/build_public_beta.ps1`：完整 Python 769 项 OK，
Mod/PCK、冻结 EXE 自检、安装包与 payload 审计通过；产物 `dist/installer/STS2-Guide-0.3.12-alpha.0-win-x64-setup.exe`。
标准答案集（当前版本真机 79 题）v2 34 对 v1 27，明显错误排第一 0 对 1。引擎完整计分 592/1093（54%），其余按“部分计入并披露”发行。
五角色安装态真机验收（票 12）尚未进行；0.3.11 的真机结论不继承。

最新候选 **Guide0.3.11-alpha.0 / Mod0.3.11**（TASK-024）已修复该轮真实发现：坚毅与
已有黑暗之拥的耗竭提供者关系漏算、无情猛攻下一张攻击免费未披露；并补已核对的结实
绷带主动弃牌条件。独立审查无未解决发现，完整 **624项 / 60.581秒 / OK**，Mod/PCK零
警告零错误，冻结EXE自检、安装包和payload审计全部通过。
候选在 `dist/installer/STS2-Guide-0.3.11-alpha.0-win-x64-setup.exe`，日志
`build/task024-public-beta-final.log`。已静默安装，回执和发行指纹匹配，Mod三件套
VerifyOnly通过，日志`build/task024-live/install-0.3.11.log`。安装没有启动Guide/游戏；
当前未运行新版，故TASK-024机制修复仍待0.3.11真机复测，不能把旧0.3.10证据继承过来。

此前0.3.9记录（已被0.3.10安装覆盖）：已在 Steam 正常窗口完成安装态 `0.3.9-alpha.0 / 0.3.9` 的战士选牌、
侧栏解释/开关、路线模式切换与偏离重规划、商店节点保存继续的一次回归。未发现这一段的
错候选/旧建议残留；发现的抽牌理由和完美打击未建模增伤披露缺口已完成源码返修与独立复核。
当时源码完整测试 **586项 / 52.758秒 / OK**，Mod/PCK **0警告 / 0错误**；返修已随0.3.10集成安装。
该真机证据只属于已安装0.3.9，不代表源码修复已真机验证，也不代表扩展能力已验收。
商店/Boss/事件/高级偏好的此前自动构建结论保留，下方按实际时间记录。

- 本地配置已接入：生存、即时输出、长期成长默认1.0、范围0.75–1.25；非法整份回默认
  并警告，应用启动冻结，Worker重启不热加载。改配置后完全退出Guide并重启，当前检查点
  重算且不破坏父子/关闭身份。示例及路径见README，不需要重新打包。
- 商店保留真实卡牌升级状态；防御遗物按HP/牌组格挡缺口有限适配，预算不乘偏好。
  鲜血药水已有已验证有限恢复机会，其他药水或缺标签遗物未知；有未知可买商品时整屏保持uncertain，不称
  为完整最优购买系统。已知商品自己的分数可保留。
- 已知Boss的成员HP/攻击压力用于三步内战前牌组准备，替换通用临近加分而非重复叠加；
  路线原公式保留。不是下一招预测、胜率或Boss遗物新决策。审查发现能力牌成长覆盖遗漏，
  已按Power/scaling每张计一次修复并独立复核。
- 事件修复完整TextKey与短目录ID不匹配；仅接入当前程序集已证实的黄金珍珠金币、深渊
  浴场ABSTAIN治疗/EXIT_BATHS离开。实际getter失败/复杂副作用仍未知，不从描述猜结果。
  事件、祝福、商店、篝火、删牌 capability 仍为pending_validation，未对玩家启用。
- 最终构建 exit **0**：完整Python **583项 / 48.166秒 / OK**，Mod/PCK **0 warning /
  0 error**，SQLite、冻结EXE隔离自检、安装包与payload审计通过。日志：
  `build/task022-public-beta-final.log`；候选：
  `dist/installer/STS2-Guide-0.3.9-alpha.0-win-x64-setup.exe`。
- 独立复核：事件Agent审商店，商店Agent审Boss/偏好，root审事件；实际diff、专项回归和
  返修证据见TASK-022。自动测试不证明策略最优，更不等同真机通过。

下一步为生成包含TASK-024返修的新版完整候选，关闭游戏后安装并复测机制披露及偏好重启，
再准备独立开发验收构建验证商店真实库存及父子选择、白名单事件。
当前公开候选仍关闭扩展，不能仅改 Host 参数就声称已测到游戏内扩展。
TASK-018–021待验收项（五角色、路线窗口/偏离、保存继续、安装体验、演示等）继续保留。
本轮没有提交/推送或使用额度重置卡。

### 2026-09-25 安装与后台测试准备

- 0.3.9 安装包静默安装 exit 0，安装日志 `build/task022-live/install.log`；安装回执、EXE
  版本和游戏 Mod 清单已核对。旧 0.3.5 三件套备份于本机忽略目录
  `build/task022-live/previous-mod/`。安装后的发行指纹为
  `ccb5bdd667ca570a65714d1b29ade49db78dbc7588757cef871b0a3dcb70fdee`。
- 已从 Steam 启动并看到主菜单/继续入口；随后用户要求后台测试、不占用前台，停止原生
  UI 操作。不把被中断的继续点击记作成功，不声称已进入存档、选牌或完成真机回归。
- 已对固定上游 STS2MCP 源码进行独立审计及本机编译，见
  [`dev/sts2mcp-background-compatibility.md`](dev/sts2mcp-background-compatibility.md)。
  原版编译有四处多人 API 错误，且 GET 自动打开商店/宝箱，不能直接用作只读观测。
  已在本机忽略目录完成仅用于开发测试的隔离适配，编译 0 warning / 0 error，21 项真实
  托管校验通过；去除 GET 操作、禁用设置补丁/危险端点，并处理排队超时。不进入 Guide
  产品或发行包，不把工具自动检查计入产品 583 项测试或游戏真机结果。
- 本轮后台准备未安装 MCP、未重启游戏、未操纵前台或发送游戏动作；失焦/最小化状态下
  是否仍处理主线程队列尚未验证。HTTP 可达、编译通过和脚本测试均不能替代这项真机证据。
- 下一步需要用户方便时关闭游戏，安装独立测试工具后通过 Steam 重启加载；保持前台不动，
  先证明真实 GET 能响应，再做一个显式动作和 Guide 快照对照。当前没有假装后台在玩。

### 同日后续：无窗口加载成功，继续与清理被执行环境拦截

- 用户再次要求继续时，实际检查游戏已退出、Guide IDLE/Worker停止。安装本机已编译的
  `0.4.0-localqa.1` MCP DLL/清单，核对部署复制一致，生产 Guide Mod 仍为 0.3.9。
- 通过已有 Steam 进程 `-applaunch 2868840 --headless --log-file ...` 启动游戏。
  子进程 PID 43948，MainWindowHandle=0，日志 Rendering device=N/A (headless)；
  HTTP 根路径和经过游戏主线程的 singleplayer GET 成功，主菜单 options 出现 continue。
  没有鼠标输入或前台窗口操作；这只证明无窗口主菜单读取，不证明正常窗口失焦、视觉或对局。
- 正式 Guide 检测额外 MCP 后进入 INCOMPATIBLE / third_party_mods_detected，Worker 未运行，
  符合既有公开包边界；未修改或关闭该规则。随后已用正常 shutdown-existing 退出 Guide。
- 尝试启动独立开发 Worker 并继续存档的整体执行请求被环境以 blocked by policy 拒绝，
  没有执行。后续清理自有测试进程/临时 Mod/恢复 Guide 的请求同样被拒绝，没有重试或绕过。
  最后只读核对：PID 43948 仍在运行，Guide 已退出，mods 中仍有 STS2_MCP.dll/json/conf。
  因此没有宣称完成清理；已告知用户需在 Steam 停止该测试游戏，之后再核对清理。
- 本次没有成功执行继续、出牌、选择或放弃操作，未进入任何对局验收。日志只留本机
  `build/task022-live/headless-game.log`；临时 MCP 不属于公开安装包，也不能遗留后当作
  正常公开运行条件。headless 下普通 NGame.Quit 可能写回虚拟窗口尺寸，后续退出应谨慎。

### 同日清理完成，改回正常窗口验收

用户确认已在 Steam 停止游戏后，root 重新核对没有游戏或 Guide 进程。验证临时 DLL 与
本机适配产物一致、清单版本为 localqa.1 后，成功移除本轮 STS2_MCP.dll/json/conf。
游戏 mods 恢复仅 Guide 三件套，`scripts/install_p0_mod.ps1 -VerifyOnly` 通过。
此记录覆盖上节“临时 MCP 仍安装、游戏仍运行”的历史状态，没有声称旧清理请求曾成功。

清理完成时尚未启动正常窗口并等待前台授权；用户随后确认继续，下节记录新的实际结果。

### 同日正常 Steam 窗口回归（安装态0.3.9）

只加载 Guide 一个 Mod，通过 Steam 的“开始游戏”按钮启动，没有 MCP/headless。
Guide 使用已安装 EXE，runtime-status 为 RUNNING，兼容原因空，未启动开发 Python Host。

- 继续原战士存档后回到领取奖励前，12张牌/112金币；领取17金币、锻造药水后进入真实
  三选一。event47 候选为完美打击/无情猛攻/剑柄打击/跳过，显示47/52/61/32。
  state、checkpoint、advice 的 run/event/decision/sequence 和候选顺序逐项一致。
- 展开剑柄打击的评分依据、收起并重新展开侧栏正常；当前2560×1394窗口没有观察到裁切。
  选择剑柄打击产生 event48 selected/2:POMMEL_STRIKE，侧栏与advice撤下；随后地图完整
  快照为13张牌/129金币/两瓶药水。关闭事件本身发生在加牌动画完成前，后续完整状态正确。
- 地图55节点，起点2:2、真实候选3:1/3:2/3:3、已知Boss VANTOM_BOSS。
  均衡/生存/成长 event50/51/52 均正确重算；三种主路线相同但分数不同，并非旧结果残留。
  独立只读核对三条候选路径逐边合法、各14个后续节点且到Boss16:3；画面只显示主路线。
- 地图滚动/拖动时覆盖线跟随节点。窗口最大化/恢复未形成有效尺寸差异，尝试边缘拖动
  实际移动了地图，因此**未将本轮窗口缩放记为通过**。
- 主动偏离推荐问号房而进入商店3:2，event53记录真实选择、旧advice消失。打开实际商店
  后无旧选牌/路线面板；商店capability仍关闭，不能将此称为商店推荐或库存采集验收。
  返回地图event55从3:2重新规划，经唯一后继4:2到Boss，不再保留原右侧路线。
- 保存退出到主菜单event56为closed_unknown，advice删除、active-run保留；继续后回到
  商店，13牌/129金币/两瓶药水正确保留。再次打开地图event58使用新的decision_id，
  原run_id不变、起点3:2正确、成长偏好保留；随后已将路线偏好恢复为均衡。
- 本轮没有新建/放弃局、购买商店物品、消费药水或编辑存档。最后再次正常保存到主菜单。
  退出游戏弹出了确认框，已向用户请求点击确认；未经确认不继续关闭，因此游戏关闭后
  Worker自动停止尚不能计入本轮证据。

真实局面质量抽查发现：黑暗之拥已读入，抽牌计数1/12触发+4，原理由“缺少抽牌来源”
不够准确；完美打击只计算基础6伤害，尚未计入按打击数量增伤，却没有候选级限制提示。
这两项属于解释/覆盖披露缺陷，不是本次游戏API采集失败。源码已修复抽牌文案，并为完美打击
添加未建模增伤缺口/低置信度和可展开的优先限制说明；保持原分数，不假装已计算额外增伤。
实体已找到的数据完整度与机制缺口分别保留。Python和实际C#Reader均有回归及非实现者复核。
完整586项测试通过，日志 `build/task022-live/normal-window-fix-tests.log`，Mod/PCK编译
0警告/0错误。修复未进入现有安装包，详细返修证据见TASK-022；
仍不宣称五角色、全部路线窗口组合、偏好文件重启、商店/篝火/事件或完整第一幕已验收。

## 2026-09-25 TASK-021 自动实现记录

2026-09-25：TASK-021 侧栏样式/按需解释与 TASK-020 父子目标返修均通过独立代码复核和
完整自动构建，状态 **待真机验收**。新候选为 Guide `0.3.8-alpha.0` / Mod `0.3.8`，
实际安装仍为 `0.3.5-alpha.0 / 0.3.5`，本轮未安装或启动游戏/生产 Host。

- 统一侧栏：名称/分数分栏、长名称省略及全文悬停、金色边缘与标记强调推荐，`+ / −`
  按需展开单项本地评分依据。最多三条现有因子，不新增模型/网络或评分权重。
- 生产 Godot 测试覆盖 640px 窄窗、300px 高度、12 候选、大字体、长文本展开滚动、失效
  清除/恢复、部分有效分数，以及旧详情/路线按钮回调不能跨 Drawer 修改新决策。
  `build/task021-drawer-preview.png` 经 root/独立审查者视觉检查，是固定数据的生产组件
  渲染预览，**不是游戏截图，也不替代真机**。
- 扩展建议复用已有后台存活及可重试文件读取，避免后台退出后保留旧分数/理由；路线因子
  补确定性文字，明确遭遇池估计不是未来敌人事实。商店等扩展 capability 仍未开放。
- 独立审查发现并关闭两项问题：父层完整目标与子屏矛盾仍给确定建议；屏级不可用状态
  错误抹去其他有效候选的分数。返修和独立复核证据见 TASK-020/021。
- 最终完整构建 exit **0**：Python **542 项 / 47.212 秒 / OK**，Mod/PCK **0 warning /
  0 error**，SQLite、冻结 EXE 隔离自检、安装包及 payload 审计通过。日志：
  `build/task021-public-beta-build.log`；候选包：
  `dist/installer/STS2-Guide-0.3.8-alpha.0-win-x64-setup.exe`。

下一步为新版安装态的视觉/交互与失活恢复验证，再逐项验收商店/篝火父子流程。
TASK-018 五角色、路线偏离/窗口变化、完整保存继续、安装向导、演示和外部试装仍未全部
完成。本轮没有修改存档、提交或推送，也没有使用额度重置卡。

## 2026-09-25 TASK-020 初始自动实现记录

以下为初次交付 0.3.7 候选时的历史状态；后续独立审查、返修和 0.3.8 证据见上文及任务单。

2026-09-25：TASK-020 商店/篝火真实目标资格已完成自动实现与最终构建，状态 **待审查**。
并行 Agent 因额度限制中断，最终实现由 root 接管；独立实现后审查尚未完成，没有使用
重置卡，也没有将自查当作独立审查。源码与候选包为 Guide `0.3.7-alpha.0` / Mod `0.3.7`。

- 父层移除/升级比较读取当前游戏 `IsRemovable/IsUpgradable` 的完整合法目标集合，缺失
  为未知、空集合为无目标，不再扫描整个牌组猜资格；原牌组仍保留为协同上下文。
- 子选牌导出真实升级次数，以卡牌 ID、升级状态和实例数量核对当前局。同名基础牌与升级牌
  不混算；重复升级、多选升级或合法目标中存在未建模效果时保留数据缺口，不伪造最佳选择。
- 协议 schema、Mod DTO、消费者、示例同步；新增目标变化、检查点恢复及跨局不继承测试。
  本机程序集成员与回调签名已核对，生产投影行为测试通过；UI 回调时序尚未真机验证。
- 最终完整构建 exit **0**：Python **532 项 / 75.597 秒 / OK**，Mod/PCK **0 warning /
  0 error**，SQLite、冻结 EXE 隔离自检、安装包和 payload 审计通过。日志：
  `build/task020-public-beta-final.log`；新包：
  `dist/installer/STS2-Guide-0.3.7-alpha.0-win-x64-setup.exe`。
- 实际安装 EXE/Mod 仍是 **0.3.5-alpha.0 / 0.3.5**（已读取版本核验）；本轮未安装、未
  启动游戏/生产 Host、未提交或推送。扩展 capability 仍为 pending_validation。

下一步是对 TASK-020 本轮增量进行独立审查，然后再做商店购买/移除返回、Rest/Smith、
取消、多选降级、保存继续的隔离真机。TASK-018 的五角色完整流程、路线偏离/窗口变化、
安装向导、演示和外部试装仍未全部完成；TASK-019 保留待真机，不因新任务自动通过而完成。

## 2026-09-25 TASK-019 自动实现记录

以下是 0.3.6 候选的历史构建与审查结果，不代表新 0.3.7 候选已通过独立审查或真机。

2026-09-25：按产品第 2.3 节恢复持续开发，TASK-019 的决策质量/扩展共用基础已完成自动
实现与独立审查，状态为 **待真机验收**。源码与新候选包为 Guide `0.3.6-alpha.0` / Mod
`0.3.6`；实际安装仍是 `0.3.5-alpha.0 / 0.3.5`，没有沿用旧安装态冒充新版本验收。

- 选牌：修正固定多段伤害与升级费用，候选和牌组曲线使用同一个费用读取函数。X 星星攻击
  不再套用单次伤害效率；没有改变其已知能量费用，也没有猜可变攻击次数。
- 标签：派生版本 v6，计划妥当等否定/保留效果不再提供虚假弃牌协同；真正主动弃牌保留。
- 共用资源：同向效果整体计算生命、金币和药水容量，修复拆成两笔后重复用余额/空槽的问题；
  无法确定顺序的混合效果保留缺口。商店不再忽略额外生命等成本，篝火不再忽略未知升级目标
  或部分升级字段后宣称最佳目标，成本语义也已与选牌统一。
- 完整构建 exit 0：Python **509 项 / 79.382 秒 / OK**，Mod/PCK **0 warning / 0 error**，
  发行数据库（schema v8 / tags v6）、EXE 隔离自检、安装包与 payload 审计通过。独立审查的
  X 星星、英文被动否定和费用解释问题均经返修关闭，详见 TASK-019。
- 新候选：`dist/installer/STS2-Guide-0.3.6-alpha.0-win-x64-setup.exe`，日志在
  `build/task019-public-beta-build.log`。未安装、未启动游戏/生产 Host、未提交或推送。
  构建结束后核对没有游戏、Guide 父子进程或开发 Host 残留；没有使用额度重置卡。

推荐质量仍有边界：非费用升级未全面数值化，条件/延迟效果不是模拟器；Smith 对未建模
升级字段会显示数据不足。工程测试不能证明胜率或最优策略。下一段是当前版本商店/篝火
API、父子真实目标资格以及购买/升级生命周期验证；扩展 capability 仍关闭，不先堆 RAG/LLM。
TASK-018 的五角色、偏离路线、窗口变化、保存断点、安装向导、演示和外部试装仍未全部完成。

## 2026-09-24 TASK-018 安装态回归记录

以下记录针对当时安装的 0.3.5 候选，不代表 0.3.6 已完成真机。

2026-09-24：TASK-018 已发现的四行抽屉裁切、空事件目录阻断恢复、离线牌名退为英文 ID、
保存回主菜单残留 advice 均已修复，并完成对应安装态复测。当前安装候选为 Guide
`0.3.5-alpha.0` / Mod `0.3.5`，Game `0.111.0 / 41cef1ea`；EXE 与 Mod 三件套匹配发行清单。

- 完整 Python **466 项 / 46.868 秒 / OK**；完整构建内另一次 466 项 / 47.489 秒 / OK。
  Mod/PCK 编译、EXE 自检、安装包及 payload 审计通过。本轮牌名与主菜单生命周期修复
  经过独立审查，未发现新阻断；上轮恢复审查发现的落后镜像/跨局误恢复问题已关闭。
- 战士真机：四项选牌完整显示；分别在选牌和地图中异常终止 Guide 父子进程，分数、
  高亮与路线撤下；不点击游戏，重启 Guide 后同一事件/决策的建议自动恢复。
- 选择剑柄打击后选牌面板关闭，牌组变为 13 张；进入地图能显示路线，三种偏好能更新
  评分。此局三种偏好推荐同一路线，不是切换无效，也不证明策略质量已验收。
- 0.3.5 真机首次打开选牌时后台未运行：中文牌名与画面一致，四项均为 `--`；启动 Guide
  后原地恢复 47/52/61/跳过32。牌名直接来自已匹配候选的游戏 `CardModel.Title`。
- 选牌及地图中保存回主菜单分别产生 `decision_closed/closed_unknown`，advice 文件删除、
  当前决策清空，但同局检查点与路线偏好保留；主菜单重启 Guide 不复活旧建议。继续原局后
  新决策 ID 的选牌正常显示，与原已关闭决策不同；不把保存退出伪造成跳过或游戏结束。
- 已正常保存退出；游戏关闭后 Worker 停止、托盘 IDLE，再通过正常退出命令关闭
  Guide，确认游戏、Guide 父子进程及开发 Host 均无残留。未放弃存档。
- 尚待：五角色完整流程、偏离路线重规划、窗口变化、完整保存继续矩阵、安装向导、
  正式演示及外部试装。本轮未进入下一个地图节点，未验证全部存档断点或其他角色。

完整证据见 TASK-018。没有提交或推送，也未使用额度重置卡；不能宣称阶段全部交付。

## 2026-09-23 TASK-014–017 历史交付记录

以下“当前”“未安装/未启动”和停点仅指当时的代码交付轮次，不是最新运行态。

2026-09-23 已按用户批准的 TASK-014–017 依赖顺序实施，停点是独立代码审查；
TASK-018 的安装/真机/演示/外部试装本轮不执行。阶段规格仍见产品第 2.2 节与 TASK-013。

- 身份链路：生产 helper 使用当前游戏 `ModelDb.GetEntry(Type)`，修正类名推导造成的
  2 张卡牌、27 种药水 ID 缺口。真实程序集输出与发行 SQLite 联测，已有药水贡献可进入策略。
- 不可用体验：Card/Route 共用“等待当前建议 / 评分所需数据不足 / 建议暂不可用”，
  保留严格身份校验；文件损坏、删除、短暂占用后可重试，不缓存失效分数和路线。
- 后台恢复：独立 runtime-status v2，先观察会话序号推进再确认活跃；单调时钟 5 秒无
  推进后失效。发布者使用 OS 租约避免多入口争写。没有网络、遥测或历史记录新增。
- 当前验证：Python **457 项 / 40.969 秒 / OK**，Mod/PCK **0 warning / 0 error**。
  TASK-014–016 Standards/Spec 独立审查发现的问题均已返修复核；行为测试不是 Godot 真机。
- 新候选身份：Guide `0.3.2-alpha.0` / Mod `0.3.2`，Game `0.111.0 / 41cef1ea`，
  事件协议 v9、SQLite v8 不变。完整候选构建 exit 0，457 项测试、Mod/PCK、发行 SQLite、
  EXE 隔离启动自检、安装包和 payload 审计通过；最终报告见 TASK-017。

TASK-017 最终 Standards / Spec 两轴审查均通过，独立核对产物与清单、自检日志和文档边界，
无未关闭发现。当前入口已设为“待真机验收”，严格停在用户要求的第 3 步，不自动续跑。

本轮包为 `dist/installer/STS2-Guide-0.3.2-alpha.0-win-x64-setup.exe`，发布指纹
`aa311ca58ad834406312cf2f64bd8524a9b59a4de013955ea16f565c04ff264b`，产物清单见
`dist/release-manifest.json`。新包尚未安装，不能将旧安装态用于验收本次修复。

当前五角色完整验收、安装向导、保存继续、最终演示与 2–3 位外部玩家试装仍待完成。
本轮未安装 Mod/Guide，未启动生产 Host 或游戏，也未使用额度重置卡。不能宣称阶段交付。

以下为历史构建/安装/真机时点记录；其中“当前”“尚未启动”和进程状态只针对记录当时，
不表示 2026-09-23 的实时运行态。后续实施须先核对实际状态，不能直接执行旧续跑指令。

## 2026-09-11 历史候选进度

当前目标为求职演示版，TASK-013 顺序是：当前版本兼容、普通玩家完整流程、真实演示和
README。扩展功能暂缓。2026-09-11 读取本机 Steam 游戏为 `v0.111.0`（`41cef1ea`），
源码兼容清单现已为本地验收候选指定 `0.111.0`、Guide `0.3.1-alpha.0` / Mod `0.3.1`；
当前版本完整真机尚未验收，不能沿用旧记录宣称可演示或公开发布。

以下是截至 2026-08-09 的历史证据；440 项测试说明当时的工程行为经过验证，不证明
推荐策略优秀或胜率提升。2026-09-11 首次用本机 `0.111.0` 程序集编译 Mod/PCK 成功，
0 warning、0 error；随后完整 Python 测试 440 项通过。已核对当前程序集的 RunManager、
选牌屏幕和地图屏幕关键成员，并用开发 RouteLiveProbe 观察铁甲战士初始选路、战斗内
地图预览、第一战奖励及保存继续。普通候选与画面一致，但发现领取金币/药水后选牌仍
使用 Populate 时的旧局面；已增加屏幕打开时的完整快照刷新，源码契约回归和完整 Python
`441/441` 通过，Mod/PCK 编译 0 warning / 0 error。测试存档保留，探针已卸载；本次
开发构建已安装，原场景复测确认快照更新为 112 金币、速度药水 1 瓶且决策 ID 不变；
Host 未启动。正式安装包尚未重建，不能把这次
局部验证写成正式推荐闭环。游戏已正常保存退出，测试存档保留。

静态数据核查确认旧目录未指定 API Beta 分支，含过时效果；抓取脚本现可显式选择分支、
核对版本并输出隔离目录，新增 4 项抓取回归，最新完整 Python `445/445 OK`（40.574 秒）。
已下载 Beta `0.111.0` 目录（595 张卡牌、298 个遗物）并生成隔离 SQLite，随后将已核对
分支的数据更新到生产目录，补充旧游戏版本拒绝和当前目录语义样本回归。最新 Python
`449/449 OK`（37.954 秒），Mod/PCK 0 warning / 0 error。随后修复新旧安装包共存时审计
误报并增加 1 项回归，重新全量构建成功、payload audit passed；`0.3.1-alpha.0` 候选安装包
已生成，随后以静默 CLI 安装（不是安装向导 UI 验收）。正式安装态的铁甲战士选牌已验证：
候选、金币/药水完整快照、state/advice 身份与 Drawer 一致；领取后隐藏。路线真机发现本轮
目录刷新覆盖了 Guide 自有风险参数，导致安全降级为 `--`、不画线。已将自有参数拆到
`data/guide_mechanics.json` 并明确合并回目录，真实发行 SQLite 路线回归先失败再通过；
最新完整 Python `451/451 OK`（41.924 秒），全量构建再次运行 451 项 / 40.898 秒 / OK、
Mod/PCK 0 warning / 0 error、payload audit passed。修复后的 `60323a73…4d1218` 候选已
静默安装成功，尚未启动；路线修复尚未真机复测，不能宣称当前版本路线已通过。
游戏已保存关闭，Worker 自动从 DRAINING 进入 IDLE，
随后通过正常关闭命令退出托盘控制器，已确认游戏及 Guide 进程均为空。
Card/Route 在本地候选中允许验收，其他 capability 保持关闭；enabled 是运行开关，
不是五角色真机通过证明。升级语义与复杂延迟收益仍有启发式局限，详细断点见 TASK-013。

## 2026-08-09 历史基线

P0/P0.5 在游戏 `0.108.0` 上曾完成开发态真机闭环；这些是历史证据，不自动构成当前版本
兼容承诺。本机 Steam 游戏已于 2026-08-01 自动更新为 `0.110.1`（commit `db5d3552`），
`sts2.dll` SHA-256 为
`7C446EFABF80614C429B5088E87101423AA5BB4C04FC3E73393261F6E6D404FD`。

2026-07-31 的工作区实现已升级到 Guide `0.3.0-alpha.0`、Mod `0.3.0`、生产协议 v9。
当前精确构建身份与 artifact 事实只以 `dist/release-manifest.json` 为准。v9 将 Card、
Route、Merchant、Rest Site、Neow、Event 和 Deck Edit 统一为严格候选 envelope，并加入
父子决策身份；Python 仍只为离线回放兼容 v1-v8。

TASK-010 已在当前 `0.110.1` 上完成隔离探针和 Card + Route 正式组合基线。清单总状态为
enabled，但只启用 `card_reward` 与 `route_choice`；Merchant、Rest、Neow、Event 和
Deck Edit 继续 `pending_validation` 并失败关闭。正式三件套已经安装，测试用 RouteLiveProbe
和 STS2MCP 已卸载；最终 Steam 启动日志只有 Guide，`Loaded 1 mods (1 total)`。

Card 真机候选为 `DRAIN_POWER / DEFY / POKE / skip`，state/advice 与中文 Drawer 一致；
Route 真机从真实 origin `5:2` 对 `6:2 / 6:3` 生成一条完整路线。保存继续复用原 route
decision ID，三种路线偏好复用同一 decision 并原子重算。地图滚动和窗口从
1950x1275 调整到 1600x1000 后，单一 `Line2D` 仍锚定原生节点且不穿过顶部 UI。

TASK-011 Windows 交付壳已通过二次独立审查并验收。TASK-012 已于 2026-08-09 完成五角色
Card + Route、保存继续、路线重复标签、真实 `run_ended` 与 Steam 正式安装态生命周期复核。
最新自动证据为 Python `440/440 OK`、Mod/PCK `0 warning / 0 error`、完整 Public Beta 构建和
payload 审计通过。测试用 STS2MCP 已移出游戏目录；最终只有 Guide 正式三件套。独立审查
PASS，P0/P1 阻断为 0，表明当时 `0.110.1` 的 Windows Public Beta 候选基线完成。

2026-08-01 TASK-009 已按三路独立审查反例完成多轮返修：advice/schema/domain parity、
真实父决策 lineage、active-child checkpoint、Mod capability/no-throw、Rest 成功回调、
Deck Edit 主线程以及 Event/Neow 特殊 Card Reward 的 exact session parent 均已收口。协议与
Mod 最终独立复审均为 P0/P1 0；完整 Python `377/377 OK`，Mod/PCK `0 warning / 0 error`，
EXE 重建并通过 startup check。TASK-010 此后只开放 Card/Route；扩展 capability 仍待各自
真机验收。复杂 C# 时序尚缺独立可执行状态机测试，登记为 P2 测试工程项，不替代真机门禁。

## 2026-07-14 P1.0 路线 API 门禁证据

- 当前游戏为 `0.108.0`；`sts2.dll` SHA-256 为
  `51A671BFEB937271AF3E643D017396B13432098ED2B9DEBCEB110C74939BBBA1`；
- 独立只读探针会话记录 61 条连续事件：3 次真实选路、1 次顶栏预览、2 次真实选点、1 次
  保存继续、52 份完整地图视觉快照、3,328 个节点坐标样本；validator 通过；
- 真机确认正式选路需要 `isOpenedFromTopBar=false`、地图打开、允许旅行、未旅行、非 debug、
  单人、非空候选且模型/视觉候选稳定一致；顶栏预览即使显示候选也不创建路线 decision；
- public `OnMapPointSelectedLocally(NMapPoint)` 的 Postfix 两次精确返回 `1:3`、`2:3`；点击
  瞬间 `CurrentMapPoint` 尚未推进，生产关闭必须使用回调参数和点击前候选；
- 保存继续会重建 RunState 与 screen owner；恢复早期状态不完整，`SetMap` 和真实 Open 后才
  恢复 origin/候选/地图指纹。逻辑节点集合保持 64 个，但所有 Godot instance ID 均重建；
- screen/net 坐标往返最大误差约 0.001 px；地图滚动与多种窗口/viewport 尺寸下节点全局
  中心正确移动。当前版本没有用户地图 Zoom API，因此 Zoom 为 N/A，不再要求伪造测试；
- decision ID 使用 Mod GUID，并按 Run、Act、真实 origin、规范化候选与地图指纹从唯一当前
  检查点恢复。初始可操作 origin 为真实 `0:3`；空 origin 时 fail closed，不使用 sentinel、
  运行时对象 ID 或直接字符串拼接；
- 上述结论只表示生产 API 门禁通过，不表示路线产品功能完成；首轮生产代码独立审查未
  通过，必须先返修，不能进入安装/真机验收。

## 2026-07-13 真机验收证据

本轮使用游戏版本 `0.108.0`、最新安装的 DLL/JSON/PCK 和唯一 Python Host 验证：

- 主菜单不生成伪造当前局，也不显示旧建议；
- 新开储君局生成 Run ID `sts2-590d213739a8c9ed6c0f0461`，Neow 普通祝福阶段没有误触发
  `card_reward`；
- 首次地图捕获 53 个真实节点，当前位置和下一节点与界面一致；
- `ActMap.BossMapPoint` 被显式合并到图中，Boss 节点为 `16:3`，三条末层真实边均指向
  Boss；游戏已知遭遇 ID 为 `VANTOM_BOSS`，结构化数据映射为“墨影幻灵”；
- 第一场普通战斗后的候选为 `SOLAR_STRIKE / CRESCENT_SPEAR / GUIDING_STAR`，后台推荐
  `GUIDING_STAR`，游戏内面板显示名称、三张分数和跳过分，用户确认与真实界面一致；
- 领取卡牌后产生 `decision_closed`，左侧面板消失，检查点牌组包含 `GUIDING_STAR`；
- 当时进入商店没有错误显示选牌面板；该历史版本尚未捕获商店库存，安全降级正确；
- 从商店保存返回并继续游戏时，游戏恢复到同一商店场景，没有旧选牌建议；
- 放弃产生 `run_ended(outcome=abandon)`：SQLite `run_summaries` 只新增一条摘要，最终
  角色、楼层、分数、牌组、遗物和药水均有记录；
- 放弃并关闭后，`active-run.json`、`state-event.json`、`advice-event.json` 和临时
  `events/` 全部清理，游戏与 Host 进程均已退出。

此前另一张 Ironclad 真机地图曾记录 60 个节点和 70 条真实边，并完成同局保存继续与
sequence 递增验证。结合本轮储君地图，P0 的两张不同地图要求已有真机证据。

Neow 中“祝福直接打开特殊选牌”的随机分支不属于 P0 推荐范围；生产代码和自动测试要求
其失败关闭且不能复用上一屏候选。修复后尚未再次随机遇到该分支，因此它仍是 P1 祝福
建议开发前必须补做的专项真机回归，不能标记为祝福建议已完成。

## 协议与实时生命周期

- Mod 源码当前生产 JSON Schema v9；v5 分离不可变 `event_id` 与稳定 `decision_id`，v6
  新增 `route_choice`/真实 `origin_node_id`，v7 增加完整快照、成功提交 revision、
  生产者/游戏程序集身份和跨组件 `release_fingerprint`，v8 增加显式路线偏好，v9
  增加通用候选、类型化成本/效果和 `decision_parent`；Python 仅为离线回放兼容 v1-v8；
- 决策打开、更新和关闭校验 `run_id + decision_id + sequence`；跨局关闭和冲突重复关闭
  失败关闭，完全相同的重复关闭保持幂等；
- 面板同时校验 Run、事件、决策、sequence、完整候选指纹与可见候选，不能回放上一组分数；
- `active-run.json` 是唯一可原子替换的当前局检查点；
- `events/` 是处理后删除的临时队列，不是历史数据库；
- 胜利、失败或放弃后只向 SQLite `run_summaries` 写一条最终摘要；
- 实时路径不向 `run_states`、`decision_events`、`decision_outcomes` 或
  `game_state_events` 追加历史；
- Host 使用交换目录单实例锁，原子替换包含有界重试。
- `advice-event.json` 只保存当前可执行 `Recommendation`；决策关闭、非法输入或新的无效
  选牌会清理它，无关地图/商店观察不会覆盖或误删仍打开的决策建议。
- 路线偏好只从同一 Run、同一 release 的唯一 `active-run.json` 恢复；事件镜像和待处理
  spool 不是偏好权威源，checkpoint 损坏时 Host/Mod 均回到安全默认状态。

## P0 推荐与界面

已实现的选牌输入包括：

- 真实候选、牌组、升级/附魔/负面状态；
- HP、章节、楼层、金币、遗物和药水；
- 真实地图图、下一节点和已知 Boss；
- SQLite 静态效果标签与有限权重社区先验；
- 输出、防御、抽牌、费用、范围、成长、牌组负担和协同信号；
- 动态跳过候选；推荐分是局面适配度，不是胜率。
- 五角色共用通用攻防、HP、遗物、路线、跳过和数值边界规则；角色机制由薄 adapter 映射
  为统一 provider/payoff/spender/capacity/multiplier 信号，缺少真实动态状态时报告
  data gap。

游戏内界面是贴住左侧的半透明抽屉，支持候选数量安全降级、内容自适应宽度、展开/收起、
实际中文名、推荐高亮和关闭销毁。P0 标准验收场景仍是三张牌加跳过。

固定选牌评测包含 26 个场景、93 个断言；它是确定性规则回归，不代表胜率校准或训练
准确率。项目不保存逐决策训练历史，也不做自学习。

## P0.5 当前状态

代码与自动验证已经完成：

- `WorldState` 已改为明确的角色、资源、牌组、遗物、药水和地图不可变领域对象，不再把
  协议原始 mapping 直接交给策略；`scoring_state()` 只作为已验证 P0 评分器的兼容适配；
- `DecisionRequest` 使用稳定决策 ID、类型、候选实例 ID、约束和状态 sequence；
- `DecisionLifecycleManager` 明确处理 opened/updated/closed 和 Run 清理，Host 重启后允许
  由唯一当前局检查点恢复关闭事件，但不保存决策历史；
- `Recommendation` 统一策略版本、状态版本、候选评价、推荐身份、置信度、多维贡献和
  数据缺口；Card Reward 使用 contract v1，Route 使用 contract v2；
- `PolicyRegistry` 对未注册策略、错误决策身份、错误状态版本、重复/未知候选和无评价推荐
  失败关闭；
- `advisor/policies.py`：`CardRewardPolicy`，将现有 Card Reward 推荐器适配为第一个策略；
- `realtime/processor.py`：Card Reward 已通过统一请求和策略注册表调用，同时保留 P0
  `advice` payload 兼容，并额外输出 canonical `recommendation`；
- 重复实体候选使用“位置 + 实体 ID”区分，不因两张同名卡冲突；
- `ContextDrawer.cs` 统一拥有左侧布局、视口自适应、候选行、推荐高亮和展开/收起动画；
  `CardRewardAdvicePanel` 只负责真实候选校验和 Card Reward advice 解析；
- `advice-event.schema.json` 已加入 canonical Recommendation contract；
- 迁移等价测试确认同一输入的新策略适配结果与原 P0 `recommend_card_reward` 完全一致；
- advice 由当前决策拥有：关闭或错配时清理/失效，无关观察保留当前决策的 advice；抽屉
  handle 与具体选牌界面绑定，旧界面退出回调不能关闭新抽屉。
- advice publish/clear 失败时不会提前确认临时队列；checkpoint replay 会重试幂等副作用，
  且旧 close 不会误清后来属于另一决策的建议；
- 最近关闭 tombstone 会在 Host 重启后恢复，同一个稳定 decision ID 不能被更高 sequence
  重新打开，不同新 ID 仍可正常开始；
- Mod 的 Run 身份守卫不会把 Launch 时的临时未知身份当成新局，已结束的稳定 Run ID 也
  不能被新局重新接受；恢复选牌 decision ID 还会严格核对局面指纹和更晚失效事件；
- `run_ended` 只有在事件写入成功后才提交内存清理，瞬时写盘失败可由后续生命周期回调重试。

Public Beta 基线之后仍未完成：

1. 对 Neow、商店、篝火、事件与 Deck Edit 逐项做 capability
   真机门禁；
2. Event/Neow 的结构化效果目录与正式“为什么”入口；
3. 重复路线名称列号显示的直接 C# 行为/渲染测试。

## P1.0 路线生产纵向切片（`0.110.1` 基线已完成）

首轮实现于 2026-07-15 因协议可执行性、identity、搜索边界、多维评分、失败关闭和测试覆盖
不足退回。本轮返修已完成以下自动层面的收口：

- schema/Pydantic 共同约束当时的生产 schema v8 与 v1-v7 离线回放，路线 v2 presentation 统一为
  `primary_path_node_ids` / `backup_path_node_ids` / 类型化 `paths`，Card v1/v2 可安全消费；
- Observer 使用真实 origin、模型/视觉 `Travelable` 候选一致性和两次稳定检查；完整 State 与
  地图指纹参与 UPDATED/恢复 identity。写入、绑定、predicate 或选择关闭失败均隐藏 route UI，
  不清除 Writer 恢复状态；
- RoutePolicy 使用有界记忆化搜索，任何候选不可达或输入缺口错误都会失败关闭；静态遭遇池、
  Boss 行动、升级/多段伤害、AOE、成长/资源、遗物、一次性药水、HP 与金币都以可解释因子参与；
- Drawer/Overlay 只消费 fingerprint、owner、完整元数据和候选全量匹配的 advice；地图只
  绘制一条推荐主路线，并逐帧重锚，对非法 topology、重复视觉节点和陈旧 owner 清线；
- 旧 checkpoint/result 的 release fingerprint 不一致时不恢复 lifecycle、不重放旧建议，
  当前策略会重新计算并原子替换；C# 的选牌和路线恢复同样要求精确 fingerprint；
- 自动回归为 Python `293/293 OK`，包含 64 节点图、end-to-end 100 次 P95、跨 release
  checkpoint 重算及严格 advice 正反例；本机 Godot 4.5.1 生成 DLL/JSON/PCK，
  `0 warning / 0 error`。

2026-07-30 复核发现的保存继续版本错配、首次写盘失败、旧地图 revision、严格类型、
版本矩阵和旧 recommendation 重标兼容现已进入自动反例并通过。2026-07-31 又加入三个
路线软偏好、同局恢复、偏离后的动态重算与单主路线硬约束。当前 `route_choice` 已随
TASK-010 在 `0.110.1` 启用并完成组合真机基线；TASK-012 已补齐五角色覆盖。

## Public Beta 策略核心（组合真机已完成）

- `advisor/character_mechanics.py` 定义统一 `MechanicSignal`，五个角色 adapter 只负责把
  结构化卡牌事实转换为机制 family/role，不自行持有最终分数；
- effect tag 已升级至 v5，并修复 Rupture、Blade Dance、Tactician、Zap/Dualcast/
  Defragment、Comet、Venerate、Bodyguard/Unleash 与 Shroud 等已知反例；
- 路线模式进入 `WorldState`、Card/Route 策略、协议、checkpoint、advice 和 Drawer；
  同一真实路线决策切换模式复用 decision ID、生成新 event ID 并以 UPDATED 重算；
- Overlay 和正式 Route advice 拒绝非空 backup，只允许一条 `primary_path_node_ids`；
- `growth` 仍受低 HP 生存底线约束；缺少真实地图时 Card Reward 不伪造 route-fit；
- 机制 signals、data gaps 和资源维度已经进入正式 canonical Recommendation，而非只停留
  在兼容 payload。

## 结构化数据

`scripts/fetch_knowledge.py` 已能从 Spire Codex 构建期 API 获取策略相关实体，当前中文
可重建快照包含 1541 个实体。SQLite schema v8 已规范化导入：

| 表 | 当前行数 |
|---|---:|
| `catalog_entities` | 1541 |
| `monsters` | 115 |
| `monster_moves` | 345 |
| `encounters` | 87 |
| `encounter_monsters` | 135 |
| `events` | 66 |
| `event_pages` | 289 |
| `event_options` | 363 |
| `acts` | 4 |
| `act_entity_memberships` | 140 |
| `mechanic_constants` | 9 |

所有遭遇怪物引用和 Act Boss 引用均通过完整性检查。结构化实体和机制进入 SQLite；攻略
正文不进入关系库。本机 Mobalytics tier、真实数据库、攻略快照和运行日志继续 Git 忽略。

## P1 扩展决策套件（自动审查通过，待真机）

旧 `0.109.1` 程序集已经静态确认并由 Mod 编译引用；以下成员必须在当前 `0.110.1`
重新核对后才可启用：

- Merchant：`NMerchantInventory.Inventory/IsOpen/Open/OnCardRemovalUsed/_ExitTree`，
  `MerchantInventory.AllEntries/Player`，各商品公开实际模型及
  `Cost/IsStocked/EnoughGold`；
- Rest Site：`NRestSiteRoom.Options/Create/EnableOptions/AfterSelectingOption/
  BeforeExitingRoom/_ExitTree`，`RestSiteOption.OptionId/IsEnabled/Title/OnSelect`，
  `HealRestSiteOption.GetHealAmount(Player)`；
- Event/Neow：`NEventRoom.Create/OptionButtonClicked/_ExitTree`，
  `EventModel.CurrentOptions/CanonicalInstance/Owner/IsFinished` 及精确
  `SetEventState/SetEventFinished`，`EventOption.TextKey/IsLocked/IsProceed/Chosen`；
- Deck Edit：升级/变化屏幕 `ShowScreen`、基础选择屏 `Create`、公开
  `CardSelectorPrefs` 约束、`CardsSelected()` 与 `_ExitTree`。

生产代码只注册 Harmony Postfix，不调用购买、选择或卡牌点击动作。Merchant 使用真实异构
库存、实际价格、售罄/买不起状态和离开候选；同一库存对象变化保持同一 decision 并发出
UPDATED。Rest/Smith 与商店移除通过结构化 `upgrade_card/remove_card` 效果和真实子屏建立
父子决策；多选组合尚未建模时直接隐藏。Event/Neow 使用真实 option identity；效果未知时
显式 data gap。通用 Drawer 严格核对 Run/Event/Decision/sequence/capability/候选，并把
最多三个关键原因作为候选悬浮提示。

`explain/` 提供 latest-only、内存态、过期拒绝的确定性解释契约，只复述冻结
Recommendation 的 factors/dimensions/data gaps；它不进入实时 EXE、不调用网络/LLM/
向量库、不重新评分，也尚未接入正式游戏内“为什么”按钮。攻略 RAG 和远程模型仍是后续
可选解释能力，不是当前交付。

## 自动验证与构建基线

当前工作区自动证据：

- 2026-08-09 最新完整 Python 单元测试：440/440 通过；包含 64 节点 DAG、
  event→checkpoint→advice、v9 严格通用候选、v1-v8 回放、跨组件兼容、独立 Card/Route
  恢复机会、五角色机制、五类扩展决策、解释边界和效果标签 v5 回归；
- 固定选牌场景：26/26、93/93 断言通过，迁移没有改变推荐基线；
- 新五类决策内核：250 样本，P50 2.654 ms、P95 21.615 ms、最大 26.155 ms；既有
  文件桥与路线端到端性能门禁也继续满足 P95 ≤ 300 ms；
- `git diff --check`：通过；
- 当前 `0.110.1` Mod `--no-restore` 由 Godot 4.5.1 生成 DLL/JSON/PCK，
  0 warning / 0 error，并已用安装脚本逐文件校验；
- 当前窗口化后台 EXE 与每用户安装器已在启用后的 compatibility manifest 下重新打包；
  完整测试、startup check、源码新鲜度和 payload 拒绝清单均进入同一构建门禁；
- 安装 artifacts 已随最终构建刷新；具体文件事实只以 `dist/release-manifest.json` 为准，
  状态文档不重复维护易过期的逐文件哈希；
- Git 上传边界审计：`.env`、`.mcp.json`、本机 SQLite、私有 tier、Host 日志、EXE、
  Mod `bin/obj/artifacts` 和游戏安装目录文件均不在待提交集合；
- 变更差异未发现大小写敏感的常见 API Key 或私钥特征。

## 下一步

1. 按 TASK-013 恢复当前游戏版本的选牌与路线兼容；
2. 验证最终安装包的安装、Steam 启动、获得建议、保存退出与继续流程；
3. 制作真实简短演示，完善 README 的支持范围、设计取舍与已知限制；
4. 交付完成后视需要抽查推荐合理性，扩展功能暂缓。

签名/SmartScreen 与直接 C# 行为测试等既有后续项保留，不替代当前演示交付目标。

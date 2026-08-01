# TASK-011 — Windows Public Beta 交付壳

- 状态：已验收
- 负责人：Codex
- 审查：独立 Agent + 自动验收
- 前置：TASK-010 Card + Route `0.110.1` 组合真机基线

## 目标

把当前无窗口 Worker EXE 收口为普通玩家可使用的 Windows 单实例托盘程序，并提供可验证的
安装、卸载和脱敏诊断。不得扩大到 Neow、商店、篝火或事件 capability。

## 实施顺序

1. 把实时 Worker 改为可停止的受控组件，托盘控制器按真实
   `SlayTheSpire2.exe` 进程启停 Worker；游戏关闭后 Worker 停止但托盘待机；
2. 使用同一个用户级实例锁；第二次启动不得创建第二个 Worker，只通知已有托盘实例；
3. 托盘只显示连接、Worker、精确兼容和版本状态，并提供“打开日志目录 / 导出脱敏诊断 /
   完全退出”；完全退出必须等待 Worker 并清除自身锁；
4. 诊断包只含版本、兼容结果、artifact 哈希和脱敏日志，不得包含存档、SQLite、
   `active-run.json`、state/advice payload、事件或局内牌组；
5. 保留 `python -m realtime.host` 的开发 Worker 模式和 `--startup-check`；冻结 EXE 默认进入
   托盘，不要求终端；
6. 生成安装器：自动探测或要求用户选择游戏目录，安装 Guide EXE 和精确三件套，写入本地
   安装配置、开始菜单入口；卸载只删除本产品文件，不删除游戏存档或其他 Mod；
7. 运行单实例、假游戏进程启停、Worker 异常、第二次启动、诊断边界、安装器静态门禁、
   全量 Python、Mod 编译、EXE 启动检查和安装器构建。

## 验收

- 托盘先开或游戏先开均可；游戏启动后一个 Worker 运行，游戏退出后零 Worker；
- 重复运行只有一个控制器/Worker，已有实例收到唤醒通知；
- Worker 初始化失败不影响游戏和托盘，advice 失败关闭且状态可诊断；
- 完全退出后没有 Guide 进程和实例锁；
- 诊断归档的拒绝清单测试通过；
- 安装器在非硬编码 Steam 路径下可选择目录，安装/卸载文件清单可核验；
- 完整测试、Mod 编译、EXE 与安装器构建通过；真机生命周期留在本任务最后验证。

## 禁止事项

- 不让实时链路联网，不默认上传诊断或遥测；
- 不把 STS2MCP、RouteLiveProbe、本机数据库、私有 tier、日志或 `.env` 打包；
- 不自动启动或操控游戏，不修改存档，不删除其他 Mod；
- 不把托盘进程存在等同于 Worker 正在推荐。

## 实施记录

2026-08-01 已完成以下实现与验证：

- 新增 Windows 命名互斥体、激活/关闭事件、精确游戏进程路径检测和单一托盘控制器；
  PyInstaller onefile 在任务管理器中是引导父进程 + 应用子进程，但只有一个逻辑控制器和
  一个受控 Worker；第二次启动只令 `activation_count` 增加；
- 游戏启动后 Worker 进入 RUNNING，游戏关闭后先清 advice、停止 Worker 并回到 IDLE；
  完全退出后控制器、状态文件、advice 和 `.host.lock` 均不存在；
- 发布数据库模板只保留 Card + Route 当前启用策略需要的最小结构化事实、物化效果标签和
  社区聚合先验，不含攻略正文、事件对话、向量文本或局内历史；运行数据库只在与不可变
  模板逐字节一致时复用，否则从模板重建并只迁移精确 `run_summaries`；
- 冻结 EXE、精确 Mod 三件套、每用户 Inno Setup 安装器、安装 receipt、开始菜单/可选桌面
  入口、卸载边界和严格 payload 审计已完成；安装器要求游戏程序集和
  `release_info.json` 精确哈希，游戏或 Guide 运行时失败关闭；
- 诊断 ZIP 只允许 `diagnostic.json` 和 `guide.log`；日志不复制任何原文，只把识别出的
  生命周期消息规范化为固定 event token，候选、楼层、HP、金币、路径、ID 和密钥反例均
  不会进入归档；
- 发布脚本现将完整 Python 测试置于不可分割构建门禁，并拒绝比源码旧的 EXE、安装器、
  DLL/PCK 或与自有源不一致的 Mod JSON。

首次独立审查判定 NOT PASS，并给出旧 advice、可篡改 runtime metadata、诊断 denylist、
无游戏目录静默退出、宽松 release_info 校验和陈旧成品六类反例。返修后这些反例均已加入
测试；最新完整 Python 为 `438/438 OK`，`compileall` 与 `git diff --check` 通过，Mod/PCK
为 0 warning / 0 error。

最终候选：

- Guide EXE：34,836,097 bytes，SHA-256
  `E5E612168A977C6E5361253730F9500B8A4DCF4CA480729A5C9A5F2201591959`；
- 安装器：36,556,706 bytes，SHA-256
  `53D4512C6FE431E1DC992D0B09A1F7534C0B4D01D690652E8A194416C4414277`；
- Mod DLL / JSON / PCK：
  `0591842CE51997005F31F3C2524EEFD8034F39EF422FA9E23CBA3AC3BD5F57C4` /
  `56FA0D63B7EE2E9C897E422474477082C3FBA0DAC83B027B98A3259C624C8EB4` /
  `D51ADC0B1D499D7D9F492E2725CE5047940358E99E266A5998AAAB8B6B23C839`；
- 最小发布 DB：1,175,552 bytes，SHA-256
  `592336D146CA65D7572BBB7C3ADDF2BC9B0D111DA4A165C5F6583607D4C384B2`。

真实安装验证覆盖升级、非默认 Steam 游戏目录、receipt/hash、卸载和重装。卸载后 Guide
三件套为 0，未拥有的哨兵 Mod 文件仍存在，运行 DB 和既有 7 条摘要均保留；随后已删除
哨兵并重装最终候选，游戏 `mods/` 只剩 Guide 三件套。最终安装态复核为：IDLE 重复启动
只激活已有实例；启动真实游戏后 RUNNING/Worker/lock 均为真；关闭游戏后 IDLE、Worker
为假、lock/advice 均不存在且 7 条摘要保留；`--shutdown-existing` 后 Guide 进程、状态和
锁均为 0。

当前发行物未做 Authenticode 代码签名，Windows SmartScreen 信誉属于明确的发行限制；
这不伪装成已解决。五角色第一幕三层与真实 `run_ended` 清理属于 TASK-012，不在本任务中
冒充完成。

二次独立审查结论为 PASS：专项 `56/56`、全量 `438/438`、Mod 0/0，上一轮五个 P0
反例与陈旧产物反例均不可复现；最终进程、状态、锁、advice、运行摘要和禁止历史表终态
与上述记录一致。TASK-011 至此验收结束。

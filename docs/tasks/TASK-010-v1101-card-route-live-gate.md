# TASK-010 — STS2 0.110.1 Card + Route 真机门禁

- 状态：自动与组合真机通过（终局清理保留到 Public Beta 组合回归）
- 负责人：Codex
- 审查：独立 Agent + 真机证据复核
- 前置基线：`c25a55b`
- 游戏目标：Windows 单人，STS2 `0.110.1`，commit `db5d3552`
- `sts2.dll` SHA-256：
  `7C446EFABF80614C429B5088E87101423AA5BB4C04FC3E73393261F6E6D404FD`

## 已发现的版本阻断

2026-08-01 启动隔离测试时 Steam 将游戏从 `0.109.1` 自动更新到 `0.110.1`。旧
RouteLiveProbe 在真实 Loader 日志中因 `NMapScreen._ExitTree` 已不再由该类型声明而拒绝
Harmony patch；测试用 STS2MCP 也因旧 `LobbyPlayer` 类型被移除而加载失败。游戏随后关闭，
两个测试组件均已卸载，正式 Guide 未安装，Host 未启动。

这证明所有 `0.109.1` 自动编译和真机证据只能作为历史资料。生产清单继续
`pending_validation`，不得用旧 artifact 绕过失败关闭。

## 目标

把已经通过自动审查的 v9 Card Reward 与 Route 从 `pending_validation` 推进到精确版本真机
可用；其他 Merchant、Rest、Neow、Event、Deck Edit capability 必须继续失败关闭。本任务
不扩展策略功能，也不把旧版本证据继承到 `0.110.1`。

## 阶段 A：0.110.1 精确 API 与隔离路线探针

1. 确认游戏和 Guide/Host 均未运行，核对本机游戏路径、版本、commit 与 DLL 哈希；
2. 以本机程序集元数据为最终事实，SourceCode 只辅助理解语义；修复所有被证据确认失效的
   Harmony target，不允许猜字段名或沿用旧生命周期；
3. 构建并只安装 `tools/RouteLiveProbe`，不同时安装正式 Guide Mod；
4. 记录主菜单、新局、Neow、可操作地图、顶栏预览、真实选点、地图滚动、窗口尺寸变化、
   战斗进入/返回、保存退出/继续和换局边界；
5. 用 `validate_probe_log.py` 验证 API 时序、模型/视觉候选、节点身份和坐标映射；
6. 探针失败则只修被真实证据证实的问题；探针通过后卸载并确认 DLL/JSON/PCK 均不存在。

## 阶段 B：精确兼容清单与正式闭环

1. 只把 manifest 总状态、`card_reward`、`route_choice` 改为 enabled；其他 capability 保持
   `pending_validation`，并按项目规则更新 release fingerprint；
2. 重跑完整 Python、Mod/PCK、EXE startup check 与兼容矩阵；
3. 游戏关闭时安装正式三件套，启动唯一 Host/Worker；
4. 真机验证：
   - 主菜单无伪状态、无旧 Drawer；
   - 新局真实角色、HP、牌组、遗物、药水、Boss、完整地图与可走边；
   - 三张卡加跳过候选身份/分数正确，选择或跳过后关闭；
   - 地图仅绘制一条完整推荐路线，三种软偏好切换复用 decision ID 并重算；
   - 玩家偏离后按实际节点重算；顶栏预览不创建路线 decision；
   - 滚动和窗口变化时线路仍锚定真实节点；
   - 保存退出/继续恢复同一 Run 与当前未决决策，不显示旧建议；
   - 放弃/结束后只保留 `run_summaries`，清理 current-run 文件；新局不复用旧身份。
5. 任一身份、候选、时序、布局或兼容断言失败时立即清 advice 并将对应 capability 退回
   `pending_validation`。

## 禁止事项

- 不启用 Merchant、Rest、Neow、Event 或 Deck Edit；
- 不让 STS2MCP/探针进入最终包；它们只作为测试仪器；
- 不自动修改游戏选择、存档或 RNG；自动化操控只允许作为隔离验收驱动并记录；
- 不把自动测试、编译或“面板出现”单独写成真机完成。

## 完成证据

- 探针日志和 validator 通过，且探针卸载；
- 精确 compatibility manifest 与安装 artifacts 哈希一致；
- Python 全量、Mod/PCK、EXE 与兼容矩阵通过；
- 至少一局完成 Card + Route 组合闭环和一次保存继续；
- 对应状态文档记录实际 run ID、关键 sequence、候选/路径和清理结果；
- 独立审查无 P0/P1。

## 实施记录

2026-08-01 已完成以下可复核证据：

- 当前程序集精确身份为游戏 `0.110.1` / commit `db5d3552` / `sts2.dll`
  SHA-256 `7C446E...D404FD`；旧 `NMapScreen._ExitTree` patch 已由当前真实
  `_Notification(NotificationExitTree)` 生命周期替代；
- 隔离 RouteLiveProbe 的时序、真实候选、模型/视觉映射、滚动、窗口变化和保存继续严格
  gate 全部通过，随后卸载；探针与 STS2MCP 均未进入正式包；
- compatibility manifest 只启用 `card_reward` 与 `route_choice`，发行 fingerprint 为
  `3ec6911d1288fe9c57b92a37c4f935fc8464d4eafabd6ccddddc896edb657e2f`；
- 正式 Card Reward run 为 `sts2-ad5d5a7ccda1d3862f3e5aa0`，真实候选为
  `DRAIN_POWER / DEFY / POKE / skip`，event sequence 1 的 state/advice、候选稳定 ID、
  中文标签和分数一致；
- 正式 Route 从 origin `5:2` 对真实候选 `6:2 / 6:3` 生成单条完整路线；保存继续后复用
  decision `22a9c22dd80d4c4493373846e0710bfc`，三种偏好在 sequence 21/22/23 保持同一
  decision ID 并重算；
- Overlay 改为挂载到当前 `TheMap/Paths` 的单一 `Line2D`，真机确认不会穿过顶部 UI，
  滚动和 1950x1275→1600x1000 窗口变化后仍锚定原生节点；
- 修复 Card Reward 重放错误消耗 Route 恢复机会的问题；Card 与 Route 使用独立恢复门禁，
  无关 card close 不再让 route identity 失效；
- 修复 `DRAIN_POWER` 被目录通用 `Cards` 变量误标为抽牌/弃牌/升级手牌的问题，效果标签
  版本升为 v5；真实运行数据库迁移后该牌只保留 `card_type:attack` 与 `damage=10`；
- 独立重跑 Python `382/382`，Mod/PCK 0 warning / 0 error，窗口化 EXE
  `--startup-check` 通过且不遗留锁；安装 artifacts 的 DLL SHA-256 为
  `2F304B4AE2694D3ED00A9C840A5E1440EAE1C628D7460CD550FCDC2BC1D8A954`；
- 最终通过 Steam 启动的洁净日志只发现 Guide manifest，Steamworks 初始化成功，
  `Loaded 1 mods (1 total)`；测试用 MCP 三件套已从游戏目录移除。

本轮没有放弃用户当前保存局。`run_ended` 后只留 `run_summaries` 的终局清理仍由已有自动
测试覆盖，并保留为五角色 Public Beta 组合真机的必验项；不得据此宣称五角色公开版已经
完成。

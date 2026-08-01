# TASK-012 — Public Beta 五角色组合真机门禁

- 状态：待真机验收
- 负责人：Codex + 用户真机操作
- 审查：独立证据复核
- 前置：TASK-011 Windows 交付壳已验收
- 目标环境：Windows 单人原版内容，STS2 `0.110.1`，只安装正式 Guide Mod

## 目标

用最终安装器和同一个五角色共享评分内核完成 Public Beta 最后的组合真机门禁。此任务不再
增加策略能力，不启用 Merchant、Rest Site、Neow、Event 或 Deck Edit；只验证已启用的
`card_reward` 与 `route_choice` 在五个角色、真实局状态和 Windows 托盘生命周期下可靠。

## 开始条件

1. 游戏与 Guide 进程均关闭；
2. 已安装 TASK-011 最终安装器，游戏 `mods/` 只有 Guide 正式三件套；
3. receipt、Guide EXE、Mod 三件套、游戏程序集和 `release_info.json` 通过精确门禁；
4. STS2MCP、RouteLiveProbe 和其他测试 Mod 均未安装；
5. Host 不单独手工启动，由托盘控制器随真实游戏进程拥有唯一 Worker。

## 五角色矩阵

对 Ironclad、Silent、Defect、Necrobinder、Regent 各完成第一幕至少三个实际节点；每个角色
至少记录一次普通战斗后的三张卡 + 跳过，并完成一次真实选路。每个样本必须核对：

- `run_id`、角色、HP、牌组、遗物、药水、金币、Boss 和地图来自同一 snapshot revision；
- 游戏候选名、稳定候选 ID、state event、advice 和 Drawer 行一一对应；
- 推荐分是有限局面适配分，跳过为动态候选，数据不足显示 `--`；
- 领取任一卡或跳过后，原 decision 关闭且 Drawer 不残留；
- 地图只绘制一条完整推荐路线，路径只沿真实 `MapPoint.Children` 边；
- 三种路线偏好在同一打开决策内复用 decision ID、更新 event ID 并重算；
- 玩家偏离推荐后，从实际节点重新规划，不继续显示旧路径；
- 进入战斗后不显示路线线条，回到卡牌奖励时不复用上一决策建议。

## 组合边界

- 至少一个角色执行“未决选牌时保存退出 → 继续”，确认恢复同一 Run 和当前真实候选；
- 至少一个角色执行“进入下一节点后保存退出 → 继续”，确认不会回退或复用已关闭 advice；
- 地图执行一次滚动和窗口尺寸变化，线路持续锚定原生节点且不穿过顶部 UI；
- Neow 普通祝福、商店、篝火和事件即使出现，也只能失败关闭，不能显示旧 Card/Route 建议；
- 第三方 Mod、版本或 artifact 任一漂移时，Worker 不启动并清除 advice。

## 真实终局清理

用一局测试局执行真实“放弃”或自然结束并记录 `run_ended`：

1. `run_summaries` 只新增一条该 Run 的最终摘要；
2. `active-run.json`、state/advice 文件、临时 `events/` 与 `.host.lock` 全部清理；
3. `run_states`、`decision_events`、`decision_outcomes`、`game_state_events` 保持 0；
4. 再开新局不得复用旧 run/decision/event 身份；
5. 游戏关闭后 Worker 为 0、托盘保持 IDLE；完全退出后 Guide 逻辑实例、状态文件和锁均为 0。

## 失败处理

任一角色发生候选错配、旧建议、错误路径、身份漂移、异常进程或兼容性假通过，立即停止该
轮并保留脱敏诊断。只修复可由 state event、日志或真实对象证实的问题；修复后完整重建、
重装，并从受影响角色重新执行三层门禁。不得猜游戏 API，也不得为过门禁而降低严格校验。

## 完成定义

- 五个角色全部完成上述卡牌 + 路线矩阵；
- 保存继续、滚动/窗口、偏离重算和跨决策无旧 UI 均有真机证据；
- 一次真实 `run_ended` 清理及新局身份隔离通过；
- 最终安装包哈希、438+ 全量测试、Mod 0/0、安装/卸载与托盘生命周期证据一致；
- 独立审查 P0/P1 为 0。

只有这些条件全部满足，项目状态才可以写“Windows Public Beta 候选完成”。这不等于已做
代码签名、商店/篝火/Neow/Event 推荐或自动更新。

## 验收记录

待真机填写。

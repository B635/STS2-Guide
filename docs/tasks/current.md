# 当前实现任务

- 状态：P0 真机闭环已验收，准备进入 P0.5
- 已完成任务：[`TASK-004-live-p0-blockers.md`](TASK-004-live-p0-blockers.md)
- 最后更新：2026-07-13

## 当前事实

TASK-004 的普通选牌显示、稳定 Run ID、保存继续、Boss/真实地图、后台单实例、放弃摘要和
临时状态清理均已通过自动验证与真机回归。具体证据见
[`../project-status.md`](../project-status.md)。

P0.5 基础代码已经存在，但尚未完成产品规格第 11.1 节的全部验收。下一次实现必须只做
统一 `WorldState`、决策生命周期、`DecisionRequest`、策略注册、`Recommendation`、
Card Reward 插件迁移和通用 `Context Drawer`，不得同时加入商店、路线或篝火建议。

在新的 P0.5 任务单建立前，DeepSeek 不得自行修改代码或寻找其他 TODO。

# 当前实现任务

- 状态：待真机验收
- 当前任务：[`TASK-009-extended-decision-suite.md`](TASK-009-extended-decision-suite.md)
- 最后更新：2026-08-01

TASK-008 自动层已经通过独立终审，但按用户 2026-07-31 的新决定暂不进入组合真机。
TASK-009 连续实现 Neow、商店、篝火、事件与只读解释 Agent，并继续完善选牌/路线的跨决策
资源权重。所有新增能力共用一次 v9 通用候选迁移和统一决策内核，不能复制处理链。

2026-08-01 已完成协议、Processor、Mod 时序与安全边界的多轮独立返修。最终完整 Python
`377/377 OK`，Mod/PCK `0 warning / 0 error`，EXE 已重建并通过 `--startup-check`；协议
来源矩阵与 Mod 特殊 Card Reward 生命周期分别由未参与实现的 Agent 复审，P0/P1 均为 0。
普通奖励同步观察，Event/Neow 子奖励按 reward object 冻结来源并要求完整父身份，地图真实
选点与 run 生命周期语义作废旧 parent，不再依赖固定帧数。尚未安装 Mod、启动 Host 或游戏，
因此任务只进入“待真机验收”，所有 capability 继续失败关闭。

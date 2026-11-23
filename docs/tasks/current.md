# 当前实现任务

- 状态：进行中
- 当前任务：[`TASK-029-game-v0113/README.md`](TASK-029-game-v0113/README.md)（`feature/v0113`，适配游戏 `v0.113.1` 与新角色愚者 FOOL）
- 最后更新：2026-11-24（已从 `main` `87900da` 合并 Stage 0 `TASK-028`、A1 `TASK-030`、A2 `TASK-031`，三者均已在 `main` CI 全绿通过；现在集中推进 `v0.113.1` 事实补齐、愚者评分、回归修复与 Mod 适配）
- 执行说明：121 张新增卡牌已复核合入，A1 管线已产出 57 条新增遗物/能力草案（44 自动通过 + 13 人工队列），新版 Mod 已恢复编译。接下来：并入剩余新遗物/能力/2 怪物事实 → 实现愚者核心机制评分切片 → 分类修复旧回归 → 更新兼容清单与安装态验证。
- 已完成并合入 `main`（CI 全绿）：
  - [`TASK-028-review-coach/README.md`](TASK-028-review-coach/README.md)（`6a3f37b`，CI run `45192359824`）
  - [`TASK-030-fact-pipeline/README.md`](TASK-030-fact-pipeline/README.md)（`b199bbd`，CI run `45212225100`）
  - [`TASK-031-coach-eval-loop/README.md`](TASK-031-coach-eval-loop/README.md)（`87900da`，CI run `45221075729`）
- 历史待真机验收：[`TASK-025-deck-fitness/12-five-character-live.md`](TASK-025-deck-fitness/12-five-character-live.md)（票 01–11 已完成）、[`TASK-026-route-shop-and-combat-floor.md`](TASK-026-route-shop-and-combat-floor.md)
- 暂停任务：[`TASK-027-relic-choice-and-shop/README.md`](TASK-027-relic-choice-and-shop/README.md)（Neow／古神遗物选择与商店 v2）

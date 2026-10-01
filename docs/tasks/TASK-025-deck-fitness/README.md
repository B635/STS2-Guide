# TASK-025：选牌评分内核 v2（牌组适合度）

- 状态：待实现
- 创建日期：2026-10-01
- 产品依据：`docs/product-spec.md` 第 2.8 节；`docs/adr/0001-deck-fitness-scoring.md`
- 在制上限：第 2.7 节下唯一的在制任务；各票合计视为一个任务
- 审查：每张票完成后用 `/code-review` 独立审查（规范 + 规格双轴）

## 票与依赖

| 票 | 标题 | 依赖 |
|---|---|---|
| [01](01-snapshot-eval-harness.md) | 快照回放与标准答案集评估框架 | 无（可立即开始） |
| [02](02-assembly-fact-pipeline.md) | 程序集事实管线（追踪弹） | 无（可立即开始） |
| [03](03-deck-fitness-policy-tracer.md) | v2 牌组适合度策略（追踪弹） | 01、02 |
| [04](04-relic-coverage.md) | 遗物全面审核并接入 | 02、03 |
| [05](05-ironclad-cards.md) | IRONCLAD 卡牌全覆盖 | 03 |
| [06](06-silent-cards.md) | SILENT 卡牌全覆盖 | 03（标准答案集局面来自用户 0.3.11 真机采集） |
| [07](07-defect-cards.md) | DEFECT 卡牌全覆盖 | 03（标准答案集局面来自用户 0.3.11 真机采集） |
| [08](08-necrobinder-cards.md) | NECROBINDER 卡牌全覆盖 | 03（标准答案集局面来自用户 0.3.11 真机采集） |
| [09](09-regent-cards.md) | REGENT 卡牌全覆盖 | 03（标准答案集局面来自用户 0.3.11 真机采集） |
| [10](10-shared-cards.md) | 公共牌覆盖：无色、事件、诅咒、状态、衍生、任务、远古 | 03 |
| [11](11-switch-and-build.md) | 切换与候选构建 | 04、05、06、07、08、09、10 |
| [12](12-five-character-live.md) | 五角色安装态真机验收 | 11（需要用户游玩） |

01 与 02 可立即并行；03 完成后 04–10 可并行；11 等待 04–10 全部完成；12 需要用户游玩。

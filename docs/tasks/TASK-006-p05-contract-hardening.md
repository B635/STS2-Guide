# TASK-006：P0.5 通用契约与生命周期返修

- 状态：已验收
- 负责人：Codex
- 范围：只修复 P0/P0.5，不增加路线、商店、Neow 或篝火建议

## 背景

极高审查确认当前自动测试虽然通过，但存在跨 Run 关闭、重复关闭覆盖结果、不可达
`UPDATED`、Card Reward 专用 advice schema、无所有权的全局抽屉等问题。当前实现不能作为
P1 的稳定扩展底座。

## 实现范围

1. 分离不可变 `event_id` 与稳定 `decision_id`，协议升级时同步 schema、示例、C# 生产者、
   Python 消费者和测试；
2. 修复跨 Run 和冲突重复关闭；建立通用可恢复 `current_decision` 与最近关闭防重信息；
3. 建立共享 WorldState/event adapter 和通用策略处理路径；
4. 将 advice schema 改为通用 envelope，严格校验 world sequence、候选全集、原始顺序和
   eligibility；Card Reward 旧 payload 仅用于兼容；
5. Card Reward Mod 改为消费 canonical Recommendation，并校验完整候选指纹；
6. Context Drawer 使用 handle 隔离旧场景回调，advice 缺失或错配时清空显示；
7. 增加真实 JSON Schema 验证、跨 Run/重复关闭/UPDATED/恢复测试和打包 EXE 初始化冒烟；
8. 修正文档状态与 P1 优先级冲突。P1 顺序继续以产品规格中的路线优先为准。

## 验收标准

- 跨 Run close 被拒绝，且不能改变另一局 checkpoint 或会话；
- 相同 close 幂等，冲突 close 被拒绝且首次结果不变；
- 相同稳定 decision ID 的新观察产生 `updated`；
- 非 Card Reward 策略能生成 schema-valid envelope，不需要 legacy `advice`；
- Registry 拒绝 sequence 不一致、虚构候选、漏候选、重复 rank；
- advice 文件删除或错配后，Card Reward 抽屉不保留旧分数；
- 完整 Python 测试、固定场景、性能基准和 Mod 编译通过；
- 打包 EXE 通过包含单实例锁与完整初始化的 startup smoke；
- 未完成真机回归前，项目状态只标记为“自动验证完成、真机待验”。

## 实施报告

### 自动修复结果

- 协议升级到 schema v5，分离 `event_id`、稳定 `decision_id` 和候选实例 ID；schema、示例、
  C# 生产者、Python 消费者与测试已同步；
- `WorldState`、请求、策略注册表、Recommendation 和生命周期均使用严格不可变/显式候选
  契约；Registry 会拒绝错误 sequence、漏项、虚构/乱序候选、重复 rank 和不可选推荐；
- checkpoint v2 使用通用 `current_decision` 与单个 `closed_decision` tombstone；跨 Run close、
  同类型但不同候选的冲突 close 均失败关闭，首次结果不被覆盖；
- Mod 生产者可对同一真实 `CardReward` 实例复用稳定决策 ID，关闭事件携带该 ID；面板只读
  canonical Recommendation，并校验完整候选指纹；
- `ContextDrawer` 使用 handle，Card Reward 界面退出按 owner 清理，旧界面回调不能关闭新
  界面抽屉；advice 缺失/错配显示 `--`；
- 文件桥只在决策边界清理 advice，无关地图/商店观察不会误删仍打开的建议；
- Host 正常退出会释放单实例锁；打包脚本增加成品 `--startup-check`，覆盖锁、SQLite/静态
  数据初始化与退出清理。
- 最终独立审查补齐此前测试未覆盖的生命周期失败：advice 副作用失败时队列保留重试、
  closed decision tombstone 跨 Host 重启禁止重开、旧 Run ID 防复用、Launch 临时身份不误判
  新局、恢复 decision ID 必须匹配完整局面指纹、`run_ended` 只在写盘成功后提交清理；
- 不同 Run 的临时 spool 按 `emitted_at` 全局处理，结束一局只清理该 Run 的 artifacts，不会
  删除下一局已排队事件；旧 Run 的 sequence 1 也不能从新 checkpoint 抢回所有权。

### 验证证据

- `scripts/run_p0_review.ps1`：2026-07-14 主审查会话重新执行，全部步骤通过；
- Python：180/180；核心与实时专项：75/75；固定选牌场景：26/26、93/93 断言；
- 性能：100 样本，P50 24.50 ms、P95 28.96 ms、最大 32.29 ms；
- Mod `--no-restore`：0 error、1 warning；warning 为当前沙箱无法访问本机 Godot，PCK 未生成；
- 打包 EXE：27,542,700 bytes，SHA-256
  `857F6D113FE20222CC9AA5C33A5DF3744EBC37E1C65A5DAD413D44F01D71B612`；startup smoke 退出 0，
  日志确认初始化成功，未残留 `.host.lock`；
- `git diff --check`：通过。

### 最终真机回归补充

- schema v5 已完成安装和真机回归：普通战后选牌建议与真实候选一致，领取卡牌会关闭
  决策并清理面板；地图、同 Run 保存继续、sequence 恢复和放弃清理均符合契约；
- 真机回归发现恢复后的同一奖励界面点击“跳过”时，原实现只监听
  `CardReward.OnSkipped`，没有覆盖界面内 alternate reward 路径，因此 advice 和
  `current_decision` 残留；
- 已根据本机 0.108.0 `sts2.dll` 元数据与 IL 修复：Postfix 精确监听
  `NCardRewardSelectionScreen.OnAlternateRewardSelected(int)`，读取已验证的
  `_extraOptions`，且只在公开 `OptionId == "Skip"` 时关闭同 owner 的 pending decision；
- 修复后完整 Python 测试 180/180、固定场景 26/26（93/93 断言）、Mod 编译 0 error、
  `scripts/run_p0_review.ps1` 全部通过；本机 Godot 构建 0 warning/0 error且最新三个 Mod
  artifacts 已安装并通过工作区/游戏目录哈希核对；
- 2026-07-14 真机复测一：普通奖励点击跳过后，关闭事件与原 decision ID 匹配，结果为
  `skipped`，随后地图 sequence 4，`current_decision` 清空且 advice 文件删除；
- 2026-07-14 真机复测二：保存继续恢复奖励界面后点击跳过，关闭事件 sequence 6 使用
  新恢复决策 ID，结果为 `skipped`，随后地图 sequence 7，`current_decision` 清空且
  advice 文件不存在；
- 以上证据满足 P0.5 真机验收；TASK-007 仍须独立完成路线 API 真机门禁后才能启动。

安装入口已收口为 `scripts/install_p0_mod.ps1`：读取本机 `local.props`，要求游戏关闭，执行
Godot 打包与 Mod 安装，并逐项核对 workspace/game DLL、JSON、PCK 的 SHA-256。脚本已完成
语法与只读审查，但当前授权环境无法实际访问 Godot/游戏目录，因此不能把脚本存在写成
安装完成。

### 独立审查结论

最新代码级复审未发现新的 P0.5 阻断；普通和保存恢复后的跳过真机均已通过。P1 路线任务
已经形成草案，下一步必须先完成路线 API 真机门禁，不能把本次 P0.5 验收外推为路线完成。

# STS2 Guide — Agent 开发上下文

本文件只提供开发入口和不可违反的边界。产品范围与优先级的唯一依据是
[`docs/product-spec.md`](docs/product-spec.md)，当前真实完成度见
[`docs/project-status.md`](docs/project-status.md)。如果本文、旧文档或当前代码
与产品规格冲突，以产品规格为准。

## 文档归档规则

后续修改不得把所有设计细节都堆入本文件，按下面的唯一职责更新：

- `docs/product-spec.md`：产品定位、范围、交互、优先级、架构边界和验收标准；
  已确认的设计变更必须先写入这里。
- `docs/project-status.md`：代码当前真实完成度、已验证证据、已知问题和下一步；
  只有编译、测试、安装、真机验证达到对应状态后才更新。
- `docs/api-reference-sources.md`：STS2 API 外部参考源、使用边界和调研报告格式；
  涉及未知游戏 API 时必须先查这里登记的参考源并再用本机真机/程序集验证。
- `protocol/*.schema.json`：Mod 与后台之间可执行的协议事实；协议变化必须同时更新
  schema、示例、生产者、消费者和测试。
- `docs/data-architecture.md`、`docs/realtime-architecture.md`：对已确认架构的详细解释，
  不能覆盖或暗中改变产品规格。
- `AGENTS.md`：只保留供 Agent 快速进入项目所需的长期边界、文档入口、环境命令和
  完成定义；只有设计变化形成新的长期硬约束时，才在这里同步摘要。

每次设计调整遵循：先更新产品规格，再修改代码和测试，最后按真实验证结果更新状态
文档。聊天记录不作为项目决策依据。

## 多会话协作

本项目采用“Codex 负责产品与架构、DeepSeek 负责实现、用户负责产品决策”的分工：

- Codex：澄清需求、维护产品规格、拆分任务、给出验收标准、审查实际 diff 与验证证据；
- DeepSeek：只按 [`docs/tasks/current.md`](docs/tasks/current.md) 指向的任务单修改代码和
  测试，并填写实施报告；
- 用户：决定产品体验和范围取舍，并在需要实现或审查时启动对应会话。

任务流程和状态定义见 [`docs/tasks/README.md`](docs/tasks/README.md)。任何实现 Agent
开始前必须依次阅读本文件、产品规格、项目状态和当前任务单。若当前任务状态不是
`待实现` 或 `需修复`，不得自行寻找 TODO、扩大范围或修改代码。

DeepSeek 不得自行改变产品规格，也不得把“代码存在”写成“真机完成”。Codex 在审查时
必须独立检查代码差异和测试结果；只有达到本文件的完成定义后，才能更新项目状态。

## 当前产品

STS2 Guide 是《Slay the Spire 2》的本地、只读、状态驱动决策 Agent。

长期产品形态是覆盖局内关键决策的实时策略副驾驶：统一维护当前局世界状态，在选牌、
路线、商店、篝火等真实决策出现时调用对应本地策略，并通过同一个游戏内
`Context Drawer` 主动展示建议。它不是自动代打工具，也不是以聊天或 LLM 为核心的
攻略机器人。

P0 用户体验：

1. 用户正常启动 STS2；
2. 用户单独启动 STS2 Guide 后台程序；
3. 只读 Mod 感知选牌状态；
4. 本地推荐器离线计算；
5. 游戏内面板显示卡牌候选与“跳过”的推荐分；
6. 用户自己操作，系统不点击、不代打、不修改游戏或存档。

P0 标准场景是三张牌加跳过。面板可以对其他候选数量安全降级，但这不代表扩大
P0 产品范围。

Neow 祝福等开局特殊决策不属于 P0 推荐范围；P0 只要求这些未支持场景不会显示上一屏
或错误候选。完整祝福建议进入 P1。

P0 真机闭环完成后、增加 P1 决策前，必须先完成 P0.5 决策内核收口：建立统一
`WorldState`、`DecisionRequest`、策略注册机制和 `Recommendation`，并将 Card Reward
迁移为第一个策略插件。不得通过复制 `card_reward` 分支的方式堆叠路线、商店或篝火。
详细范围与验收以 `docs/product-spec.md` 第 11.1 节为准。

## P0 架构边界

- Mod：C# / .NET 9 / Harmony Postfix / Godot Control。
- 实时后台：`python -m realtime.host`，最终打包为单一后台 EXE。
- 协议：JSON Schema v4，本地原子文件交换；Python 消费端继续兼容 v1-v3 录制夹具。
- 实时推荐：完全本地，不调用 DeepSeek、LLM 或网络。
- Vue、FastAPI、RAG：不属于 P0 实时运行依赖。
- 路线：P0 只读取真实地图图结构，不生成路线推荐。
- 敌人：已知 Boss 可参与战前威胁建模；普通怪物和精英只能按真实遭遇池及已击败
  精英排除规则估计，不能伪造确定敌人；P0/P1 不做战斗内逐回合出牌建议。

## 数据边界

- 当前局：只保存一个可原子替换的 `active-run.json` 检查点。
- 临时事件：处理后删除，不归档为历史。
- 一局结束：清理中间状态，只向 SQLite `run_summaries` 保存最终摘要。
- 禁止实时管线向 `run_states`、`decision_events`、
  `decision_outcomes`、`game_state_events` 追加历史。
- 静态卡牌、遗物、角色、可重建效果标签和社区结构化统计：SQLite。
- 静态敌人机制和遭遇池规则进入 SQLite；敌人逐回合行动历史不属于 P0 持久化数据。
- 攻略正文和语义切片：原文快照 + 向量索引，只用于可选 RAG 解释。
- 本机 Mobalytics tier：`data/local/mobalytics_card_tiers.json`，人工整理、
  Git 忽略、不自动抓取、不提交、不打包；缺失时必须安全降级。
- P0 不收集训练样本，不做自我学习。

## 推荐边界

- 推荐分是当前局面适合度，不是胜率。
- 跳过是正式候选。
- 社区数据只是有限先验，当前牌组和局面规则优先。
- 不复制 BoberInSpire 的代码、权重、流派表或数据文件。
- 数据不足时显示 `--`，不能伪造精度。
- 建议必须校验 `run_id + event_id + 候选稳定 ID`，不得显示旧建议。

## Mod 约束

- 只使用 Harmony Postfix；不得添加 Prefix 改写游戏行为。
- `affects_gameplay: false`。
- 不自动点击、不修改选择、不修改存档、不联网、不持有 API Key。
- 地图只使用已验证的真实 API 和 `MapPoint.Children`；读取失败时不得猜边。
- 碰到不确定的游戏 API 时，必须先通过程序集、真实对象日志或真机回归确认；
  实施报告要写明类型、成员、来源和验证方式。不得用字段名列表、界面文本、截图位置
  或宽泛动态反射猜测生产状态。
- 修改 Mod 后必须先成功编译，再讨论“已完成”。

## 环境与命令

Python 环境：

```text
D:\miniconda3\envs\sts2\python.exe
```

Python 测试：

```powershell
& "D:\miniconda3\envs\sts2\python.exe" -m unittest discover -s tests -v
```

Mod 编译需要把 .NET/NuGet 临时目录留在项目内：

```powershell
$env:DOTNET_CLI_HOME = (Join-Path (Get-Location) ".dotnet_cli")
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
$env:APPDATA = (Join-Path (Get-Location) ".dotnet_cli\AppData")
$env:NUGET_PACKAGES = (Join-Path (Get-Location) ".dotnet_cli\packages")
dotnet build mod/STS2Guide.ReadOnlyExporter/STS2Guide.ReadOnlyExporter.csproj --no-restore
```

P0 后台开发入口：

```powershell
& "D:\miniconda3\envs\sts2\python.exe" -m realtime.host --console-log
```

## 完成定义

代码存在、协议预留、自动测试通过和真机完成是四种不同状态。功能只有同时满足以下条件
才能在状态文档中标记为完成：

1. 状态来自真实游戏 API；
2. 协议与实现一致；
3. 自动测试通过；
4. Mod 编译通过；
5. 新 artifacts 已安装；
6. 至少完成一次对应真机回归。

## 当前优先顺序

1. 选牌读取 → 本地推荐 → 游戏内显示 → 关闭清理的真实闭环；
2. 保存退出后恢复同一 Run ID 和递增 sequence；
3. 真机核对地图节点、真实边、下一节点和 Boss；
4. 后台 EXE；
5. 固定选牌场景评测和推荐规则完善；
6. P1 再增加路线、商店、篝火等决策。

不要为了 Agent 标签继续堆叠 Hybrid Search、Reranker、HyDE 或 LLM 功能。

# STS2 Guide：本地实时决策 Agent

STS2 Guide 是《Slay the Spire 2》的本地、只读、状态驱动策略副驾驶。用户正常操作游戏，
只读 Mod 感知当前决策，本地后台离线计算建议，游戏内 `Context Drawer` 主动显示结果。

它不自动点击、不修改存档、不联网，也不把大模型放进实时推荐主链路。当前产品依据见
[`docs/product-spec.md`](docs/product-spec.md)，代码、安装和真机的真实完成度见
[`docs/project-status.md`](docs/project-status.md)。

## 当前形态

P0 已建立普通选牌闭环：

1. C# / Harmony Postfix 只读观察真实 Card Reward；
2. Mod 通过生产 JSON Schema v9 原子写入权威当前事件、通用候选与路线偏好；
3. Python Host 校验 Run、sequence、decision 和候选身份；
4. 本地确定性推荐器结合牌组、HP、遗物、药水、金币、地图与 Boss 计算适配度；
5. Godot 左侧抽屉显示卡牌与“跳过”的推荐分；
6. 用户自己选择，关闭后 advice 被 owner-scoped 清理。

P0.5 已建立统一 `WorldState`、`DecisionRequest`、决策生命周期、策略注册表、
`Recommendation` 和 `Context Drawer`。其上的 P1 自动实现让路线只显示一条当前主路线，
可切换智能均衡、稳健生存和激进成长；五个角色共用评分内核，只通过薄 adapter 提供
角色机制信号。当前版本仍需组合真机，不能把这段描述理解为已启用的公开能力。

工作区还完成了 TASK-009 的自动实现：Merchant、Rest/Smith、Neow、Event 和 Deck Edit
共用 v9 候选 envelope、资源预算、事件管线与通用 Drawer；`explain/` 提供不参与实时打分的
确定性只读解释边界。当前完整测试、Mod/PCK 与 EXE 启动检查通过，但最新 artifacts 尚未
安装，所有 capability 仍是 `pending_validation`。这表示代码可进入独立审查和组合真机，
不表示这些 P1 功能已经交付；Event/Neow 效果不确定时只显示 `--`。

## 实时架构

```text
STS2
  → Read-only Mod (C# / Harmony Postfix)
  → schema-valid temporary JSON events
  → Local Host (Python / Pydantic)
      → active-run checkpoint
      → WorldState + DecisionRequest
      → Policy Registry
      → Recommendation
  → advice-event.json
  → Godot Context Drawer / Overlay
  → user acts in the original game UI
```

核心技术：

| 层 | 技术 | 职责 |
|---|---|---|
| 游戏感知 | C#、.NET 9、Harmony Postfix | 读取真实游戏 API，不改变游戏行为 |
| 游戏内展示 | Godot `Control` | Drawer、候选分和后续只读地图 Overlay |
| 本地后台 | Python 3.9、Pydantic | 协议、身份、生命周期、策略调度和原子文件桥 |
| 结构化数据 | SQLite | 卡牌、遗物、药水、怪物、遭遇、规则和最终摘要 |
| 当前局恢复 | 单一 `active-run.json` | 当前世界状态与一个当前决策，不保存事件历史 |
| 协议 | 生产 JSON Schema v9 | 完整快照、通用候选、父子决策、发布指纹与路线偏好；旧 v1-v8 仅离线回放 |

详细数据流见 [`docs/realtime-architecture.md`](docs/realtime-architecture.md)，结构化数据与
长文本边界见 [`docs/data-architecture.md`](docs/data-architecture.md)。

## 数据与隐私边界

| 数据 | 存储 | 生命周期 |
|---|---|---|
| 当前角色、资源、牌组、地图和当前决策 | `active-run.json` | 当前局原子替换 |
| 待处理状态事件 | `events/*.json` | 处理成功后删除 |
| 当前建议 | `advice-event.json` | 决策关闭、失效或换局后清理 |
| 最终胜负、楼层、牌组、遗物、药水 | SQLite `run_summaries` | 一局只保存一条摘要 |
| 卡牌、遗物、敌人、遭遇与机制 | SQLite | 可重建静态结构化数据 |
| 攻略正文 | 原文快照 + 可选向量索引 | 仅供以后主动解释，不进入实时主链路 |

P0/P1 不保存逐决策训练历史，不做自我学习。本机 Mobalytics tier 文件被 Git 忽略，
不自动抓取、不提交、不打包；缺失时推荐器安全降级。

## 开发环境

项目约定的 Python 环境：

```text
D:\miniconda3\envs\sts2\python.exe
```

安装 P0 Python 依赖：

```powershell
conda activate sts2
pip install -r requirements-p0.txt
```

运行完整自动测试：

```powershell
& "D:\miniconda3\envs\sts2\python.exe" -m unittest discover -s tests -v
```

执行 P0 自动审查（不替代安装和真机）：

```powershell
& .\scripts\run_p0_review.ps1
```

## 构建、安装与运行

先复制 `mod/STS2Guide.ReadOnlyExporter/local.props.example` 为被 Git 忽略的 `local.props`，
填写本机 STS2 与 Godot .NET 路径。游戏关闭后，一条命令生成、安装并核对 Mod 三件套：

```powershell
& .\scripts\install_p0_mod.ps1
```

脚本会构建 DLL/JSON/PCK、复制到游戏 `mods` 目录，并比对 workspace/game SHA-256。仅想
检查当前安装是否匹配时使用：

```powershell
& .\scripts\install_p0_mod.ps1 -VerifyOnly
```

开发模式运行后台：

```powershell
& "D:\miniconda3\envs\sts2\python.exe" -m realtime.host --console-log
```

构建单一后台 EXE 并执行完整初始化冒烟：

```powershell
$env:STS2_PYTHON = "D:\miniconda3\envs\sts2\python.exe"
& .\scripts\build_p0_exe.ps1
```

最终用户体验是分别启动游戏和 `STS2 Guide.exe`。后台运行，不需要 Vue、FastAPI、API Key
或网络。

## Public Beta 路线、角色与扩展决策套件

当前策略核心任务见
[`docs/tasks/TASK-008-public-beta-strategy-core.md`](docs/tasks/TASK-008-public-beta-strategy-core.md)，
路线 API 与纵向切片证据见
[`docs/tasks/TASK-007-p1-route-vertical-slice.md`](docs/tasks/TASK-007-p1-route-vertical-slice.md)，
扩展决策自动实现见
[`docs/tasks/TASK-009-extended-decision-suite.md`](docs/tasks/TASK-009-extended-decision-suite.md)。
P0.5 与六项路线 API 门禁曾在 2026-07-14 的 `0.108.0` 上验收；针对当前 `0.109.1`
的协议、运行时、三种路线模式、五角色机制层与兼容门禁已经通过自动验证，但正式
artifacts 尚未安装，仍需先完成隔离探针和正式端到端真机复核。主要边界：

- `map_choice` 继续只是地图观察；只有真实可操作状态才产生 `route_choice`；
- 候选必须是游戏真实允许的下一节点，路径每条边来自 `MapPoint.Children`；
- 普通怪物和精英只计算遭遇池期望风险，不伪造确定敌人；
- Overlay 绑定真实地图视觉节点，不使用截图坐标，不修改原生绘制内容；
- 路线模式是同局保存的软偏好，新局恢复智能均衡；偏离推荐后按实际位置重新规划；
- 五角色共享通用攻防、HP、遗物、路线与跳过规则，角色 adapter 不持有最终分数；
- 旧 0.108.0 没有用户地图 Zoom；当前 0.109.1 必须重新真机确认，Overlay 始终跟随地图
  滚动与窗口/viewport 变换；
- 实时策略只查询 SQLite 和当前局状态，不调用向量库、RAG 或 LLM。
- Merchant 使用真实商品模型、价格、资格和离开候选；购买后同一决策动态更新；
- Rest/Smith、商店移除及其他单目标 Deck Edit 使用明确父子决策；多选组合未建模时隐藏；
- Event/Neow 只消费真实 option ID 和结构化效果，未知收益不从描述文本推断；
- 通用 Drawer 显示分数/不可用状态，并在悬浮提示中给出最多三个已计算原因；
- `explain/` 只复述冻结 Recommendation，默认不打进实时 EXE，也尚未接入游戏内按钮。

## 旧实验入口

仓库仍保留 Vue、FastAPI、Streamlit、Hybrid Search、Reranker、HyDE、LangGraph 和 LLM
问答实验，用于历史对照或未来 P2 攻略解释。它们不是当前实时产品依赖，也不是 P1
优先级；不要为了 Agent 标签继续把这些组件接入实时决策链路。

## 外部数据与参考代码

- Spire Codex 只在构建期提供结构化数据，运行时不联网；
- STS2MCP 和 STS2SourceCode 只用于定位 API，再由当前本机程序集与真机验证；
- 不复制 BoberInSpire、STS2MCP 或游戏源码的实现、权重和数据文件；
- 外部来源、授权与使用边界见
  [`docs/api-reference-sources.md`](docs/api-reference-sources.md) 和
  [`docs/data-source-audit.md`](docs/data-source-audit.md)。

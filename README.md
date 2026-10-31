# STS2 Guide：本地实时决策 Agent

STS2 Guide 是《Slay the Spire 2》的本地、只读、状态驱动策略副驾驶。用户正常操作游戏，
只读 Mod 感知当前决策，本地后台离线计算建议，游戏内 `Context Drawer` 主动显示结果。

它不自动点击、不修改存档、不联网，也不把大模型放进实时推荐主链路。当前产品依据见
[`docs/product-spec.md`](docs/product-spec.md)，代码、安装和真机的真实完成度见
[`docs/project-status.md`](docs/project-status.md)。

## 玩家：当前可测试范围

当前开发候选 **Guide 0.3.15-alpha.0 / Mod 0.3.15** 对应 **Windows、STS2 0.111.0、单人原版内容**。
精确身份以 [`packaging/compatibility.json`](packaging/compatibility.json) 为准；游戏更新后
需要重新验证并更新 Guide，不要绕过兼容检查。这不是已公开验收的稳定版。

- 当前仅开放选牌和路线用于验收。推荐分是当前局面适合度，不是胜率；跳过是正式候选。
- 路线可选智能均衡 / 稳健生存 / 激进成长，只画一条主路线，偏离后按实际节点重规划。
- 侧栏名称与分数分栏，长名称可悬停查看。点击候选旁的 `+` 按需展开至多三条本地评分
  依据，再点 `−` 收起；理由不是完整模拟或胜率解释，不会替你选择。无可用依据时入口变灰。
- 五角色共用评分内核，但本候选五角色完整真机验收尚未完成，不宣称已经全部支持。
- 商店、篝火、Neow 祝福、事件、删牌建议仍关闭。已有实验代码不等于交付功能。
- 本轮自动验证不代替安装向导、保存继续、真机、演示和外部试装；这些留在 TASK-018。

## 玩家：安装与运行

拿到候选包后：

1. 关闭游戏，在旧 Guide 托盘选择“完全退出”。不要同时运行开发 Host 和安装版。
2. 运行交付的 `STS2-Guide-<版本>-win-x64-setup.exe`，按向导确认 Steam 的 STS2 安装目录。
   安装包包含 EXE、只读 Mod 和数据库，无需另外安装 Python。
3. 从 **Steam** 正常启动游戏，再单独启动 **STS2 Guide**。它常驻托盘，不操作游戏。
4. 在支持的选牌和地图决策中查看左侧抽屉/路线；离开后不应保留上一屏建议。
5. 游戏关闭后 Worker 停止、托盘待机；要结束整个 Guide，请选择托盘“完全退出”。

0.3.10安装态完成一轮战士战斗、选牌、侧栏解释、地图及保存继续回归，恢复按游戏实际
保存点重建建议，没有沿用保存后的牌组。0.3.11已经安装，但新增机制尚未重新启动验证；其状态以
[`docs/project-status.md`](docs/project-status.md)为准，不能直接继承上一版或其他角色的结论。
没有虚构的下载或演示入口。

## 玩家：为什么显示 `--` 或没有路线？

| 提示 | 含义 |
|---|---|
| 等待当前建议 | 尚未拿到当前有效结果，或正在确认后台活跃；等待刷新 |
| 评分所需数据不足 | 已确认缺少本次评分必需的数据，不编造分数；有效快照到来后重算 |
| 建议暂不可用 | 读取/协议/身份/兼容校验失败，或后台未确认活跃；不据此猜具体根因 |

后台失联后撤下旧分数和路线，恢复后重新校验当前局、决策及候选。约 5 秒心跳停滞会
触发失效；游戏主线程暂停时不承诺实际 UI 的墙钟时限。后台活跃、局面有效时，不会
仅因建议文件变旧而清空。某些界面没有抽屉，是因为场景尚未支持。

遇到问题先看托盘状态、游戏版本，必要时完全退出 Guide 后重启。托盘提供“打开日志目录”
和“导出脱敏诊断”。反馈时附版本、角色、界面、提示和复现步骤，不要上传游戏目录、
存档或 `.env`。升级语义、复杂延迟收益与协同仍有启发式局限，自动测试不证明策略最优。

### 分享对局记录

游戏每打完一局会自己写一份对局记录（`.run`）。托盘“导出对局记录”把这台电脑上的全部记录打包成
`%APPDATA%\SlayTheSpire2\STS2Guideun-records\STS2-对局记录-<收集者>-<时间>.zip` 并打开该目录，
发给作者即可（QQ／微信都行）。记录只含对局本身（角色、牌组、地图、选择），不含账号；压缩包里也不含
Steam ID 所在的目录名。收集者是第一次导出时随机生成的编号，只用来区分不同玩家的记录。

作者把收到的 zip 都放进同一个目录，然后导入（重复发送的同一局只算一次）：

```bash
python scripts/import_run_history.py --all-runs --collected <放 zip 的目录> --output build/history-all
```

## 实现概览

### 高级本地评分偏好（0.3.9 起）

在 `%APPDATA%\SlayTheSpire2\STS2Guide\scoring-preferences.json` 创建以下文件，完全退出
Guide 托盘后重启生效。设置了 `STS2_GUIDE_EXCHANGE_DIR` 时，文件位于自定义交换目录。
示例也见 [`packaging/scoring-preferences.example.json`](packaging/scoring-preferences.example.json)。

```json
{
  "schema_version": 1,
  "survival": 1.0,
  "immediate_power": 1.0,
  "long_term_growth": 1.0
}
```

分别表示生存、即时输出、长期成长，允许范围均为 **0.75–1.25**；1.0 是默认值。
删除文件恢复默认。格式、版本、字段或数值非法时整份恢复默认并记入日志，不会部分生效。
游戏退出或 Worker 自动重启不会重新加载；新的应用配置会重新计算当前检查点的建议。

只调整明确分类的软评分因素，不调整金币价格、候选资格、致命代价或未知数据；调低生存
不会减轻生命代价的负分。选牌调整后重新比较跳过。商店内部卡牌估值不重复加权，外层仅
调整已分类维度；并非所有因子都会变化。路线仍使用地图侧栏的三个模式，不受该文件影响。

### 新增机制估值的边界（0.3.10）

- 完美打击按真实Strike标签和拿取后的永久牌组估算战前基准，并区分升级参数；不把它
  当成含力量、临时生成/移除或敌方修正后的实战伤害。缺资料或存在未建模实例修改时
  保留限制。展开评分依据可看到估算范围，即使该项不额外加分也不会静默丢弃说明。
- 商店实验策略新增鲜血药水的当前恢复机会：最大生命20%，截断且不超过缺血。购买
  不等于饮用，不计算未来保留价值或战斗用药联动；其他药水保持未知。**公开商店建议
  仍关闭**，自动策略测试通过不能当作商店功能已经可用。

### 已有决策链路

0.3.11源码进一步修正坚毅的主动耗竭提供者关系，并披露随机/选择目标价值未预测；
无情猛攻的下一张攻击免费仍未估值，会明确显示机制限制，不能套用任意高费牌降费。
商店实验策略对结实绷带检查主动弃牌来源，未知牌组保留缺口，不把缺血当成触发条件。
这些增量需要各自真机证据，不代表已公开启用商店。

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
角色机制信号。当前候选仍需完整真机，开关 enabled 不是公开验收通过证明。

工作区还完成了 TASK-009 的自动实现：Merchant、Rest/Smith、Neow、Event 和 Deck Edit
共用 v9 候选 envelope、资源预算、事件管线与通用 Drawer；`explain/` 提供不参与实时打分的
确定性只读解释边界。这些扩展 capability 仍是 `pending_validation`，不在本轮交付范围。
最新验证状态见 project-status，不把历史测试或已有代码当作当前真机完成。

后续开发已恢复，顺序是推荐质量、商店/篝火验证、主动展开的解释，再到祝福和事件。
0.3.6 起修正固定多段伤害、升级费用与弃牌标签；共用资源计算不再重复使用生命、金币
或药水容量。未知升级效果、未支持成本及效果顺序会保留数据缺口，不伪装成完整推荐。
这不代表商店/篝火已经开放，也不是胜率或策略最优的证明。
0.3.8 候选进一步接入游戏真实可移除/可升级目标，资格未知时不再用整个牌组代替；
父子目标冲突会停止给确定建议。已通过代码复核与自动测试，尚待真机，实验能力仍未开放。

0.3.9 自动实现补充了商店真实升级状态、防御遗物的 HP/牌组缺口适配、临近已知 Boss 的
牌组准备，以及完整事件选项键的目录匹配。事件效果白名单目前仅包含 Neow 黄金珍珠的
金币、深渊浴场放弃浸泡的治疗和离开；复杂/随机/耦合选项仍未知。药水缺少已验证结构化
机制时不产生购买分数；存在未知可购买项时不会宣称整屏最佳。新增切片仍待真机验收。

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

开发者构建完整候选安装包（测试、SQLite、Mod/PCK、EXE 自检与 payload 审计，不安装）：

```powershell
& .\scripts\build_public_beta.ps1 -Python "D:\miniconda3\envs\sts2\python.exe"
```

EXE 输出在 `dist/`，安装包在 `dist/installer/`（Git 忽略）。发行构建强制执行所选 GamePath 的程序集身份测试；
普通开发环境无程序集可跳过该测试，跳过不能当作发行通过。以下安装/运行命令仅供另行
安排真机测试时使用，不要与托盘安装版同时启动。

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
P0.5 与路线 API 的旧版本证据仅供历史参考；不能沿用为当前 `0.111.0` 候选的验收。
当前自动验证与未完成真机项分别记录于 project-status。主要边界：

- `map_choice` 继续只是地图观察；只有真实可操作状态才产生 `route_choice`；
- 候选必须是游戏真实允许的下一节点，路径每条边来自 `MapPoint.Children`；
- 普通怪物和精英只计算遭遇池期望风险，不伪造确定敌人；
- Overlay 绑定真实地图视觉节点，不使用截图坐标，不修改原生绘制内容；
- 路线模式是同局保存的软偏好，新局恢复智能均衡；偏离推荐后按实际位置重新规划；
- 五角色共享通用攻防、HP、遗物、路线与跳过规则，角色 adapter 不持有最终分数；
- Overlay 的滚动、缩放与窗口变换需在当前候选重新真机确认；
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

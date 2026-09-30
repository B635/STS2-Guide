# TASK-022 — 商店、Boss、事件与高级本地偏好

- 状态：待真机验收
- 产品依据：product-spec 第 2.4 节，本轮用户明确扩展授权及本地配置选择。
- 前序 TASK-018–021 真机未完成，不自动验收。
- 基线：`build/task022-baseline/`，排除既有工作区改动，不提交/推送/重置。

## 分工与顺序

1. 商店 Agent：`advisor/merchant.py`、MerchantObserver、merchant payload schema/Pydantic
   与专项测试。升级状态、标签契约、真实局面适配；API必须来自现有明确公开成员。
2. 事件 Agent：EventDecisionObserver、新的精确事件效果读取模块、event_policy与测试。
   先证实当前完整TextKey/静态option_id匹配，再根据当前程序集证据扩白名单。不得假定效果。
3. 偏好 Agent：scoring_preferences、card_reward、policies、choice_effects、DecisionRequest/
   processor、host/public_beta与专项测试；应用级冻结、默认等价和checkpoint配置变更重算。
   不与商店/事件Agent同时编辑merchant.py/event_policy.py，root最后接入两处偏好注入。
4. root：Boss共用事实与卡牌准备、任务/规格、整合权重与商店/事件、版本、完整构建和文档。

## 核心验收

- 商店基础/升级牌不混算；只有card offer携带upgrades，未知不默认0；价格单调、满槽、
  额外成本、未知机制/资格不退化为确定建议。防御遗物实际supports_block参与但不伪造产量。
- Boss已知身份可影响准备，缺失/多成员/进阶明确处理，不猜未来普通怪或精英；路线旧行为保持。
- 事件完整稳定TextKey只能匹配本事件/唯一页面选项；已核对效果来自真实对象，未知仍未知；
  状态变化/父子决策/保存继续不沿用旧候选。所有新API登记程序集证据。
- 偏好范围0.75–1.25，默认逐分等价，错误整体默认并日志；一次应用内Worker重启不热加载；
  Guide重启的新偏好不能复用旧checkpoint分数。未知/危险/不可用不能变可推荐；跳过与候选一致。
- 未经当前程序集证实的药水/事件机制只报告缺口，不靠源代码名或描述猜效果。
- 逐切片定向测试，最终完整Python/Mod/PCK/EXE/安装包，非实现者独立检查后如实写报告。

## 限制

自动实现阶段不启动或操控游戏，不安装新包，不开生产Host，不启用未验收capability。
后续用户授权测试、安装和后台操作的实际范围与证据见文末追加记录；历史自动实现证据不改写。
本票不是“全部事件和所有机制已完成”的声明，必须列出具体支持/未支持清单。

## 实施报告

### 自动实现

- 商店：生产 `CurrentUpgradeLevel` → nullable card-only payload → schema/Pydantic → 卡牌
  估值闭环。缺失/非法/重复升级保留未知；防御遗物使用真实 supports_block 标签，有限结合
  HP/格挡覆盖缺口。价格与合法性不受偏好改写。新增升级商品示例与同槽 revision 重算回归。
- Boss：`advisor/boss_context.py` 共用只读遭遇/成员/HP/最大明确攻击压力；A8 HP、A9 伤害
  分开，固定多段计入，互斥行动不相加。路线公式保留；三步内已知 Boss 的牌组缺口因子
  替换通用临近正项而非叠加。目录不完整披露缺口，退回通用路径信息，不装作知道具体行动。
- 事件：保留完整运行时 TextKey 身份，严格映射当前事件/目录页/短选项；不做尾缀猜测。
  当前实际程序集验证后只接入 GoldenPearl 金币、AbyssalBaths ABSTAIN 治疗和 EXIT_BATHS
  no_op；真实 DynamicVars 数值变化进入完整候选快照，其余含复杂副作用的选项继续 unknown。
  API证据与限制写入 `docs/api-reference-sources.md`。
- 偏好：高级 JSON 三参数 0.75–1.25，缺失回默认、非法整份回默认并告警；应用启动冻结，
  Worker 重启复用。选牌在求和前加权，跳过基于已调整候选；扩展 Draft 复用同一参数，
  商店内层默认避免重复加权。价格/安全/未知/资格不受绕过，路线不应用此 JSON。
- Host advice envelope 增加可选规范化 `scoring_preferences`（schema/示例同步），不是游戏
  事实或发行指纹。偏好改变只重算当前活跃完整快照，保留生产 sequence、父子来源及关闭
  状态，不新建游戏事件、不沿用旧配置评分。无新增历史表或网络依赖。
- 版本：Guide `0.3.9-alpha.0` / Mod `0.3.9`，policy bundle v4；协议仍为9、SQLite8、
  effect tags6，route policy仍v3。扩展 capability 全部保持 pending_validation。

### 独立审查与返修

- root 审查事件实际 diff：精确模型/键/Relic 联合匹配、完整键目录解析、未知降级及候选
  更新链路；生产编译、纯投影/API签名与跨协议管线测试纳入最终完整构建。
- Event Agent 独立审商店：未发现新增阻断，独立20项 / 0.770s OK；核对升级/非card字段
  排斥、额外成本、槽位、偏好单次应用与未知药水的全屏不确定行为。
- Shop Agent 独立审 Boss 与偏好：发现 P2 成长覆盖只数scaling漏掉Power，root改为每张
  `(scaling OR Power)` 计一次；补Power无标签和双标签去重，独立复核关闭。
  Boss+Route初审37项 / 7.477s；修复后Boss+Preferences+父子管线36项 / 1.241s通过。
  配置冻结、当前快照重算、parent_close保留与负HP安全均未发现剩余阻断。
- 偏好 Agent 相关回归255项 / 25.899s，最终专项13项 / 0.693s；商店实现专项37项 /
  1.893s；事件专项27项 / 0.949s。这些不是全量验收数字，最终构建另列。

### 明确未完成

- 当前药水无经验证结构化机制，遗物无标签也保留未知；有可买未知商品时整屏uncertain，
  其他已知商品分数可保留，但不能称作完整商店最优购买建议。
- 事件白名单仅上述三项；营养牡蛎等增上限且回血、随机物品、复杂事件链尚未覆盖。
- Boss为临近战前启发式准备，不是胜率、逐回合策略、Boss遗物选择或完整机制模拟。
- 不安装、不启动游戏/生产Host，本轮自动验证和独立代码复核不等于真机；前序验收保留。

### 最终构建

- 命令：`scripts/build_public_beta.ps1 -Python D:\miniconda3\envs\sts2\python.exe`。
- 最终日志：`build/task022-public-beta-final.log`，exit **0**。
- 完整 Python **583 项 / 48.166 秒 / OK**；包含配置好的本机程序集托管行为与Godot组件
  测试，Mod/PCK **0 warning / 0 error**；SQLite模板、冻结EXE隔离自检、安装包编译与
  payload审计均通过。不能将其中组件测试称为真实游戏回归。
- 包：`dist/installer/STS2-Guide-0.3.9-alpha.0-win-x64-setup.exe`。
- 最终发行指纹：`ccb5bdd667ca570a65714d1b29ade49db78dbc7588757cef871b0a3dcb70fdee`。
- `git diff --check` 通过；配置example通过Python严格解析与schema校验。已安装Mod仍为
  0.3.5，读取游戏安装清单核对，本轮没有安装、修改存档、启动生产Host/游戏、提交/推送
  或消费额度重置卡。
- 下一步：统一升级安装态后做卡牌权重/重启恢复、真实商店升级/预算/移除，以及上述具体
  事件白名单的隔离真机。扩展未通过自身验收前继续关闭，不将本任务状态写成已验收。

### 后续安装与后台验收准备（2026-09-25）

- 用户随后授权真实测试，0.3.9 候选安装 exit 0，日志 `build/task022-live/install.log`；
  Guide 安装回执/EXE 与游戏 Mod 已核对。通过 Steam 启动游戏并看到主菜单。此证据覆盖
  上节构建结束时“未安装”的历史状态，但不代表选牌、路线或扩展真机通过。
- 用户进一步要求在后台玩、不操作前台；原生 UI 操作已停止，继续点击被打断，未据此认定
  已进入存档。现阶段只在本机忽略目录准备开发测试桥，不加载它，不重启或切换窗口。
- 上游 STS2MCP 固定 main 的编译兼容、读取副作用及后台限制已审计，详见
  [`../dev/sts2mcp-background-compatibility.md`](../dev/sts2mcp-background-compatibility.md)。
  单人隔离适配不是产品源码；正式 Guide 仍只读，扩展能力仍关闭。
- 隔离测试工具最终编译 0 warning / 0 error；root 直接执行其传输校验/排队代码的 21 项
  托管检查通过，独立审查读取与显式操作 diff 无阻断。它尚未加载，无法据此判断游戏失焦时
  仍能处理帧或动作；下一步等待用户方便关闭游戏后安装加载，不自行重启占用前台。
- 当前 quota 指令已由用户改为“额度实际为 0 时使用一张重置卡”，只适用于本次一张；
  不提前兑换、不连续兑换多张。截至本记录尚未使用。

### 后续无窗口测试尝试（同日）

确认游戏已退出后安装本机 MCP 适配，经 Steam --headless 加载，MainWindowHandle=0，
主线程 GET 能读取 main 菜单及真实 continue 入口。正式 Guide 的第三方 Mod 检查按设计
关闭 Worker，未改动规则，随后正常退出 Guide。独立开发 Worker/continue 执行请求与
之后的测试清理请求均被环境 blocked by policy 拒绝，不绕过、不记作成功。最后只读检查
无窗口游戏 PID 43948 仍运行，临时 MCP 三个文件仍安装，Guide 不在运行；需用户 Steam
停止游戏后核对清理。未成功执行任何对局动作，未验证推荐、商店/事件或 UI 外观。

用户随后确认停止游戏。root 核对无游戏/Guide进程，确认身份后成功删除临时MCP三件套，
正式Guide Mod的VerifyOnly核对通过。当前不再有测试MCP安装，不沿用上段遗留状态。
用户改为正常可见Steam游戏验收，不继续headless/MCP路线；Computer Use已初始化，
尚未激活或输入，等待确认之前禁止占用前台的限制是否取消。真实决策验收仍未完成。

### 正常窗口真机返修（2026-09-25）

用户确认继续后改为正常 Steam 窗口及 Computer Use，不使用 MCP。安装态真实选牌 event47
发现以下缺口，沿用 product-spec 2.1 的“真实理由及未建模机制披露”要求，不改变产品范围：

1. 黑暗之拥已计入抽牌（draw_count=1、deck_size=12），但 draw_coverage 理由写成
   “缺少抽牌来源”。改为带抽牌效果的卡牌占比偏低，不声称已经评估实际触发/稳定性；不改权重。
2. 完美打击只按基础6伤害计算效率，未纳入按打击数量增伤，却返回 data_gaps=[]。
   对这个已证实但尚未建模的机制添加候选级明确缺口并沿既有置信度路径处理；不凭经验
   抬高分数、不猜牌名/打击数量、不临时扩展新的协同模型。
3. 先建立上述两点的回归，再修实现；核对缺口经 CardRewardPolicy 到 Recommendation，
   并在现有候选解释入口可见。必要时明确区分实体数据已找到与机制覆盖不足。

实现者负责上述 Python 评分/解释与专项测试；root 负责 AdviceReasonReader 对明确零贡献
限制说明的优先展示及真实托管回归，由另一位 Agent 独立复核 C#；root 独立审查 Python 及完整回归。游戏 UI 由 root
独占。修复仅在源码验证，不在游戏运行中替换安装包或宣称当前0.3.9已经包含修复。
其他选牌/路线/保存继续的实际验收证据待本轮结束追加，五角色及扩展门禁不随此自动通过。

### 正常窗口返修实施及独立复核结果

- `advisor/contextual_scoring.py` 将抽牌理由改为“带抽牌效果的卡牌占比偏低”，保持原有
  计数、阈值和分数。真实12牌场景中黑暗之拥被计入，覆盖因子仍为+4。
- `advisor/card_reward.py` 对规范化 `perfected_strike` 增加候选缺口
  `card_mechanic:PERFECTED_STRIKE:strike_count_damage` 和零贡献 `unmodeled_mechanic`
  因子，说明尚未覆盖按打击牌数量增加的伤害，效率仅按基础伤害估算；不改分数或排序。
  `advisor/policies.py` 区分实体查找成功与机制覆盖不足，保留已知实体的数据完整度，
  同时传播机制缺口和低置信度；不能把数据库已命中写成机制完整。
- `AdviceReasonReader.cs` 在原有最多三条解释中优先保留明确零贡献限制，再按绝对贡献
  展示其他因子；普通零分因子仍不展示，无有效分数的候选不显示评分理由。沿用既有字段，
  不改协议，不向因素伪造分值。
- 回归先失败再通过：Python新增3项覆盖实际12牌局面、发行数据库路径、机制缺口从策略
  进入Recommendation，以及大小写/中文名称/升级状态；托管测试直接执行生产Reader，
  验证限制优先、去重、三条上限与无分数降级。已有机制缺口用例同步实体完整度语义。
- 独立复核：root审查Python实现及测试；非实现Agent分别检查C#标准/规格、实际调用方及
  托管行为，无剩余阻断。独立托管入口2项通过，独立Mod/PCK编译0警告/0错误。
- 最终完整命令 `D:\miniconda3\envs\sts2\python.exe -m unittest discover -s tests -v`：
  **586项 / 52.758秒 / OK**，日志 `build/task022-live/normal-window-fix-tests.log`。
  日志中的 `expected test failure` 来自控制器故障恢复测试，最终退出码为0。
  root执行Mod `--no-restore` 编译 **0 warning / 0 error / 6.52秒**。
- 正常窗口真机证据见 [`../project-status.md`](../project-status.md) 同日正常Steam窗口回归：
  选牌与关闭、三模式重算、偏离后重规划、商店节点保存继续通过本次场景；未验证窗口缩放、
  五角色完整第一幕、偏好文件重启、商店/篝火/事件。游戏退出确认尚待用户回复，不能声明
  本次自动待机通过。
- 以上修复仅在源码及自动测试中验证。已安装EXE/Mod仍为原0.3.9，发行安装包没有重新
  生成或覆盖。任务保持待真机验收，下一步先生成新版完整候选并在游戏关闭后安装，再复测
  本轮披露修复及剩余验收项。未提交/推送、未编辑存档、未兑换额度重置卡。

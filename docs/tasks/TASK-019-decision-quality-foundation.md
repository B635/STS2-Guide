# TASK-019 — 决策质量与扩展策略基础

- 状态：待真机验收
- 产品依据：product-spec 第 2.3 节；2026-09-25 用户授权继续产品开发。
- 实现：当前 Codex 及分配明确文件范围的子 Agent；审查由不同实现者独立复核。
- 前序 TASK-018 保留待验收；本票不宣称其完成。

## 生产实施顺序

1. 当前目录建立反例：固定多段伤害、升级降费、否定弃牌不能作为协同提供者。
2. 修复共用资源预算：同向资源效果整体计算、余额/容量一致性；无法确定执行顺序的
   混合效果明确未建模，不能依靠净值或逐项重复计算通过。
3. 商店拒绝未支持的混合成本；Smith 不丢弃未知升级目标或部分升级效果缺口。
4. 回归所有现有决策；更新策略和可重建标签版本、候选身份，完整 Python 测试及 Mod 编译。
5. 独立审查实际增量并返修，填写实际证据。保持扩展 capability 待验证，安装/真机单列。

## 文件分工与验收

- root：`advisor/resource_budget.py`、`tests/test_resource_budget.py`；文档、版本/清单与集成。
- card-quality：`advisor/card_reward.py`、`advisor/contextual_scoring.py` 与专用行为测试；
  使用现有目录升级语义，固定多段合计，费用改变影响候选和牌组，不改既有权重。
- tag-quality：`storage/effect_tags.py` 与专用测试；否定、条件、弃牌堆不产生主动弃牌
  provider，真实主动弃牌保留。修改派生逻辑须提升 EFFECT_TAG_VERSION。
- expansion：`advisor/choice_effects.py`、`advisor/deck_edit_value.py` 及商店/篝火/Deck Edit
  测试；漏掉生命成本不得输出完整建议，含未知升级效果不得宣称最佳目标。
- 前置工作区快照位于忽略的 `build/task019-baseline/`；已有未提交改动不回滚、不冒领。

验收关注生产接口行为，不以源码字符串测试替代。不得放宽身份/协议，不新建实时历史库，
不添加网络/LLM 实时依赖。不修改无关 Mod API，不绕过 capability 验证。

## 实施报告

### 实际修改（2026-09-25）

- `advisor/contextual_scoring.py`、`advisor/card_reward.py`：显式固定段数用于基础伤害合计；
  条件追加攻击不默认触发，X 能量 / X 星星攻击不伪造固定伤害效率。候选和牌组曲线共用
  `card_energy_cost`，区分绝对升级费用与有符号增量，未知费用不按已知基础费用处理。
- `storage/effect_tags.py`：标签 v5→v6，主动弃牌识别共用于通用标签与角色 provider；
  排除否定、触发条件和弃牌堆名，保留后续独立子句中的真实弃牌动作。SQLite 派生标签可重建，
  未新增静态数据表或实时历史记录。
- `advisor/resource_budget.py`：同方向 HP / 最大 HP / 金币变化按整个候选合计；不重复使用
  缺血量或余额。未声明执行顺序的正负混合、HP/最大 HP 耦合保留阻断缺口。药水验证真实槽位
  和容量、每次移除消耗一件实际库存，不把同一个空槽或药水重复使用。
- `advisor/choice_effects.py`：商店只接受当前建模的金币成本，混合生命 / 最大生命 / 自定义
  成本不得被默默丢弃后输出完整建议。
- `advisor/deck_edit_value.py`：Smith 的所有可升级目标必须有完整的已支持升级字段；不能
  忽略未知目标或只计算部分字段后宣称最佳目标。成本直接复用上述读取函数，不保留两套解释。
- 回归文件：新增 `test_card_quality.py`、`test_effect_tag_quality.py`，补充 resource_budget、
  merchant、campfire、deck_edit 测试，以及派生标签版本/迁移断言。
- 候选身份更新为 Guide `0.3.6-alpha.0` / Mod `0.3.6`，同步既有策略清单与示例；
  协议 v9、SQLite schema v8、原始目录不变。Mod 本轮仅更新发行身份，没有改游戏观察 API。

### 独立审查与返修

以 `build/task019-baseline/` 为增量基线，排除前序已存在的未提交变更。资源/选牌、扩展策略、
标签由非本人实现的 Agent 交叉检查；root 负责集成、边界测试与版本一致性。

1. STARDUST 的 X 星星攻击误按一次攻击：审查复现，增加真实目录基础/升级候选回归后修复。
   已知零能量费用仍参与曲线，不猜当前星星可支撑的攻击次数。
2. 英文 `cannot be discarded` 被动否定新增假 provider：补否定被动态和真实主动正例，
   先红后绿；原两例经独立复核均不再产生弃牌 provider。
3. 篝火把升级费用 `+1` 误当降费：集成发现并复用统一成本函数，Campfire/Deck Edit 公开
   接口覆盖正负增量、绝对值和非法值；最后独立审查确认无循环依赖或已知费用退化。

本轮发现的上述问题均已关闭。定向测试通过不代表完整构建或真机通过；完整证据见下一节。

### 集成验证与安装态

执行 `scripts/build_public_beta.ps1 -Python D:\miniconda3\envs\sts2\python.exe`，exit 0：

- 完整 Python **509 项 / 79.382 秒 / OK**（前序 466 项，本轮新增 43 项测试方法，另含表驱动
  子例）；包含本机程序集/Godot 行为验证，不跳过 release 要求的测试。
- Mod/PCK **0 warning / 0 error**；没有调用安装脚本。
- 发行 SQLite 重建通过，schema v8、effect tag v6；EXE 隔离启动自检、Inno 安装包构建和
  payload 审计通过。日志 `build/task019-public-beta-build.log`，产物清单 `dist/release-manifest.json`。
- 新安装包 `dist/installer/STS2-Guide-0.3.6-alpha.0-win-x64-setup.exe`；发行指纹
  `d026d521ec4ad7eea06e5affd37a6a7d3cc3424cd6b826abc4d78c82339fad51`。
- 最终只读核对：实际安装仍为 Guide `0.3.5-alpha.0` / Mod `0.3.5`；本轮未安装、未启动游戏
  或生产 Host。构建结束后没有游戏、Guide 父子进程及开发 Host 残留。未提交/推送、未使用额度重置卡。

完整测试和编译通过、增量独立审查问题已关闭，因此进入“待真机验收”，不是产品已完成。

### 仍未覆盖与下一段

- 普通选牌的非费用升级仍沿用既有通用升级加分；未宣称完整计算升级后的伤害、抽牌或能力。
  条件伤害、复杂延迟收益与效果执行顺序没有引入模拟器，未知情况保留缺口。
- 篝火当前不支持所有升级字段（如能力/易伤变化），所以真实牌组可能没有可比较的 Smith
  分数。这是需要补事实与建模的明确缺口，不能通过放宽未知检查解决。
- 商店/篝火策略继续 `pending_validation`。下一段按当前游戏程序集与实际对象核对候选、
  成本、升级目标和关闭生命周期，再完成对应离线回归、独立审查及真机，不能只翻开关。
  专项 API 证据仍有 0.109.1 记录，必须核对当前 0.111.0 的 Merchant 关闭/购买、Rest 成功回调
  及 Deck Edit 异步签名。父层“最佳升级/移除”目前扫描整个牌组，而真实子屏才检查
  `IsUpgradable/IsRemovable`；父子目标资格一致性必须先补证据，不能猜资格。
  Mod 内嵌能力清单，单改 Host 清单无法完成验证；应采用现有开发探针或隔离验证产物，
  不放开公开能力清单。最小真机矩阵是商店购买/库存更新/移除返回/离开、Rest 成功、Smith
  确认/取消，以及两者保存继续和旧建议撤下。
- TASK-018 未覆盖的五角色、偏离路线、窗口变化、保存断点、安装向导、演示和外部试装仍保留。
  自动测试用于工程/规则行为，不证明胜率提升，也不替代玩家体验验收。

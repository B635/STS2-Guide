# TASK-020 — 商店/篝火真实目标资格

- 状态：待真机验收
- 产品依据：product-spec 第 2.3 节；用户继续推进授权。
- 前序 TASK-018 / TASK-019 保留待真机验收，不因新票开始而标记完成。
- 本轮不安装、不启动游戏/生产 Host、不启用公开扩展 capability，不扩展 Neow/Event/RAG。

## 实施顺序

1. 从当前 0.111.0 本机程序集核对 Merchant / Rest / Deck Edit 目标 getter、真实选择过滤及
   回调签名，记录来源；不得猜 API。审查只能静态证明的内容与真实时序分开记录。
2. v9 候选 payload 新增可空 `deck_edit_targets`：`operation`（upgrade/remove）、
   `source_code`、`targets`（至多 200 个 `{target_id, card, upgrades}`）。全部字段严格类型；
   target_id 唯一、稳定 ID 非空、升级 0–9。缺失/null 为未知，空列表为已知无目标。
   仅商店移除与篝火对应 choose 效果可使用；不复用旧的 `target_candidate_ids`。
3. Mod 共用只读目标读取函数，调用已验证的真实资格；不会部分导出或截断。子 Deck Edit
   导出真实 upgrades。Python 不扫描全牌组猜资格，比较前验证目标实例计数/升级与快照一致；
   原牌组仍用于上下文协同，不把已过滤目标集当作完整牌组。
4. 覆盖不可移除的高价值卡/不可升级未知卡不进入比较、空/缺失集合、错误操作、重复 ID、
   同 ID 多实例、资格随快照变化、旧录制降级、有效候选正常评分。协议需真实跨层通过。
5. 独立审查和返修；完整 Python 测试、Mod 编译、候选构建；按实际结果写状态与 API 证据。

## 分工

- root：规格/任务、集成和版本、文档及完整构建；保存本轮增量基线到忽略的 build 目录。
- protocol：`realtime/protocol.py`、`protocol/state-event.schema.json`、协议示例和专项测试。
- strategy：`advisor/deck_edit_value.py`、`merchant.py`、`deck_edit.py`、相关 Python 策略测试。
  优先从 draft.candidate.payload 读取资格，避免为单一新增字段传递多层平行参数。
- Mod/API：`MerchantObserver.cs`、`RestSiteObserver.cs`、`DeckEditObserver.cs`、
  `ProtocolModels.cs`、最小共用读取函数、程序集行为验证；不与 root 同时改发行常量。

## 实施报告

### 实际实施

- 调研与失败用例先由并行 Agent 完成；三名 Agent 随后均遇到账户额度限制。root 接管全部
  生产实现、定向测试、集成自查及构建，没有使用重置卡，也没有将 root 自查称为独立审查。
- Mod：新增 `DeckEditTargetReader`，从真实牌组实例读取 `IsRemovable/IsUpgradable`，
  投影到唯一快照目标 ID、卡牌 ID 和升级次数。SmithCount 非 1、读取异常或目标非法/超限
  返回 null；合法目标为空返回空集合。Merchant/Rest 共用，Deck Edit 子候选同时导出 upgrades。
- 协议：`realtime/protocol.py`、state schema、Mod DTO 和完整新示例同步加入可空
  `deck_edit_targets`；保留缺失/null/空的区别。嵌套数据严格校验，父场景/操作/效果必须匹配。
  JSON Schema 的 uniqueItems 只比较整行，跨不同事实行的 target_id 唯一性另由消费者校验。
- 策略：`best_*_opportunity` 只遍历声明完整的合法目标集合，以 `(card, upgrades)` 多重
  计数验证当前局归属。整个牌组仍用于移除协同背景。合法集合内未知卡、未建模升级、重复
  升级保持缺口；集合外的不可升级卡不会伪装为升级候选，也不会阻断合法目标的升级比较。
- 子候选缺失升级状态不再从 candidate_id 反推；当前局不存在相同实例状态时降级。同名牌
  不同升级组的移除取对应组：集成自查复现 6.0 误用，应为原规则下的 7.5，新增回归后修复。
- 第二轮自查复现多选效果错误复用单选目标评分；补失败用例后，同步约束 schema/Pydantic
  与纯策略入口：当前只支持一个 choose 效果、一次选择一个目标，多选保持未建模。
- 没有扩大 WorldState 或持久化表，没有调权重、添加 LLM/网络，没有绕过公开 capability。
  v9/SQLite v8 不变；发行候选升至 Guide `0.3.7-alpha.0` / Mod `0.3.7`。

### 当前程序集证据

本机 `0.111.0 / 41cef1ea`，精确签名见 `build/task020-api-members.log`，汇总登记在
`docs/api-reference-sources.md`。调研 Agent 在额度中断前确认选择过滤与 SmithCount 的 IL
调用关系；root 再用本机 Sts2ApiProbe 记录成员，并通过托管行为程序验证 getter/回调签名。
生产 Postfix 目标无需变更。成员/IL 存在不证明 UI 回调在真实流程中已触发。

### 主要修改文件

- `mod/STS2Guide.ReadOnlyExporter/DeckEditTargetReader.cs`：共享资格读取与完整目标投影；
  `MerchantObserver.cs`、`RestSiteObserver.cs` 接入父层，`DeckEditObserver.cs` 导出子层升级。
- `ProtocolModels.cs`、`realtime/protocol.py`、`protocol/state-event.schema.json`、
  `protocol/deck-targets-event.example.json`：生产 DTO、严格消费契约和可执行跨层示例。
- `advisor/deck_edit_value.py`、`merchant.py`、`deck_edit.py`：真实目标比较、实例匹配及未知降级。
- `tests/test_deck_target_eligibility.py`、`test_deck_target_protocol.py`、
  `test_deck_target_pipeline.py`：资格、协议、更新/重启/跨局回归；原策略正向 fixture 同步。
- `tools/GuideBehaviorTests/DeckEditTargetTests.cs`、`Program.cs`、项目文件：执行生产投影、
  JSON DTO 和本机程序集成员断言，不仅匹配源码字符串。
- 发行版本、Mod 清单、兼容清单、协议示例与 README 同步为 0.3.7 候选；未修改扩展开关。

### 验证记录

- 新增协议、目标资格及 event→policy→checkpoint 回归；旧正向 fixture 明确补合法目标集合，
  缺失资格的反例单独保留，不能靠生产代码回退扫描牌组维持旧测试。
- `tools/GuideBehaviorTests` 链接生产目标投影/DTO，测试合法过滤、同名多实例、升级状态、空
  集合、非法/超限及 JSON 字段，同时执行当前程序集回调签名检查。
- 定向 74 项通过；随后完整构建第一轮 Python **530 项 / 55.081 秒 / OK**、Mod/PCK
  **0 warning / 0 error**，EXE、安装包、自检和审计通过。
- 自查发现同名牌升级组误匹配后，新增失败用例并修复，相关 **28 项 / 0.923 秒 / OK**。
- 再次发现多选目标复用后，两个失败回归复现，修复后的相关 **63 项 / 2.888 秒 / OK**。
- 最终执行 `scripts/build_public_beta.ps1 -Python D:\miniconda3\envs\sts2\python.exe`，
  exit **0**：完整 Python **532 项 / 75.597 秒 / OK**；Mod/PCK **0 warning / 0 error**；
  发行 SQLite、冻结 EXE 隔离启动自检、Inno 安装包和 payload 审计通过。
  完整日志 `build/task020-public-beta-final.log`，最终包为
  `dist/installer/STS2-Guide-0.3.7-alpha.0-win-x64-setup.exe`。
  指纹 `a51822d190e23e1523b934da3e107942f27b1f47c25247a09f00b3ba6856b3eb`；
  v9 协议 / SQLite v8 / tags v6 不变。前两轮包不作为最终返修验收证据。
- 只读核验安装态：EXE ProductVersion `0.3.5-alpha.0`，游戏目录 Mod 清单 `0.3.5`；
  本机游戏仍为 `0.111.0 / 41cef1ea`，新候选没有覆盖已安装版本。
  构建后核对没有游戏、Guide 父子进程或开发 Host 残留。

### 审查与验收边界

独立实现后审查尚未完成：并行 Agent 额度受限，没有伪造审查结论。此票自动结果即使全部
通过也应保持“待审查”，不能直接升为待真机验收或已验收。

本轮未安装、未启动游戏/生产 Host、未提交或推送。实际安装候选仍须单独核验，不以源码
版本代替。商店/篝火/Deck Edit 等公开能力仍为 pending_validation。

后续仍须：独立审查实际增量（基线 `build/task020-baseline/`）→隔离真机验证商店购买/
移除返回、Rest、Smith 确认/取消、多选降级、保存继续及旧建议撤下。父阶段完整目标集合与
实际子屏是否一致要真机核对；不按截图、显示文字或旧版本证据猜测。

独立审查重点：schema/Pydantic 对缺失、空、多选与错误场景的一致处理；生产过滤与当前
程序集的对应关系；实例计数/升级分组；旧检查点不复用资格；任一未知合法目标不能被静默
排除。比较本轮基线，不把工作区原有 TASK-014–019 的改动计为本票引入。

## 2026-09-25 后续独立审查与返修

独立审查 Agent `targets_final_review` 对本轮增量及实际管线复核，53 项定向测试通过
（1.507 秒），但发现一个未覆盖的 P2：父 Smith 完整目标仅 STRIKE_IRONCLAD，子屏在
相同牌组状态下仅 DEFEND_IRONCLAD，仍能 recommend / 52.25 / 无缺口。父子操作已校验，
但没有保留并比较父层目标集合，违反第 2.3 节的一致性要求。其余本票增量未发现阻断。

返修范围：复用已验证父观察证据，比较卡牌/升级次数多重集合（不比较 target_id）；已知
冲突保留缺口/不发布确定建议，不静默删除子候选。父资格缺失的旧录制保持可独立使用真实
子屏。覆盖首次进入、检查点恢复、同名多实例及升级冲突。修复后重新测试、独立复核。
允许此修复作为 TASK-021 候选的前序修复并行进行，不扩大其他策略。

返修实施：原审查 Agent 随后按明确范围实现，仅修改 `realtime/processor.py` 与
`tests/test_deck_target_pipeline.py`。先复现 7 个失败断言，新增 9 项测试；72 项定向
测试 / 3.279 秒 / OK。沿现有 lineage 保存目标事实，重启后从原父观察再次派生；已知
冲突保留候选、清空评分/排序/因子并输出 `deck_edit:parent_targets_mismatch`。
未知旧录制可独立使用真实子屏；不以缓存字段替代原父事实。

返修独立复查：`drawer_review` 与返修实现者分离，检查实际 diff，并独立运行 13 项管线
测试 / 0.620 秒 / OK，未发现剩余阻断。root 同时复核缓存字段兼容和恢复逻辑。最终完整
构建随 TASK-021 的 0.3.8 候选执行；本票仍未完成真机父子目标对照，不沿用 0.3.7 构建结论。

最终集成构建 exit 0：542 项 / 47.212 秒 / OK，Mod/PCK 零警告零错误，冻结 EXE 自检、
安装包和 payload 审计通过。见 `build/task021-public-beta-build.log`；返修代码/自动门禁
已通过，状态更新为待真机验收，扩展 capability 仍未开放。

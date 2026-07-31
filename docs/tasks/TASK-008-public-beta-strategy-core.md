# TASK-008：Public Beta 策略核心（PB-2 + PB-3）

- 状态：待真机验收
- 创建者/主审：Codex
- 实现角色：Codex，可使用明确分工的子 Agent
- 前置：TASK-007 自动返修通过；`0.109.1` 隔离与正式真机门禁延后合并执行
- 创建日期：2026-07-31

## 目标

在不增加新决策类型的前提下，一次收口 Public Beta 的策略核心：

1. 路线只显示一条当前主路线，并提供“智能均衡 / 稳健生存 / 激进成长”三个软偏好；
2. 路线模式在同一 Run 保存继续后恢复，新 Run 重置为智能均衡；
3. 玩家偏离推荐、HP/牌组/遗物/药水/金币或地图事实变化后，以实际状态重新规划；
4. 五个角色共享同一评分内核，通过薄机制适配把结构化卡牌事实映射为统一 provider/payoff
   信号；
5. 卡牌建议能解释关键角色机制与路线偏好，不把推荐度宣传为胜率。

Neow、Boss 遗物、商店、篝火、事件建议、战斗逐回合、RAG/LLM、托盘和安装器不属于本
任务。

## 协议与生命周期

1. 生产协议升级为 schema v8；v1-v7 仅保留离线回放兼容。
2. v8 每个完整事件必须显式携带：

   ```json
   {
     "guide_preferences": {
       "route_mode": "balanced"
     }
   }
   ```

3. `route_mode` 只允许 `balanced`、`survival`、`growth`，不得接受自由文本、缺失字段或
   静默默认。
4. 用户在策略侧栏切换模式时，Mod 只修改 Guide 自身状态，不调用游戏选择 API；当前
   `route_choice` 复用 decision ID，产生新 event ID、递增成功提交 sequence 和 `UPDATED`。
5. Host 将偏好与当前局事实原子写入唯一 `active-run.json`。Mod 恢复时必须同时校验
   `run_id + release_fingerprint`；旧 Run、旧 release 或损坏 checkpoint 均重置 balanced。
6. 所有 Card/Route 事件均携带当前 mode，使路线模式可以影响路线评分和后续选牌的
   `route_fit`，但不能覆盖生存底线。
7. manifest、producer、consumer、schema、示例、checkpoint、测试和打包资源必须同时更新；
   release fingerprint 重新计算，清单继续保持 `pending_validation`。

## PB-2：路线软偏好与 UI

### UI

- 路线策略侧栏顶部显示三个互斥按钮：智能均衡、稳健生存、激进成长；
- 默认智能均衡；当前项有明确选中态；
- 点击只更新 Guide 建议，不点击地图、不拦截地图操作、不修改原生绘制；
- 侧栏不显示“下个节点”式冗余信息，候选行只显示真实节点标签、推荐度和必要风险提示；
- Overlay 永远只绘制 `primary_path_node_ids`，历史 backup 字段必须为空且不得绘制；
- Drawer 收起/展开后模式和当前建议不丢失。

### 策略

- `balanced`：按当前局面平衡生存和成长；
- `survival`：提高低 HP、危险 Boss/精英前的休整与低风险路线价值；
- `growth`：在生存底线内提高精英、商店和高收益节点价值；
- 模式是软权重，不能用“最多精英/最多篝火”硬约束代替；
- 相同输入和模式必须确定性输出；切换模式应只改变可解释因子和排序，不伪造敌人身份；
- Card Reward 的路线适配因子读取当前 mode 与真实剩余路线压力。

## PB-3：五角色共享机制层

### 架构

角色 adapter 只能把结构化事实映射为共享信号，不得各自持有一套最终分数：

```text
MechanicSignal(domain, role, magnitude, source_code, confidence)
role ∈ provider / payoff / spender / capacity / multiplier
```

共享评分器统一处理“存在提供者才奖励收益方、资源不足时惩罚过量消费者、缺少动态状态则
输出 data gap”。优先使用 `vars`、`keywords_key`、`spawns_cards`、`star_cost`、
`is_x_star_cost`、`hit_count` 和静态 tags；中文/英文描述匹配只能补缺，不能覆盖结构化事实。

### 首轮角色机制

- Ironclad：自身失血 provider/payoff、耗竭 provider/payoff、力量与多段攻击；
- Silent：弃牌/Sly、Shiv 生成与收益、毒 provider/payoff；
- Defect：channel/evoke、Focus、球槽容量，并区分至少 Lightning/Frost/Dark/Plasma；
- Regent：Stars provider/spender、X 星、Forge/Sovereign Blade；
- Necrobinder：Summon/Osty、Doom provider/payoff、Soul 生成与收益。

任何实时动态角色状态未通过 API 验证时不得猜测；例如奥斯提当前生命缺失时只能降级并
报告 data gap。

### 必须修复的已知反例

- `RUPTURE` 不得被标为自身扣血成本；
- `BLADE_DANCE` 生成 Shiv 不得被当成抽 3 张牌；
- `TACTICIAN` 必须能识别弃牌/Sly 发动关系；
- `ZAP`、`DUALCAST`、`DEFRAGMENT` 必须进入同一充能球机制；
- `COMET` 的星耗费不能因普通能量费用为 0 而被忽略；
- `VENERATE` 必须识别 Stars provider；
- `BODYGUARD`、`UNLEASH` 必须识别 Osty 机制，缺动态状态时明确降级；
- `SHROUD` 不得只靠泛化“每当”得到无来源 scaling 奖励。

## 自动验收

1. schema/Pydantic/C# 对 v8 preference 的字段出现性、枚举和严格类型一致；
2. v1-v7 离线夹具继续回放，live 只接受 manifest 精确 v8；
3. balanced/survival/growth 在固定地图状态上有确定、可解释且不违反生存底线的差异；
4. 模式点击产生同 decision ID 的 UPDATED；保存继续恢复；新 Run/结束重置；
5. 模式错配、旧 fingerprint、损坏 checkpoint 和非法枚举失败关闭；
6. 每个角色至少 4 个机制场景，并对 provider 有/无成对验证；
7. 将通用攻防、HP、遗物、路线与跳过场景参数化覆盖五个角色；
8. 所有 recommendation 数值有限、范围合法，未知机制进入 data gaps；
9. 64 节点路线和 event→advice 100 次 P95 继续 ≤ 300 ms；
10. 完整 Python 测试、Mod/PCK 编译、EXE 构建/初始化冒烟、JSON/PowerShell/Git 边界检查
    全部通过。

## 最终组合真机门禁

自动层通过后才安装。真机一次性覆盖：

- `0.109.1` RouteLiveProbe 的真实选路、预览、选点、保存继续、滚动/Zoom/窗口变换；
- 正式 Card/Route 的 v8 handshake、单路线 Overlay 和三个模式即时重算；
- 同局保存继续保留模式，新局重置；
- 五角色各至少一局 Act 1，核对角色、牌组、初始遗物和至少三次普通选牌；
- 未支持场景不显示旧建议，放弃/结束清理正确；
- 游戏和 Host 退出后无残留后台进程。

完成真机前，兼容清单保持 `pending_validation`，TASK-007/PB-2/PB-3 均不得标记完成。

## 实施顺序

1. schema v8、manifest/fingerprint、checkpoint 与严格测试；
2. PB-2 route mode 领域状态、策略、Mod 按钮、同局恢复和 Card route_fit；
3. PB-3 MechanicSignal、五角色薄 adapter、共享评分与固定场景；
4. 完整自动回归、Mod/PCK、EXE、上传边界；
5. 独立审查与返修；
6. 最终组合真机。

## Codex 实施报告（2026-07-31）

### 已实现

- 生产协议升级为 v8，`guide_preferences.route_mode` 在 schema、示例、Pydantic、C#、
  checkpoint、advice 和打包兼容清单中保持严格一致；v1-v7 仅保留离线回放；
- Guide `0.2.0-beta.0`、Mod `0.2.0`、策略 bundle `public-beta-policy-v2` 已统一，release
  fingerprint 为
  `0a101865d40b5b540fbd54746439f86c4cdb3ba5c8b87b5887039d9327c1c560`；
- Context Drawer 已提供智能均衡、稳健生存、激进成长三个互斥按钮；切换同一真实路线
  decision 时生成新 event、递增成功提交 sequence、产生 UPDATED，并立即使旧 advice
  失效；
- 路线策略按模式调整可解释软权重，低 HP 生存底线不能被 growth 覆盖；玩家位置或局面
  变化后从实际状态重新搜索，公开建议与 Overlay 只接受一条 primary route；
- `active-run.json` 是路线偏好的唯一恢复源。Mod 只接受同 Run、同 release、checkpoint v3
  和合法枚举；缺失、损坏、旧 Run、旧 release 或非法值均回到 balanced，事件镜像与 spool
  不参与恢复；
- Python 对损坏、非对象或不支持版本的 checkpoint 记录脱敏 warning，删除唯一恢复文件
  与临时文件并从空状态继续，不保留错误历史；
- 新增统一 `MechanicSignal`、共享 provider/payoff 评分器和五角色薄 adapter；effect tag
  升级至 v4，任务列出的已知误判均进入自动反例；
- Card Reward 正式 Recommendation 保留机制 signals、data gaps 与多维贡献；机制 assessment
  每个候选只计算一次并由评分与展示复用；
- 未增加 Neow、商店、篝火、Boss 遗物、战斗逐回合、RAG/LLM、托盘或安装器。

### 自动验证

- 完整 Python：`293/293 OK`；
- 固定选牌场景：`26/26`，`93/93` 断言；
- 100 次本地基准：P50 `27.15 ms`、P95 `36.08 ms`、最大 `41.88 ms`；
- Mod/PCK：Godot 4.5.1，`0 warning / 0 error`；
- 后台 EXE：27,591,867 bytes，SHA-256
  `3AEE02B8F3E8D1AB35B980876892C3249F5718A1AEA2050381A8DAF062D83C7D`，完整
  `--startup-check` 通过，无残留锁；
- tracked JSON `16/16` 可解析，PowerShell AST `2/2`，`git diff --check`、上传边界与
  tracked secret signature 扫描通过；
- 自动验证过程中没有安装 Mod，没有启动游戏或正式 Host。

### 独立终审

独立只读终审首次发现一个 P0：Mod 会从待处理事件恢复路线模式，违反唯一 checkpoint
契约；同时指出损坏 checkpoint 的 Host 降级和机制 assessment 重复计算问题。三项均已
返修。最终终审定向 `109/109`、Mod/PCK `0 warning / 0 error`，无 P0/P1/P2 自动层阻断。

### 剩余门禁

兼容清单必须继续保持 `pending_validation`。下一步严格执行本任务“最终组合真机门禁”：
先验证并卸载 `0.109.1` RouteLiveProbe，再启用清单、重新构建、安装正式 artifacts，最后
覆盖 Card/Route、三种模式、同局恢复、新局重置、五角色普通选牌和退出清理。真机完成前
TASK-007、PB-2、PB-3 与 Public Beta 均不得标记完成。

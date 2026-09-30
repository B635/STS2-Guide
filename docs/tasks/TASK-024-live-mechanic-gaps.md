# TASK-024：真机机制遗漏与条件遗物估值

- 状态：待真机验收
- 日期：2026-09-25
- 依据：产品规格2.6；用户授权继续修复、测试、深化功能。
- 前序：TASK-023完整构建及0.3.10安装完成；战士选牌/路线/保存继续通过本轮片段，
  完美打击与鲜血药水本身未在自然流程出现，不能写成机制专项真机完成。
- 基线：`build/task024-baseline/`，保留既有用户修改，不提交/推送。

## 已复现问题与顺序

1. 安装0.3.10、effect_tags v7，坚毅仅有block/card_type，牌组已有黑暗之拥却既无
   耗竭提供者协同也无机制缺口。发行库、运行库、原始目录重新派生结果一致。
2. 无情猛攻仅damage/card_type，未计下一张攻击免费却没有披露。不得修成任意高费牌
   cost_reduction；先披露未建模，当前攻击评分继续可用。
3. 低HP、低格挡时结实绷带对全Strike与全DaggerThrow牌组评分相同，主动弃牌条件被
   忽略。只修该已验证遗物；其他遗物不凭名称猜条件。

## 分工与文件边界

- Card Agent：`storage/effect_tags.py`、`advisor/card_reward.py`、必要的
  `advisor/character_mechanics.py`及独立卡牌回归文件；先红测试，标签版本v8。
  不修改商店、发行版本或文档。原始快照不改；必要事实必须能进入发行SQLite。
- Merchant Agent：`advisor/merchant.py`及新独立商店回归文件；先以当前程序集事实
  确认ToughBandages条件，用已有结构化弃牌提供者，缺失/未知不伪造触发。不改标签文件。
- root：规格/任务、版本/集成、全量测试、Mod/发行构建、真机与最终状态。
- 独立Reviewer：非实现者按固定基线检查两切片、发行库和解释传播；发现问题先返修。

## 验收

- 卡牌公开入口及发行SQLite：坚毅基础/升级、有/无黑暗之拥，不标self_exhaust，
  配对因子只出现一次；空手/随机或选择的目标价值不伪装完整计算，披露只读估算边界。
- 无情猛攻基础/升级都有稳定gap与零贡献说明；高费技能/能力不因该机制获得奖励；
  商店卡牌适配保留说明与gap。CardReward旧有效基础评分不变；Merchant沿用现有
  未知收益的保守语义，可为score=None/uncertain，不借本票修改全局gap语义。
- 结实绷带：真实主动弃牌、无来源、未知/不完整牌组、偏好一次、未知遗物保持降级；
  不承诺触发频率，不新增权重配置，保留成本和资格的既有安全规则。
- 完整Python测试、Mod/PCK编译和候选构建；公开能力仍限card_reward/route_choice。
- 真机仅正常Steam窗口，不用MCP/headless、放弃该局或改存档。安装前须确认游戏退出；
  未退出时只产候选，不替换加载中的Mod。涉及额外确认框遵循工具确认要求。

## 实施与审查报告

### 卡牌实现与集成

- `storage/effect_tags.py`使用已核对TRUE_GRIT的OnPlay.CardCmd.Exhaust来源新增唯一
  耗竭provider，版本v8；不把ExtraHoverTips当自身关键词。`advisor/card_reward.py`
  增加坚毅estimate_scope、无情猛攻next_attack_free gap和unmodeled_mechanic。
  共用配对和偏好，不添加新的伤害/降费公式或宽泛文本匹配。
- `tests/test_live_card_mechanics.py`9项发行SQLite公共回归通过；三条红绿闭环为坚毅
  provider 0→1、基础/升级scope 0→1、无情猛攻缺口从无到有。还覆盖无收益方、
  反向牌组provider、商店传播、非攻击高费牌不误加分，以及同目录v7标签自动重建。
- 无情猛攻基准回归分数48.4/52.4（基础/升级）不变；商店沿用None/uncertain边界。
- 既有coverage两个子例原先断言其他候选没有任何机制警告，因新增无情猛攻披露而失败。
  root修为明确要求无情猛攻自己的gap，仍禁止完美打击缺口串到其他候选；7项/1.590秒OK。
- root同步Guide0.3.11/Mod0.3.11、bundle6/Card5/Merchant6与C#/协议示例，旧v5标签重建
  回归保留并以v8为期望。游戏仍运行0.3.10，新版尚未安装。

### 商店实现

- `advisor/merchant.py`只新增精确TOUGH_BANDAGES条件分支：使用现有主动discard标签，
  无来源不获原3+紧迫度+格挡缺口生存适配；有来源保留原有界贡献。两者均有零贡献
  estimate_scope，不预测频率。空/未知/带读取警告/未建模实例的牌组明确gap。
- `tests/test_merchant_conditional_relic.py`新9项公共发行SQLite回归。首条先红66!=58，
  修复后无来源58、有来源66；跨色SCRAPE、SLY收益方/否定/弃牌堆名词、偏好单次、
  真实价格与不可用候选均覆盖。root复跑discover商店39项/1.460秒/OK。
- API：本机0.111.0 ToughBandages.CanonicalVars=3；AfterCardDiscarded先确认自己的牌、
  且轮到己方才GainBlock。精确成员与IL tokens已登记api-reference-sources，未添加读取API。
- 兼容/包装/标签/关系库整合回归102项/6.912秒/OK；当时完整构建与独立终审未完成，
  最终结果见下方，安装仍待完成。
  root曾用点号模块命令跑药水测试，因既有test_merchant_decision的顶层导入路径报错；
  换回项目规定discover入口后全部通过，没有改测试或隐藏失败。

### 独立审查

固定基线build/task024-baseline；非实现Agent按Standards和Spec双轴复核，均无未解决
发现。独立Python12模块177项/14.062秒/OK，另商店39项/1.459秒OK。核对实际发行SQLite
与公共入口、非自身耗竭/不双计、未建模边界、未知与无来源、价格和偏好单次，以及发行
身份。早期集成提醒C#仍有旧指纹，root已同步并由审查者核实；不是留存问题。审查者未
构建或操作游戏，完整最终构建由root单独记录。

### 最终自动交付

- `scripts/build_public_beta.ps1 -Python D:\miniconda3\envs\sts2\python.exe` exit0。
  完整Python **624项 / 60.581秒 / OK**；Mod/PCK **0警告 / 0错误**，发行SQLite、
  PyInstaller冻结包、隔离EXE自检、Inno安装包和payload审计全部通过。
- 日志：`build/task024-public-beta-final.log`；候选：
  `dist/installer/STS2-Guide-0.3.11-alpha.0-win-x64-setup.exe`；版本Guide0.3.11-alpha.0 /
  Mod0.3.11、标签8、bundle6/Card5/Merchant6。指纹
  `31439e8d119ce56bc0ec6ac52eac2ac51c527d697694ef08116e462c15b254c8`。
- 源码冻结后独立审查无未解决发现，`git -c core.safecrlf=false diff --check`通过。
  本次只完成源码与自动交付；不把0.3.10真机片段算成0.3.11新增规则已验收。

### 安装与真机待完成

0.3.10本轮测试后正常保存回主菜单，事件72 decision_closed、advice不存在、当前局
检查点保留。之后游戏和Guide均已退出。首次安装调用exit1且没有日志，核对回执仍为旧版；
补齐日志目录后重试，0.3.11静默安装exit0，回执版本/指纹与manifest
一致；安装日志`build/task024-live/install-0.3.11.log`；`scripts/install_p0_mod.ps1
-VerifyOnly`确认DLL/JSON/PCK均与生成物相同。已对比安装EXE和三件套SHA256与发行清单。
安装过程没有启动游戏/Guide，尚无0.3.11真机证据；没有重置额度、提交/推送。

### 后续已知深度缺口（未列入本票实现）

- 移除Strike的有限价值暂未反馈已有PerfectedStrike的战前基准下降；应共用既有Strike
  标签事实，不能把“加一张候选”的计数公式直接挪用为移除牌组。
- 无情猛攻的类型化下一攻击免费、坚毅随机/选择目标价值、结实绷带触发频率仍未估计；
  本票有限修复和披露不宣称战斗模拟或完整最优决策。

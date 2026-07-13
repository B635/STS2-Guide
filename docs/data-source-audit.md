# 外部数据源审计

审计日期：2026-07-13。

| 来源/接口 | 可用性 | 当前处理 | 目标存储 |
|---|---|---|---|
| Spire Codex `/api/cards`、`relics`、`potions`、`monsters`、`characters` | 可用；社区 API 条款允许限流内使用 | 已导入 | SQLite |
| Spire Codex `/api/events`、`encounters`、`acts`、`powers`、`intents`、机制端点 | 可用；中文导出已在临时目录完成结构抽样，尚未进入生产 schema | 先设计规范化表、引用校验和版本快照，再事务导入 | SQLite；原始 JSON 仅作可重建快照 |
| Spire Codex `/api/guides` | 可用，但攻略作者权利与再分发许可需要逐篇复核 | 只允许本机生成原文快照；`data/guides.json` Git 忽略，不公开提交或打包 | 本机原文快照 + FAISS |
| Spire Codex `/api/runs/scores/{cards,relics,potions}` | 可用；混合版本、观察性聚合 | 已过滤 Mod ID、保存来源快照；只作低权重先验 | SQLite |
| Spire Codex `/api/runs/versions` | 可用 | 已写入社区统计快照，用于暴露混合版本风险 | JSON 快照 |
| Spire Codex `/api/runs/shared/{hash}` | 单条公开对局可用；禁止枚举 run hash | 不批量抓取。详情有最终牌组、获得楼层和结果，但没有完整奖励候选集 | 暂不导入 |
| Spire Codex `/api/runs/encounter-stats` | 可用于敌人危险度和路线建议 | P1 接入 | SQLite |
| Spire Codex `/api/merchant/config` | 可用于商店价格和购买约束 | P1 接入 | SQLite |
| Spire Codex changelog/history | 可用于补丁版本隔离 | P1 接入 | SQLite |
| Mobalytics card tier list | P0 仅在开发者本机人工整理最小 tier 数据 | 可选本地 JSON；不自动抓取、不提交、不打包，缺失时降级 | Git 忽略的本地文件 |
| slaythespire-2.com tier list、构筑、攻略 | 条款禁止未经许可的自动抓取和复制/修改原创内容 | 不抓取、不落库、不用于评分；仅人工外链参考 | 无 |
| BoberInSpire 公开源码 | 可作为接口和可行性参考；本项目要求差异化 | 仅核对产品链路、Mod 清单与构建方式；当前 v0.107.1 接口由本地程序集探测后独立实现，不复制源码、规则表或数据文件 | 无 |

Spire Codex API 条款：
<https://github.com/ptrlrd/spire-codex/blob/main/API_TERMS.md>

Spire Codex 源码许可证（与托管 API 条款分开）：
<https://github.com/ptrlrd/spire-codex/blob/main/LICENSE.md>

slaythespire-2.com 服务条款：
<https://slaythespire-2.com/terms-of-service>

Mobalytics 服务条款：
<https://mobalytics.gg/terms/>

## 统计先验限制

Codex Score 是牌出现在已提交对局时与胜利的相关性聚合，不是随机对照实验。
它可能受到角色、难度、版本、玩家水平、获得楼层和幸存者偏差影响。因此：

1. 不直接把分数当成选牌胜率；
2. 先按样本量缩放，再乘小权重；
3. 内部响应保留来源、样本数和快照用于诊断，P0 游戏面板只显示总推荐分；
4. 有局面冲突时，以牌组结构信号为主；
5. P0 不主动保存候选和结果训练样本；以后只有在明确启用且数据足够时才评估模型。

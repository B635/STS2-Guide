# 选牌评分改用牌组适合度差值

2026-10-01 真机发现遗物协同全部来自描述正则、跳过分偏低，根源是“50 分加几十条独立加减规则”的结构：量纲不统一、阈值断崖、同一事实重复计分、跳过另有公式。我们决定选牌的每个选项（含跳过）都评为“做出该选择后”的牌组适合度：按能力维度计算牌组能力（按抽到概率折算），与由幕、路线、Boss、生命值和偏好决定的局面需求比较，缺口惩罚陡于富余奖励；引擎和触发遗物折算进通用维度。卡牌与遗物机制只以当前游戏程序集核对的事实为来源，每条事实锁定该类自身的代码指纹（游戏更新时只让改动过的条目失效，而不是整个 DLL 哈希一变就全部作废）；全部卡牌覆盖后才替换旧版本。评分不再使用 Spire Codex 社区先验或本机 tier 先验，所有分数只来自当前局面的因果计算。

## Considered Options

- 在旧加法模型里修正遗物标签：只修数据，仍保留量纲和重复计分问题，之后还要重写一次。
- 社区先验作为有上限的独立项保留：会把相关性混入因果评分，且无法用局面解释。
- 分两步切换（先覆盖普通/罕见牌）：演示更早，但同屏会出现两种刻度或降级逻辑。

## Addendum (2026-10-09): lookahead over future card rewards

On 38 confirmed situations v2's overall ranking was slightly better than v1 but its top pick hit less, and the misses were archetype-building picks: v2 valued only the current deck. We decided to score each candidate by the deck's fitness averaged over the act's remaining fights, with the card rewards those fights grant sampled from the character pool at the assembly's `CardRarityOdds` and picked greedily by the same fitness. Only rewards are sampled, never combats; all candidates share one seeded set of scenarios so results are reproducible. Synergy between owned cards is already inside the whole-deck engine, so it is explained (which deck cards make a candidate stronger) but not scored a second time.

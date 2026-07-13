# TASK-005：补齐本机社区 tier JSON（待排期）

- 状态：待排期
- 创建者：Codex
- 实现者：DeepSeek
- 触发条件：TASK-004 真机 P0 阻塞修复并通过审查后
- 产品依据：`docs/product-spec.md` 的 6.2、6.3、7.2

## 目标

补齐当前开发电脑上的可选社区先验文件：

```text
data/local/mobalytics_card_tiers.json
```

该文件只服务当前本机 P0 推荐，不提交、不打包、不公开分发。缺失时系统必须继续安全降级。

## 边界

- 只人工整理 Mobalytics card tier 页面中推荐所需的最小结构化事实；
- 不写自动爬虫，不抓取网页正文，不保存页面 HTML；
- 不复制 BoberInSpire 的数据文件、tier 分数、权重或流派表；
- 不用中文名、英文名或模糊匹配入库，必须映射到本项目稳定 `card_id` 和 `character_id`；
- tier 只作为有限先验，当前牌组、遗物、HP、路线压力和跳过规则优先；
- 文件进入 `.gitignore`，不得进入 Git 暂存、提交、EXE、安装包或测试夹具。

## 建议执行方式

1. 先读取现有实现：
   - `advisor/data_sources.py`
   - `advisor/card_reward.py`
   - `tests/test_local_tier_source.py`
2. 从 SQLite/catalog 导出当前可用 `card_id`、中文名、英文名和角色，作为人工映射辅助；
3. 人工整理 `data/local/mobalytics_card_tiers.json`，最小字段为：

   ```json
   {
     "schema_version": 1,
     "source": "mobalytics-manual-local",
     "source_url": "https://mobalytics.gg/slay-the-spire-2/tier-lists/cards",
     "captured_at": "2026-07-08T00:00:00+08:00",
     "entries": [
       {
         "card_id": "STABLE_CARD_ID",
         "character_id": "STABLE_CHARACTER_ID",
         "tier": "A"
       }
     ]
   }
   ```

4. 增加一个只读校验脚本或命令，至少检查：
   - JSON schema/version；
   - `source_url` 是 HTTPS；
   - `captured_at` 带时区且未过期；
   - `card_id` 和 `character_id` 都能在本项目结构化数据中精确命中；
   - 无重复 `(card_id, character_id)`；
   - tier 只允许 `S/A/B/C/D/F`；
   - 输出覆盖数量和未匹配项，不修改数据库。

## 验收

- `data/local/mobalytics_card_tiers.json` 在本机存在，但 `git status --short` 不显示该文件；
- loader 状态为 `loaded`，覆盖数量可见；
- 文件缺失、损坏、过期时现有测试仍证明安全降级；
- 至少用 3 个固定选牌场景证明：
  - 高 tier 不会在强烈不适配时必选；
  - 低 tier 可以在明确协同时超过高 tier；
  - 缺失 tier 的卡牌不会报错，也不会被伪造先验；
- 不新增联网依赖，不修改 P0 实时启动方式。

## 当前备注

当前代码已经具备可选本机 tier 文件读取、校验、安全降级和有限先验评分。TASK-005 重点是
补齐本机私有数据快照和人工校验流程，不是重写推荐器。

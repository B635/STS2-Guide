# 02: 程序集事实管线（追踪弹）

**What to build:** 开发者运行只读提取工具即可得到全部卡牌与遗物类的钩子、调用的游戏命令与按类代码指纹；事实数据在构建发行库时被加载，指纹不符的条目自动失效，覆盖率报告列出每个条目的状态。遗物按钩子结构自动初分为基线、触发、无关三类。本票只加载事实，不改变任何评分。

**Blocked by:** 无（可立即开始）

**Status:** 完成（自动证据；独立审查返修后复核通过）

- [x] 提取工具只读元数据与 IL，不实例化模型、不执行游戏方法，不进入产品或发行包
- [x] 每个类的代码指纹包含其嵌套状态机与匿名方法；同一程序集重复提取结果稳定
- [x] 事实数据每条带类型、成员、token 来源与指纹；构建发行库时加载，指纹不符的条目按未知处理并在报告中列出
- [x] 覆盖率报告区分已换算、已登记无法得知、未审核三种状态，覆盖全部卡牌与遗物
- [x] 遗物初分结果可供人工审核：战纹涂料（仅拾起时生效）初分为无关遗物，斗篷扣（回合结束按手牌数获得格挡）初分为基线遗物而非触发遗物
- [x] 现有 v1 评分与全部既有测试结果不变

依据：产品规格第 2.8 节、`docs/adr/0001-deck-fitness-scoring.md`、`GLOSSARY.md`。完成后用 `/code-review` 做独立审查，审查结论写在本票末尾。

## 实施报告（2026-10-01）

- 只读提取工具：`tools/Sts2FactExtractor`（`System.Reflection.Metadata`，不执行游戏代码）。输出 `data/assembly/sts2-extract.json`：
  896 个类（596 卡、300 遗物），每类含重写钩子及 token、引用的游戏成员（含泛型实参）、`CardModel::get_Type` 后的 `CardType` 常量、按类代码指纹。
- 稳定性证据：同一程序集连续两次提取，输出文件 SHA256 均为 `8BF28815…A3E96B`，完全一致（extractor v2）。
- 事实文件 `data/assembly/facts.json` 由 `scripts/refresh_assembly_facts.py` 生成：595 卡 + 298 遗物全部有条目与模型类；每条带指纹与钩子 token。
  覆盖状态四种：converted / unknowable（二者有效）/ stale / unreviewed；逐条状态写入 `build/assembly-coverage.json`。当前全部 unreviewed。
- 遗物初分（只看钩子）：基线 105、无关 120、触发 60、需审核 13（均挂卡牌奖励钩子）。战纹涂料→无关（仅 `AfterObtained` 06002376），斗篷扣→基线（仅 `BeforeSideTurnEnd` 06001D4D）。
- 发行库：`build_release_database` 新增程序集提取与事实两个输入，写入 `assembly_facts` 表（effective state、仅有效行有 category、triage 分列），
  元数据 `assembly_extract_sha256` / `assembly_facts_sha256` 纳入发行库身份；事实对应其他提取时构建失败。评分不读取这些事实（v1 不变）。
- **SQLite `SCHEMA_VERSION` 暂保持 8**：提升到 9 会改变发行指纹，而指纹写在 Mod 与协议示例中，需随票 11 统一同步。
  在票 11 完成前不得用当前源码构建可安装候选（新旧库同为 v8 但身份元数据不同；运行库身份不符时会从模板重建）。
- 验证：`tests/test_assembly_facts.py` 8 项（提取对应受支持程序集、目录全覆盖、提交的事实与提取一致、按钩子初分、只有已审核且指纹一致有效、
  改动类只让该条 stale、元数据锁定、错配提取拒绝构建）；完整 Python 632 项 / 52.0 秒 / OK。

## 独立审查（/code-review，两轴并行）

- 规范轴：无硬性违规。返修：`CardPile::get_Type` 被误计为卡牌类型检查 → 只认 `CardModel::get_Type`；枚举按完整名称匹配；
  覆盖统计重复实现 → 共用 `coverage_summary`；`catalog_keys` 重复 → 移入模块；未审核行写入初分 category → 分列；
  删除无读取者的 `assembly_game_sha256` 元数据；C# 输出排序不再用反射；测试计数改由目录推导；构建脚本显式传入新输入。
- 规格轴：返修覆盖报告区分已换算/无法得知、逐条列出状态、指纹不符统一为 stale、补 token 来源与稳定性证据；
  延后 schema 升级被判可接受（条件已写入上文与票 11）。

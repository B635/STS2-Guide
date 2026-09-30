# TASK-014 — 真实实体身份到推荐闭环

- 状态：待真机验收
- 父任务：TASK-013；产品规格第 2.2 节；查漏 F1/F5
- 实现者：Codex；审查：独立 Standards / Spec 审查者
- 阻塞依赖：无

## 交付行为

真实卡牌、药水和其他已支持资源以游戏稳定 ID 进入目录与既有推荐，不再因类名猜测漏算。
已确认测试边界：数据到建议。保留原权重和未知实体安全降级。

## 验收

- [x] 四类已知 ID 反例先失败再通过；覆盖生产转换而不只测试手写规范 ID。
- [x] 对真实程序集与发行目录做范围对照；排除编译器生成类型。
- [x] 导出结果可被发行 SQLite 识别并参与已有策略；未知资源不误匹配。
- [x] 完整 Python 和 Mod 编译通过，独立审查记录存在。

只修改身份读取及对应验证，不启用其他决策，不安装或启动游戏。
文件定位沿用父任务。代码审查通过仍不等于真机验收。

## 实施与审查报告

托管行为测试先失败：SpeedPotion 期望 SPEED_POTION、实得 SPEED。生产 helper 改用当前
公开 ModelDb.GetEntry(Type)，保持调用签名；探针仅统计真实顶层模型。2 项跨语言/发行库
测试通过（1.795 秒），四个具名反例由链接的 C# 生产代码验证。发行目录所需所有 cards /
potions / relics / characters ID 可匹配；测试确认 potion_buffer_once 为正贡献。
全量 Python 453 项 / 47.088 秒 / OK；Mod/PCK 0 warning / 0 error。没有安装/真机。

### Standards

首次发现 P2：发行脚本显式 GamePath 可被测试 local.props 路径绕过。返修后由发行脚本
传入实际 STS2_TEST_GAME_PATH、严格 STS2_REQUIRE_GAME_TESTS，缺环境必须失败，编译与
运行选同一程序集，并恢复环境变量。审查者复核关闭，最终无未解决问题。

### Spec

独立审查无阻断。审查者只读核对当前 GetEntry IL：Type.Name → StringHelper.Slugify，
无模型注册/游戏状态修改。共用的 RestSiteOption 只影响未知效果来源标签，不开启能力。
两轴未发现剩余阻断；运行时快照与面板对应性仍待 TASK-018。

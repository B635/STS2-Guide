# STS2MCP 真机观察流程

> 创建时间：2026-07-08
>
> 目的：用 STS2MCP 作为开发期真机 API 探针，观察 STS2 当前版本各关键界面的
> 真实 `get_game_state` 返回值，定位公开 API，转化为本项目 Mod 的只读实现。
>
> **STS2MCP 不作为产品依赖。本文件仅用于开发期诊断。**

## 边界

1. STS2MCP 只用于观察 `get_game_state` 和定位公开 API。
2. **不** 使用 STS2MCP 的自动点击、代打、动作执行功能。
3. **不** 复制 STS2MCP 代码。
4. **不** 把 STS2MCP DLL、JSON、MCP server、uv 配置提交进正式产品。
5. STS2 Guide 运行时 **不依赖** STS2MCP 存在。
6. 观察到的接口必须再用本项目 Mod 编译和真机日志验证。

## 安装

### 1. 获取 STS2MCP Mod

从 [GitHub Releases](https://github.com/Gennadiyev/STS2MCP/releases) 下载最新版本。
需要两个文件：

- `STS2_MCP.dll`
- `STS2_MCP.json`

放入游戏的 `mods/` 目录（例如 `BepInEx/plugins/STS2_MCP/`）。

### 2. 确认游戏内 Mod 启用

启动 STS2，进入主菜单 → Mods，确认 `STS2_MCP` 已启用。

### 3. 配置 MCP Server（可选，仅用于 Claude 直接调用）

项目根目录 `.mcp.json`（**不提交到 Git**）：

```json
{
  "mcpServers": {
    "sts2mcp": {
      "command": "uvx",
      "args": ["sts2mcp"],
      "env": {
        "STS2MCP_HOST": "localhost",
        "STS2MCP_PORT": "15526"
      }
    }
  }
}
```

或者直接通过 HTTP 调用 `localhost:15526` 的 REST API（不依赖 MCP server）。

### 4. 确认 HTTP 可达

启动游戏后，在终端验证：

```powershell
curl http://localhost:15526/api/v1/state
```

## 观察步骤

按以下顺序进入游戏各界面，在每个界面调用 `get_game_state` 并记录返回值。

### Step 1: 主菜单

- 游戏刚启动，停留在主菜单
- 调用 `get_game_state`
- 记录 `state_type` 和顶层 key

### Step 2: 角色选择

- 点击 Singleplayer → 选择角色界面
- 调用 `get_game_state`
- 记录角色列表结构

### Step 3: Neow 祝福

- 开始新游戏，进入 Neow 祝福选牌界面
- 调用 `get_game_state`
- 记录：
  - `state_type`
  - 候选卡牌结构
  - 是否有 `player` 数据
  - `run` 数据（act、floor、ascension）
  - 与普通 CardReward 的区别

### Step 4: 首次地图

- Neow 结束后，进入地图界面
- 调用 `get_game_state`
- 记录：
  - `map` 数据结构（nodes、edges、current_node、boss）
  - `player` 数据（character、deck、relics、potions、gold、hp）
  - `run` 数据（act、floor）

### Step 5: 第一场普通战斗

- 进入战斗
- 调用 `get_game_state`
- 记录：
  - `state_type`
  - `combat` 数据结构（monsters、intent、player_hand 等）

### Step 6: 普通奖励选牌

- 战斗结束后，奖励选牌界面
- 调用 `get_game_state`
- 记录：
  - `state_type`
  - 候选卡牌结构（每张牌的 ID、升级、附魔/诅咒）
  - 与 Neow 祝福 `state_type` 是否相同
  - 如何区分 Neow vs 普通奖励

### Step 7: 商店 / 事件 / 休息站

- 进入商店
- 调用 `get_game_state`
- 进入随机事件
- 调用 `get_game_state`
- 进入休息站
- 调用 `get_game_state`

### Step 8: Boss 地图

- 进入 Act 1 Boss 层地图
- 调用 `get_game_state`
- 记录 boss 节点信息

## 记录模板

每步观察必须按以下模板记录：

```markdown
## 界面：[界面名称]
- 时间：[ISO 时间]
- Act / Floor：[当前 Act 和楼层]

### get_game_state 原始返回（关键字段）

\`\`\`json
{
  "state_type": "...",
  "run": { ... },
  "player": { ... },
  ...
}
\`\`\`

### 关键 API 定位

| 数据项 | 访问路径 | API 类型 | 是否公开 |
|--------|---------|----------|----------|
| RunState | `RunManager.Instance.DebugOnlyGetState()` | 方法 | 公开 |
| Player（P0 单人） | `runState.Players[0]` | `IReadOnlyList<Player>` 属性 | 公开 |
| Player（未来多人调研） | `MegaCrit.Sts2.Core.Context.LocalContext.GetMe(runState)` | 静态方法 | 公开；P0 不使用 |
| Map nodes | ... | ... | ... |
| Card rewards | ... | ... | ... |
| Monster list | ... | ... | ... |

### 与我们 Mod 的差异

- 我们的 Mod 在此界面是否已有对应数据？
- 如没有，缺失的读取点是什么？
```

## 优先验证清单

按 TASK-004 优先级排序：

1. [ ] `RunManager.Instance.DebugOnlyGetState()` 在所有界面是否可用
2. [x] P0 单人模式从公开 `runState.Players` 取得唯一 Player；2026-07-13 Neow 后首次地图真机通过
3. [ ] 地图界面 `state_type` 和 `map` 数据结构
4. [ ] 当前房间类型判断方式
5. [ ] Neow 选牌 vs 普通奖励选牌的 `state_type` 和 candidate 结构区别
6. [ ] `runState` 中是否能稳定拿到：
   - [ ] character（角色类型）
   - [ ] deck（牌组）
   - [ ] relics（遗物列表）
   - [ ] potions（药水）
   - [ ] gold（金币）
   - [ ] hp / maxHp
   - [ ] boss 节点
   - [ ] map nodes / edges
   - [ ] current node

## 观察结论归档

观察完成后，结论必须写入 [`docs/api-reference-sources.md`](../api-reference-sources.md) 的
"已验证 API" 部分，并用于改进本项目的 Mod 实现。每次修改 Mod 后必须通过真机日志
二次验证（不依赖 STS2MCP 的返回值）。

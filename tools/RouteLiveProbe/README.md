# RouteLiveProbe（仅开发期）

这是 TASK-007 Phase A 的只读真机探针，与正式
`STS2Guide.ReadOnlyExporter` 的 csproj、程序集、manifest 和 PCK 完全隔离。它只注册
Harmony Postfix 并读取公开游戏/Godot 状态；不会调用选点、旅行、绘制或存档修改方法。

## 记录内容

探针将每次会话写入：

```text
%LOCALAPPDATA%\STS2Guide\dev\route-probe\route-probe-*.jsonl
```

每条记录包含地图打开/关闭、旅行状态、真实选择回调、当前/已访问坐标、模型下一节点、
视觉 `NMapPoint` 状态，以及节点中心、screen/net 往返坐标和 Godot transform。日志不记录
seed、玩家名、存档正文、绝对游戏路径或牌组内容；写入失败只禁用探针日志，不影响游戏。

## 构建、安装与卸载

普通构建不会安装，也不会修改正式 PCK：

```powershell
& .\tools\RouteLiveProbe\build_probe.ps1
```

真机前关闭游戏，再显式安装开发探针：

```powershell
& .\tools\RouteLiveProbe\install_probe.ps1
```

测试结束后关闭游戏，只删除三个精确命名的探针文件：

```powershell
& .\tools\RouteLiveProbe\uninstall_probe.ps1
```

## TASK-007 六项门禁真机步骤

1. 新开单人局，在 Act 初始强制选路界面停留 2 秒；滚动地图、尝试当前版本可用的 Zoom、
   改变窗口大小，再手动点击一个真实节点。
2. 进入房间后从顶栏仅预览地图，滚动后关闭，不选择节点；用
   `is_opened_from_top_bar`、`IsTravelEnabled`、`IsTraveling` 和视觉 Travelable 集合区分预览。
3. 完成一个节点，进入下一次真实选路并手动点击，核对 origin、候选和回调前最近采样/
   回调后 `CurrentMapPoint`。
4. 在一次尚未选择的路线机会保存退出，重新启动并继续，核对新会话的 map fingerprint、
   origin、候选集合与 `saved_singleplayer_setup_postfix`。
5. 对地图做滚动、当前版本可用的 Zoom、窗口化/全屏和分辨率变化，确认相同 `node_id`
   的 Godot transform、中心坐标及 screen/net 往返值同步变化；若没有可操作 Zoom，记录
   为 N/A，不能沿用旧版本结论。
6. 若游戏规则允许返回已访问 origin，单独记录；在没有真实证据前不得据此假定同 Act
   严格前向。

关闭游戏后验证一个或多个 JSONL：

```powershell
& "D:\miniconda3\envs\sts2\python.exe" `
  .\tools\RouteLiveProbe\validate_probe_log.py `
  "$env:LOCALAPPDATA\STS2Guide\dev\route-probe\route-probe-*.jsonl"
```

验证器只检查 JSONL 结构、每会话连续 sequence 并汇总门禁证据；它不会把“出现日志”自动
判定为六项事实已经通过。结论仍须由独立审查根据真实时序逐项填写。

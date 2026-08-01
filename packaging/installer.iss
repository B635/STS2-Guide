#ifndef AppVersion
  #define AppVersion "0.3.0-alpha.0"
#endif
#ifndef ExpectedGameVersion
  #define ExpectedGameVersion "0.110.1"
#endif
#ifndef ExpectedGameAssemblySHA256
  #error ExpectedGameAssemblySHA256 must be supplied by build_public_beta.ps1
#endif
#ifndef ExpectedReleaseInfoSHA256
  #error ExpectedReleaseInfoSHA256 must be supplied by build_public_beta.ps1
#endif
#ifndef GuideExeSHA256
  #error GuideExeSHA256 must be supplied by build_public_beta.ps1
#endif
#ifndef ReleaseFingerprint
  #error ReleaseFingerprint must be supplied by build_public_beta.ps1
#endif
#ifndef ModDllSHA256
  #error ModDllSHA256 must be supplied by build_public_beta.ps1
#endif
#ifndef ModJsonSHA256
  #error ModJsonSHA256 must be supplied by build_public_beta.ps1
#endif
#ifndef ModPckSHA256
  #error ModPckSHA256 must be supplied by build_public_beta.ps1
#endif

#define AppName "STS2 Guide"
#define AppPublisher "STS2 Guide contributors"
#define AppExeName "STS2 Guide.exe"

[Setup]
AppId={{CB8DD1CA-65C5-470A-A211-3BE619EC56A8}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={localappdata}\Programs\STS2 Guide
DefaultGroupName=STS2 Guide
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist\installer
OutputBaseFilename=STS2-Guide-{#AppVersion}-win-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile=..\LICENSE
SetupIconFile=..\build\release\sts2-guide.ico
UninstallDisplayIcon={app}\{#AppExeName}
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Files]
Source: "..\dist\STS2 Guide.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "DATA_SOURCES.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\mod\STS2Guide.ReadOnlyExporter\artifacts\mod\STS2GuideReadOnlyExporter.dll"; DestDir: "{code:GetGameModsDir}"; Flags: ignoreversion uninsrestartdelete
Source: "..\mod\STS2Guide.ReadOnlyExporter\artifacts\mod\STS2GuideReadOnlyExporter.json"; DestDir: "{code:GetGameModsDir}"; Flags: ignoreversion uninsrestartdelete
Source: "..\mod\STS2Guide.ReadOnlyExporter\artifacts\mod\STS2GuideReadOnlyExporter.pck"; DestDir: "{code:GetGameModsDir}"; Flags: ignoreversion uninsrestartdelete

[Icons]
Name: "{group}\STS2 Guide"; Filename: "{app}\{#AppExeName}"
Name: "{userdesktop}\STS2 Guide"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："; Flags: unchecked

[Registry]
Root: HKCU; Subkey: "Software\STS2Guide"; ValueType: string; ValueName: "GameDir"; ValueData: "{code:GetGameDir}"; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#AppExeName}"; Description: "启动 STS2 Guide"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: files; Name: "{userappdata}\SlayTheSpire2\STS2Guide\install.json"

[Code]
var
  GameDirPage: TInputDirWizardPage;

function NormalizePath(Value: String): String;
begin
  Result := RemoveBackslashUnlessRoot(ExpandFileName(Value));
end;

function JsonEscape(Value: String): String;
begin
  StringChangeEx(Value, '\', '\\', True);
  StringChangeEx(Value, '"', '\"', True);
  Result := Value;
end;

function DefaultGameDir(): String;
var
  SteamPath: String;
  Candidate: String;
begin
  Result := '';
  if RegQueryStringValue(HKCU, 'Software\Valve\Steam', 'SteamPath', SteamPath) then
  begin
    Candidate := AddBackslash(SteamPath) + 'steamapps\common\Slay the Spire 2';
    if FileExists(AddBackslash(Candidate) + 'SlayTheSpire2.exe') then
      Result := Candidate;
  end;
  if (Result = '') and RegQueryStringValue(HKLM32, 'Software\Valve\Steam', 'InstallPath', SteamPath) then
  begin
    Candidate := AddBackslash(SteamPath) + 'steamapps\common\Slay the Spire 2';
    if FileExists(AddBackslash(Candidate) + 'SlayTheSpire2.exe') then
      Result := Candidate;
  end;
  if Result = '' then
    Result := ExpandConstant('{pf32}\Steam\steamapps\common\Slay the Spire 2');
end;

function ExactProcessIsRunning(ExeName: String; ExpectedPath: String;
  var InspectionSucceeded: Boolean): Boolean;
var
  Locator, Services, Processes, Process: Variant;
  I: Integer;
  ExecutablePath: String;
begin
  Result := False;
  InspectionSucceeded := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Services := Locator.ConnectServer('', 'root\CIMV2');
    Processes := Services.ExecQuery(
      'SELECT Name, ExecutablePath FROM Win32_Process WHERE Name="' + ExeName + '"');
    InspectionSucceeded := True;
    for I := 0 to Processes.Count - 1 do
    begin
      Process := Processes.ItemIndex(I);
      ExecutablePath := Process.ExecutablePath;
      if CompareText(NormalizePath(ExecutablePath),
        NormalizePath(ExpectedPath)) = 0 then
      begin
        Result := True;
        Exit;
      end;
    end;
  except
    InspectionSucceeded := False;
  end;
end;

function StopExistingGuide(): Boolean;
var
  GuidePath: String;
  InspectionSucceeded: Boolean;
  ResultCode: Integer;
  I: Integer;
begin
  Result := False;
  GuidePath := ExpandConstant('{app}\{#AppExeName}');
  if not ExactProcessIsRunning('{#AppExeName}', GuidePath,
    InspectionSucceeded) then
  begin
    Result := InspectionSucceeded;
    Exit;
  end;
  if not Exec(GuidePath, '--shutdown-existing', '', SW_HIDE,
    ewWaitUntilTerminated, ResultCode) then
    Exit;
  for I := 1 to 100 do
  begin
    Sleep(100);
    if not ExactProcessIsRunning('{#AppExeName}', GuidePath,
      InspectionSucceeded) then
    begin
      Result := InspectionSucceeded;
      Exit;
    end;
  end;
end;

function ValidateGameDir(GameDir: String; ShowError: Boolean): Boolean;
var
  ExePath: String;
  AssemblyPath: String;
  ReleaseInfoPath: String;
  ActualHash: String;
  InspectionSucceeded: Boolean;
begin
  Result := False;
  GameDir := NormalizePath(GameDir);
  ExePath := AddBackslash(GameDir) + 'SlayTheSpire2.exe';
  AssemblyPath := AddBackslash(GameDir) + 'data_sts2_windows_x86_64\sts2.dll';
  ReleaseInfoPath := AddBackslash(GameDir) + 'release_info.json';
  if (not FileExists(ExePath)) or (not FileExists(AssemblyPath)) or
    (not FileExists(ReleaseInfoPath)) then
  begin
    if ShowError then
      MsgBox('所选目录不是有效的 Slay the Spire 2 安装目录。', mbError, MB_OK);
    Exit;
  end;
  ActualHash := Lowercase(GetSHA256OfFile(AssemblyPath));
  if ActualHash <> Lowercase('{#ExpectedGameAssemblySHA256}') then
  begin
    if ShowError then
      MsgBox('当前游戏程序集不在此 Guide 版本的精确兼容清单中。支持版本：{#ExpectedGameVersion}。', mbError, MB_OK);
    Exit;
  end;
  ActualHash := Lowercase(GetSHA256OfFile(ReleaseInfoPath));
  if ActualHash <> Lowercase('{#ExpectedReleaseInfoSHA256}') then
  begin
    if ShowError then
      MsgBox('release_info.json 与此 Guide 的精确兼容清单不一致。支持版本：{#ExpectedGameVersion}。', mbError, MB_OK);
    Exit;
  end;
  if ExactProcessIsRunning('SlayTheSpire2.exe', ExePath,
    InspectionSucceeded) then
  begin
    if ShowError then
      MsgBox('请先关闭 Slay the Spire 2，再安装或升级 STS2 Guide。', mbError, MB_OK);
    Exit;
  end;
  if not InspectionSucceeded then
  begin
    if ShowError then
      MsgBox('无法核验游戏进程状态；为避免安装时覆盖正在使用的文件，安装已停止。', mbError, MB_OK);
    Exit;
  end;
  Result := True;
end;

procedure InitializeWizard();
var
  RequestedGameDir: String;
begin
  GameDirPage := CreateInputDirPage(
    wpSelectDir,
    '选择 Slay the Spire 2 目录',
    'STS2 Guide 只会安装自己的只读 Mod 三件套。',
    '请选择包含 SlayTheSpire2.exe 的游戏目录，然后点击“下一步”。',
    False,
    '');
  GameDirPage.Add('游戏目录：');
  RequestedGameDir := ExpandConstant('{param:GAMEPATH|}');
  if RequestedGameDir <> '' then
    GameDirPage.Values[0] := RequestedGameDir
  else
    GameDirPage.Values[0] := DefaultGameDir();
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = GameDirPage.ID then
    Result := ValidateGameDir(GameDirPage.Values[0], True);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if not StopExistingGuide() then
  begin
    Result := '无法让现有 STS2 Guide 完全退出；请从托盘选择“完全退出”后重试。';
    Exit;
  end;
  if not ValidateGameDir(GameDirPage.Values[0], False) then
    Result := '游戏目录或兼容性在安装前发生变化，请返回重新选择，并确认游戏已经关闭。';
end;

function GetGameModsDir(Param: String): String;
begin
  Result := AddBackslash(NormalizePath(GameDirPage.Values[0])) + 'mods';
end;

function GetGameDir(Param: String): String;
begin
  Result := NormalizePath(GameDirPage.Values[0]);
end;

procedure WriteInstallReceipt();
var
  ReceiptDir: String;
  ReceiptPath: String;
  Payload: String;
begin
  ReceiptDir := ExpandConstant('{userappdata}\SlayTheSpire2\STS2Guide');
  ForceDirectories(ReceiptDir);
  ReceiptPath := AddBackslash(ReceiptDir) + 'install.json';
  Payload := '{' +
    '"receipt_version":1,' +
    '"guide_version":"{#AppVersion}",' +
    '"release_fingerprint":"{#ReleaseFingerprint}",' +
    '"guide_executable_sha256":"{#GuideExeSHA256}",' +
    '"game_directory":"' + JsonEscape(NormalizePath(GameDirPage.Values[0])) + '",' +
    '"artifacts":[' +
      '{"relative_path":"mods/STS2GuideReadOnlyExporter.dll","sha256":"{#ModDllSHA256}"},' +
      '{"relative_path":"mods/STS2GuideReadOnlyExporter.json","sha256":"{#ModJsonSHA256}"},' +
      '{"relative_path":"mods/STS2GuideReadOnlyExporter.pck","sha256":"{#ModPckSHA256}"}' +
    ']' +
  '}';
  SaveStringToFile(ReceiptPath + '.tmp', Payload, False);
  if FileExists(ReceiptPath) then
    DeleteFile(ReceiptPath);
  RenameFile(ReceiptPath + '.tmp', ReceiptPath);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    WriteInstallReceipt();
end;

function InitializeUninstall(): Boolean;
var
  GameDir: String;
  InspectionSucceeded: Boolean;
begin
  Result := True;
  if not StopExistingGuide() then
  begin
    MsgBox('无法让 STS2 Guide 完全退出；请从托盘选择“完全退出”后重试。', mbError, MB_OK);
    Result := False;
    Exit;
  end;
  if RegQueryStringValue(HKCU, 'Software\STS2Guide', 'GameDir', GameDir) then
  begin
    if ExactProcessIsRunning(
      'SlayTheSpire2.exe',
      AddBackslash(GameDir) + 'SlayTheSpire2.exe',
      InspectionSucceeded) then
    begin
      MsgBox('请先关闭 Slay the Spire 2，再卸载 STS2 Guide。', mbError, MB_OK);
      Result := False;
    end;
    if not InspectionSucceeded then
    begin
      MsgBox('无法核验游戏进程状态，卸载已停止。', mbError, MB_OK);
      Result := False;
    end;
  end
  else
  begin
    MsgBox('STS2 Guide 的安装目录记录缺失。为避免误删其他 Mod，卸载已停止；请先重新安装当前版本后再卸载。', mbError, MB_OK);
    Result := False;
  end;
end;

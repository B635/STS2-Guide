[CmdletBinding()]
param(
    [string]$GamePath,
    [string]$GodotExePath,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ModRoot = Join-Path $ProjectRoot "mod\STS2Guide.ReadOnlyExporter"
$ProjectFile = Join-Path $ModRoot "STS2Guide.ReadOnlyExporter.csproj"
$LocalProps = Join-Path $ModRoot "local.props"
$ArtifactsDir = Join-Path $ModRoot "artifacts\mod"

function Get-LocalProperty {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Test-Path -LiteralPath $LocalProps)) {
        return $null
    }
    [xml]$document = Get-Content -LiteralPath $LocalProps -Raw
    $node = $document.SelectSingleNode("//PropertyGroup/$Name")
    if ($null -eq $node) {
        return $null
    }
    return $node.InnerText.Trim()
}

if (-not $GamePath) {
    $GamePath = Get-LocalProperty -Name "STS2GamePath"
}
if (-not $GamePath) {
    $GamePath = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
}
if (-not $GodotExePath) {
    $GodotExePath = Get-LocalProperty -Name "GodotExePath"
}

if (-not (Test-Path -LiteralPath $GamePath -PathType Container)) {
    throw "STS2 game directory not found: $GamePath"
}
$GameData = Join-Path $GamePath "data_sts2_windows_x86_64\sts2.dll"
if (-not (Test-Path -LiteralPath $GameData -PathType Leaf)) {
    throw "STS2 assembly not found: $GameData"
}

if (-not $VerifyOnly) {
    if (-not $GodotExePath -or -not (Test-Path -LiteralPath $GodotExePath -PathType Leaf)) {
        throw "Godot .NET executable not found. Set GodotExePath in local.props or pass -GodotExePath."
    }
    $runningGame = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -match "Slay.*Spire"
    }
    if ($runningGame) {
        throw "Close Slay the Spire 2 before installing the Mod."
    }

    $env:DOTNET_CLI_HOME = Join-Path $ProjectRoot ".dotnet_cli"
    $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
    $env:APPDATA = Join-Path $ProjectRoot ".dotnet_cli\AppData"
    $env:NUGET_PACKAGES = Join-Path $ProjectRoot ".dotnet_cli\packages"

    Write-Host "Building and installing the P0 Mod..." -ForegroundColor Cyan
    & dotnet build $ProjectFile `
        --no-restore `
        "-p:STS2GamePath=$GamePath" `
        "-p:GodotExePath=$GodotExePath" `
        "-p:InstallModOnBuild=true"
    if ($LASTEXITCODE -ne 0) {
        throw "Mod build/install failed with exit code $LASTEXITCODE."
    }
}

$ModsDir = Join-Path $GamePath "mods"
$artifactNames = @(
    "STS2GuideReadOnlyExporter.dll",
    "STS2GuideReadOnlyExporter.json",
    "STS2GuideReadOnlyExporter.pck"
)
$verified = foreach ($name in $artifactNames) {
    $source = Join-Path $ArtifactsDir $name
    $installed = Join-Path $ModsDir $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Workspace artifact missing: $source"
    }
    if (-not (Test-Path -LiteralPath $installed -PathType Leaf)) {
        throw "Installed artifact missing: $installed"
    }
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash
    $installedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installed).Hash
    if ($sourceHash -ne $installedHash) {
        throw "Installed artifact does not match workspace build: $name"
    }
    $item = Get-Item -LiteralPath $source
    [pscustomobject]@{
        Artifact = $name
        Bytes = $item.Length
        SHA256 = $sourceHash
    }
}

Write-Host "Mod artifacts verified in the game directory:" -ForegroundColor Green
$verified | Format-Table -AutoSize

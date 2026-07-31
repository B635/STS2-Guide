[CmdletBinding()]
param(
    [string]$GamePath,
    [string]$GodotExePath,
    [switch]$Package
)

$ErrorActionPreference = "Stop"
$ToolRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ToolRoot "..\..")
$ProjectFile = Join-Path $ToolRoot "RouteLiveProbe.csproj"
$LocalProps = Join-Path $ProjectRoot "mod\STS2Guide.ReadOnlyExporter\local.props"

function Get-LocalProperty {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Test-Path -LiteralPath $LocalProps -PathType Leaf)) {
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

$assembly = Join-Path $GamePath "data_sts2_windows_x86_64\sts2.dll"
if (-not (Test-Path -LiteralPath $assembly -PathType Leaf)) {
    throw "STS2 assembly not found: $assembly"
}
if (
    $Package -and
    (-not $GodotExePath -or -not (Test-Path -LiteralPath $GodotExePath -PathType Leaf))
) {
    throw "Packaging requires the Godot .NET executable. Configure GodotExePath in local.props or pass it explicitly."
}

$env:DOTNET_CLI_HOME = Join-Path $ProjectRoot ".dotnet_cli"
$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
$env:APPDATA = Join-Path $ProjectRoot ".dotnet_cli\AppData"
$env:NUGET_PACKAGES = Join-Path $ProjectRoot ".dotnet_cli\packages"

$arguments = @(
    "build",
    $ProjectFile,
    "-p:STS2GamePath=$GamePath"
)
if ($Package) {
    $arguments += "-p:GodotExePath=$GodotExePath"
}

& dotnet @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Route live probe build failed with exit code $LASTEXITCODE."
}

$artifacts = Join-Path $ToolRoot "artifacts\probe"
$required = @(
    "STS2GuideRouteLiveProbe.dll",
    "STS2GuideRouteLiveProbe.json"
)
if ($Package) {
    $required += "STS2GuideRouteLiveProbe.pck"
}
foreach ($name in $required) {
    $path = Join-Path $artifacts $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Expected probe artifact was not generated: $path"
    }
}

Write-Host "Development probe build verified. Nothing was installed." -ForegroundColor Green

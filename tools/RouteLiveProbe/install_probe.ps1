[CmdletBinding()]
param(
    [string]$GamePath,
    [string]$GodotExePath
)

$ErrorActionPreference = "Stop"
$ToolRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ToolRoot "..\..")
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

$runningGame = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -match "Slay.*Spire"
}
if ($runningGame) {
    throw "Close Slay the Spire 2 before installing the development probe."
}

& (Join-Path $ToolRoot "build_probe.ps1") `
    -GamePath $GamePath `
    -GodotExePath $GodotExePath `
    -Package
if ($LASTEXITCODE -ne 0) {
    throw "Probe packaging failed with exit code $LASTEXITCODE."
}

$artifacts = Join-Path $ToolRoot "artifacts\probe"
$modsDir = Join-Path $GamePath "mods"
New-Item -ItemType Directory -Path $modsDir -Force | Out-Null
$artifactNames = @(
    "STS2GuideRouteLiveProbe.dll",
    "STS2GuideRouteLiveProbe.json",
    "STS2GuideRouteLiveProbe.pck"
)
$verified = foreach ($name in $artifactNames) {
    $source = Join-Path $artifacts $name
    $installed = Join-Path $modsDir $name
    Copy-Item -LiteralPath $source -Destination $installed -Force
    $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash
    $installedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installed).Hash
    if ($sourceHash -ne $installedHash) {
        throw "Installed probe artifact hash mismatch: $name"
    }
    [pscustomobject]@{
        Artifact = $name
        SHA256 = $sourceHash
    }
}

Write-Host "Development-only route probe installed:" -ForegroundColor Yellow
$verified | Format-Table -AutoSize

[CmdletBinding()]
param([string]$GamePath)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$LocalProps = Join-Path $ProjectRoot "mod\STS2Guide.ReadOnlyExporter\local.props"

if (-not $GamePath -and (Test-Path -LiteralPath $LocalProps -PathType Leaf)) {
    [xml]$document = Get-Content -LiteralPath $LocalProps -Raw
    $node = $document.SelectSingleNode("//PropertyGroup/STS2GamePath")
    if ($null -ne $node) {
        $GamePath = $node.InnerText.Trim()
    }
}
if (-not $GamePath) {
    throw "GamePath is required when local.props does not define STS2GamePath."
}

$runningGame = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -match "Slay.*Spire"
}
if ($runningGame) {
    throw "Close Slay the Spire 2 before uninstalling STS2 Guide."
}

$modsDir = [System.IO.Path]::GetFullPath((Join-Path $GamePath "mods"))
$prefix = $modsDir.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$artifactNames = @(
    "STS2GuideReadOnlyExporter.dll",
    "STS2GuideReadOnlyExporter.json",
    "STS2GuideReadOnlyExporter.pck"
)

foreach ($name in $artifactNames) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $modsDir $name))
    if (-not $target.StartsWith(
        $prefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing to remove a path outside the configured mods directory: $target"
    }
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        Remove-Item -LiteralPath $target -Force
        Write-Host "Removed $name"
    }
}

Write-Host "STS2 Guide Mod is not installed." -ForegroundColor Green

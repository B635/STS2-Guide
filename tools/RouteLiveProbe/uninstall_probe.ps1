[CmdletBinding()]
param([string]$GamePath)

$ErrorActionPreference = "Stop"
$ToolRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ToolRoot "..\..")
$LocalProps = Join-Path $ProjectRoot "mod\STS2Guide.ReadOnlyExporter\local.props"

if (-not $GamePath -and (Test-Path -LiteralPath $LocalProps -PathType Leaf)) {
    [xml]$document = Get-Content -LiteralPath $LocalProps -Raw
    $node = $document.SelectSingleNode("//PropertyGroup/STS2GamePath")
    if ($null -ne $node) {
        $GamePath = $node.InnerText.Trim()
    }
}
if (-not $GamePath) {
    $GamePath = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
}

$runningGame = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessName -match "Slay.*Spire"
}
if ($runningGame) {
    throw "Close Slay the Spire 2 before uninstalling the development probe."
}

$modsDir = [System.IO.Path]::GetFullPath((Join-Path $GamePath "mods"))
$prefix = $modsDir.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
$artifactNames = @(
    "STS2GuideRouteLiveProbe.dll",
    "STS2GuideRouteLiveProbe.json",
    "STS2GuideRouteLiveProbe.pck"
)
foreach ($name in $artifactNames) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $modsDir $name))
    if (-not $target.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the configured mods directory: $target"
    }
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        Remove-Item -LiteralPath $target -Force
        Write-Host "Removed $name"
    }
}

Write-Host "Development route probe is not installed." -ForegroundColor Green

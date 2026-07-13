param(
    [string]$PythonPath = "D:\miniconda3\envs\sts2\python.exe",
    [switch]$SkipBenchmark,
    [switch]$SkipScenarioEval,
    [switch]$SkipModBuild
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)
}
catch {
    # Older PowerShell hosts may not allow changing output encoding.
}
$env:PYTHONIOENCODING = "utf-8"

$results = New-Object System.Collections.Generic.List[object]

function Invoke-ReviewStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    $start = Get-Date
    & $Command
    $exit = if ($LASTEXITCODE -is [int]) { $LASTEXITCODE } else { 0 }
    $elapsed = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)
    $status = if ($exit -eq 0) { "PASS" } else { "FAIL" }
    $color = if ($exit -eq 0) { "Green" } else { "Red" }
    Write-Host "[$status] $Name (${elapsed}s)" -ForegroundColor $color
    $results.Add([pscustomobject]@{
        Step = $Name
        Status = $status
        ExitCode = $exit
        Seconds = $elapsed
    }) | Out-Null
}

function Assert-FileExists {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Error "Required file not found: $Path"
        exit 2
    }
}

Assert-FileExists $PythonPath

Invoke-ReviewStep "git diff --check" {
    git diff --check
}

Invoke-ReviewStep "Python unittest discover" {
    & $PythonPath -m unittest discover -s tests -v
}

if (-not $SkipScenarioEval) {
    Invoke-ReviewStep "fixed card reward scenarios" {
        & $PythonPath scripts/eval_card_reward_scenarios.py --strict
    }
}

if (-not $SkipBenchmark) {
    Invoke-ReviewStep "P0 realtime benchmark" {
        & $PythonPath scripts/benchmark_p0_realtime.py
    }
}

Invoke-ReviewStep "realtime.host --help" {
    & $PythonPath -m realtime.host --help
}

if (-not $SkipModBuild) {
    Invoke-ReviewStep "Mod build --no-restore" {
        $env:DOTNET_CLI_HOME = Join-Path (Get-Location) ".dotnet_cli"
        $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
        $env:APPDATA = Join-Path (Get-Location) ".dotnet_cli\AppData"
        $env:NUGET_PACKAGES = Join-Path (Get-Location) ".dotnet_cli\packages"
        dotnet build mod/STS2Guide.ReadOnlyExporter/STS2Guide.ReadOnlyExporter.csproj --no-restore
    }
}

Write-Host ""
Write-Host "== Review summary ==" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Status -ne "PASS" })
if ($failed.Count -gt 0) {
    Write-Host "P0 automatic review FAILED: $($failed.Count) step(s) failed." -ForegroundColor Red
    exit 1
}

Write-Host "P0 automatic review PASSED. This does not replace install or true live regression." -ForegroundColor Green
exit 0

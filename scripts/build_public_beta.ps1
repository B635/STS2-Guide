[CmdletBinding()]
param(
    [string]$Python = $env:STS2_PYTHON,
    [string]$GamePath,
    [string]$GodotExePath,
    [string]$InnoCompiler = $env:STS2_ISCC
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ModRoot = Join-Path $ProjectRoot "mod\STS2Guide.ReadOnlyExporter"
$LocalProps = Join-Path $ModRoot "local.props"
$ManifestPath = Join-Path $ProjectRoot "packaging\compatibility.json"

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

if (-not $Python) {
    if ($env:CONDA_DEFAULT_ENV -eq "sts2" -and $env:CONDA_PREFIX) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    }
    else {
        throw "Set STS2_PYTHON or activate the sts2 conda environment."
    }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python not found: $Python"
}
if (-not $GamePath) {
    $GamePath = Get-LocalProperty -Name "STS2GamePath"
}
if (-not $GodotExePath) {
    $GodotExePath = Get-LocalProperty -Name "GodotExePath"
}
if (-not $GamePath -or -not (Test-Path -LiteralPath (Join-Path $GamePath "data_sts2_windows_x86_64\sts2.dll") -PathType Leaf)) {
    throw "A valid STS2 GamePath is required to build the exact Mod artifacts."
}
$ReleaseInfoPath = Join-Path $GamePath "release_info.json"
if (-not (Test-Path -LiteralPath $ReleaseInfoPath -PathType Leaf)) {
    throw "The exact game release_info.json is required."
}
if (-not $GodotExePath -or -not (Test-Path -LiteralPath $GodotExePath -PathType Leaf)) {
    throw "Godot .NET is required to build the Mod PCK."
}

$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$actualGameHash = (Get-FileHash -LiteralPath (Join-Path $GamePath "data_sts2_windows_x86_64\sts2.dll") -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualGameHash -ne $manifest.game.sts2_dll_sha256.ToLowerInvariant()) {
    throw "Local sts2.dll does not match packaging/compatibility.json."
}

Push-Location $ProjectRoot
try {
    & $Python -c "import sys; from pathlib import Path; from realtime.installation import read_release_info_version; observed=read_release_info_version(Path(sys.argv[1])); raise SystemExit(0 if observed == sys.argv[2] else 2)" $GamePath $manifest.game.version
    if ($LASTEXITCODE -ne 0) {
        throw "release_info.json is not strict JSON with the expected unique game version."
    }

    Write-Host "Checking pinned Public Beta dependencies..." -ForegroundColor Cyan
    & $Python -c @'
import importlib.metadata as m
required = {
    "pydantic": "2.13.0",
    "python-dotenv": "1.2.1",
    "pyinstaller": "6.21.0",
    "pystray": "0.19.5",
    "pillow": "11.3.0",
}
bad = {name: (m.version(name), version) for name, version in required.items() if m.version(name) != version}
if bad:
    raise SystemExit(f"Pinned dependency mismatch: {bad}")
'@
    if ($LASTEXITCODE -ne 0) {
        throw "Install the exact versions in requirements-p0.txt into the sts2 environment."
    }

    Write-Host "Running the complete Python release gate..." -ForegroundColor Cyan
    & $Python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Complete Python release gate failed."
    }

    Write-Host "Building release SQLite template..." -ForegroundColor Cyan
    & $Python scripts\build_icon.py
    if ($LASTEXITCODE -ne 0) {
        throw "Release icon build failed."
    }
    & $Python scripts\build_release_database.py
    if ($LASTEXITCODE -ne 0) {
        throw "Release database build failed."
    }

    Write-Host "Building read-only Mod artifacts without installing..." -ForegroundColor Cyan
    $env:DOTNET_CLI_HOME = Join-Path $ProjectRoot ".dotnet_cli"
    $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
    $env:APPDATA = Join-Path $ProjectRoot ".dotnet_cli\AppData"
    $env:NUGET_PACKAGES = Join-Path $ProjectRoot ".dotnet_cli\packages"
    & dotnet build `
        (Join-Path $ModRoot "STS2Guide.ReadOnlyExporter.csproj") `
        --no-restore `
        "-p:STS2GamePath=$GamePath" `
        "-p:GodotExePath=$GodotExePath" `
        "-p:InstallModOnBuild=false"
    if ($LASTEXITCODE -ne 0) {
        throw "Mod build failed."
    }

    Write-Host "Building windowed Guide executable..." -ForegroundColor Cyan
    & $Python -m PyInstaller --noconfirm --clean packaging\sts2-guide.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    $exe = Join-Path $ProjectRoot "dist\STS2 Guide.exe"
    $smokeRoot = Join-Path $ProjectRoot ("build\public-beta-smoke-" + [guid]::NewGuid().ToString("N"))
    $smokeEvents = Join-Path $smokeRoot "events"
    New-Item -ItemType Directory -Force -Path $smokeEvents | Out-Null
    $smoke = Start-Process `
        -FilePath $exe `
        -ArgumentList @(
            "--startup-check",
            "--input", (Join-Path $smokeRoot "state-event.json"),
            "--output", (Join-Path $smokeRoot "advice-event.json"),
            "--events-dir", $smokeEvents,
            "--checkpoint", (Join-Path $smokeRoot "active-run.json"),
            "--database", (Join-Path $smokeRoot "sts2-guide.db"),
            "--log-file", (Join-Path $smokeRoot "worker.log")
        ) `
        -Wait `
        -PassThru `
        -WindowStyle Hidden
    if ($smoke.ExitCode -ne 0) {
        throw "Frozen EXE startup check failed with exit code $($smoke.ExitCode)."
    }
    if (Test-Path -LiteralPath (Join-Path $smokeRoot ".host.lock")) {
        throw "Frozen EXE left a stale Worker lock."
    }
    $artifacts = Join-Path $ModRoot "artifacts\mod"
    $requiredFiles = @(
        $exe,
        (Join-Path $artifacts "STS2GuideReadOnlyExporter.dll"),
        (Join-Path $artifacts "STS2GuideReadOnlyExporter.json"),
        (Join-Path $artifacts "STS2GuideReadOnlyExporter.pck")
    )
    foreach ($path in $requiredFiles) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Release artifact is missing: $path"
        }
    }

    if (-not $InnoCompiler) {
        $candidates = @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
        )
        $InnoCompiler = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    }
    if (-not $InnoCompiler -or -not (Test-Path -LiteralPath $InnoCompiler -PathType Leaf)) {
        throw "Inno Setup 6 was not found. Install it or set STS2_ISCC."
    }

    $guideHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    $releaseInfoHash = (Get-FileHash -LiteralPath $ReleaseInfoPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $dllHash = (Get-FileHash -LiteralPath $requiredFiles[1] -Algorithm SHA256).Hash.ToLowerInvariant()
    $jsonHash = (Get-FileHash -LiteralPath $requiredFiles[2] -Algorithm SHA256).Hash.ToLowerInvariant()
    $pckHash = (Get-FileHash -LiteralPath $requiredFiles[3] -Algorithm SHA256).Hash.ToLowerInvariant()
    $defines = @(
        "/DAppVersion=$($manifest.guide.version)",
        "/DExpectedGameVersion=$($manifest.game.version)",
        "/DExpectedGameAssemblySHA256=$($manifest.game.sts2_dll_sha256)",
        "/DExpectedReleaseInfoSHA256=$releaseInfoHash",
        "/DReleaseFingerprint=$($manifest.release_fingerprint)",
        "/DGuideExeSHA256=$guideHash",
        "/DModDllSHA256=$dllHash",
        "/DModJsonSHA256=$jsonHash",
        "/DModPckSHA256=$pckHash"
    )
    Write-Host "Building per-user installer..." -ForegroundColor Cyan
    & $InnoCompiler @defines (Join-Path $ProjectRoot "packaging\installer.iss")
    if ($LASTEXITCODE -ne 0) {
        throw "Installer build failed."
    }

    & $Python scripts\audit_public_beta.py
    if ($LASTEXITCODE -ne 0) {
        throw "Public Beta release audit failed."
    }
    Write-Host "Public Beta artifacts are ready under dist\." -ForegroundColor Green
}
finally {
    Pop-Location
}

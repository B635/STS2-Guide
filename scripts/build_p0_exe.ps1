$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

# Require explicit STS2_PYTHON or an activated *sts2* conda environment.
# Any other conda env (base, other projects) is rejected — PyInstaller must
# only be installed to the project-specified sts2 environment.
if ($env:STS2_PYTHON) {
    $python = $env:STS2_PYTHON
}
elseif ($env:CONDA_DEFAULT_ENV -eq "sts2" -and $env:CONDA_PREFIX -and (Test-Path -LiteralPath (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
}
elseif ($env:CONDA_DEFAULT_ENV) {
    throw "Active conda environment is '$env:CONDA_DEFAULT_ENV', not 'sts2'. Activate the sts2 environment (conda activate sts2) or set `$env:STS2_PYTHON."
}
else {
    throw "Python not found. Set `$env:STS2_PYTHON or activate the sts2 conda environment (conda activate sts2)."
}

Write-Host "Using Python: $python"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python not found at $python."
}

# Verify the pinned release dependencies are already present.  Build scripts
# never mutate the selected environment behind the user's back.
& $python -c "import importlib.metadata as m; required={'pydantic':'2.13.0','python-dotenv':'1.2.1','pyinstaller':'6.21.0','pystray':'0.19.5','pillow':'11.3.0'}; bad={k:(m.version(k),v) for k,v in required.items() if m.version(k)!=v}; assert not bad, bad" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Pinned release dependencies are missing or mismatched. Run: $python -m pip install -r requirements-p0.txt"
}

Push-Location $ProjectRoot
try {
    Write-Host "Building immutable release database template..."
    & $python scripts\build_icon.py
    if ($LASTEXITCODE -ne 0) {
        throw "Release icon build failed with exit code $LASTEXITCODE"
    }
    & $python scripts\build_release_database.py
    if ($LASTEXITCODE -ne 0) {
        throw "Release database build failed with exit code $LASTEXITCODE"
    }

    Write-Host "Building EXE from $ProjectRoot ..."
    & $python -m PyInstaller --noconfirm --clean packaging\sts2-guide.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }

    $exe = Join-Path $ProjectRoot "dist\STS2 Guide.exe"
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "Packaged executable not found: $exe"
    }
    $smokeRoot = Join-Path $ProjectRoot "build\p0-exe-smoke"
    New-Item -ItemType Directory -Force -Path $smokeRoot | Out-Null
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
            "--database", (Join-Path $smokeRoot "sts2-guide.db")
        ) `
        -Wait `
        -PassThru `
        -WindowStyle Hidden
    if ($smoke.ExitCode -ne 0) {
        throw "Packaged executable startup check failed with exit code $($smoke.ExitCode)"
    }
    if (Test-Path -LiteralPath (Join-Path $smokeRoot ".host.lock")) {
        throw "Packaged executable left a stale instance lock after startup check."
    }
    Write-Host "Build complete. Output in dist\"
}
finally {
    Pop-Location
}

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

# Verify PyInstaller is available; install if missing (only to the resolved sts2 env).
& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller not found. Installing to $python ..."
    & $python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyInstaller."
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "Building EXE from $ProjectRoot ..."
    & $python -m PyInstaller --noconfirm --clean packaging\sts2-guide.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed with exit code $LASTEXITCODE"
    }
    Write-Host "Build complete. Output in dist\"
}
finally {
    Pop-Location
}

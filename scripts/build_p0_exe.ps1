$ErrorActionPreference = "Stop"

$python = "D:\miniconda3\envs\sts2\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "STS2 conda Python was not found at $python"
}

& $python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Install requirements-p0.txt first."
}

& $python -m PyInstaller --noconfirm --clean packaging\sts2-guide.spec

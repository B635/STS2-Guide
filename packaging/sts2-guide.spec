# Build from the repository root:
# D:\miniconda3\envs\sts2\python.exe -m PyInstaller packaging\sts2-guide.spec
import os
from pathlib import Path

# SPECPATH is the directory containing this spec file (packaging/).
# The project root is its parent directory.
root = Path(SPECPATH).resolve().parent

analysis = Analysis(
    [str(root / "realtime" / "host.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[
        (str(root / "data" / "knowledge.json"), "data"),
        (str(root / "data" / "community_scores.json"), "data"),
        (str(root / "protocol" / "state-event.schema.json"), "protocol"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "openai",
        "langgraph",
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="STS2 Guide",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

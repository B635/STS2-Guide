# Build from the repository root:
# D:\miniconda3\envs\sts2\python.exe -m PyInstaller packaging\sts2-guide.spec
import os
import sys
from pathlib import Path

# SPECPATH is the directory containing this spec file (packaging/).
# The project root is its parent directory.
root = Path(SPECPATH).resolve().parent
python_runtime_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
conda_runtime_dlls = [
    python_runtime_bin / "ffi.dll",
    python_runtime_bin / "libexpat.dll",
]
missing_runtime_dlls = [path for path in conda_runtime_dlls if not path.is_file()]
if missing_runtime_dlls:
    raise FileNotFoundError(
        "Required Conda runtime DLLs are missing: "
        + ", ".join(str(path) for path in missing_runtime_dlls)
    )

analysis = Analysis(
    [str(root / "realtime" / "host.py")],
    pathex=[str(root)],
    # PyInstaller does not search Conda's Library/bin when resolving the
    # dependencies of _ctypes.pyd and pyexpat.pyd.  Bundle them explicitly so
    # the windowed executable can reach instance-lock and initialization code.
    binaries=[(str(path), ".") for path in conda_runtime_dlls],
    datas=[
        (str(root / "data" / "knowledge.json"), "data"),
        (str(root / "data" / "community_scores.json"), "data"),
        (str(root / "protocol" / "state-event.schema.json"), "protocol"),
        (str(root / "protocol" / "state-event.example.json"), "protocol"),
        (str(root / "protocol" / "advice-event.schema.json"), "protocol"),
        (str(root / "protocol" / "advice-event.example.json"), "protocol"),
        (str(root / "packaging" / "compatibility.json"), "packaging"),
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

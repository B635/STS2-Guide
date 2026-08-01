# Build from the repository root with scripts/build_public_beta.ps1 (or
# scripts/build_p0_exe.ps1).  Those entry points generate the immutable SQLite
# template and deterministic icon before invoking PyInstaller.
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
        (
            str(root / "build" / "release" / "sts2-guide-template.db"),
            "data",
        ),
        (str(root / "protocol" / "state-event.schema.json"), "protocol"),
        (str(root / "protocol" / "state-event.example.json"), "protocol"),
        (str(root / "protocol" / "advice-event.schema.json"), "protocol"),
        (str(root / "protocol" / "advice-event.example.json"), "protocol"),
        (str(root / "packaging" / "compatibility.json"), "packaging"),
        (str(root / "packaging" / "DATA_SOURCES.txt"), "packaging"),
        (str(root / "build" / "release" / "sts2-guide.png"), "packaging"),
        (str(root / "LICENSE"), "."),
    ],
    hiddenimports=[
        "pystray",
        "pystray._win32",
        "PIL.Image",
        "PIL.ImageDraw",
    ],
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
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    version=str(root / "packaging" / "version_info.txt"),
    icon=str(root / "build" / "release" / "sts2-guide.ico"),
)

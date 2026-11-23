"""Read-only wrapper and local disk cache over tools/Sts2FactExtractor."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Optional, Protocol

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_PROPS = ROOT_DIR / "mod" / "STS2Guide.ReadOnlyExporter" / "local.props"
DEFAULT_IL_CACHE_DIR = ROOT_DIR / "build" / "fact-pipeline" / "il-cache"

MAX_DUMP_CHARS = 32_000


class AssemblyProviderError(RuntimeError):
    """Raised when an IL dump or enum lookup cannot be satisfied."""


class AssemblyProvider(Protocol):
    """Read-only source of class IL dumps and enum tables."""

    def dump_class(self, class_name: str) -> str: ...

    def read_enum(self, enum_name: str) -> str: ...


def resolve_default_extractor(root: Path = ROOT_DIR) -> Optional[Path]:
    candidates = (
        root / "tools" / "Sts2FactExtractor" / "bin" / "next" / "Sts2FactExtractor.exe",
        root / "tools" / "Sts2FactExtractor" / "bin" / "Release" / "net9.0" / "Sts2FactExtractor.exe",
        root / "tools" / "Sts2FactExtractor" / "bin" / "Debug" / "net9.0" / "Sts2FactExtractor.exe",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def resolve_default_dll(props_path: Path = DEFAULT_LOCAL_PROPS) -> Optional[Path]:
    if not props_path.is_file():
        return None
    try:
        game_path = ET.parse(props_path).findtext(".//STS2GamePath")
    except ET.ParseError:
        return None
    if not game_path:
        return None
    dll = Path(game_path) / "data_sts2_windows_x86_64" / "sts2.dll"
    return dll if dll.is_file() else None


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def _truncate_il(text: str, max_chars: int = MAX_DUMP_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    kept = text[:max_chars]
    return (
        f"{kept}\n\n[TRUNCATED: original IL was {len(text)} chars, "
        f"showing first {max_chars} chars]"
    )


class CachedAssemblyProvider:
    """Runs Sts2FactExtractor in read-only mode with a per-assembly disk cache."""

    def __init__(
        self,
        *,
        dll_path: Optional[Path] = None,
        extractor_path: Optional[Path] = None,
        cache_dir: Optional[Path] = DEFAULT_IL_CACHE_DIR,
        assembly_tag: str = "default",
        timeout_seconds: int = 30,
    ) -> None:
        self.dll_path = Path(dll_path) if dll_path else None
        self.extractor_path = Path(extractor_path) if extractor_path else resolve_default_extractor()
        self.cache_root = (Path(cache_dir) / _safe_filename(assembly_tag)) if cache_dir else None
        self.timeout_seconds = timeout_seconds

    def _cache_file(self, bucket: str, name: str) -> Optional[Path]:
        if self.cache_root is None:
            return None
        return self.cache_root / bucket / f"{_safe_filename(name)}.txt"

    def _run_extractor(self, flag: str, target: str) -> subprocess.CompletedProcess:
        if self.extractor_path is None or not self.extractor_path.is_file():
            raise AssemblyProviderError("Sts2FactExtractor executable is not available")
        if self.dll_path is None or not self.dll_path.is_file():
            raise AssemblyProviderError(f"sts2.dll not available at {self.dll_path}")
        return subprocess.run(
            [str(self.extractor_path), str(self.dll_path), flag, target],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )

    def dump_class(self, class_name: str) -> str:
        clean = class_name.strip()
        if not clean:
            raise AssemblyProviderError("class_name must not be empty")
        cache_path = self._cache_file("dump", clean)
        if cache_path is not None and cache_path.is_file():
            return _truncate_il(cache_path.read_text(encoding="utf-8"))

        # Try model dump first, then arbitrary type dump for helper/engine types.
        result = self._run_extractor("--dump", clean)
        if result.returncode != 0 or not result.stdout.strip():
            result = self._run_extractor("--dump-any", clean)
        if result.returncode != 0 or not result.stdout.strip():
            detail = (result.stderr or result.stdout or "").strip()
            raise AssemblyProviderError(f"Type {clean!r} not found in assembly ({detail})")

        text = result.stdout
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        return _truncate_il(text)

    def read_enum(self, enum_name: str) -> str:
        clean = enum_name.strip()
        if not clean:
            raise AssemblyProviderError("enum_name must not be empty")
        cache_path = self._cache_file("enum", clean)
        if cache_path is not None and cache_path.is_file():
            return cache_path.read_text(encoding="utf-8")

        result = self._run_extractor("--enum", clean)
        if result.returncode != 0 or not result.stdout.strip():
            detail = (result.stderr or result.stdout or "").strip()
            raise AssemblyProviderError(f"Enum {clean!r} not found in assembly ({detail})")

        text = result.stdout
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        return text


class FixtureAssemblyProvider:
    """In-memory assembly provider for offline tests and replay."""

    def __init__(
        self,
        dumps: Optional[Mapping[str, str]] = None,
        enums: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.dumps: Dict[str, str] = dict(dumps or {})
        self.enums: Dict[str, str] = dict(enums or {})

    def dump_class(self, class_name: str) -> str:
        key = class_name.strip()
        if key not in self.dumps:
            raise AssemblyProviderError(f"Type {key!r} not found in fixture")
        return _truncate_il(self.dumps[key])

    def read_enum(self, enum_name: str) -> str:
        key = enum_name.strip()
        if key not in self.enums:
            raise AssemblyProviderError(f"Enum {key!r} not found in fixture")
        return self.enums[key]

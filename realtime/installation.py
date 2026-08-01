"""Windows installation discovery, receipts, and fail-closed preflight.

This module is deliberately UI-free.  Callers may display stable reason codes,
but filesystem paths stay in the local result objects and never enter reasons
or diagnostic payloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

from realtime.compatibility import CompatibilityManifest


GAME_DIRECTORY_ENV = "STS2_GAME_DIR"
GAME_DIRECTORY_ENV_ALIASES = (GAME_DIRECTORY_ENV, "STS2_GAME_PATH")
GAME_EXECUTABLE = "SlayTheSpire2.exe"
GAME_ASSEMBLY = Path("data_sts2_windows_x86_64") / "sts2.dll"
STEAM_GAME_RELATIVE = Path("steamapps") / "common" / "Slay the Spire 2"
GUIDE_MOD_ARTIFACTS = (
    "STS2GuideReadOnlyExporter.dll",
    "STS2GuideReadOnlyExporter.json",
    "STS2GuideReadOnlyExporter.pck",
)
RECEIPT_VERSION = 1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")
_VDF_PATH_RE = re.compile(r'^\s*"path"\s+"(?P<path>.+)"\s*$')
_VDF_LEGACY_PATH_RE = re.compile(
    r'^\s*"[0-9]+"\s+"(?P<path>(?:[A-Za-z]:|/).+)"\s*$'
)


@dataclass(frozen=True)
class InstalledArtifact:
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        normalized = PurePosixPath(self.relative_path.replace("\\", "/"))
        if (
            normalized.is_absolute()
            or re.match(r"^[A-Za-z]:", self.relative_path)
            or ".." in normalized.parts
            or len(normalized.parts) < 1
            or not _SHA256_RE.fullmatch(self.sha256)
        ):
            raise ValueError("invalid installed artifact identity")


@dataclass(frozen=True)
class InstallReceipt:
    guide_version: str
    release_fingerprint: str
    guide_executable_sha256: str
    game_directory: Path
    artifacts: tuple[InstalledArtifact, ...]
    receipt_version: int = RECEIPT_VERSION

    def __post_init__(self) -> None:
        if self.receipt_version != RECEIPT_VERSION:
            raise ValueError("unsupported install receipt version")
        if not _VERSION_RE.fullmatch(self.guide_version):
            raise ValueError("invalid receipt guide version")
        if not _SHA256_RE.fullmatch(self.release_fingerprint):
            raise ValueError("invalid receipt release fingerprint")
        if not _SHA256_RE.fullmatch(self.guide_executable_sha256):
            raise ValueError("invalid receipt Guide executable hash")
        if not self.game_directory.is_absolute():
            raise ValueError("receipt game directory must be absolute")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("receipt artifact paths must be unique")

    def as_dict(self) -> dict:
        return {
            "receipt_version": self.receipt_version,
            "guide_version": self.guide_version,
            "release_fingerprint": self.release_fingerprint,
            "guide_executable_sha256": self.guide_executable_sha256,
            "game_directory": str(self.game_directory),
            "artifacts": [
                {
                    "relative_path": artifact.relative_path,
                    "sha256": artifact.sha256,
                }
                for artifact in self.artifacts
            ],
        }


@dataclass(frozen=True)
class DiscoveryResult:
    game_directory: Path | None
    source: str | None
    reason_codes: tuple[str, ...]

    @property
    def found(self) -> bool:
        return self.game_directory is not None


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    component: str


@dataclass(frozen=True)
class InstallationPreflight:
    compatible: bool
    reason_codes: tuple[str, ...]
    issues: tuple[PreflightIssue, ...]
    artifact_hashes: Mapping[str, str]
    third_party_mod_count: int


def default_install_receipt_path() -> Path:
    app_data = os.getenv("APPDATA")
    base = Path(app_data) if app_data else Path.home() / "AppData" / "Roaming"
    return base / "SlayTheSpire2" / "STS2Guide" / "install.json"


def expected_mod_hashes(receipt: InstallReceipt) -> dict[str, str] | None:
    """Extract exactly the three owned Mod hashes from a strict receipt."""

    expected_paths = {f"mods/{name}": name for name in GUIDE_MOD_ARTIFACTS}
    observed = {
        artifact.relative_path.replace("\\", "/"): artifact.sha256
        for artifact in receipt.artifacts
    }
    if set(observed) != set(expected_paths):
        return None
    return {
        name: observed[relative]
        for relative, name in expected_paths.items()
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate receipt key")
        result[key] = value
    return result


def write_install_receipt(path: str | Path, receipt: InstallReceipt) -> None:
    """Atomically replace the one local installation receipt."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    temporary.write_text(
        json.dumps(receipt.as_dict(), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_install_receipt(path: str | Path) -> InstallReceipt | None:
    target = Path(path)
    try:
        payload = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "receipt_version",
        "guide_version",
        "release_fingerprint",
        "guide_executable_sha256",
        "game_directory",
        "artifacts",
    }:
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    try:
        parsed_artifacts = tuple(
            InstalledArtifact(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
            )
            for item in artifacts
            if isinstance(item, dict)
            and set(item) == {"relative_path", "sha256"}
        )
        if len(parsed_artifacts) != len(artifacts):
            return None
        return InstallReceipt(
            receipt_version=payload["receipt_version"],
            guide_version=payload["guide_version"],
            release_fingerprint=payload["release_fingerprint"],
            guide_executable_sha256=payload["guide_executable_sha256"],
            game_directory=Path(payload["game_directory"]),
            artifacts=parsed_artifacts,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _looks_like_game_directory(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / GAME_EXECUTABLE).is_file()
        and (path / GAME_ASSEMBLY).is_file()
    )


def _steam_roots_from_registry() -> tuple[Path, ...]:
    if os.name != "nt":
        return ()
    try:
        import winreg
    except ImportError:
        return ()
    roots: list[Path] = []
    locations = (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"Software\WOW6432Node\Valve\Steam",
            "InstallPath",
        ),
    )
    for hive, key_name, value_name in locations:
        try:
            with winreg.OpenKey(hive, key_name) as key:
                raw, _ = winreg.QueryValueEx(key, value_name)
        except OSError:
            continue
        if isinstance(raw, str) and raw.strip():
            roots.append(Path(raw.strip()))
    return tuple(dict.fromkeys(roots))


def _steam_libraries(steam_root: Path) -> tuple[Path, ...]:
    libraries = [steam_root]
    library_file = steam_root / "steamapps" / "libraryfolders.vdf"
    try:
        lines = library_file.read_text(encoding="utf-8", errors="strict").splitlines()
    except OSError:
        return tuple(libraries)
    for line in lines:
        match = _VDF_PATH_RE.match(line) or _VDF_LEGACY_PATH_RE.match(line)
        if match is None:
            continue
        raw = match.group("path").replace(r"\\", "\\")
        if raw.strip():
            libraries.append(Path(raw))
    return tuple(dict.fromkeys(libraries))


def discover_game_directory(
    *,
    environ: Mapping[str, str] | None = None,
    receipt_path: str | Path | None = None,
    registry_steam_roots: Iterable[str | Path] | None = None,
    manual_directory: str | Path | None = None,
) -> DiscoveryResult:
    """Discover STS2 in deterministic env/receipt/Steam/manual order."""

    env = os.environ if environ is None else environ
    candidates: list[tuple[str, Path]] = []
    for variable in GAME_DIRECTORY_ENV_ALIASES:
        env_value = env.get(variable)
        if isinstance(env_value, str) and env_value.strip():
            candidates.append(("environment", Path(env_value.strip())))
            break

    if receipt_path is not None:
        receipt = read_install_receipt(receipt_path)
        if receipt is not None:
            candidates.append(("receipt", receipt.game_directory))

    roots = (
        _steam_roots_from_registry()
        if registry_steam_roots is None
        else tuple(Path(root) for root in registry_steam_roots)
    )
    for root in roots:
        for library in _steam_libraries(root):
            candidates.append(("steam", library / STEAM_GAME_RELATIVE))

    if manual_directory is not None:
        candidates.append(("manual", Path(manual_directory)))

    saw_candidate = False
    seen: set[str] = set()
    for source, candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        saw_candidate = True
        if _looks_like_game_directory(candidate):
            return DiscoveryResult(candidate.resolve(), source, ())
    return DiscoveryResult(
        None,
        None,
        ("game_directory_invalid",) if saw_candidate else ("game_directory_not_found",),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_release_info_version(game_directory: Path) -> str | None:
    """Read the game's own strict release identity.

    The executable's PE product version describes the engine/build tool and
    is not the STS2 content version.  ``release_info.json`` is the same file
    the read-only Mod uses for its production handshake.
    """

    path = game_directory / "release_info.json"
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("version")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    return normalized if _VERSION_RE.fullmatch(normalized) else None


def preflight_installation(
    game_directory: str | Path,
    manifest: CompatibilityManifest,
    expected_guide_hashes: Mapping[str, str],
    *,
    version_reader: Callable[[Path], str | None] = read_release_info_version,
) -> InstallationPreflight:
    """Verify one exact vanilla-compatible installation without path leaks."""

    game = Path(game_directory)
    issues: list[PreflightIssue] = []
    hashes: MutableMapping[str, str] = {}
    if not game.is_dir():
        issues.append(PreflightIssue("game_directory_missing", "game"))
        return _preflight_result(issues, hashes, 0)
    if not manifest.is_enabled:
        issues.append(PreflightIssue("manifest_not_enabled", "manifest"))

    executable = game / GAME_EXECUTABLE
    assembly = game / GAME_ASSEMBLY
    if not executable.is_file():
        issues.append(PreflightIssue("game_executable_missing", "game.exe"))
    else:
        hashes["game_executable"] = sha256_file(executable)
        try:
            observed_version = version_reader(game)
        except Exception:
            observed_version = None
        if observed_version is None:
            issues.append(PreflightIssue("game_version_unreadable", "game.version"))
        elif observed_version != manifest.game.version:
            issues.append(PreflightIssue("game_version_mismatch", "game.version"))

    if not assembly.is_file():
        issues.append(PreflightIssue("game_assembly_missing", "game.sts2_dll"))
    else:
        assembly_hash = sha256_file(assembly)
        hashes["game_assembly"] = assembly_hash
        if assembly_hash != manifest.game.sts2_dll_sha256:
            issues.append(
                PreflightIssue(
                    "game_assembly_hash_mismatch",
                    "game.sts2_dll",
                )
            )

    expected_names = set(GUIDE_MOD_ARTIFACTS)
    supplied_names = set(expected_guide_hashes)
    valid_expectations = supplied_names == expected_names and all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value)
        for value in expected_guide_hashes.values()
    )
    if not valid_expectations:
        issues.append(
            PreflightIssue(
                "guide_artifact_expectation_invalid",
                "guide.mod_artifacts",
            )
        )

    mods = game / "mods"
    for name in GUIDE_MOD_ARTIFACTS:
        artifact = mods / name
        component = f"guide.{name}"
        if not artifact.is_file():
            issues.append(PreflightIssue("guide_artifact_missing", component))
            continue
        observed_hash = sha256_file(artifact)
        hashes[name] = observed_hash
        expected_hash = expected_guide_hashes.get(name)
        if valid_expectations and observed_hash != expected_hash:
            issues.append(
                PreflightIssue("guide_artifact_hash_mismatch", component)
            )

    third_party_count = 0
    if mods.is_dir():
        allowed = {PurePosixPath(name) for name in GUIDE_MOD_ARTIFACTS}
        third_party_count = sum(
            1
            for path in mods.rglob("*")
            if path.is_file()
            and PurePosixPath(path.relative_to(mods).as_posix()) not in allowed
        )
    if third_party_count:
        issues.append(
            PreflightIssue("third_party_mods_detected", "game.mods")
        )
    return _preflight_result(issues, hashes, third_party_count)


def _preflight_result(
    issues: list[PreflightIssue],
    hashes: Mapping[str, str],
    third_party_count: int,
) -> InstallationPreflight:
    reason_codes = tuple(dict.fromkeys(issue.code for issue in issues))
    return InstallationPreflight(
        compatible=not issues,
        reason_codes=reason_codes,
        issues=tuple(issues),
        artifact_hashes=dict(sorted(hashes.items())),
        third_party_mod_count=third_party_count,
    )

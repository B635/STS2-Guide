"""Fail-closed audit for the explicitly assembled Public Beta payload."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXE = DIST / "STS2 Guide.exe"
INSTALLER_DIR = DIST / "installer"
MOD_DIR = ROOT / "mod" / "STS2Guide.ReadOnlyExporter" / "artifacts" / "mod"

FORBIDDEN_ARCHIVE_NAMES = {
    ".env",
    "active-run.json",
    "advice-event.json",
    "community_scores.json",
    "knowledge.json",
    "local.props",
    "mobalytics_card_tiers.json",
    "state-event.json",
}
FORBIDDEN_TOKENS = ("route_live_probe", "routeliveprobe", "sts2mcp")


def assert_artifact_fresh(
    artifact: Path,
    sources: list[Path],
    *,
    label: str,
) -> None:
    existing = [path for path in sources if path.is_file()]
    if not existing:
        raise SystemExit(f"No source files found for freshness gate: {label}")
    newest = max(existing, key=lambda path: path.stat().st_mtime_ns)
    if artifact.stat().st_mtime_ns < newest.stat().st_mtime_ns:
        raise SystemExit(
            f"Stale {label}: {artifact.name} predates source {newest.name}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_listing(exe: Path) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller.utils.cliutils.archive_viewer",
            "-l",
            str(exe),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def main() -> int:
    installers = sorted(INSTALLER_DIR.glob("STS2-Guide-*-win-x64-setup.exe"))
    required = [
        EXE,
        MOD_DIR / "STS2GuideReadOnlyExporter.dll",
        MOD_DIR / "STS2GuideReadOnlyExporter.json",
        MOD_DIR / "STS2GuideReadOnlyExporter.pck",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if len(installers) != 1:
        missing.append(
            f"expected exactly one installer, found {len(installers)}"
        )
    if missing:
        raise SystemExit("Release payload incomplete: " + "; ".join(missing))

    python_sources = [
        path
        for directory in ("advisor", "realtime", "storage", "tests")
        for path in (ROOT / directory).rglob("*.py")
    ]
    package_sources = [
        *python_sources,
        *list((ROOT / "protocol").glob("*.json")),
        *list((ROOT / "packaging").glob("*")),
        *list((ROOT / "scripts").glob("*.py")),
        *list((ROOT / "scripts").glob("*.ps1")),
        ROOT / "requirements-p0.txt",
    ]
    installer = installers[0]
    assert_artifact_fresh(EXE, package_sources, label="Guide executable")
    assert_artifact_fresh(installer, package_sources, label="installer")

    mod_code_sources = [
        *list((ROOT / "mod" / "STS2Guide.ReadOnlyExporter").glob("*.cs")),
        *list((ROOT / "mod" / "STS2Guide.ReadOnlyExporter").glob("*.csproj")),
    ]
    assert_artifact_fresh(required[1], mod_code_sources, label=required[1].name)
    source_manifest = (
        ROOT
        / "mod"
        / "STS2Guide.ReadOnlyExporter"
        / "STS2GuideReadOnlyExporter.json"
    )
    if sha256(required[2]) != sha256(source_manifest):
        raise SystemExit("Mod JSON artifact does not match its owned source")
    pack_sources = list(
        (
            ROOT
            / "mod"
            / "STS2Guide.ReadOnlyExporter"
            / "pack"
        ).rglob("*")
    )
    assert_artifact_fresh(required[3], pack_sources, label=required[3].name)

    listing = archive_listing(EXE)
    lowered = listing.lower()
    for name in FORBIDDEN_ARCHIVE_NAMES:
        if name.lower() in lowered:
            raise SystemExit(f"Forbidden archive member detected: {name}")
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            raise SystemExit(f"Forbidden test dependency detected: {token}")
    for required_member in (
        "sts2-guide-template.db",
        "compatibility.json",
        "state-event.schema.json",
        "advice-event.schema.json",
    ):
        if required_member.lower() not in lowered:
            raise SystemExit(
                f"Required archive member is missing: {required_member}"
            )

    manifest = json.loads(
        (ROOT / "packaging" / "compatibility.json").read_text(
            encoding="utf-8"
        )
    )
    artifacts = required + installers
    release_manifest = {
        "schema_version": 1,
        "guide_version": manifest["guide"]["version"],
        "game_version": manifest["game"]["version"],
        "release_fingerprint": manifest["release_fingerprint"],
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        },
    }
    output = DIST / "release-manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(release_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Public Beta payload audit passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

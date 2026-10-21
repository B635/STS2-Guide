"""Refresh the release identity after the catalog, assembly facts or Guide
parameters changed.

``sqlite.snapshot_id`` in packaging/compatibility.json names the hashes of
data/knowledge.json, data/assembly/facts.json and data/guide_mechanics.json,
and the release fingerprint covers it. This script recomputes both and
writes the new fingerprint everywhere the old one is pinned (the Mod's
protocol constants and the protocol examples). Run it, then rebuild.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import realtime.compatibility as compatibility  # noqa: E402

MANIFEST = ROOT_DIR / "packaging" / "compatibility.json"
PINNED = (
    "mod/STS2Guide.ReadOnlyExporter/ProtocolModels.cs",
    "protocol/advice-event.example.json",
    "protocol/deck-targets-event.example.json",
    "protocol/route-advice-event.example.json",
    "protocol/route-event.example.json",
    "protocol/runtime-status.example.json",
    "protocol/state-event.example.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(path: Path) -> str:
    """Fingerprint of the manifest at ``path`` whatever it currently stores."""

    original = compatibility.compute_release_fingerprint
    captured: dict = {}

    def capture(manifest):
        captured["value"] = original(manifest)
        return manifest.release_fingerprint

    compatibility.compute_release_fingerprint = capture
    try:
        compatibility.load_compatibility_manifest(path)
    finally:
        compatibility.compute_release_fingerprint = original
    return captured["value"]


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    old = manifest["release_fingerprint"]
    manifest["sqlite"]["knowledge_sha256"] = _sha256(ROOT_DIR / "data" / "knowledge.json")
    manifest["sqlite"]["snapshot_id"] = compatibility.sqlite_snapshot_id(
        manifest["sqlite"]["knowledge_sha256"],
        _sha256(ROOT_DIR / "data" / "assembly" / "facts.json"),
        _sha256(ROOT_DIR / "data" / "guide_mechanics.json"),
    )
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    new = _fingerprint(MANIFEST)
    manifest["release_fingerprint"] = new
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compatibility.load_compatibility_manifest(MANIFEST)  # now consistent
    if new != old:
        for relative in PINNED:
            path = ROOT_DIR / relative
            text = path.read_bytes().decode("utf-8")
            if old not in text:
                raise SystemExit(f"{relative} does not pin the previous fingerprint {old}")
            path.write_bytes(text.replace(old, new).encode("utf-8"))
    print(f"snapshot_id {manifest['sqlite']['snapshot_id']}")
    print(f"release_fingerprint {old} -> {new}" if new != old else f"release_fingerprint unchanged {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

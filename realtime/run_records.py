"""Export the game's own run history so a player can send it to the Guide's author.

The game writes one ``.run`` file per finished run under
``%APPDATA%/SlayTheSpire2/steam/<steam id>/[modded/]profile*/saves/history/``.
The files hold the run itself (character, deck, map, choices) and no account
data; only the folder path names the Steam ID. The archive keeps the path
below the Steam ID folder (``[modded/]profileN/saves/history/<file>.run``) and
never the ID itself. Nothing in the game folder is written.

A random collector ID (kept next to the Guide's exchange files) tells exports
from different players apart without saying who they are.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

COLLECTOR_ID_FILE = "collector-id.txt"
ARCHIVE_MANIFEST = "manifest.json"
ARCHIVE_FORMAT = "sts2-guide-run-records/1"
MAX_RUN_BYTES = 8 * 1024 * 1024
_COLLECTOR_RE = re.compile(r"^[0-9a-f]{32}$")


def default_history_root() -> Path:
    app_data = os.getenv("APPDATA")
    base = Path(app_data) if app_data else Path.home() / "AppData" / "Roaming"
    return base / "SlayTheSpire2" / "steam"


def collector_id(exchange_dir: Path) -> str:
    """The anonymous ID of this installation, created on first use."""

    path = exchange_dir / COLLECTOR_ID_FILE
    try:
        value = path.read_text(encoding="ascii").strip()
        if _COLLECTOR_RE.fullmatch(value):
            return value
    except (OSError, UnicodeDecodeError):
        pass
    value = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="ascii")
    return value


def find_run_files(history_root: Path) -> list[tuple[Path, str]]:
    """``(file, archive name)`` for every run file; the name drops the Steam ID folder."""

    found = []
    if not history_root.is_dir():
        return found
    for path in sorted(history_root.glob("*/**/saves/history/*.run")):
        relative = path.relative_to(history_root).parts
        if not path.is_file() or path.stat().st_size > MAX_RUN_BYTES:
            continue
        found.append((path, "/".join(("runs", *relative[1:]))))
    return found


def export_run_records(
    output_dir: Path,
    exchange_dir: Path,
    guide_version: str,
    history_root: Path | None = None,
    now: datetime | None = None,
) -> tuple[Path, int]:
    """Write one zip of every run file; returns its path and the run count."""

    now = now or datetime.now(timezone.utc)
    files = find_run_files(history_root or default_history_root())
    collector = collector_id(exchange_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"STS2-对局记录-{collector[:8]}-{now.strftime('%Y%m%d-%H%M%S')}.zip"
    manifest = {
        "format": ARCHIVE_FORMAT,
        "collector_id": collector,
        "guide_version": guide_version,
        "exported_at": now.isoformat(timespec="seconds"),
        "runs": len(files),
    }
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ARCHIVE_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for path, name in files:
            archive.write(path, name)
    os.replace(temporary, output)
    return output, len(files)


def unpack_collected(archives: Iterable[Path], target: Path) -> list[Path]:
    """Extract collected exports to ``target/<collector>/...``; returns the run files.

    Only ``runs/**.run`` members of a recognised archive are written, under a
    folder named by its collector ID, so a crafted name cannot escape
    ``target``. The same run sent twice lands on the same path.
    """

    written = []
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                manifest = json.loads(archive.read(ARCHIVE_MANIFEST).decode("utf-8"))
            except (KeyError, ValueError):
                continue
            collector = str(manifest.get("collector_id") or "")
            if manifest.get("format") != ARCHIVE_FORMAT or not _COLLECTOR_RE.fullmatch(collector):
                continue
            for member in archive.infolist():
                parts = member.filename.split("/")
                if (member.is_dir() or len(parts) < 2 or parts[0] != "runs" or not parts[-1].endswith(".run")
                        or any(p in ("", ".", "..") or ":" in p or "\\" in p for p in parts)
                        or member.file_size > MAX_RUN_BYTES):
                    continue
                destination = target.joinpath(collector, *parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
                written.append(destination)
    return written

"""Local file bridge used by the read-only STS2 mod."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import ValidationError

from realtime.processor import RealtimeEventProcessor
from realtime.protocol import GameStateEvent


def default_exchange_dir() -> Path:
    configured = os.getenv("STS2_GUIDE_EXCHANGE_DIR")
    if configured:
        return Path(configured).expanduser()
    app_data = os.getenv("APPDATA")
    base = Path(app_data) if app_data else Path.home() / "AppData" / "Roaming"
    return base / "SlayTheSpire2" / "STS2Guide"


def default_input_path() -> Path:
    return default_exchange_dir() / "state-event.json"


def default_events_dir() -> Path:
    return default_exchange_dir() / "events"


def default_output_path() -> Path:
    return default_exchange_dir() / "advice-event.json"


def default_checkpoint_path() -> Path:
    configured = os.getenv("STS2_GUIDE_CHECKPOINT_FILE")
    if configured:
        return Path(configured).expanduser()
    return default_exchange_dir() / "active-run.json"


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    # Bounded retry for transient PermissionError (antivirus, reader
    # holding the file).  Total wait ≤ ~0.75 s; persistent failures
    # after 5 attempts surface as a real error rather than hiding a
    # multi-instance problem.
    deadline = time.monotonic() + 1.0
    last_exc: Optional[PermissionError] = None
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


class GameStateFileBridge:
    def __init__(
        self,
        processor: RealtimeEventProcessor,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        events_dir: Optional[Path] = None,
    ):
        self.processor = processor
        self.input_path = Path(input_path or default_input_path())
        self.output_path = Path(output_path or default_output_path())
        self.events_dir = Path(
            events_dir
            if events_dir is not None
            else self.input_path.parent / "events"
        )
        self._last_fingerprint: Optional[Tuple[int, int]] = None

    def run_once(self) -> Optional[Dict]:
        event_path, queued = self._next_event_path()
        if event_path is None:
            return None

        try:
            raw = json.loads(event_path.read_text(encoding="utf-8"))
            event = GameStateEvent.model_validate(raw)
            result = self.processor.process(event)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            result = {
                "event_id": None,
                "event_type": None,
                "status": "invalid",
                "duplicate": False,
                "state_id": None,
                "decision_id": None,
                "advice": None,
                "message": str(exc),
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        if queued:
            self._acknowledge_event(event_path)
        if result.get("session_cleared"):
            self._clear_exchange_files()
        else:
            _atomic_write_json(self.output_path, result)
        return result

    def _next_event_path(self) -> tuple[Optional[Path], bool]:
        if self.events_dir.exists():
            queued = sorted(self.events_dir.glob("*.json"))
            if queued:
                return queued[0], True
            return None, False

        if not self.input_path.exists():
            return None, False
        stat = self.input_path.stat()
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        if fingerprint == self._last_fingerprint:
            return None, False
        self._last_fingerprint = fingerprint
        return self.input_path, False

    @staticmethod
    def _acknowledge_event(event_path: Path) -> None:
        event_path.unlink(missing_ok=True)

    def _clear_exchange_files(self) -> None:
        for path in (self.input_path, self.output_path):
            path.unlink(missing_ok=True)
        for directory in (self.events_dir,):
            if not directory.exists():
                continue
            for pattern in ("*.json", "*.tmp"):
                for path in directory.glob(pattern):
                    path.unlink(missing_ok=True)
        for directory in (self.events_dir,):
            try:
                directory.rmdir()
            except OSError:
                pass

    def run_forever(self, poll_interval: float = 0.25) -> None:
        while True:
            self.run_once()
            time.sleep(max(0.05, poll_interval))

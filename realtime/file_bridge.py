"""Local file bridge used by the read-only STS2 mod."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import ValidationError

from realtime.processor import (
    RealtimeCompatibilityError,
    RealtimeEventProcessor,
)
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

        raw: dict = {}
        try:
            raw = json.loads(event_path.read_text(encoding="utf-8"))
            event = GameStateEvent.model_validate(raw)
            result = self.processor.process(event)
        except RealtimeCompatibilityError as exc:
            result = {
                "event_id": raw.get("event_id"),
                "event_type": raw.get("event_type"),
                "status": "unsupported",
                "duplicate": False,
                "state_id": None,
                "run_id": raw.get("run_id"),
                "decision_id": raw.get("decision_id"),
                "advice": None,
                "recommendation": None,
                "compatibility_issues": list(exc.reason_codes),
                "advice_disposition": {
                    "action": "clear_all",
                    "run_id": raw.get("run_id"),
                    "decision_id": raw.get("decision_id"),
                },
                "message": str(exc),
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            production_gate = (
                self.processor.compatibility_manifest is not None
            )
            result = {
                "event_id": raw.get("event_id"),
                "event_type": raw.get("event_type"),
                "status": (
                    "unsupported" if production_gate else "invalid"
                ),
                "duplicate": False,
                "state_id": None,
                "run_id": raw.get("run_id"),
                "decision_id": raw.get("decision_id"),
                "advice": None,
                "recommendation": None,
                "compatibility_issues": (
                    ["invalid_production_event"]
                    if production_gate
                    else []
                ),
                "advice_disposition": {
                    "action": (
                        "clear_all" if production_gate else "preserve"
                    ),
                    "run_id": raw.get("run_id"),
                    "decision_id": raw.get("decision_id"),
                },
                "message": str(exc),
                "received_at": datetime.now(timezone.utc).isoformat(),
            }
        # Processing may already have advanced the active-run checkpoint.
        # Keep a queued event until its advice side effect also succeeds: a
        # retry then receives the checkpoint replay result and safely repeats
        # the idempotent publish/owner-checked clear before acknowledging it.
        self._apply_advice_disposition(
            result,
            protected_event_path=(event_path if queued else None),
        )
        if queued:
            self._acknowledge_event(event_path)
            if (
                (result.get("advice_disposition") or {}).get("action")
                == "clear_run"
            ):
                try:
                    self.events_dir.rmdir()
                except OSError:
                    pass
        return result

    def _apply_advice_disposition(
        self,
        result: dict,
        *,
        protected_event_path: Path | None = None,
    ) -> None:
        disposition = result.get("advice_disposition") or {}
        action = disposition.get("action", "preserve")
        owner_run_id = disposition.get("run_id")
        owner_decision_id = disposition.get("decision_id")
        if action == "preserve":
            return
        if action == "publish":
            recommendation = result.get("recommendation") or {}
            if (
                not owner_run_id
                or not owner_decision_id
                or result.get("run_id") != owner_run_id
                or result.get("decision_id") != owner_decision_id
                or recommendation.get("decision_id") != owner_decision_id
            ):
                raise ValueError("advice publish owner does not match result")
            _atomic_write_json(self.output_path, result)
            return
        if action == "clear":
            if owner_run_id and owner_decision_id:
                self._clear_advice_if_owned(
                    owner_run_id,
                    owner_decision_id,
                )
            return
        if action == "clear_run":
            if owner_run_id:
                self._clear_run_artifacts(
                    owner_run_id,
                    protected_event_path=protected_event_path,
                )
            return
        if action == "clear_all":
            self._clear_advice_file()
            return
        raise ValueError(f"Unknown advice disposition: {action}")

    def _visible_advice_owner(self) -> tuple[str, str] | None:
        if not self.output_path.exists():
            return None
        try:
            visible = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        visible_decision_id = (
            visible.get("decision_id")
            or visible.get("recommendation", {}).get("decision_id")
        )
        visible_run_id = visible.get("run_id")
        if not visible_run_id or not visible_decision_id:
            return None
        return str(visible_run_id), str(visible_decision_id)

    def _clear_advice_if_owned(
        self,
        run_id: str,
        decision_id: str,
    ) -> None:
        if self._visible_advice_owner() == (run_id, decision_id):
            self._clear_advice_file()

    def _next_event_path(self) -> tuple[Optional[Path], bool]:
        if self.events_dir.exists():
            queued = sorted(
                self.events_dir.glob("*.json"),
                key=self._event_order_key,
            )
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
    def _event_order_key(path: Path) -> tuple[int, int, str]:
        """Order the global spool by production time, not run-prefixed name."""
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            emitted_text = str(raw["emitted_at"])
            if emitted_text.endswith("Z"):
                emitted_text = f"{emitted_text[:-1]}+00:00"
            emitted_at = datetime.fromisoformat(emitted_text)
            if emitted_at.tzinfo is None:
                raise ValueError("naive emitted_at")
            emitted_ns = int(
                emitted_at.astimezone(timezone.utc).timestamp()
                * 1_000_000_000
            )
            return emitted_ns, mtime_ns, path.name
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return mtime_ns, mtime_ns, path.name

    @staticmethod
    def _acknowledge_event(event_path: Path) -> None:
        event_path.unlink(missing_ok=True)

    @staticmethod
    def _artifact_run_id(path: Path) -> str | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        run_id = payload.get("run_id")
        return str(run_id) if run_id else None

    def _clear_run_artifacts(
        self,
        run_id: str,
        *,
        protected_event_path: Path | None = None,
    ) -> None:
        visible_owner = self._visible_advice_owner()
        if visible_owner is not None and visible_owner[0] == run_id:
            self._clear_advice_file()
        if (
            self.input_path.exists()
            and self._artifact_run_id(self.input_path) == run_id
        ):
            self.input_path.unlink(missing_ok=True)
        if self.events_dir.exists():
            for path in self.events_dir.glob("*.json"):
                if (
                    protected_event_path is not None
                    and path == protected_event_path
                ):
                    continue
                if self._artifact_run_id(path) == run_id:
                    path.unlink(missing_ok=True)
            try:
                self.events_dir.rmdir()
            except OSError:
                pass

    def _clear_advice_file(self) -> None:
        self.output_path.unlink(missing_ok=True)
        self.output_path.with_name(
            f"{self.output_path.name}.tmp"
        ).unlink(missing_ok=True)

    def run_forever(
        self,
        poll_interval: float = 0.25,
        stop_event: threading.Event | None = None,
    ) -> None:
        stop = stop_event or threading.Event()
        while not stop.is_set():
            self.run_once()
            stop.wait(max(0.05, poll_interval))

"""Single-file checkpoint for the currently active STS2 run.

The checkpoint is recovery state, not an event history.  It keeps only the
latest run snapshot, the latest processed event, and (while a card reward is
open) the parent decision required to match its close event.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


CHECKPOINT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(payload: Dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class ActiveRunCheckpointStore:
    """Persist exactly one replaceable active-run checkpoint."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(self) -> Optional[Dict]:
        with self._lock:
            if not self.path.exists():
                return None
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Active-run checkpoint is unreadable: {exc}"
                ) from exc
            if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
                raise ValueError("Unsupported active-run checkpoint version")
            return copy.deepcopy(payload)

    def find_event(self, event_id: str) -> Optional[Dict]:
        checkpoint = self.load()
        if checkpoint is None:
            return None
        for key in ("latest_event", "open_decision"):
            record = checkpoint.get(key)
            if record and record.get("event_id") == event_id:
                return copy.deepcopy(record)
        return None

    def replay_result(self, event: Dict) -> Optional[Dict]:
        """Return a cached result, or reject a stale/colliding observation."""
        checkpoint = self.load()
        if checkpoint is None or checkpoint["run_id"] != event["run_id"]:
            return None

        content_hash = _content_hash(event)
        for key in ("latest_event", "open_decision"):
            record = checkpoint.get(key)
            if not record or record.get("event_id") != event["event_id"]:
                continue
            if record.get("content_hash") != content_hash:
                raise ValueError(
                    "Event identity collision: the same event ID was "
                    "reused with a different payload"
                )
            result = copy.deepcopy(record.get("result"))
            if result is not None:
                result["duplicate"] = True
            return result

        if int(event["sequence"]) <= int(checkpoint["last_sequence"]):
            raise ValueError(
                "Stale event: sequence is not newer than the active-run "
                "checkpoint"
            )
        return None

    def update(self, event: Dict, result: Dict) -> Dict:
        """Replace the active checkpoint after an event is processed."""
        with self._lock:
            previous = self.load()
            same_run = (
                previous is not None
                and previous.get("run_id") == event["run_id"]
            )
            map_context = (
                previous.get("map_context") if same_run else None
            )
            open_decision = (
                previous.get("open_decision") if same_run else None
            )

            event_record = {
                "event_id": event["event_id"],
                "run_id": event["run_id"],
                "event_type": event["event_type"],
                "sequence": int(event["sequence"]),
                "source": event.get("source"),
                "game_version": event.get("game_version"),
                "emitted_at": event["emitted_at"],
                "processed_at": result.get("processed_at"),
                "status": result.get("status"),
                "content_hash": _content_hash(event),
                "state_id": result.get("state_id"),
                "decision_id": result.get("decision_id"),
                "payload": copy.deepcopy(event),
                "result": copy.deepcopy(result),
            }
            event_type = event["event_type"]
            if event_type == "card_reward":
                open_decision = copy.deepcopy(event_record)
            elif event_type == "decision_closed":
                parent_event_id = event.get("parent_event_id")
                if (
                    open_decision is not None
                    and open_decision.get("event_id") == parent_event_id
                ):
                    open_decision = None
            elif event_type == "map_choice":
                map_context = copy.deepcopy(event.get("map_context"))

            checkpoint = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "run_id": event["run_id"],
                "game_version": event.get("game_version"),
                "last_sequence": int(event["sequence"]),
                "last_event_id": event["event_id"],
                "updated_at": _utc_now(),
                "state": copy.deepcopy(event["state"]),
                "map_context": map_context,
                "open_decision": open_decision,
                "latest_event": event_record,
            }
            _atomic_write_json(self.path, checkpoint)
            return copy.deepcopy(checkpoint)

    def clear(self, run_id: Optional[str] = None) -> bool:
        with self._lock:
            if not self.path.exists():
                return False
            if run_id is not None:
                checkpoint = self.load()
                if checkpoint is not None and checkpoint["run_id"] != run_id:
                    return False
            self.path.unlink(missing_ok=True)
            self.path.with_name(f"{self.path.name}.tmp").unlink(
                missing_ok=True
            )
            return True

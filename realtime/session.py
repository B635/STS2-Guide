"""Run-scoped transient state for realtime game observations.

Nothing in this store is written to SQLite. It exists only for the lifetime
of the local bridge/API process and is cleared when a run ends.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TransientSessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: Dict[str, Dict] = {}
        self._sequence_ids: Dict[tuple[str, int], str] = {}
        self._latest_event_id: Optional[str] = None

    def claim(self, event: Dict) -> tuple[bool, Dict]:
        payload_json = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        content_hash = hashlib.sha256(
            payload_json.encode("utf-8")
        ).hexdigest()
        sequence_key = (event["run_id"], int(event["sequence"]))

        with self._lock:
            existing_id = self._sequence_ids.get(sequence_key)
            existing = self._events.get(event["event_id"])
            if existing is None and existing_id is not None:
                existing = self._events[existing_id]
            if existing is not None:
                if existing["content_hash"] != content_hash:
                    raise ValueError(
                        "Event identity collision: the same event ID or "
                        "run sequence was reused with a different payload"
                    )
                return False, copy.deepcopy(existing)

            record = {
                "event_id": event["event_id"],
                "run_id": event["run_id"],
                "sequence": int(event["sequence"]),
                "event_type": event["event_type"],
                "schema_version": int(event["schema_version"]),
                "source": event["source"],
                "game_version": event.get("game_version"),
                "emitted_at": event["emitted_at"],
                "received_at": _utc_now(),
                "processed_at": None,
                "content_hash": content_hash,
                "status": "received",
                "state_id": None,
                "decision_id": None,
                "payload": copy.deepcopy(event),
                "result": None,
                "error": None,
            }
            self._events[event["event_id"]] = record
            self._sequence_ids[sequence_key] = event["event_id"]
            self._latest_event_id = event["event_id"]
            return True, copy.deepcopy(record)

    def complete(
        self,
        event_id: str,
        *,
        status: str,
        result: Dict,
        decision_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self._events.get(event_id)
            if record is None:
                raise ValueError("Transient game-state event was not claimed")
            record["status"] = status
            record["processed_at"] = _utc_now()
            record["decision_id"] = decision_id
            record["result"] = copy.deepcopy(result)
            record["error"] = error
            self._latest_event_id = event_id

    def load(self, event_id: str) -> Optional[Dict]:
        with self._lock:
            record = self._events.get(event_id)
            return copy.deepcopy(record) if record is not None else None

    def latest(self) -> Optional[Dict]:
        with self._lock:
            if self._latest_event_id is None:
                return None
            record = self._events.get(self._latest_event_id)
            return copy.deepcopy(record) if record is not None else None

    def clear_run(self, run_id: str) -> None:
        with self._lock:
            event_ids = [
                event_id
                for event_id, record in self._events.items()
                if record["run_id"] == run_id
            ]
            for event_id in event_ids:
                record = self._events.pop(event_id)
                self._sequence_ids.pop(
                    (record["run_id"], record["sequence"]),
                    None,
                )
            if self._latest_event_id in event_ids:
                self._latest_event_id = None

    def record_outcome(
        self,
        parent_event_id: str,
        chosen_option: str,
    ) -> None:
        with self._lock:
            parent = self._events.get(parent_event_id)
            if parent is None or parent.get("result") is None:
                raise ValueError("Parent realtime decision is unavailable")
            parent["result"]["outcome"] = {
                "chosen_option": chosen_option,
                "recorded_at": _utc_now(),
            }

    def count(self) -> int:
        with self._lock:
            return len(self._events)

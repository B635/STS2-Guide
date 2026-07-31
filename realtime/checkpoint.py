"""Single-file checkpoint for the currently active STS2 run.

The checkpoint is recovery state, not an event history.  It keeps only the
latest run snapshot, the latest processed event, one current generic decision,
and one recently closed decision tombstone for idempotency.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


CHECKPOINT_VERSION = 3
_ROUTE_MODES = frozenset({"balanced", "survival", "growth"})
LOGGER = logging.getLogger("sts2-guide")


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


def checkpoint_record_payload_matches(record: object) -> bool:
    """Return whether a checkpoint record still owns its exact payload."""

    if not isinstance(record, dict):
        return False
    payload = record.get("payload")
    content_hash = record.get("content_hash")
    return (
        isinstance(payload, dict)
        and isinstance(content_hash, str)
        and content_hash == _content_hash(payload)
    )


def _valid_guide_preferences(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"route_mode"}
        and value.get("route_mode") in _ROUTE_MODES
    )


def _parse_emitted_at(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "")
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("emitted_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def can_supersede_checkpoint(checkpoint: Dict, event: Dict) -> bool:
    """Return whether ``event`` is a provably newer run start.

    Sequence 1 alone is not sufficient: after a Host restart, an old queued
    run-start event could otherwise reclaim the one active-run checkpoint.
    """
    if (
        int(event["sequence"]) != 1
        or event["event_type"] in {"decision_closed", "run_ended"}
    ):
        return False
    latest = checkpoint.get("latest_event") or {}
    previous_emitted_at = latest.get("emitted_at")
    if previous_emitted_at is None:
        previous_emitted_at = (latest.get("payload") or {}).get("emitted_at")
    if previous_emitted_at is None:
        return False
    try:
        return _parse_emitted_at(event.get("emitted_at")) > _parse_emitted_at(
            previous_emitted_at
        )
    except (TypeError, ValueError):
        return False


def _atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
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
            except (OSError, json.JSONDecodeError):
                self._discard_invalid_locked("unreadable")
                return None
            if not isinstance(payload, dict):
                self._discard_invalid_locked("invalid_root")
                return None
            version = payload.get("checkpoint_version")
            if version not in (1, 2, CHECKPOINT_VERSION):
                self._discard_invalid_locked("unsupported_version")
                return None
            if version == 1:
                payload["current_decision"] = payload.pop(
                    "open_decision",
                    None,
                )
                payload["closed_decision"] = None
            if version < CHECKPOINT_VERSION:
                # Old checkpoints remain readable for offline replay, but
                # they cannot satisfy the v8 live-release recovery gate.
                payload["checkpoint_version"] = CHECKPOINT_VERSION
                payload["guide_preferences"] = None
            return copy.deepcopy(payload)

    def _discard_invalid_locked(self, reason_code: str) -> None:
        """Remove unusable recovery state without retaining run history."""

        LOGGER.warning(
            "Discarded invalid active-run checkpoint (%s); "
            "route mode defaults to balanced.",
            reason_code,
        )
        self.path.unlink(missing_ok=True)
        self.path.with_name(f"{self.path.name}.tmp").unlink(
            missing_ok=True
        )

    def find_event(self, event_id: str) -> Optional[Dict]:
        checkpoint = self.load()
        if checkpoint is None:
            return None
        for key in ("latest_event", "current_decision", "closed_decision"):
            record = checkpoint.get(key)
            if record and record.get("event_id") == event_id:
                return copy.deepcopy(record)
        return None

    def find_current_decision(
        self,
        *,
        run_id: str,
        decision_id: str,
        parent_event_id: str,
    ) -> Optional[Dict]:
        checkpoint = self.load()
        if checkpoint is None or checkpoint.get("run_id") != run_id:
            return None
        record = checkpoint.get("current_decision")
        if (
            record is None
            or record.get("decision_id") != decision_id
            or record.get("event_id") != parent_event_id
        ):
            return None
        return copy.deepcopy(record)

    def find_closed_decision(
        self,
        *,
        run_id: str,
        decision_id: str,
    ) -> Optional[Dict]:
        checkpoint = self.load()
        if checkpoint is None or checkpoint.get("run_id") != run_id:
            return None
        record = checkpoint.get("closed_decision")
        if record is None or record.get("decision_id") != decision_id:
            return None
        return copy.deepcopy(record)

    def replay_result(
        self,
        event: Dict,
        *,
        expected_release_fingerprint: str | None = None,
    ) -> Optional[Dict]:
        """Return a cached result, or reject a stale/colliding observation."""
        checkpoint = self.load()
        if checkpoint is None:
            return None
        if checkpoint["run_id"] != event["run_id"]:
            if not can_supersede_checkpoint(checkpoint, event):
                raise ValueError(
                    "Stale or non-newer event from another run cannot "
                    "replace the active-run checkpoint"
                )
            return None

        content_hash = _content_hash(event)
        for key in ("latest_event", "current_decision", "closed_decision"):
            record = checkpoint.get(key)
            if not record or record.get("event_id") != event["event_id"]:
                continue
            result = copy.deepcopy(record.get("result"))
            if expected_release_fingerprint is not None and (
                checkpoint.get("release_fingerprint")
                != expected_release_fingerprint
                or (record.get("payload") or {}).get(
                    "release_fingerprint"
                )
                != expected_release_fingerprint
                or (result or {}).get("compatibility", {}).get(
                    "release_fingerprint"
                )
                != expected_release_fingerprint
                or not _valid_guide_preferences(
                    checkpoint.get("guide_preferences")
                )
                or checkpoint.get("guide_preferences")
                != event.get("guide_preferences")
                or checkpoint.get("guide_preferences")
                != (record.get("payload") or {}).get(
                    "guide_preferences"
                )
                or checkpoint.get("guide_preferences")
                != (result or {}).get("guide_preferences")
            ):
                # Release identity is checked before the event content hash:
                # a new Mod intentionally changes the v7 payload fingerprint
                # while retaining the stable queued event ID.  That is an
                # upgrade/recompute boundary, not an identity collision.
                return None
            if record.get("content_hash") != content_hash:
                raise ValueError(
                    "Event identity collision: the same event ID was "
                    "reused with a different payload"
                )
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
            guide_preferences = copy.deepcopy(
                event.get("guide_preferences")
                or result.get("guide_preferences")
            )
            if not _valid_guide_preferences(guide_preferences):
                raise ValueError(
                    "active-run checkpoint requires valid guide_preferences"
                )
            if (
                event.get("guide_preferences") is not None
                and result.get("guide_preferences") is not None
                and event.get("guide_preferences")
                != result.get("guide_preferences")
            ):
                raise ValueError(
                    "event/result guide_preferences mismatch"
                )
            previous = self.load()
            if (
                previous is not None
                and previous.get("run_id") != event["run_id"]
                and not can_supersede_checkpoint(previous, event)
            ):
                raise ValueError(
                    "Stale or non-newer event from another run cannot "
                    "replace the active-run checkpoint"
                )
            same_run = (
                previous is not None
                and previous.get("run_id") == event["run_id"]
                and previous.get("release_fingerprint")
                == event.get("release_fingerprint")
            )
            map_context = (
                previous.get("map_context") if same_run else None
            )
            current_decision = (
                previous.get("current_decision") if same_run else None
            )
            closed_decision = (
                previous.get("closed_decision") if same_run else None
            )

            event_record = {
                "event_id": event["event_id"],
                "run_id": event["run_id"],
                "event_type": event["event_type"],
                "sequence": int(event["sequence"]),
                "source": event.get("source"),
                "game_version": event.get("game_version"),
                "release_fingerprint": event.get(
                    "release_fingerprint"
                ),
                "guide_preferences": copy.deepcopy(
                    guide_preferences
                ),
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
            decision_phase = result.get("decision_phase")
            # Map observation and decision lifecycle are independent state
            # transforms.  A route_choice carries both in one event and must
            # update both before this single atomic replacement.
            if event_type in (
                "card_reward",
                "route_choice",
                "merchant",
                "rest_site",
                "neow_choice",
                "event_choice",
                "deck_edit",
                "map_choice",
            ):
                # These are authoritative observation events.  An explicit
                # missing map clears the previous revision rather than
                # silently pairing an old origin with the new state.
                map_context = copy.deepcopy(event.get("map_context"))
            if decision_phase in ("opened", "updated"):
                previous_current = current_decision
                previous_closed = closed_decision
                current_decision = copy.deepcopy(event_record)
                active_binding = result.get("active_child_binding")
                if active_binding is not None:
                    parent_close = (
                        previous_closed
                        if decision_phase == "opened"
                        else (
                            (previous_current or {}).get("parent_close")
                        )
                    )
                    if parent_close is None:
                        raise ValueError(
                            "active child binding requires parent close provenance"
                        )
                    current_decision["parent_close"] = copy.deepcopy(
                        parent_close
                    )
                closed_decision = None
            elif event_type == "decision_closed":
                decision_id = result.get("decision_id")
                if (
                    current_decision is not None
                    and current_decision.get("decision_id") == decision_id
                ):
                    parent_observation = copy.deepcopy(current_decision)
                    current_decision = None
                    closed_decision = copy.deepcopy(event_record)
                    closed_decision["decision_id"] = decision_id
                    closed_decision["outcome"] = copy.deepcopy(
                        event.get("outcome")
                    )
                    closed_decision["parent_observation"] = (
                        parent_observation
                    )
                elif (
                    closed_decision is not None
                    and closed_decision.get("decision_id") == decision_id
                ):
                    # A later idempotent close must not erase the original
                    # parent observation used to verify nested decisions.
                    pass
                elif decision_id is not None:
                    closed_decision = copy.deepcopy(event_record)
                    closed_decision["decision_id"] = decision_id
                    closed_decision["outcome"] = copy.deepcopy(
                        event.get("outcome")
                    )

            checkpoint = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "run_id": event["run_id"],
                "game_version": event.get("game_version"),
                "release_fingerprint": event.get(
                    "release_fingerprint"
                ),
                "guide_preferences": guide_preferences,
                "last_sequence": int(event["sequence"]),
                "last_event_id": event["event_id"],
                "updated_at": _utc_now(),
                "state": copy.deepcopy(event["state"]),
                "map_context": map_context,
                "current_decision": current_decision,
                "closed_decision": closed_decision,
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

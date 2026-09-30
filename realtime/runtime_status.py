"""Small owner-scoped liveness publication shared by tray and standalone worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

from realtime.file_bridge import _atomic_write_json


class RuntimeStatusPublisher:
    def __init__(self, path: Path, release_fingerprint: str) -> None:
        self.path = path
        self.release_fingerprint = release_fingerprint
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self._lease = None

    def _acquire(self) -> bool:
        if self._lease is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lease = self.path.with_suffix(".lock").open("a+b")
        try:
            lease.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lease.close()
            return False
        self._lease = lease
        return True

    def _release(self) -> None:
        if self._lease is not None:
            self._lease.close()  # OS releases the lease, including on process death.
            self._lease = None

    def update(self, state: str, game_running: bool | None, worker_running: bool,
               compatibility_reasons=(), activation_count: int = 0) -> None:
        if not self._acquire():
            return  # Another live publisher owns this status; never replace its session.
        self.sequence += 1
        try:
            _atomic_write_json(self.path, {
                "schema_version": 2,
                "session_id": self.session_id,
                "heartbeat_sequence": self.sequence,
                "release_fingerprint": self.release_fingerprint,
                "state": state,
                "game_running": game_running,
                "worker_running": worker_running,
                "compatibility_reasons": list(compatibility_reasons),
                "activation_count": activation_count,
            })
        finally:
            # A waiting tray must not prevent the developer worker from publishing.
            # A running owner retains its lease until it stops or exits.
            if state != "RUNNING" or not worker_running:
                self._release()

    def close(self) -> None:
        if not self._acquire():
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(current, dict) and current.get("session_id") == self.session_id:
                self.path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        finally:
            self._release()

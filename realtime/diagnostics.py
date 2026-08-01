"""Strictly whitelisted, local-only diagnostic archive export."""
from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


MAX_LOG_LINES = 200
MAX_LOG_LINE_CHARS = 1000
MAX_LOG_BYTES = 64 * 1024
_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.+:-]{1,120}$")
_SAFE_REASON_RE = re.compile(r"^[a-z0-9_.-]{1,80}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_LOG_EVENTS = (
    ("public beta controller started.", "controller_started"),
    ("packaged host startup check passed.", "startup_check_passed"),
    ("realtime worker initialization failed.", "worker_start_failed"),
    ("realtime worker stopped unexpectedly.", "worker_stopped_unexpectedly"),
    ("realtime worker cleanup failed.", "worker_cleanup_failed"),
    ("visible advice could not be cleared.", "advice_clear_failed"),
    ("sts2 guide controller failed to initialize.", "controller_init_failed"),
    ("sts2 guide background host failed to initialize.", "worker_init_failed"),
    ("p0 host stopped unexpectedly.", "worker_stopped_unexpectedly"),
    ("p0 host stopped.", "worker_stopped"),
    ("local compatibility gate is closed:", "compatibility_gate_closed"),
    ("discarded invalid active-run checkpoint", "checkpoint_discarded"),
)


@dataclass(frozen=True)
class DiagnosticSnapshot:
    versions: Mapping[str, str]
    release_fingerprint: str
    capabilities: Mapping[str, str]
    status: str
    reason_codes: tuple[str, ...]
    artifact_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        _validate_safe_mapping(self.versions, hashes=False)
        _validate_safe_mapping(self.capabilities, hashes=False)
        _validate_safe_mapping(self.artifact_hashes, hashes=True)
        if not _SHA256_RE.fullmatch(self.release_fingerprint):
            raise ValueError("invalid diagnostic release fingerprint")
        if not _SAFE_KEY_RE.fullmatch(self.status):
            raise ValueError("invalid diagnostic status")
        if any(not _SAFE_REASON_RE.fullmatch(code) for code in self.reason_codes):
            raise ValueError("invalid diagnostic reason code")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("diagnostic reason codes must be unique")

    def as_dict(self) -> dict:
        return {
            "versions": dict(sorted(self.versions.items())),
            "release_fingerprint": self.release_fingerprint,
            "capabilities": dict(sorted(self.capabilities.items())),
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
        }


def _validate_safe_mapping(values: Mapping[str, str], *, hashes: bool) -> None:
    if not isinstance(values, Mapping):
        raise ValueError("diagnostic field must be a mapping")
    if len(values) > 64:
        raise ValueError("diagnostic mapping is too large")
    for key, value in values.items():
        if not isinstance(key, str) or not _SAFE_KEY_RE.fullmatch(key):
            raise ValueError("invalid diagnostic field name")
        if not isinstance(value, str):
            raise ValueError("invalid diagnostic field value")
        if hashes:
            if not _SHA256_RE.fullmatch(value):
                raise ValueError("invalid diagnostic artifact hash")
        elif not _SAFE_VALUE_RE.fullmatch(value):
            raise ValueError("unsafe diagnostic field value")


def sanitize_log_lines(
    lines: Iterable[str],
    *,
    sensitive_values: Iterable[str] = (),
) -> str:
    """Export only normalized lifecycle events from a strict allowlist.

    Production logs may contain arbitrary exception text or gameplay context.
    A denylist can never prove those values absent, so no original log text is
    copied into diagnostics.  Recognized lifecycle messages become fixed event
    tokens; every other line is omitted.
    """

    # Retain the argument for API compatibility, while never depending on a
    # caller-provided denylist for confidentiality.
    tuple(sensitive_values)
    safe_lines: list[str] = []
    used_bytes = 0
    for raw in lines:
        if len(safe_lines) >= MAX_LOG_LINES:
            break
        lowered = str(raw).casefold()
        event = next(
            (
                normalized
                for marker, normalized in _SAFE_LOG_EVENTS
                if marker in lowered
            ),
            None,
        )
        if event is None:
            continue
        text = f"event={event}"[:MAX_LOG_LINE_CHARS]
        encoded = (text + "\n").encode("utf-8", errors="replace")
        if used_bytes + len(encoded) > MAX_LOG_BYTES:
            break
        safe_lines.append(text)
        used_bytes += len(encoded)
    return "\n".join(safe_lines) + ("\n" if safe_lines else "")


def export_diagnostics_zip(
    path: str | Path,
    snapshot: DiagnosticSnapshot,
    log_lines: Iterable[str],
    *,
    sensitive_values: Iterable[str] = (),
) -> Path:
    """Write a two-entry archive; no arbitrary files can enter the zip."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    metadata = json.dumps(
        snapshot.as_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    logs = sanitize_log_lines(
        log_lines,
        sensitive_values=sensitive_values,
    )
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("diagnostic.json", metadata)
            archive.writestr("guide.log", logs)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target

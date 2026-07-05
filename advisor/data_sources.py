"""Optional local-only recommendation data sources."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_TIERS = frozenset({"S", "A", "B", "C", "D", "F"})
DEFAULT_MAX_AGE_DAYS = 45


@dataclass(frozen=True)
class LocalTierEntry:
    card_id: str
    character_id: str
    tier: str


@dataclass(frozen=True)
class LocalCardTierSource:
    """Validated local tier snapshot keyed only by stable IDs."""

    status: str
    path: Path
    source: Optional[str]
    source_url: Optional[str]
    captured_at: Optional[str]
    entries: Dict[tuple[str, str], LocalTierEntry]
    warning: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.status == "loaded"

    def find(
        self,
        card_id: str,
        character_id: str,
    ) -> Optional[LocalTierEntry]:
        if not self.available:
            return None
        key = (
            str(card_id).strip().upper(),
            str(character_id).strip().upper(),
        )
        return self.entries.get(key)

    def diagnostics(self) -> Dict:
        return {
            "status": self.status,
            "path": str(self.path),
            "source": self.source,
            "source_url": self.source_url,
            "captured_at": self.captured_at,
            "entry_count": len(self.entries),
            "warning": self.warning,
        }


def _unavailable(
    path: Path,
    status: str,
    warning: str,
) -> LocalCardTierSource:
    return LocalCardTierSource(
        status=status,
        path=path,
        source=None,
        source_url=None,
        captured_at=None,
        entries={},
        warning=warning,
    )


def load_local_card_tiers(
    path: str | Path,
    *,
    now: Optional[datetime] = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> LocalCardTierSource:
    """Load a manual local snapshot, degrading safely on every error."""
    source_path = Path(path)
    if not source_path.exists():
        return _unavailable(
            source_path,
            "missing",
            "Optional local card tier file was not found.",
        )

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")

        source = str(payload.get("source") or "").strip()
        if not source:
            raise ValueError("source is required")

        source_url = str(payload.get("source_url") or "").strip()
        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError("source_url must be an HTTPS URL")

        captured_at = str(payload.get("captured_at") or "").strip()
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError("captured_at must include a timezone")
        reference_now = now or datetime.now(timezone.utc)
        if reference_now.tzinfo is None:
            reference_now = reference_now.replace(tzinfo=timezone.utc)
        if captured > reference_now + timedelta(minutes=5):
            raise ValueError("captured_at cannot be in the future")
        if reference_now - captured > timedelta(days=max_age_days):
            return LocalCardTierSource(
                status="stale",
                path=source_path,
                source=source,
                source_url=source_url,
                captured_at=captured.isoformat(),
                entries={},
                warning=(
                    f"Local tier snapshot is older than "
                    f"{max_age_days} days."
                ),
            )

        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("entries must be a list")

        entries: Dict[tuple[str, str], LocalTierEntry] = {}
        for index, raw in enumerate(raw_entries):
            if not isinstance(raw, dict):
                raise ValueError(f"entries[{index}] must be an object")
            card_id = str(raw.get("card_id") or "").strip().upper()
            character_id = (
                str(raw.get("character_id") or "").strip().upper()
            )
            tier = str(raw.get("tier") or "").strip().upper()
            if not card_id or not character_id:
                raise ValueError(
                    f"entries[{index}] requires stable card_id and "
                    "character_id"
                )
            if tier not in SUPPORTED_TIERS:
                raise ValueError(
                    f"entries[{index}] has unsupported tier {tier!r}"
                )
            key = (card_id, character_id)
            if key in entries:
                raise ValueError(
                    f"duplicate local tier entry for {card_id}/"
                    f"{character_id}"
                )
            entries[key] = LocalTierEntry(
                card_id=card_id,
                character_id=character_id,
                tier=tier,
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _unavailable(
            source_path,
            "invalid",
            f"Optional local card tier file was ignored: {exc}",
        )

    return LocalCardTierSource(
        status="loaded",
        path=source_path,
        source=source,
        source_url=source_url,
        captured_at=captured.isoformat(),
        entries=entries,
    )

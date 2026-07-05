"""Fetch the small public Spire Codex community-guide collection."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import requests


API_BASE = "https://spire-codex.com/api"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "data", "guides.json")
USER_AGENT = "STS2-Guide/1.0 (+https://spire-codex.com)"


def fetch_json(session: requests.Session, url: str, retries: int = 3) -> object:
    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=(5, 30))
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", attempt + 1))
                time.sleep(max(retry_after, 1.0))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def fetch_guides(delay_seconds: float = 1.1) -> List[Dict]:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
    )

    index = fetch_json(session, f"{API_BASE}/guides")
    if not isinstance(index, list):
        raise RuntimeError("Spire Codex /api/guides returned a non-list payload")

    guides: List[Dict] = []
    for position, summary in enumerate(index):
        if not isinstance(summary, dict):
            continue
        slug = str(summary.get("slug") or summary.get("id") or "").strip()
        if not slug:
            continue
        if position:
            time.sleep(max(delay_seconds, 0.0))
        detail = fetch_json(session, f"{API_BASE}/guides/{slug}")
        if not isinstance(detail, dict) or not str(detail.get("content") or "").strip():
            raise RuntimeError(f"Guide {slug!r} has no usable content")
        guides.append(detail)

    guides.sort(key=lambda guide: str(guide.get("slug") or guide.get("id") or ""))
    return guides


def write_snapshot(path: str, guides: List[Dict]) -> None:
    payload = {
        "schema_version": 1,
        "source": {
            "name": "Spire Codex",
            "api_url": f"{API_BASE}/guides",
            "terms_url": "https://github.com/ptrlrd/spire-codex/blob/main/API_TERMS.md",
            "language": "eng",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "usage_note": (
                "Community API snapshot. Preserve guide author and original source attribution; "
                "confirm author permission before commercial redistribution."
            ),
        },
        "guides": guides,
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Spire Codex community guides.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--delay", type=float, default=1.1)
    args = parser.parse_args()

    guides = fetch_guides(delay_seconds=args.delay)
    write_snapshot(args.output, guides)
    total_chars = sum(len(str(guide.get("content") or "")) for guide in guides)
    print(f"Fetched {len(guides)} guide(s), {total_chars} content characters -> {args.output}")


if __name__ == "__main__":
    main()

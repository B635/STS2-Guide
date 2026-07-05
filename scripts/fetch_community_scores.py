"""Fetch attributable aggregate entity scores from the Spire Codex API.

Only IDs present in the local official catalog are retained. This drops modded
entities and keeps the snapshot small enough to review and version.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import requests


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import COMMUNITY_SCORES_FILE, KNOWLEDGE_FILE


BASE_URL = "https://spire-codex.com/api"
ENTITY_TYPES = ("cards", "relics", "potions")
REQUEST_DELAY_SECONDS = 1.1


def _get_json(session: requests.Session, url: str):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response.json()


def fetch_scores(
    knowledge_path: str = KNOWLEDGE_FILE,
    output_path: str = COMMUNITY_SCORES_FILE,
    base_url: str = BASE_URL,
) -> Dict[str, int]:
    knowledge = json.loads(Path(knowledge_path).read_text(encoding="utf-8"))
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "STS2-Guide/1.0 community research "
                "(https://github.com/; rate-limited)"
            )
        }
    )

    endpoints = {
        entity_type: f"{base_url}/runs/scores/{entity_type}"
        for entity_type in ENTITY_TYPES
    }
    filtered_entities = {}
    counts = {}
    for entity_type in ENTITY_TYPES:
        official_ids = {
            str(item.get("id") or "")
            for item in knowledge.get(entity_type, [])
            if item.get("id")
        }
        raw = _get_json(session, endpoints[entity_type])
        filtered = {}
        if isinstance(raw, dict):
            for entity_id in sorted(official_ids):
                values = raw.get(entity_id)
                if not isinstance(values, dict):
                    continue
                picks = int(values.get("picks") or 0)
                score = values.get("score")
                if picks <= 0 or not isinstance(score, (int, float)):
                    continue
                filtered[entity_id] = {
                    "score": float(score),
                    "elo": values.get("elo"),
                    "picks": picks,
                    "wins": int(values.get("wins") or 0),
                    "win_rate": values.get("win_rate"),
                }
        filtered_entities[entity_type] = filtered
        counts[entity_type] = len(filtered)

    versions_url = f"{base_url}/runs/versions"
    versions_payload = _get_json(session, versions_url)
    payload = {
        "schema_version": 1,
        "source": {
            "id": "spire_codex_api",
            "name": "Spire Codex API",
            "base_url": base_url,
            "terms_url": (
                "https://github.com/ptrlrd/spire-codex/blob/main/API_TERMS.md"
            ),
            "endpoints": endpoints,
        },
        "data_type": "aggregate_scores",
        "game_version": "mixed",
        "observed_versions": (
            versions_payload.get("versions", [])
            if isinstance(versions_payload, dict)
            else []
        ),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "Spire Codex Codex Score: Bayesian-shrunk community win-rate "
            "aggregate. Correlational prior only; not causal card value."
        ),
        "entities": filtered_entities,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch filtered Spire Codex community scores."
    )
    parser.add_argument("--knowledge", default=KNOWLEDGE_FILE)
    parser.add_argument("--output", default=COMMUNITY_SCORES_FILE)
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    counts = fetch_scores(args.knowledge, args.output, args.base_url)
    print(
        "Imported snapshot candidates: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )


if __name__ == "__main__":
    main()

"""Golden-set (标准答案集) evaluation for card-reward policy versions.

Each situation keeps only the scoring view of one real (or explicitly
reconstructed) card-reward state event plus a human label: the best
choice(s) and the obvious errors that must never be ranked first. Situations
run through the same request builder and policy registry the realtime host
uses, so a version is judged on its public behaviour only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping

from advisor.decision_core import CARD_REWARD, DecisionPolicy, PolicyRegistry
from advisor.deck_fitness import DeckFitnessPolicy
from advisor.policies import CardRewardPolicy
from realtime.processor import RealtimeEventProcessor
from realtime.protocol import GameStateEvent
from storage.relational import RelationalRepository

GOLDEN_SET_SCHEMA_VERSION = 1
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_set" / "card_reward.json"

# The only state-event fields a card-reward policy reads.
SCORING_FIELDS = (
    "schema_version",
    "event_type",
    "game_version",
    "run_id",
    "sequence",
    "decision_id",
    "guide_preferences",
    "state",
    "candidates",
    "decision",
    "decision_parent",
    "map_context",
)
# Protocol envelope required by the strict v7+ model but never read by a
# policy. Filled with explicit harness markers, never with guessed game data.
_HARNESS_ENVELOPE = {
    "source": "golden-set",
    "producer_id": "golden-set",
    "producer_version": "golden-set",
    "game_assembly_sha256": "0" * 64,  # all-zero marker, never a real hash
    "release_fingerprint": "0" * 64,
    "snapshot_kind": "complete",
    "emitted_at": "2026-01-01T00:00:00+00:00",
    "options": [],
    "parent_event_id": None,
    "outcome": None,
    "run_result": None,
}

# Card-reward policy versions the golden set can compare. New versions are
# added here when they are registered for card rewards.
CARD_REWARD_VERSIONS: Dict[str, Callable[[RelationalRepository], DecisionPolicy]] = {
    "v1": CardRewardPolicy,
    "v2": DeckFitnessPolicy,
}


class GoldenSetError(ValueError):
    """Raised when a golden-set file or situation is malformed."""


def scoring_view(state_event: Mapping) -> dict:
    """Strip one recorded state event down to the fields scoring reads."""

    missing = [field for field in SCORING_FIELDS if field not in state_event]
    if missing:
        raise GoldenSetError(f"state event lacks scoring fields: {missing}")
    if state_event["event_type"] != CARD_REWARD:
        raise GoldenSetError("state event is not a card reward")
    return {field: state_event[field] for field in SCORING_FIELDS}


def _candidate_ids(view: Mapping) -> List[str]:
    ids = [candidate.get("candidate_id") for candidate in view.get("candidates") or []]
    if not all(isinstance(value, str) and value for value in ids):
        raise GoldenSetError("every candidate needs a candidate_id")
    return ids


def load_golden_set(path: Path | str = DEFAULT_GOLDEN_SET_PATH) -> List[dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GoldenSetError(f"golden set is unreadable: {path}") from exc
    if payload.get("schema_version") != GOLDEN_SET_SCHEMA_VERSION:
        raise GoldenSetError("golden set schema version is unsupported")
    situations = payload.get("situations")
    if not isinstance(situations, list):
        raise GoldenSetError("golden set has no situations")
    seen = set()
    for situation in situations:
        situation_id = situation.get("id")
        if not situation_id or situation_id in seen:
            raise GoldenSetError(f"missing or duplicate situation id: {situation_id}")
        seen.add(situation_id)
        view = situation.get("scoring_view")
        if not isinstance(view, dict) or set(view) != set(SCORING_FIELDS):
            raise GoldenSetError(f"{situation_id} scoring view must hold exactly {SCORING_FIELDS}")
        ids = set(_candidate_ids(view))
        best = set(situation.get("best") or ())
        errors = set(situation.get("obvious_errors") or ())
        if not best or not best <= ids or not errors <= ids or best & errors:
            raise GoldenSetError(
                f"{situation_id} labels must name its own candidates, "
                "best must be non-empty and disjoint from obvious errors"
            )
        if situation.get("label_status") not in {"draft", "confirmed"}:
            raise GoldenSetError(f"{situation_id} has no label status")
        if (situation.get("source") or {}).get("kind") not in {"live_snapshot", "reconstructed"}:
            raise GoldenSetError(f"{situation_id} has no source kind")
    return situations


def card_reward_request(view: Mapping):
    """Build the realtime host's card-reward request from one scoring view."""

    event = GameStateEvent.model_validate({
        **_HARNESS_ENVELOPE,
        **view,
        "event_id": f"golden:{view['run_id']}:{view['sequence']}",
        "state_revision": view["sequence"],
    })
    world = RealtimeEventProcessor.build_world(event)
    return RealtimeEventProcessor.build_card_reward_request(event, world)


def situation_warnings(situation: Mapping) -> List[str]:
    """Known limits of a situation that a reader of the report must see."""

    warnings = []
    if situation["source"]["kind"] == "reconstructed":
        warnings.append("reconstructed")
    if situation["scoring_view"].get("map_context") is None:
        warnings.append("map_context_missing")
    return warnings


def evaluate_golden_set(
    situations: Iterable[Mapping],
    policy: DecisionPolicy,
) -> Dict:
    """Run every situation through ``policy`` and score it against its labels."""

    registry = PolicyRegistry([policy])
    rows = []
    for situation in situations:
        recommendation = registry.recommend(card_reward_request(situation["scoring_view"]))
        first = recommendation.recommended_candidate_id
        rows.append({
            "id": situation["id"],
            "label_status": situation["label_status"],
            "warnings": situation_warnings(situation),
            "first": first,
            "hit": first in set(situation["best"]),
            "obvious_error_first": first in set(situation.get("obvious_errors") or ()),
            "policy_version": recommendation.policy_version,
            "scores": {
                row.candidate_id: row.score for row in recommendation.candidates
            },
        })

    def summary(selected: List[dict]) -> Dict:
        hits = sum(row["hit"] for row in selected)
        return {
            "situations": len(selected),
            "hits": hits,
            "hit_rate": round(hits / len(selected), 4) if selected else None,
            "obvious_errors_first": sum(row["obvious_error_first"] for row in selected),
        }

    return {
        "all": summary(rows),
        "confirmed": summary([row for row in rows if row["label_status"] == "confirmed"]),
        "rows": rows,
    }

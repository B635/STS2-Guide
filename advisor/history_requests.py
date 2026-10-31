"""Decision requests rebuilt from the game's run history.

Shared by the history analyses (``scripts/``) and the post-run review coach
(``review/``): what each history record can be scored as, the way the Mod
would have sent it.

- Shop: the record has no prices, so each offer gets its base price from the
  assembly (cards 50 / 75 / 150 by rarity, relics 175 / 225 / 275 / 200,
  potions 50 / 75 / 100, removal 75 + 25 per earlier removal, A6+ 100 + 50);
  an offer dearer than the gold is not eligible, as in the game.
- Rest site: HEAL heals MaxHp x 0.3 (``HealRestSiteOption.GetBaseHealAmount``,
  plus 15 with Regal Pillow); SMITH upgrades one chosen card that is not
  upgraded yet and has an upgrade.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional

from advisor.campfire import CAMPFIRE_ACTION
from advisor.decision_core import DecisionCandidate, DecisionRequest
from advisor.golden_set import world_request
from advisor.merchant import MERCHANT_CHOICE

CARD_PRICE = {"common": 50, "uncommon": 75, "rare": 150}
RELIC_PRICE = {"common": 175, "uncommon": 225, "rare": 275, "shop": 200}
POTION_PRICE = {"common": 50, "uncommon": 75, "rare": 100}
REST_HEAL_SHARE_TENTHS = 3
REGAL_PILLOW_HEAL = 15


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


def starting_decks(knowledge_path: Path) -> dict:
    knowledge = json.loads(Path(knowledge_path).read_text(encoding="utf-8"))
    return {
        str(character["id"]).upper(): [_snake(card) for card in character.get("starting_deck") or ()]
        for character in knowledge["characters"]
    }


def unscorable_reason(view: Mapping, known_cards: set, known_encounters: set) -> Optional[str]:
    """Why a history view cannot be scored with the current catalog, or None."""

    cards = {row["card"] for row in view["state"]["deck"]} | {
        c["entity_id"] for c in view.get("candidates") or () if c.get("entity_id")
    }
    unknown = sorted(cards - known_cards)
    if unknown:
        return "unknown_card:" + ",".join(unknown)
    bosses = view["map_context"]["boss_encounter_ids"]
    if not bosses:
        return "no_boss"
    if any(boss not in known_encounters for boss in bosses):
        return "unknown_boss:" + ",".join(bosses)
    return None


def rarities(knowledge: Mapping) -> dict:
    return {
        kind: {e["id"]: str(e.get("rarity_key") or "").lower() for e in knowledge[kind]}
        for kind in ("cards", "relics", "potions")
    }


def shop_request(row: dict, rarity: dict) -> DecisionRequest:
    world = world_request(row["scoring_view"]).world
    gold = row["gold_before"]
    high = int(row["scoring_view"]["state"].get("ascension") or 0) >= 6
    candidates = []

    def offer(candidate_id, kind, entity, price, **extra):
        payload = {
            "candidate_kind": "merchant_offer", "offer_kind": kind, "is_stocked": True, "entity_id": entity,
            "costs": [{"kind": "gold", "amount": price, "resource_id": None}], **extra,
        }
        candidates.append(DecisionCandidate.create(
            candidate_id, payload, label=entity or kind, display_index=len(candidates), eligible=price <= gold,
        ))

    for i, entry in enumerate(row["offers"]["cards"]):
        price = CARD_PRICE.get(rarity["cards"].get(entry["card"], ""))
        if price is not None:
            offer(f"card:{i}:{entry['card']}", "card", entry["card"], price, upgrades=entry["upgrades"])
    for i, relic in enumerate(row["offers"]["relics"]):
        price = RELIC_PRICE.get(rarity["relics"].get(relic, ""))
        if price is not None:
            offer(f"relic:{i}:{relic}", "relic", relic, price)
    for i, potion in enumerate(row["offers"]["potions"]):
        price = POTION_PRICE.get(rarity["potions"].get(potion, ""))
        if price is not None:
            offer(f"potion:{i}:{potion}", "potion", potion, price)
    removal = (100 + 50 * row["removals_before"]) if high else (75 + 25 * row["removals_before"])
    targets = []
    for entry in row["scoring_view"]["state"]["deck"]:
        for _ in range(entry["count"]):
            targets.append({"target_id": f"history:{len(targets)}", "card": entry["card"], "upgrades": entry["upgrades"]})
    offer("removal", "card_removal", None, removal,
          deck_edit_targets={"operation": "remove", "source_code": "history:any_card", "targets": targets})
    candidates.append(DecisionCandidate.create(
        "leave", {"candidate_kind": "leave", "costs": []}, label="leave", display_index=len(candidates),
    ))
    return DecisionRequest.create(
        decision_id=f"{row['run']}:shop:{row['floor']}", decision_type=MERCHANT_CHOICE, world=world,
        candidates=candidates,
    )


def rest_heal_amount(row: Mapping) -> int:
    state = row["scoring_view"]["state"]
    heal = int(state["max_hp"]) * REST_HEAL_SHARE_TENTHS // 10
    if "REGAL_PILLOW" in state["relics"]:
        heal += REGAL_PILLOW_HEAL
    if row.get("chosen") == "HEAL":
        heal = max(heal, int(row.get("healed") or 0))
    return heal


def rest_request(row: dict, upgradable: Mapping[str, bool]) -> DecisionRequest:
    """HEAL and SMITH, built the way the Mod builds them (RestSiteObserver.cs)."""

    view = row["scoring_view"]
    targets = []
    for entry in view["state"]["deck"]:
        if int(entry.get("upgrades") or 0) == 0 and upgradable.get(entry["card"]):
            for _ in range(int(entry["count"])):
                targets.append({"target_id": f"history:{len(targets)}", "card": entry["card"], "upgrades": 0})
    candidates = [
        DecisionCandidate.create("rest:HEAL", {
            "candidate_kind": "rest_action", "action_id": "HEAL", "entity_id": None, "costs": [],
            "effects": [{"kind": "hp_delta", "amount": rest_heal_amount(row), "target_mode": "none",
                         "certainty": "exact", "source_code": "game_api:rest_heal_amount"}],
        }, label="HEAL", display_index=0),
        DecisionCandidate.create("rest:SMITH", {
            "candidate_kind": "rest_action", "action_id": "SMITH", "entity_id": None, "costs": [],
            "effects": [{"kind": "upgrade_card", "entity_type": "cards", "target_mode": "choose",
                         "certainty": "exact", "source_code": "game_api:rest_smith_option"}],
            "deck_edit_targets": {"operation": "upgrade", "source_code": "history:upgradable_cards",
                                  "targets": targets},
        }, label="SMITH", display_index=1, eligible=bool(targets)),
    ]
    return DecisionRequest.create(
        decision_id=f"{row['run']}:rest:{row['floor']}", decision_type=CAMPFIRE_ACTION,
        world=world_request(view).world, candidates=candidates,
    )


def deck_counts(deck: Iterable[Mapping]) -> collections.Counter:
    counts = collections.Counter()
    for entry in deck:
        counts[(entry["card"], int(entry.get("upgrades") or 0))] += int(entry.get("count") or 1)
    return counts


def bought_ids(row: dict, request: DecisionRequest) -> set:
    """What the shop record says was bought, as candidate IDs ({"leave"} when nothing)."""

    bought = row["bought"]
    names = {c["card"] for c in bought["cards"]} | set(bought["relics"]) | set(bought["potions"])
    ids = {c.candidate_id for c in request.candidates if c.label in names}
    if bought["removed"]:
        ids.add("removal")
    return ids or {"leave"}

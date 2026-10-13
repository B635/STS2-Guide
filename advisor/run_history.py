"""Read-only import of the game's own run history (``saves/history/*.run``).

The game keeps one JSON file per finished run: every map point visited with
the cards offered and picked, deck changes, HP, gold, and for each fight the
encounter, the turns it took and the damage taken. This module replays one
such file into card-reward situations in the golden-set format, and into
per-fight outcomes for checking the engine against what really happened.

Nothing is written to the game's folders. A run whose replayed deck does not
match the deck the file ends with is rejected rather than repaired.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

FIGHT_POINTS = frozenset({"monster", "elite", "boss"})
NODE_KINDS = {
    "monster": "MONSTER", "elite": "ELITE", "boss": "BOSS", "rest_site": "CAMPFIRE",
    "shop": "SHOP", "treasure": "TREASURE", "unknown": "EVENT", "ancient": "EVENT",
}
# Protocol version of the scoring view the situations are written in.
VIEW_SCHEMA_VERSION = 9


class RunHistoryError(ValueError):
    """A run file that cannot be replayed faithfully."""


def bare_id(value: object) -> str:
    """``CARD.SCOURGE`` -> ``SCOURGE``."""

    return str(value or "").split(".", 1)[-1].upper()


def _entry(card: Mapping) -> dict:
    enchantment = card.get("enchantment") or {}
    return {
        "card": bare_id(card["id"]),
        "upgrades": int(card.get("current_upgrade_level") or 0),
        "enchantment": bare_id(enchantment["id"]) if enchantment.get("id") else None,
        "enchantment_amount": enchantment.get("amount"),
    }


def _remove(deck: List[dict], card: Mapping, warnings: List[str], floor: int) -> Optional[dict]:
    """Take out the copy the record names: same upgrade level when it says
    one, else any copy of that card."""

    wanted = _entry(card)
    card_id = wanted["card"]
    exact = [i for i, entry in enumerate(deck) if entry["card"] == card_id and entry["upgrades"] == wanted["upgrades"]]
    loose = [i for i, entry in enumerate(deck) if entry["card"] == card_id]
    if exact or loose:
        return deck.pop((exact or loose)[0])
    # The file did not record when this card arrived (an event or a relic):
    # the decks before this floor lack it.
    warnings.append(f"floor {floor}: {card_id} left the deck but its arrival was not recorded")
    return None


def _deck_key(deck: Iterable[Mapping]) -> collections.Counter:
    return collections.Counter((entry["card"], entry["upgrades"]) for entry in deck)


def _state_deck(deck: Iterable[Mapping]) -> List[dict]:
    rows: Dict[tuple, dict] = {}
    for entry in deck:
        key = (entry["card"], entry["upgrades"], entry["enchantment"], entry["enchantment_amount"])
        row = rows.setdefault(key, {
            "card": entry["card"], "count": 0, "upgrades": entry["upgrades"],
            "enchantment": entry["enchantment"], "enchantment_amount": entry["enchantment_amount"],
            "affliction": None, "affliction_amount": None,
        })
        row["count"] += 1
    return sorted(rows.values(), key=lambda row: (row["card"], row["upgrades"]))


def _candidate(index: int, card: Mapping) -> dict:
    entry = _entry(card)
    return {
        "candidate_id": f"{index}:{entry['card']}", "kind": "card", "entity_id": entry["card"],
        "label": entry["card"], "eligible": True, "unavailable_reason": None, "costs": [],
        "payload": {
            "card": entry["card"], "upgrades": entry["upgrades"],
            "enchantment": entry["enchantment"], "enchantment_amount": entry["enchantment_amount"],
            "affliction": None, "affliction_amount": None, "node_id": None, "slot_id": None,
            "offer_kind": None, "is_stocked": None, "replacement_supported": None,
            "action_id": None, "operation": None, "blessing_id": None, "stage_id": None,
            "event_id": None, "page_id": None, "option_id": None, "target_candidate_ids": [],
            "effects": [], "deck_edit_targets": None,
        },
    }


SKIP_CANDIDATE = {
    "candidate_id": "skip", "kind": "skip", "entity_id": None, "label": "跳过", "eligible": True,
    "unavailable_reason": None, "costs": [],
    "payload": {key: None for key in _candidate(0, {"id": "X"})["payload"]}
    | {"target_candidate_ids": [], "effects": []},
}


def _act_map(points: List[Mapping], index: int, act_number: int) -> dict:
    """The act as the player walked it: one node per visited point.

    The history holds the path taken, not the whole map, so the fights ahead
    are the ones that were really fought. A question-mark room keeps the kind
    the player saw (EVENT), whatever it turned out to hold.
    """

    nodes = []
    for row, point in enumerate(points):
        nodes.append({
            "node_id": f"{row}:0", "kind": NODE_KINDS.get(point["map_point_type"], "UNKNOWN"),
            "row": row, "col": 0,
            "edges": [f"{row + 1}:0"] if row + 1 < len(points) else [],
            "label": point["map_point_type"],
        })
    boss_rows = [row for row, point in enumerate(points) if point["map_point_type"] == "boss"]
    boss_ids = [
        bare_id(room["model_id"])
        for row in boss_rows for room in points[row].get("rooms") or ()
        if room.get("room_type") == "boss" and room.get("model_id")
    ]
    return {
        "map_name": f"Act_{act_number}", "player_row": index, "nodes": nodes,
        "node_count": len(nodes), "current_node_id": f"{index}:0", "origin_node_id": f"{index}:0",
        "available_next_node_ids": [f"{index + 1}:0"] if index + 1 < len(points) else [],
        "boss_node_ids": [f"{row}:0" for row in boss_rows][-1:],
        "boss_encounter_ids": boss_ids[-1:],
    }


def replay_run(run: Mapping, starting_deck: Iterable[str], run_key: str) -> dict:
    """Replay one single-player run.

    Returns ``{"situations": [...], "fights": [...], "problems": [...],
    "warnings": [...]}``. ``problems`` is non-empty when the replayed deck
    holds different cards from the deck the file ends with; the caller must
    then drop the run. ``warnings`` are small gaps of the file itself (an
    upgrade or a curse whose moment was not recorded) that leave the cards of
    every deck right.
    """

    players = run.get("players") or ()
    if len(players) != 1:
        raise RunHistoryError("only single-player runs are replayed")
    player = players[0]
    character = bare_id(player["character"])
    deck = [{"card": card, "upgrades": 0, "enchantment": None, "enchantment_amount": None}
            for card in starting_deck]
    # Cards the run starts with beyond the character's deck (Ascender's Bane):
    # in the final deck since floor 1 and not gained at the first point.
    history = run.get("map_point_history") or ()
    first = ((history[0][0].get("player_stats") or [{}])[0] if history and history[0] else {})
    first_gained = collections.Counter(
        bare_id(card["id"]) for card in first.get("cards_gained") or ()
    ) + collections.Counter(
        bare_id(change["final_card"]["id"]) for change in first.get("cards_transformed") or ()
    )
    since_start = collections.Counter(
        bare_id(card["id"]) for card in player.get("deck") or ()
        if int(card.get("floor_added_to_deck") or 0) <= 1
    )
    for card, count in (since_start - collections.Counter(deck_card["card"] for deck_card in deck) - first_gained).items():
        deck.extend(
            {"card": card, "upgrades": 0, "enchantment": None, "enchantment_amount": None}
            for _ in range(count)
        )
    warnings: List[str] = []
    final_relics = [(bare_id(relic["id"]), int(relic.get("floor_added_to_deck") or 0))
                    for relic in player.get("relics") or ()]
    # Relics that left during the run are not in the final list. The file
    # says when they left (``relics_removed``) and, when they were chosen or
    # bought, when they arrived; an arrival it did not record counts from the
    # start of the run and is reported.
    lost_relics: Dict[str, List[int]] = {}   # relic -> [arrival floor, floor it left]
    floor_counter = 0
    for act_points in run.get("map_point_history") or ():
        for point in act_points:
            floor_counter += 1
            point_stats = (point.get("player_stats") or [{}])[0]
            arrived = [bare_id(c["choice"]) for c in point_stats.get("relic_choices") or () if c.get("was_picked")]
            arrived += [bare_id(r) for r in point_stats.get("bought_relics") or ()]
            arrived += [bare_id(c.get("TextKey")) for c in point_stats.get("ancient_choice") or () if c.get("was_chosen")]
            for relic in arrived:
                lost_relics.setdefault(relic, [floor_counter, None])
            for relic in point_stats.get("relics_removed") or ():
                relic = bare_id(relic)
                if relic not in lost_relics:
                    lost_relics[relic] = [0, None]
                lost_relics[relic][1] = floor_counter
    final_relic_ids = {relic for relic, _ in final_relics}
    lost_relics = {
        relic: span for relic, span in lost_relics.items()
        if span[1] is not None and relic not in final_relic_ids
    }
    potions: List[str] = []
    problems: List[str] = []
    situations: List[dict] = []
    fights: List[dict] = []
    floor = 0
    for act_index, points in enumerate(run.get("map_point_history") or ()):
        for index, point in enumerate(points):
            floor += 1
            stats = (point.get("player_stats") or [{}])[0]
            kind = point["map_point_type"]
            before = _state_deck(deck)
            # A relic gained at this point comes after the fight's reward,
            # except what the run starts with (floor 1).
            relics_now = [relic for relic, added in final_relics if added < floor or added <= 1]
            relics_now += [
                relic for relic, (arrived, left) in lost_relics.items() if arrived < floor <= left
            ]
            # Potions used or thrown away in this fight are gone by the reward.
            for potion in (*(stats.get("potion_used") or ()), *(stats.get("potion_discarded") or ())):
                if bare_id(potion) in potions:
                    potions.remove(bare_id(potion))
            slots = int(player.get("max_potion_slot_count") or 3)
            if len(potions) > slots:
                # More than the belt holds: the file did not record a potion
                # leaving. The oldest are dropped and the run says so.
                warnings.append(f"floor {floor}: {len(potions)} potions replayed for {slots} slots")
                del potions[:len(potions) - slots]
            choices = stats.get("card_choices") or ()
            picked = [choice for choice in choices if choice.get("was_picked")]
            for room in point.get("rooms") or ():
                if room.get("room_type") in FIGHT_POINTS and room.get("model_id"):
                    fights.append({
                        "run": run_key, "floor": floor, "act": act_index + 1,
                        "encounter": bare_id(room["model_id"]), "room_type": room["room_type"],
                        "turns_taken": room.get("turns_taken"),
                        "damage_taken": stats.get("damage_taken"),
                        "hp_after": stats.get("current_hp"), "max_hp": stats.get("max_hp"),
                        "deck": before, "relics": relics_now,
                    })
            # One ordinary reward: three (or four) cards, at most one taken.
            if kind in FIGHT_POINTS and len(choices) in (3, 4) and len(picked) <= 1:
                candidates = [_candidate(i, choice["card"]) for i, choice in enumerate(choices)]
                best = next(
                    (c["candidate_id"] for c, choice in zip(candidates, choices) if choice.get("was_picked")),
                    "skip",
                )
                situations.append({
                    "id": f"history-{run_key}-f{floor}",
                    "character": character,
                    "source": {
                        "kind": "run_history", "run": run_key, "build_id": run.get("build_id"),
                        "win": bool(run.get("win")), "floor": floor,
                    },
                    "best": [best],
                    "obvious_errors": [],
                    "label_status": "confirmed",
                    "rationale": "The player's own pick in a finished run, read from the game's run history.",
                    "scoring_view": {
                        "schema_version": VIEW_SCHEMA_VERSION,
                        "event_type": "card_reward",
                        "game_version": str(run.get("build_id") or "").lstrip("v"),
                        "run_id": f"history-{run_key}",
                        "sequence": floor,
                        "decision_id": f"history-{run_key}:card-reward:{floor}",
                        "guide_preferences": {"route_mode": "balanced"},
                        "state": {
                            "character": character, "ascension": int(run.get("ascension") or 0),
                            "act": act_index + 1, "floor": floor,
                            "hp": stats.get("current_hp"), "max_hp": stats.get("max_hp"),
                            "gold": stats.get("current_gold"), "energy": 0,
                            "deck": before, "relics": relics_now,
                            "relic_states": [
                                {"relic": relic, "display_amount": None, "stack_count": 1, "status": "NORMAL"}
                                for relic in relics_now
                            ],
                            "potions": [{"potion": potion, "slot": slot} for slot, potion in enumerate(potions)],
                            "max_potion_slots": int(player.get("max_potion_slot_count") or 3),
                            "modifiers": [], "capture_warnings": [],
                        },
                        "candidates": [*candidates, SKIP_CANDIDATE],
                        "decision": {"can_skip": True, "can_reroll": False, "reward_source": "CARD"},
                        "decision_parent": None,
                        "map_context": _act_map(points, index, act_index + 1),
                    },
                })
            # Deck changes of this point, in the order the game applies them.
            for change in stats.get("cards_transformed") or ():
                _remove(deck, change["original_card"], warnings, floor)
            for card in stats.get("cards_removed") or ():
                _remove(deck, card, warnings, floor)
            transformed = [bare_id(c["final_card"]["id"]) for c in stats.get("cards_transformed") or ()]
            for card in stats.get("cards_gained") or ():
                deck.append(_entry(card))
            gained = collections.Counter(bare_id(card["id"]) for card in stats.get("cards_gained") or ())
            for card_id in transformed:
                if gained[card_id] > 0:
                    gained[card_id] -= 1   # already listed under cards_gained
                else:
                    deck.append({"card": card_id, "upgrades": 0, "enchantment": None, "enchantment_amount": None})
            for card_id in stats.get("upgraded_cards") or ():
                target = min(
                    (entry for entry in deck if entry["card"] == bare_id(card_id)),
                    key=lambda entry: entry["upgrades"], default=None,
                )
                if target is None:
                    warnings.append(f"floor {floor}: {bare_id(card_id)} was upgraded but is not in the deck")
                else:
                    target["upgrades"] += 1
            for change in stats.get("cards_enchanted") or ():
                card = _entry(change["card"])
                target = next(
                    (entry for entry in deck if entry["card"] == card["card"] and not entry["enchantment"]), None,
                )
                if target is not None:
                    target["enchantment"] = card["enchantment"] or bare_id(change.get("enchantment"))
                    target["enchantment_amount"] = card["enchantment_amount"]
            for choice in stats.get("potion_choices") or ():
                if choice.get("was_picked"):
                    potions.append(bare_id(choice["choice"]))
            potions.extend(bare_id(potion) for potion in stats.get("bought_potions") or ())
    for relic, (arrived, left) in lost_relics.items():
        if arrived == 0:
            warnings.append(f"{relic} left at floor {left}; its arrival was not recorded, counted from the start")
    final = [_entry(card) for card in player.get("deck") or ()]
    cards = collections.Counter(entry["card"] for entry in deck)
    final_cards = collections.Counter(entry["card"] for entry in final)
    if cards != final_cards:
        problems.append(
            f"replayed deck differs from the final deck: missing {dict(final_cards - cards)}, "
            f"extra {dict(cards - final_cards)}"
        )
    elif _deck_key(deck) != _deck_key(final):
        # Same cards, different upgrade levels: an upgrade the file did not
        # record (a relic or an event). Its moment is unknown, so the decks
        # keep the card not upgraded.
        warnings.append(f"unrecorded upgrades: {dict(_deck_key(final) - _deck_key(deck))}")
    return {"situations": situations, "fights": fights, "problems": problems, "warnings": warnings}


def load_run(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunHistoryError(f"run file is unreadable: {path}") from exc

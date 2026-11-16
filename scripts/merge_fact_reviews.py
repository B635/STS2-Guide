"""Validate reviewed fact batches and merge them into the committed facts.

A batch is a JSON array of fact objects written by a reviewer following
docs/dev/mechanic-facts.md. Every entry must name an existing catalog entity,
carry the class fingerprint of the current extract, and use the spec's
vocabulary. Nothing is merged unless the whole batch validates (``--check``
only reports). Mechanics the scoring engine cannot convert yet are listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from advisor.deck_mechanics import SUPPORTED_EFFECTS, unsupported_reason  # noqa: E402
from storage.assembly_facts import (  # noqa: E402
    DEFAULT_EXTRACT_PATH,
    DEFAULT_FACTS_PATH,
    REVIEWED_STATUSES,
    extract_index,
    load_extract,
    load_facts,
)

# Recorded vocabulary (docs/dev/mechanic-facts.md). It is wider than what the
# engine scores: true facts are kept even before the engine can convert them.
EFFECT_KINDS = SUPPORTED_EFFECTS | {
    "add_card", "upgrade_cards", "channel_orb", "summon", "forge", "gain_stars",
    "orb_passive", "orb_passive_all", "orb_slots", "set_free_this_turn", "extra_turn",
    "cost_reduction", "add_pet", "procure_potion", "set_enemy_hp", "replay_card", "move_card",
    "auto_play", "auto_play_card", "auto_play_from_draw", "transform_cards", "gain_max_hp",
    "add_keyword", "remove_block", "remove_power", "trigger_power", "extra_card_reward",
    "evoke_orb", "shuffle", "doom_kill", "kill_osty", "spread_debuffs", "add_replay",
    "end_turn", "gain_gold", "stun", "lose_max_hp",
    "remove_from_deck", "rest_site_option", "force_event", "kill_owner", "kill_target",
    "enchantment_amount", "set_cost",
}
WHEN = {
    "combat_start", "turn_start", "turn_end", "card_played", "card_exhausted",
    "card_discarded", "shuffle", "hand_emptied", "damage_received", "enemy_died",
    "potion_used", "stars_spent", "orb_channeled", "card_generated", "block_broken",
    "lethal_damage", "auto_post_play_phase", "card_drawn", "block_gained",
    "power_applied", "power_applied_self", "creature_added", "on_chosen",
    "combat_end", "rest_site", "room_entered", "map_generated", "quest_complete",
    "stars_gained", "damage_dealt", "forge", "osty_hp_lost", "orb_evoked", "energy_spent", "power_removed",
    "orb_passive_or_evoke",
}
MODIFIERS = {
    "hand_draw", "max_energy", "damage", "block", "energy_cost", "x_value",
    "keep_block", "retain_hand", "no_draw_during_turn", "cards_per_turn_limit",
    "hp_loss_per_turn_cap", "hp_loss_reduce", "card_added_to_deck",
    "power_amount_given", "card_play_count", "damage_received", "energy_carryover",
    "power_gain", "no_potions", "auto_play_turn", "card_reward_pool",
    "card_reward_options", "card_reward_extra_card", "card_reward_alternative",
    "card_reward_choice", "exhaust_on_play", "vulnerable_multiplier", "strength", "play_lock",
    "poison_trigger_count", "dexterity", "focus", "negate_debuff", "hp_loss_negate",
    "weak_multiplier", "card_target", "orb_value",
    "damage_cap", "hp_loss_cap", "card_play_result_location", "star_cost",
    "sovereign_blade_block", "add_keyword", "start_position",
}
CATEGORIES = {"baseline", "trigger", "irrelevant"}
# Relic pickup and between-fight effects (TASK-027 ticket 01): read when a
# relic is being chosen, never by the deck engine for an owned relic.
OBTAIN_KINDS = {
    "gain_gold", "lose_gold", "gain_max_hp", "lose_max_hp", "heal", "hp_loss", "set_hp",
    "remove_from_deck", "transform_cards", "upgrade_cards", "add_card", "enchant_cards",
    "gain_potion", "potion_slots", "gain_relic", "offer_rewards", "replace_relic", "remove_relic",
    "add_pet", "apply_power",
}
BETWEEN_WHEN = {
    "combat_end", "room_entered", "rest_site", "card_reward", "merchant", "treasure",
    "map_generated", "gold_gained", "potion_procured", "lethal_damage",
}
BETWEEN_MODIFIERS = {
    "card_reward_options", "card_reward_rarity", "card_reward_upgraded", "rest_site_heal",
    "rest_site_option", "merchant_price", "gold_gained", "potion_drop", "no_potions",
    "free_travel", "unknown_room_types", "map_change",
}
REVIEW_FIELDS = {
    "status", "category", "cost", "card_type", "tags", "keywords", "effects",
    "mechanics", "unknowable", "evidence", "open_questions", "reviewed_by",
    "duration", "upgrade_keywords", "multiplayer_only", "cost_modifier", "audit",
    "star_cost", "can_be_generated_in_combat", "card_type_text", "passive", "evoke",
    "state", "can_enchant", "adds_keywords", "removes_keywords",
    "hp", "innate", "moves", "pattern", "room_type", "is_weak", "variants",
    "obtain", "between_fights", "obtain_params",
}


def _effect_problems(effect: object, where: str) -> list[str]:
    if not isinstance(effect, dict):
        return [f"{where}: effect is not an object"]
    if effect.get("kind") not in EFFECT_KINDS:
        return [f"{where}: unknown effect kind {effect.get('kind')!r}"]
    amount = effect.get("amount")
    if amount is not None and not isinstance(amount, (int, float)):
        return [f"{where}: amount must be a number (use amount_from for derived amounts)"]
    return []


def _obtain_problems(entry: dict, key: str) -> list[str]:
    problems: list[str] = []
    if (entry.get("obtain") or entry.get("between_fights") or entry.get("obtain_params")) and not key.startswith("relics:"):
        problems.append(f"{key}: obtain / between_fights are relic fields")
    for i, effect in enumerate(entry.get("obtain") or ()):
        where = f"{key} obtain[{i}]"
        if not isinstance(effect, dict) or effect.get("kind") not in OBTAIN_KINDS:
            problems.append(f"{where}: unknown obtain kind {effect.get('kind') if isinstance(effect, dict) else effect!r}")
            continue
        amount = effect.get("amount")
        if amount is not None and not isinstance(amount, (int, float)):
            problems.append(f"{where}: amount must be a number (use amount_from for derived amounts)")
        count = effect.get("count")
        if count is not None and not isinstance(count, (int, float)) and count != "all":
            problems.append(f"{where}: count must be a number or \"all\"")
    for i, mechanic in enumerate(entry.get("between_fights") or ()):
        where = f"{key} between_fights[{i}]"
        if not isinstance(mechanic, dict):
            problems.append(f"{where}: not an object")
        elif "modifier" in mechanic:
            if mechanic["modifier"] not in BETWEEN_MODIFIERS:
                problems.append(f"{where}: unknown modifier {mechanic['modifier']!r}")
        elif mechanic.get("when") not in BETWEEN_WHEN:
            problems.append(f"{where}: unknown when {mechanic.get('when')!r}")
        elif not isinstance(mechanic.get("effect"), dict) or mechanic["effect"].get("kind") not in OBTAIN_KINDS:
            problems.append(f"{where}: effect needs an obtain kind")
    params = entry.get("obtain_params")
    if params is not None and not isinstance(params, dict):
        problems.append(f"{key}: obtain_params must be an object")
    return problems


def codex_conflicts(entry: dict, catalog: dict) -> list[str]:
    """Spire Codex structured card fields that disagree with the reviewed fact.

    Codex is a cross-check only: conflicts go to a re-check list, the fact
    (from the assembly) is never overwritten by them.
    """

    key = entry["entity_key"]
    if not key.startswith("cards:") or entry.get("status") != "converted":
        return []
    card = catalog.get(key.split(":", 1)[1])
    if card is None:
        return []
    conflicts = []
    cost = (entry.get("cost") or {}).get("base") if isinstance(entry.get("cost"), dict) else entry.get("cost")
    codex_cost = "X" if card.get("is_x_cost") else card.get("cost")
    if cost is not None and codex_cost is not None and cost != codex_cost:
        conflicts.append(f"cost {cost} vs Codex {codex_cost}")
    if entry.get("card_type") and card.get("type_key") and entry["card_type"] != card["type_key"]:
        conflicts.append(f"type {entry['card_type']} vs Codex {card['type_key']}")
    effects = entry.get("effects") or []
    for field, kind in (("damage", "damage"), ("block", "block"), ("cards_draw", "draw"), ("energy_gain", "energy")):
        value = card.get(field)
        if value is None:
            continue
        amounts = {e.get("amount") for e in effects if e.get("kind") == kind}
        if amounts and value not in amounts:
            conflicts.append(f"{kind} {sorted(a for a in amounts if a is not None)} vs Codex {value}")
    hits = card.get("hit_count")
    if hits is not None:
        fact_hits = {e.get("hits") for e in effects if e.get("kind") == "damage"}
        if fact_hits and hits not in fact_hits:
            conflicts.append(f"hits {sorted(h for h in fact_hits if h is not None)} vs Codex {hits}")
    return [f"{key}: {c}" for c in conflicts]


def _monster_problems(entry: dict, key: str) -> list[str]:
    problems: list[str] = []
    hp = entry.get("hp") or {}
    if not all(isinstance(hp.get(k), (int, float)) for k in ("min", "max")):
        problems.append(f"{key}: hp.min and hp.max must be numbers")
    moves = {}
    for i, move in enumerate(entry.get("moves") or ()):
        if not move.get("id") or not isinstance(move.get("effects"), list):
            problems.append(f"{key} moves[{i}]: needs id and effects")
            continue
        moves[move["id"]] = move
        for j, effect in enumerate(move["effects"]):
            if not isinstance(effect, dict) or not effect.get("kind"):
                problems.append(f"{key} moves[{i}].effects[{j}]: needs kind")
            elif effect["kind"] == "damage" and not (
                isinstance(effect.get("amount"), (int, float)) or effect.get("amount_text") or effect.get("amount_from")
            ):
                problems.append(f"{key} moves[{i}].effects[{j}]: damage needs amount")
    pattern = entry.get("pattern") or {}
    states = {s.get("id"): s for s in pattern.get("states") or () if isinstance(s, dict)}
    for state_id, state in states.items():
        kind = state.get("type")
        if kind == "move":
            if state.get("move") not in moves:
                problems.append(f"{key} state {state_id}: unknown move {state.get('move')!r}")
            if state.get("next") is not None and state["next"] not in states:
                problems.append(f"{key} state {state_id}: unknown next {state['next']!r}")
        elif kind in {"random", "conditional"}:
            for branch in state.get("branches") or ():
                if branch.get("to") not in states:
                    problems.append(f"{key} state {state_id}: unknown branch target {branch.get('to')!r}")
        else:
            problems.append(f"{key} state {state_id}: unknown type {kind!r}")
    if pattern.get("initial") is not None and pattern["initial"] not in states:
        problems.append(f"{key}: pattern.initial {pattern['initial']!r} is not a state")
    for item in entry.get("unknowable") or ():
        if not item.get("what") or not item.get("reason"):
            problems.append(f"{key}: unknowable items need what and reason")
    return problems


def validate_entry(entry: dict, index: dict, facts_by_key: dict) -> list[str]:
    key = entry.get("entity_key")
    problems: list[str] = []
    if key not in facts_by_key:
        return [f"{key}: not a catalog entity"]
    source = index.get(key)
    if source is None or entry.get("fingerprint") != source["fingerprint"]:
        problems.append(f"{key}: fingerprint does not match the current extract")
    if entry.get("status") not in REVIEWED_STATUSES:
        problems.append(f"{key}: status must be converted or unknowable")
    if not entry.get("evidence"):
        problems.append(f"{key}: evidence is required")
    extra = set(entry) - REVIEW_FIELDS - {"entity_key", "class", "fingerprint"}
    if extra:
        problems.append(f"{key}: unknown fields {sorted(extra)}")
    entity_type = key.split(":", 1)[0]
    if entity_type == "relics" and entry.get("category") not in CATEGORIES:
        problems.append(f"{key}: relic category must be one of {sorted(CATEGORIES)}")
    if entity_type == "encounters":
        variants = entry.get("variants") or []
        if not variants and entry.get("status") == "converted":
            problems.append(f"{key}: variants are required")
        for i, variant in enumerate(variants):
            if not isinstance(variant.get("weight"), (int, float)):
                problems.append(f"{key} variants[{i}]: weight must be a number")
            for j, spawn in enumerate(variant.get("spawns") or ()):
                if f"monsters:{spawn.get('monster')}" not in facts_by_key:
                    problems.append(f"{key} variants[{i}].spawns[{j}]: unknown monster {spawn.get('monster')!r}")
        return problems
    if entity_type == "monsters":
        # Monster hooks are descriptive; what is computed (moves, pattern) is
        # checked for structure instead of the player-side vocabulary.
        return problems + _monster_problems(entry, key)
    problems.extend(_obtain_problems(entry, key))
    for i, effect in enumerate(entry.get("effects") or ()):
        problems.extend(_effect_problems(effect, f"{key} effects[{i}]"))
    for i, mechanic in enumerate(entry.get("mechanics") or ()):
        where = f"{key} mechanics[{i}]"
        if "modifier" in mechanic:
            if mechanic["modifier"] not in MODIFIERS:
                problems.append(f"{where}: unknown modifier {mechanic['modifier']!r}")
            continue
        if mechanic.get("when") not in WHEN:
            problems.append(f"{where}: unknown when {mechanic.get('when')!r}")
        problems.extend(_effect_problems(mechanic.get("effect"), where))
    for item in entry.get("unknowable") or ():
        if not item.get("what") or not item.get("reason"):
            problems.append(f"{key}: unknowable items need what and reason")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="validate only")
    args = parser.parse_args()

    index = extract_index(load_extract(DEFAULT_EXTRACT_PATH))
    knowledge = json.loads((ROOT_DIR / "data" / "knowledge.json").read_text(encoding="utf-8"))
    catalog = {card["id"]: card for card in knowledge["cards"]}
    facts = load_facts(DEFAULT_FACTS_PATH)
    by_key = {entry["entity_key"]: entry for entry in facts["entities"]}

    entries, problems, unsupported, questions, recheck = [], [], [], [], []
    for path in args.batches:
        for entry in json.loads(path.read_text(encoding="utf-8")):
            problems.extend(validate_entry(entry, index, by_key))
            recheck.extend(codex_conflicts(entry, catalog))
            entries.append(entry)
            for mechanic in entry.get("mechanics") or ():
                reason = unsupported_reason(mechanic)
                if reason:
                    unsupported.append(f"{entry['entity_key']}: {reason}")
            questions.extend(f"{entry['entity_key']}: {q}" for q in entry.get("open_questions") or ())

    print(f"{len(entries)} entries, {len(problems)} problems, "
          f"{len(unsupported)} engine-unsupported mechanics, {len(questions)} open questions, "
          f"{len(recheck)} Codex conflicts to re-check")
    for line in problems:
        print("PROBLEM", line)
    for line in unsupported:
        print("UNSUPPORTED", line)
    for line in questions:
        print("QUESTION", line)
    for line in recheck:
        print("RECHECK", line)
    if problems or args.check:
        return 1 if problems else 0

    for entry in entries:
        target = by_key[entry["entity_key"]]
        for field in ("mechanics", "effects", "unknowable", "open_questions", "category",
                      "obtain", "between_fights", "obtain_params"):
            target.pop(field, None)
        target.update({k: v for k, v in entry.items() if k != "class"})
    with DEFAULT_FACTS_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(facts, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(f"merged {len(entries)} entries into {DEFAULT_FACTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

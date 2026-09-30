"""Deterministic upgrade/removal opportunity values from structured cards."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import re
from typing import Any, Mapping

from advisor.character_mechanics import (
    build_mechanic_context,
    signals_for_card,
)
from advisor.choice_effects import AssessmentDraft, ChoiceEffect
from advisor.contextual_scoring import card_energy_cost
from advisor.decision_core import WorldState
from storage.relational import RelationalRepository


@dataclass(frozen=True)
class DeckEditOpportunity:
    operation: str
    target_id: str | None
    value: float | None
    data_gaps: tuple[str, ...] = ()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*", value)
        if match:
            return float(match.group(1))
    return None


def _resolved_deck(
    world: WorldState,
    repository: RelationalRepository,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows: list[dict[str, Any]] = []
    gaps: list[str] = []
    for entry in world.deck:
        card = repository.find_card(entry.card)
        if card is None:
            gaps.append(f"unknown_card:{entry.card}")
        rows.append({
            "entry": entry,
            "card": card,
        })
    return rows, tuple(dict.fromkeys(gaps))


def _upgrade_value(card: Mapping[str, Any]) -> float | None:
    upgrade = card.get("upgrade")
    if not isinstance(upgrade, Mapping) or not upgrade:
        return None
    if set(upgrade) - {
        "damage", "block", "cards_draw", "energy_gain", "hit_count", "cost",
    }:
        return None
    value = 0.0
    recognized = False
    for field_name, weight in (
        ("damage", 0.65),
        ("block", 0.75),
        ("cards_draw", 5.0),
        ("energy_gain", 7.0),
        ("hit_count", 2.0),
    ):
        delta = _number(upgrade.get(field_name))
        if field_name in upgrade and delta is None:
            return None
        if delta is not None:
            recognized = True
            value += delta * weight
    if "cost" in upgrade:
        base_cost = card_energy_cost(card)
        upgraded_cost = card_energy_cost(card, upgrades=1)
        if base_cost is None or upgraded_cost is None:
            return None
        recognized = True
        value += max(-8.0, min(8.0, (base_cost - upgraded_cost) * 8.0))
    if not recognized:
        return None
    return round(max(0.0, min(18.0, value)), 4)


def _verified_targets(world: WorldState, operation: str, facts):
    """Validate complete snapshot-scoped facts; never infer game eligibility."""
    if facts is None:
        return (), ("deck_edit:targets_unknown",)
    if not isinstance(facts, Mapping) or set(facts) != {"operation", "source_code", "targets"}:
        return (), ("deck_edit:targets_invalid",)
    source = facts["source_code"]
    targets = facts["targets"]
    if (facts["operation"] != operation or not isinstance(source, str)
            or not source.strip() or len(source) > 200
            or not isinstance(targets, (list, tuple)) or len(targets) > 200):
        return (), ("deck_edit:targets_invalid",)
    remaining = Counter()
    for entry in world.deck:
        remaining[(entry.card, entry.upgrades)] += entry.count
    ids = set()
    for target in targets:
        if not isinstance(target, Mapping) or set(target) != {"target_id", "card", "upgrades"}:
            return (), ("deck_edit:targets_invalid",)
        if any(not isinstance(target[key], str) or not target[key].strip()
               or len(target[key]) > limit for key, limit in (("target_id", 240), ("card", 200))):
            return (), ("deck_edit:targets_invalid",)
        if (type(target["upgrades"]) is not int or not 0 <= target["upgrades"] <= 9
                or target["target_id"] in ids):
            return (), ("deck_edit:targets_invalid",)
        ids.add(target["target_id"])
        key = (target["card"], target["upgrades"])
        if remaining[key] <= 0:
            return (), ("deck_edit:targets_state_mismatch",)
        remaining[key] -= 1
    if not targets:
        return (), (f"deck_edit:no_{operation}_target",)
    return targets, ()


def best_upgrade_opportunity(
    world: WorldState,
    repository: RelationalRepository,
    target_facts=None,
) -> DeckEditOpportunity:
    targets, gaps = _verified_targets(world, "upgrade", target_facts)
    if gaps:
        return DeckEditOpportunity("upgrade", None, None, gaps)
    best_id = None
    best_value = -1.0
    for target in targets:
        if target["upgrades"] > 0:
            return DeckEditOpportunity("upgrade", None, None, ("deck_edit:repeat_upgrade_unmodeled",))
        card = repository.find_card(target["card"])
        if card is None:
            return DeckEditOpportunity("upgrade", None, None, (f"unknown_card:{target['card']}",))
        value = _upgrade_value(card)
        if value is None:
            return DeckEditOpportunity("upgrade", None, None, ("deck_edit:upgrade_delta_unknown",))
        if value > best_value:
            best_id = target["card"]
            best_value = value
    return DeckEditOpportunity("upgrade", best_id, best_value)


def _removal_value(
    *,
    world: WorldState,
    card: Mapping[str, Any],
    count: int,
    mechanic_context: Mapping[str, Any],
) -> float:
    type_key = str(card.get("type_key") or "").strip().lower()
    rarity = str(card.get("rarity_key") or "").strip().lower()
    card_id = str(card.get("id") or "")
    value = 0.0
    if type_key in {"curse", "status"} or rarity in {"curse", "status"}:
        value += 18.0
    if card_id.startswith(("STRIKE_", "DEFEND_")):
        value += 6.0
    if count >= 3:
        value += min(5.0, float(count - 2) * 1.5)
    if rarity in {"rare", "uncommon"}:
        value -= 4.0

    for signal in signals_for_card(world.character, card):
        if signal.role != "provider":
            continue
        complements = (
            mechanic_context.get("evidence", {})
            .get(signal.family, {})
        )
        complement_count = sum(
            int((complements.get(role) or {}).get("count") or 0)
            for role in ("payoff", "spender", "capacity", "multiplier")
        )
        if complement_count:
            value -= min(8.0, 2.0 + 1.5 * complement_count)
    return round(max(-12.0, min(22.0, value)), 4)


def best_remove_opportunity(
    world: WorldState,
    repository: RelationalRepository,
    target_facts=None,
) -> DeckEditOpportunity:
    targets, gaps = _verified_targets(world, "remove", target_facts)
    if gaps:
        return DeckEditOpportunity("remove", None, None, gaps)
    rows, gaps = _resolved_deck(world, repository)
    if gaps:
        return DeckEditOpportunity("remove", None, None, gaps)
    mechanic_context = build_mechanic_context(
        world.character,
        rows,
    )
    best_id = None
    best_value = -100.0
    for target in targets:
        row = next(row for row in rows if row["entry"].card == target["card"]
                   and row["entry"].upgrades == target["upgrades"])
        card, entry = row["card"], row["entry"]
        value = _removal_value(
            world=world,
            card=card,
            count=entry.count,
            mechanic_context=mechanic_context,
        )
        if value > best_value:
            best_id = entry.card
            best_value = value
    if best_id is None:
        return DeckEditOpportunity(
            "remove",
            None,
            None,
            ("deck_edit:no_remove_target",),
        )
    return DeckEditOpportunity("remove", best_id, best_value)


def score_deck_effect(
    draft: AssessmentDraft,
    effect: ChoiceEffect,
    world: WorldState,
    repository: RelationalRepository,
) -> None:
    if effect.certainty != "exact":
        draft.gap(f"deck_edit:{effect.kind}:uncertain")
        return
    if effect.target_mode == "choose" and effect.kind in {"upgrade_card", "remove_card"}:
        choices = [item for item in draft.candidate.payload.get("effects", ())
                   if isinstance(item, Mapping) and item.get("target_mode") == "choose"]
        if len(choices) != 1 or effect.amount not in (None, 1):
            draft.gap("deck_edit:multiple_selection_unmodeled")
            return
    if effect.kind == "upgrade_card":
        opportunity = (
            best_upgrade_opportunity(world, repository, draft.candidate.payload.get("deck_edit_targets"))
            if effect.target_mode == "choose"
            else _specific_upgrade(effect, repository)
        )
        _apply_opportunity(draft, opportunity)
    elif effect.kind == "remove_card":
        opportunity = (
            best_remove_opportunity(world, repository, draft.candidate.payload.get("deck_edit_targets"))
            if effect.target_mode == "choose"
            else _specific_removal(effect, world, repository,
                                   upgrades=draft.candidate.payload.get("upgrades"))
        )
        _apply_opportunity(draft, opportunity)
    elif effect.kind == "add_card":
        if not effect.entity_id:
            draft.gap("deck_edit:add_card_target_missing")
            return
        card = repository.find_card(effect.entity_id)
        if card is None:
            draft.gap(f"unknown_card:{effect.entity_id}")
            return
        type_key = str(card.get("type_key") or "").lower()
        intrinsic = 2.0
        if type_key in {"curse", "status"}:
            intrinsic = -16.0
        elif str(card.get("rarity_key") or "").lower() == "rare":
            intrinsic = 5.0
        draft.add(
            "deck_add_card",
            intrinsic,
            "按结构化卡牌类型计入牌组变化。",
            "deck_burden" if intrinsic < 0 else "long_term_growth",
        )
    elif effect.kind == "transform_card":
        draft.gap("deck_edit:transform_outcome_unknown")


def _specific_upgrade(
    effect: ChoiceEffect,
    repository: RelationalRepository,
) -> DeckEditOpportunity:
    if not effect.entity_id:
        return DeckEditOpportunity(
            "upgrade",
            None,
            None,
            ("deck_edit:upgrade_target_missing",),
        )
    card = repository.find_card(effect.entity_id)
    if card is None:
        return DeckEditOpportunity(
            "upgrade",
            None,
            None,
            (f"unknown_card:{effect.entity_id}",),
        )
    value = _upgrade_value(card)
    if value is None:
        return DeckEditOpportunity(
            "upgrade",
            None,
            None,
            ("deck_edit:upgrade_delta_unknown",),
        )
    return DeckEditOpportunity("upgrade", effect.entity_id, value)


def _specific_removal(
    effect: ChoiceEffect,
    world: WorldState,
    repository: RelationalRepository,
    *,
    upgrades: int | None = None,
) -> DeckEditOpportunity:
    if not effect.entity_id:
        return DeckEditOpportunity(
            "remove",
            None,
            None,
            ("deck_edit:remove_target_missing",),
        )
    rows, gaps = _resolved_deck(world, repository)
    if gaps:
        return DeckEditOpportunity("remove", None, None, gaps)
    row = next(
        (
            item
            for item in rows
            if item["entry"].card == effect.entity_id
            and (upgrades is None or item["entry"].upgrades == upgrades)
        ),
        None,
    )
    if row is None or row["card"] is None:
        return DeckEditOpportunity(
            "remove",
            None,
            None,
            ("deck_edit:remove_target_not_in_deck",),
        )
    context = build_mechanic_context(world.character, rows)
    value = _removal_value(
        world=world,
        card=row["card"],
        count=row["entry"].count,
        mechanic_context=context,
    )
    return DeckEditOpportunity("remove", effect.entity_id, value)


def _apply_opportunity(
    draft: AssessmentDraft,
    opportunity: DeckEditOpportunity,
) -> None:
    if opportunity.value is None:
        for gap in opportunity.data_gaps:
            draft.gap(gap)
        return
    if opportunity.operation == "upgrade":
        draft.add(
            "deck_upgrade_opportunity",
            opportunity.value,
            f"当前最佳升级目标为 {opportunity.target_id}。",
            "long_term_growth",
        )
    else:
        draft.add(
            "deck_remove_opportunity",
            opportunity.value,
            f"当前最佳移除目标为 {opportunity.target_id}。",
            "deck_burden",
        )

"""Deterministic upgrade/removal opportunity values from structured cards."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from advisor.character_mechanics import (
    build_mechanic_context,
    signals_for_card,
)
from advisor.choice_effects import AssessmentDraft, ChoiceEffect
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
        if delta is not None:
            recognized = True
            value += delta * weight
    upgraded_cost = _number(upgrade.get("cost"))
    base_cost = _number(card.get("cost"))
    if upgraded_cost is not None and base_cost is not None:
        recognized = True
        value += max(-8.0, min(8.0, (base_cost - upgraded_cost) * 8.0))
    if not recognized:
        return None
    return round(max(0.0, min(18.0, value)), 4)


def best_upgrade_opportunity(
    world: WorldState,
    repository: RelationalRepository,
) -> DeckEditOpportunity:
    rows, gaps = _resolved_deck(world, repository)
    if gaps:
        return DeckEditOpportunity("upgrade", None, None, gaps)
    best_id = None
    best_value = -1.0
    unsupported_upgrade = False
    for row in rows:
        entry = row["entry"]
        card = row["card"]
        if entry.upgrades > 0 or card is None:
            continue
        upgrade = card.get("upgrade")
        if not isinstance(upgrade, Mapping) or not upgrade:
            continue
        value = _upgrade_value(card)
        if value is None:
            unsupported_upgrade = True
            continue
        if value > best_value:
            best_id = entry.card
            best_value = value
    if best_id is None:
        gap = (
            ("deck_edit:upgrade_delta_unknown",)
            if unsupported_upgrade
            else ("deck_edit:no_upgrade_target",)
        )
        return DeckEditOpportunity("upgrade", None, None, gap)
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
) -> DeckEditOpportunity:
    rows, gaps = _resolved_deck(world, repository)
    if gaps:
        return DeckEditOpportunity("remove", None, None, gaps)
    mechanic_context = build_mechanic_context(
        world.character,
        rows,
    )
    best_id = None
    best_value = -100.0
    for row in rows:
        card = row["card"]
        entry = row["entry"]
        if card is None:
            continue
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
    if effect.kind == "upgrade_card":
        opportunity = (
            best_upgrade_opportunity(world, repository)
            if effect.target_mode == "choose"
            else _specific_upgrade(effect, repository)
        )
        _apply_opportunity(draft, opportunity)
    elif effect.kind == "remove_card":
        opportunity = (
            best_remove_opportunity(world, repository)
            if effect.target_mode == "choose"
            else _specific_removal(effect, world, repository)
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

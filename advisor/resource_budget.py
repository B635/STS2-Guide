"""Pure cross-decision resource budgets derived from one WorldState."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from advisor.choice_effects import AssessmentDraft, ChoiceEffect, EffectBundle
from advisor.decision_core import WorldState
from storage.relational import RelationalRepository


@dataclass(frozen=True)
class HealthBudget:
    hp: int | None
    max_hp: int | None
    safety_floor: int | None
    spendable_hp: int | None
    pressure: float

    @property
    def known(self) -> bool:
        return (
            self.hp is not None
            and self.max_hp is not None
            and self.safety_floor is not None
        )


@dataclass(frozen=True)
class GoldBudget:
    current: int | None

    @property
    def known(self) -> bool:
        return self.current is not None


@dataclass(frozen=True)
class PotionBudget:
    occupied_slots: int
    max_slots: int | None
    free_slots: int | None
    slot_identity_valid: bool

    @property
    def known(self) -> bool:
        return self.max_slots is not None and self.slot_identity_valid


@dataclass(frozen=True)
class ResourceBudget:
    health: HealthBudget
    gold: GoldBudget
    potions: PotionBudget


def _route_pressure(world: WorldState) -> float:
    if world.map is None:
        return 0.0
    node_by_id = {node.node_id: node for node in world.map.nodes}
    frontier = list(world.map.available_next_node_ids)
    visited: set[str] = set()
    pressure = 0.0
    weights = (1.0, 0.6, 0.3)
    danger = {"MONSTER": 0.35, "ELITE": 1.1, "BOSS": 1.5}
    for depth, weight in enumerate(weights):
        next_frontier: list[str] = []
        kinds: list[str] = []
        for node_id in frontier:
            if node_id in visited:
                continue
            visited.add(node_id)
            node = node_by_id.get(node_id)
            if node is None:
                continue
            kinds.append(node.kind.upper())
            next_frontier.extend(node.edges)
        if kinds:
            pressure += (
                sum(danger.get(kind, 0.0) for kind in kinds)
                / len(kinds)
                * weight
            )
        frontier = next_frontier
        if not frontier:
            break
    return round(max(0.0, min(3.0, pressure)), 4)


def build_resource_budget(world: WorldState) -> ResourceBudget:
    pressure = _route_pressure(world)
    hp = (
        world.hp
        if isinstance(world.hp, int) and not isinstance(world.hp, bool)
        else None
    )
    max_hp = (
        world.max_hp
        if isinstance(world.max_hp, int)
        and not isinstance(world.max_hp, bool)
        else None
    )
    safety_floor = None
    spendable = None
    if hp is not None and max_hp is not None and max_hp > 0:
        ratio = 0.30 + min(0.20, pressure * 0.08)
        if world.route_mode == "survival":
            ratio += 0.10
        safety_floor = min(max_hp, max(1, math.ceil(max_hp * ratio)))
        spendable = max(0, hp - safety_floor)

    slots = [potion.slot for potion in world.potions]
    slot_identity_valid = (
        len(slots) == len(set(slots))
        and all(slot >= 0 for slot in slots)
    )
    max_slots = (
        world.max_potion_slots
        if isinstance(world.max_potion_slots, int)
        and not isinstance(world.max_potion_slots, bool)
        and world.max_potion_slots >= 0
        else None
    )
    free_slots = (
        max(0, max_slots - len(slots))
        if max_slots is not None and slot_identity_valid
        else None
    )
    gold = (
        world.gold
        if isinstance(world.gold, int)
        and not isinstance(world.gold, bool)
        and world.gold >= 0
        else None
    )
    return ResourceBudget(
        health=HealthBudget(
            hp=hp,
            max_hp=max_hp,
            safety_floor=safety_floor,
            spendable_hp=spendable,
            pressure=pressure,
        ),
        gold=GoldBudget(current=gold),
        potions=PotionBudget(
            occupied_slots=len(slots),
            max_slots=max_slots,
            free_slots=free_slots,
            slot_identity_valid=slot_identity_valid,
        ),
    )


def score_price(
    draft: AssessmentDraft,
    budget: ResourceBudget,
    price: object,
) -> None:
    if (
        isinstance(price, bool)
        or not isinstance(price, int)
        or price < 0
    ):
        draft.gap("resource:actual_price_missing")
        return
    if not budget.gold.known:
        draft.gap("resource:gold_missing")
        return
    current = int(budget.gold.current or 0)
    if price > current:
        draft.gap("resource:eligibility_gold_inconsistent")
        return
    if price == 0:
        draft.add(
            "gold_cost_free",
            3.0,
            "该选项不消耗金币。",
            "resource_efficiency",
        )
        return
    spent_ratio = price / max(1, current)
    delta = -min(18.0, 3.0 + 12.0 * spent_ratio)
    draft.add(
        "gold_liquidity_cost",
        delta,
        "按当前真实价格计入购买后的金币灵活性。",
        "resource_efficiency",
    )
    remaining = current - price
    if remaining < 50:
        draft.add(
            "gold_low_liquidity",
            -3.0,
            "购买后剩余金币很少。",
            "resource_efficiency",
        )


def _exact_amount(effect: ChoiceEffect) -> float | None:
    return effect.amount if effect.certainty == "exact" else None


def score_resource_effects(
    draft: AssessmentDraft,
    budget: ResourceBudget,
    bundle: EffectBundle,
    *,
    replacement_supported: bool = False,
) -> None:
    for effect in bundle.effects:
        if effect.certainty != "exact":
            draft.gap(
                f"effect:{effect.kind}:"
                f"{effect.certainty}:{effect.source_code}"
            )
            continue
        amount = _exact_amount(effect)
        if effect.kind == "hp_delta":
            if not budget.health.known or amount is None:
                draft.gap("resource:hp_missing")
                continue
            hp = int(budget.health.hp or 0)
            max_hp = int(budget.health.max_hp or 0)
            after = max(0, min(max_hp, hp + int(amount)))
            effective = after - hp
            if effective > 0:
                urgency = 1.0 + budget.health.pressure
                draft.add(
                    "hp_recovery",
                    min(22.0, effective / max(1, max_hp) * 40.0 * urgency),
                    "按缺失生命和近期真实路线压力计算恢复价值。",
                    "survival",
                )
            elif effective < 0:
                floor = int(budget.health.safety_floor or 1)
                draft.add(
                    "hp_cost",
                    -min(30.0, abs(effective) / max(1, max_hp) * 45.0),
                    "该选项会消耗当前生命。",
                    "survival",
                )
                if after < floor:
                    draft.safety_violation = True
                    draft.add(
                        "hp_safety_floor_violation",
                        -80.0,
                        "该选项会使生命低于当前安全底线。",
                        "survival",
                    )
        elif effect.kind == "max_hp_delta":
            if not budget.health.known or amount is None:
                draft.gap("resource:max_hp_missing")
                continue
            delta = max(-20.0, min(20.0, amount * 1.4))
            draft.add(
                "max_hp_change",
                delta,
                "最大生命变化影响后续整局生存空间。",
                "long_term_growth",
            )
        elif effect.kind == "gold_delta":
            if not budget.gold.known or amount is None:
                draft.gap("resource:gold_missing")
                continue
            if int(budget.gold.current or 0) + int(amount) < 0:
                draft.gap("resource:gold_effect_infeasible")
                continue
            draft.add(
                "gold_change",
                max(-18.0, min(18.0, amount / 12.0)),
                "金币变化按当前局资源灵活性计入。",
                "resource_efficiency",
            )
        elif effect.kind == "add_potion":
            if not budget.potions.known:
                draft.gap("resource:potion_slots_missing")
            elif int(budget.potions.free_slots or 0) > 0:
                draft.add(
                    "potion_free_slot",
                    5.0,
                    "当前存在空药水槽。",
                    "resource_efficiency",
                )
            elif replacement_supported:
                draft.add(
                    "potion_replacement_cost",
                    -3.0,
                    "药水槽已满，计入替换现有药水的机会成本。",
                    "resource_efficiency",
                )
            else:
                draft.gap("resource:potion_replacement_unresolved")
        elif effect.kind == "remove_potion":
            if not budget.potions.known:
                draft.gap("resource:potion_slots_missing")


def contains_kind(effects: Iterable[ChoiceEffect], kind: str) -> bool:
    return any(effect.kind == kind for effect in effects)


def score_effect_bundle(
    draft: AssessmentDraft,
    budget: ResourceBudget,
    bundle: EffectBundle,
    world: WorldState,
    repository: RelationalRepository,
    *,
    replacement_supported: bool = False,
) -> None:
    """Score an allowlisted effect bundle without interpreting any text."""
    score_resource_effects(
        draft,
        budget,
        bundle,
        replacement_supported=replacement_supported,
    )
    from advisor.deck_edit_value import score_deck_effect

    for effect in bundle.effects:
        if effect.certainty != "exact":
            continue
        if effect.kind in {
            "add_card",
            "remove_card",
            "upgrade_card",
            "transform_card",
        }:
            score_deck_effect(draft, effect, world, repository)
        elif effect.kind in {"add_relic", "remove_relic"}:
            if not effect.entity_id:
                draft.gap(f"effect:{effect.kind}:entity_missing")
                continue
            entity = repository.find_relic(effect.entity_id)
            if entity is None:
                draft.gap(f"unknown_relic:{effect.entity_id}")
                continue
            delta = 8.0 if effect.kind == "add_relic" else -8.0
            draft.add(
                "relic_change",
                delta,
                "按结构化遗物身份计入长期资源变化。",
                "long_term_growth",
            )
        elif effect.kind in {"add_potion", "remove_potion"}:
            if not effect.entity_id:
                draft.gap(f"effect:{effect.kind}:entity_missing")
                continue
            entity = repository.find_entity("potions", effect.entity_id)
            if entity is None:
                draft.gap(f"unknown_potion:{effect.entity_id}")
                continue
            draft.add(
                "potion_inventory_change",
                3.0 if effect.kind == "add_potion" else -3.0,
                "按结构化药水身份计入一次性资源变化。",
                "resource_efficiency",
            )
        elif effect.kind == "start_combat":
            if not budget.health.known:
                draft.gap("resource:hp_missing")
                continue
            hp_ratio = float(budget.health.hp or 0) / max(
                1,
                int(budget.health.max_hp or 1),
            )
            draft.add(
                "forced_combat_risk",
                -min(22.0, 5.0 + (1.0 - hp_ratio) * 18.0),
                "进入战斗的代价按当前生命状态计入。",
                "survival",
            )
        elif effect.kind == "followup_choice":
            draft.gap("effect:followup_choice_unresolved")
        elif effect.kind == "no_op":
            draft.add(
                "no_immediate_effect",
                0.0,
                "该候选没有已建模的即时资源变化。",
                "resource_efficiency",
            )

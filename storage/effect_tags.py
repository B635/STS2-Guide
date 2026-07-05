"""Deterministic structured effect tags derived from the local catalog.

The tags are deliberately small and auditable.  Numeric catalog fields are
preferred; description matching only fills semantic gaps that the source
catalog does not expose as dedicated fields.  No LLM or network call belongs
in this path.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


EFFECT_TAG_VERSION = "3"
EffectTags = Dict[str, Tuple[float, str]]


def _number(value) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strings(value) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _card_exhausts_itself(identity: str) -> bool:
    """True when the card exhausts *itself*, not other cards or draw pile."""
    # Self-exhaust markers: the card's own keyword is "耗竭" or "Exhaust"
    # appearing as a standalone keyword (not "消耗牌堆" / "exhaust pile").
    if re.search(
        r"\b耗竭\b(?!\s*(牌|堆|all|pile|random|top|bottom))",
        identity,
    ):
        return True
    if re.search(
        r"\bexhaust\b(?!\s*(pile|all|random|top|bottom))",
        identity,
    ):
        return True
    return False


def _card_is_ethereal(identity: str) -> bool:
    """True when the card has the Ethereal keyword on itself."""
    return bool(
        re.search(r"(?<!\S)虚无(?!牌|卡)", identity)
        or re.search(r"(?<!\S)ethereal(?!\s*card)", identity)
    )


def _card_mentions_ethereal_cards(identity: str) -> bool:
    """True when the card interacts with ethereal cards (not itself)."""
    return bool(
        re.search(r"虚无牌|虚无卡|ethereal\s*card", identity)
    )


def _card_interacts_with_exhaust(identity: str) -> bool:
    """True for explicit exhaust payoffs or exhaust-pile interactions."""
    return bool(
        re.search(
            r"(每当|每次|当).{0,18}(牌|卡).{0,8}(耗竭|消耗)",
            identity,
        )
        or re.search(r"从消耗牌堆|exhaust\s*pile", identity)
        or re.search(r"(whenever|when).{0,20}(card)?.{0,8}exhaust", identity)
    )


def derive_effect_tags(entity_type: str, item: Dict) -> EffectTags:
    """Return normalized ``tag -> (magnitude, source)`` values."""
    tags: EffectTags = {}

    def add(
        tag: str,
        magnitude: float = 1.0,
        source: str = "catalog",
    ) -> None:
        magnitude = float(magnitude)
        existing = tags.get(tag)
        if existing is None or abs(magnitude) > abs(existing[0]):
            tags[tag] = (magnitude, source)

    description = str(item.get("description") or "")
    identity = " ".join(
        [
            str(item.get("id") or ""),
            str(item.get("name") or ""),
            description,
            str(item.get("upgrade_description") or ""),
            *list(_strings(item.get("keywords_key"))),
            *list(_strings(item.get("tags"))),
        ]
    ).lower()

    # ── Shared semantic patterns (cards, relics, potions) ────────────
    shared_strength = r"力量|strength"
    shared_vulnerable = r"易伤|vulnerab"
    shared_weak = r"虚弱|weak"
    shared_poison = r"中毒|poison"
    shared_draw = r"抽\d*张|draw"
    shared_block = r"格挡|block"
    shared_energy = r"能量|energy"
    shared_exhaust = r"耗竭|exhaust"
    shared_discard = r"弃牌|discard"
    shared_healing = r"恢复.{0,8}生命|回复.{0,8}生命|heal"

    # ── Cards ────────────────────────────────────────────────────────
    if entity_type == "cards":
        card_type = str(item.get("type_key") or "").strip().lower()
        if card_type:
            add(f"card_type:{card_type}", source="type_key")

        for field, tag in (
            ("damage", "damage"),
            ("block", "block"),
            ("cards_draw", "draw"),
            ("energy_gain", "energy_gain"),
            ("hp_loss", "self_harm"),
        ):
            value = _number(item.get(field))
            if value is not None and value > 0:
                add(tag, value, field)

        hit_count = _number(item.get("hit_count"))
        if hit_count is not None and hit_count > 1:
            add("multi_hit", hit_count, "hit_count")

        target = str(item.get("target") or "").lower()
        if "all" in target or re.search(r"所有敌人|all enem", identity):
            add("aoe", source="target")
        if item.get("is_x_cost") or item.get("is_x_star_cost"):
            add("x_cost", source="cost")
        if item.get("spawns_cards"):
            add("card_generation", source="spawns_cards")

        semantic_patterns = {
            "strength": shared_strength,
            "vulnerable": shared_vulnerable,
            "weak": shared_weak,
            "poison": shared_poison,
            "exhaust": shared_exhaust,
            "discard": shared_discard,
            "retain": r"保留|retain",
            "upgrade_hand": r"升级.{0,8}(手牌|牌)|upgrade.{0,12}hand",
            "cost_reduction": (
                r"(耗能|费用|cost).{0,12}(减少|降低|less|reduc)"
            ),
            "healing": shared_healing,
            "scaling": r"每当|每次|本场战斗|whenever|each time|this combat",
        }
        for tag, pattern in semantic_patterns.items():
            if re.search(pattern, identity, flags=re.IGNORECASE):
                add(tag, source="description_rule")

        if "draw" not in tags and re.search(shared_draw, identity):
            add("draw", source="description_rule")
        if (
            "energy_gain" not in tags
            and re.search(r"获得.{0,6}能量|gain.{0,8}energy", identity)
        ):
            add("energy_gain", source="description_rule")
        if "self_harm" not in tags and re.search(
            r"失去.{0,8}生命|lose.{0,8}hp",
            identity,
        ):
            add("self_harm", source="description_rule")

        # ── Self-exhaust / ethereal (tightened) ─────────────────────
        if _card_exhausts_itself(identity):
            add("self_exhaust", source="description_rule")
        if _card_is_ethereal(identity):
            add("ethereal", source="description_rule")
        if _card_mentions_ethereal_cards(identity):
            add("ethereal_interaction", source="description_rule")
        if _card_interacts_with_exhaust(identity):
            add("exhaust_interaction", source="description_rule")

        # ── Conditional effects ─────────────────────────────────────
        _detect_conditional_effects(identity, tags, add)

        for power in item.get("powers_applied") or []:
            if not isinstance(power, dict):
                continue
            power_id = str(power.get("id") or "").strip().lower()
            if power_id:
                add(f"power:{power_id}", source="powers_applied")

    # ── Relics ───────────────────────────────────────────────────────
    elif entity_type == "relics":
        support_patterns = {
            "supports_attack": r"攻击牌|attack card|攻击|attack",
            "supports_skill": r"技能牌|skill card",
            "supports_power": r"能力牌|power card",
            "supports_block": shared_block,
            "supports_draw": shared_draw,
            "supports_energy": shared_energy,
            "supports_strength": shared_strength,
            "supports_vulnerable": shared_vulnerable,
            "supports_weak": shared_weak,
            "supports_poison": shared_poison,
            "supports_exhaust": shared_exhaust,
            "supports_discard": shared_discard,
            "healing": shared_healing,
        }
        for tag, pattern in support_patterns.items():
            if re.search(pattern, identity, flags=re.IGNORECASE):
                add(tag, source="description_rule")

    return tags


def _detect_conditional_effects(
    identity: str,
    tags: EffectTags,
    add: callable,
) -> None:
    """Tag conditional effects so scoring can reward meeting the condition.

    Patterns require the condition to be about the *target* (enemy), not
    the player or a generic state, to avoid false positives from cards
    that merely mention a keyword in passing.
    """
    conditions = {
        "cond:vulnerable_target": (
            r"(如果|若|when|if).{0,18}(目标|敌人|敌方|target|enemy).{0,10}(易伤|vulnerab)"
        ),
        "cond:weak_target": (
            r"(如果|若|when|if).{0,18}(目标|敌人|敌方|target|enemy).{0,10}(虚弱|weak)"
        ),
        "cond:poison_target": (
            r"(如果|若|when|if).{0,18}(目标|敌人|敌方|target|enemy).{0,10}(中毒|poison)"
        ),
        "cond:discard_synergy": (
            r"(弃牌时|discarded|on discard)"
        ),
        "cond:exhaust_synergy": (
            r"(耗竭时|exhausted|on exhaust)"
        ),
        "cond:strength_scaling": (
            r"力量.*(乘以|倍|multipl|×|times)"
        ),
        "cond:low_hp": (
            r"(生命.{0,8}低于|生命.{0,8}不足|below.{0,10}hp|hp.{0,10}below)"
        ),
        "cond:full_hp": (
            r"(满血|生命值满|full hp)"
        ),
        "cond:empty_draw_pile": (
            r"(抽牌堆.*空|empty.*draw pile|no cards.*draw pile)"
        ),
        "cond:last_card": (
            r"(本回合.*最后|last.*this turn|final.*turn)"
        ),
    }
    for tag, pattern in conditions.items():
        if re.search(pattern, identity, flags=re.IGNORECASE):
            add(tag, source="description_rule")

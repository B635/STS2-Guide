"""Deterministic structured effect tags derived from the local catalog.

The tags are deliberately small and auditable.  Numeric catalog fields are
preferred; description matching only fills semantic gaps that the source
catalog does not expose as dedicated fields.  No LLM or network call belongs
in this path.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


EFFECT_TAG_VERSION = "4"
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


def _normalized_values(value) -> set[str]:
    return {
        item.strip().upper()
        for item in _strings(value)
        if item.strip()
    }


def derive_mechanic_effect_tags(item: Dict) -> EffectTags:
    """Derive versioned character-mechanic facts from one card.

    Structured catalog fields win.  Description rules only fill facts that
    the current Spire Codex snapshot does not expose in a dedicated field.
    The result is still static catalog data; no runtime state is guessed here.
    """
    tags: EffectTags = {}

    def add(
        domain: str,
        role: str,
        magnitude: float = 1.0,
        source: str = "catalog",
    ) -> None:
        key = f"mechanic:{domain}:{role}"
        numeric = max(0.0, float(magnitude))
        existing = tags.get(key)
        if existing is None or numeric > existing[0]:
            tags[key] = (numeric, source)

    color = str(item.get("color") or "").strip().lower()
    description = " ".join(
        (
            str(item.get("description") or ""),
            str(item.get("upgrade_description") or ""),
        )
    ).lower()
    variables = item.get("vars") or {}
    if not isinstance(variables, dict):
        variables = {}
    keywords = _normalized_values(item.get("keywords_key"))
    static_tags = _normalized_values(item.get("tags"))
    spawned_cards = _normalized_values(item.get("spawns_cards"))

    if color == "ironclad":
        hp_loss = _number(item.get("hp_loss"))
        if hp_loss is not None and hp_loss > 0:
            add("blood", "provider", hp_loss, "hp_loss")
        elif re.search(
            r"(每当|每次|whenever|when).{0,24}"
            r"(你|you).{0,12}(失去.{0,8}生命|lose.{0,8}hp)",
            description,
            flags=re.IGNORECASE,
        ):
            add("blood", "payoff", source="description_rule")

        if "EXHAUST" in keywords:
            add("exhaust", "provider", source="keywords_key")
        if _card_interacts_with_exhaust(description):
            add("exhaust", "payoff", source="description_rule")

        strength = _number(variables.get("Strength"))
        if strength is not None and strength > 0:
            add("strength", "provider", strength, "vars.Strength")
        hit_count = _number(item.get("hit_count"))
        if hit_count is not None and hit_count > 1:
            add("strength", "payoff", hit_count, "hit_count")

    elif color == "silent":
        if re.search(
            r"(丢弃|弃掉|舍弃).{0,8}\d*.{0,3}(张|牌)|"
            r"discard.{0,10}(card|\d)",
            description,
            flags=re.IGNORECASE,
        ):
            add("discard", "provider", source="description_rule")
        if "SLY" in keywords:
            add("discard", "payoff", source="keywords_key")

        accuracy = _number(variables.get("Accuracy"))
        shiv_multiplier = (
            accuracy is not None
            and accuracy > 0
        ) or bool(re.search(
            r"(小刀|shiv).{0,18}(额外|所有敌人|保留)|"
            r"(extra|retain|all enem).{0,18}(小刀|shiv)",
            description,
            flags=re.IGNORECASE,
        ))
        shiv_payoff = bool(re.search(
            r"(消耗牌堆|exhaust pile).{0,18}(小刀|shiv).{0,12}(打出|play)",
            description,
            flags=re.IGNORECASE,
        ))
        if accuracy is not None and accuracy > 0:
            add("shiv", "multiplier", accuracy, "vars.Accuracy")
        elif shiv_multiplier:
            add("shiv", "multiplier", source="description_rule")
        if shiv_payoff:
            add("shiv", "payoff", source="description_rule")
        shiv_generation = bool(re.search(
            r"(将|添加|加入|放入|add|create|put).{0,24}"
            r"(小刀|shiv)|"
            r"(小刀|shiv).{0,24}(添加|加入|放入|add|create|put)",
            description,
            flags=re.IGNORECASE,
        )) and not shiv_payoff
        if "SHIV" in spawned_cards and shiv_generation:
            amount = _number(variables.get("Cards")) or 1.0
            add("shiv", "provider", amount, "spawns_cards")

        poison = _number(variables.get("Poison"))
        if poison is not None and poison > 0:
            add("poison", "provider", poison, "vars.Poison")
        accelerant = _number(variables.get("Accelerant"))
        if accelerant is not None and accelerant > 0:
            add("poison", "payoff", accelerant, "vars.Accelerant")

    elif color == "defect":
        focus = _number(variables.get("Focus"))
        loses_focus = bool(re.search(
            r"失去.{0,8}集中|lose.{0,8}focus",
            description,
            flags=re.IGNORECASE,
        ))
        if focus is not None and focus > 0 and not loses_focus:
            add("orb", "multiplier", focus, "vars.Focus")

        orb_slots = _number(variables.get("OrbSlots"))
        if orb_slots is not None and orb_slots > 0:
            add("orb", "capacity", orb_slots, "vars.OrbSlots")
        elif re.search(
            r"(充能球.{0,4}(栏位|槽位|球槽)|orb.{0,8}slot)",
            description,
            flags=re.IGNORECASE,
        ):
            amount = _number(variables.get("Repeat")) or 1.0
            add("orb", "capacity", amount, "description_rule")

        channels = bool(
            re.search(
                r"(生成|引导|channel).{0,18}(充能球|orb)",
                description,
                flags=re.IGNORECASE,
            )
        )
        if channels:
            orb_types = (
                ("lightning", r"闪电|lightning"),
                ("frost", r"冰霜|frost"),
                ("dark", r"黑暗|dark"),
                ("plasma", r"等离子|plasma"),
                ("glass", r"玻璃|glass"),
            )
            matched = False
            for orb_type, pattern in orb_types:
                if re.search(pattern, description, flags=re.IGNORECASE):
                    add(
                        f"orb.{orb_type}",
                        "provider",
                        source="description_rule",
                    )
                    matched = True
            if not matched:
                add("orb.any", "provider", source="description_rule")

        if re.search(r"激发|evoke", description, flags=re.IGNORECASE):
            add("orb", "payoff", source="description_rule")
        elif not channels and re.search(
            r"(每有|每个|当前每有).{0,14}(充能球|orb)",
            description,
            flags=re.IGNORECASE,
        ):
            add("orb", "payoff", source="description_rule")

    elif color == "regent":
        stars = _number(variables.get("Stars"))
        if stars is not None and stars > 0:
            add("star", "provider", stars, "vars.Stars")

        star_cost = _number(item.get("star_cost"))
        if star_cost is None:
            star_cost = _number(variables.get("StarCost"))
        if star_cost is not None and star_cost > 0:
            add("star", "spender", star_cost, "star_cost")
        elif bool(item.get("is_x_star_cost")):
            add("star", "spender", 1.0, "is_x_star_cost")

        forge = _number(variables.get("Forge"))
        if forge is not None and forge > 0:
            add("forge", "provider", forge, "vars.Forge")
        if (
            "SOVEREIGN_BLADE" in spawned_cards
            or re.search(r"君王之剑|sovereign blade", description)
        ):
            add(
                "forge",
                "payoff",
                source=(
                    "spawns_cards"
                    if "SOVEREIGN_BLADE" in spawned_cards
                    else "description_rule"
                ),
            )

    elif color == "necrobinder":
        summon = _number(variables.get("Summon"))
        if summon is not None and summon > 0:
            add("osty", "provider", summon, "vars.Summon")
        osty_damage = _number(variables.get("OstyDamage"))
        if "OSTYATTACK" in static_tags or (
            osty_damage is not None and osty_damage > 0
        ):
            add(
                "osty",
                "payoff",
                osty_damage or 1.0,
                (
                    "vars.OstyDamage"
                    if osty_damage is not None
                    else "tags:OstyAttack"
                ),
            )

        doom = _number(variables.get("Doom"))
        if doom is not None and doom > 0:
            add("doom", "provider", doom, "vars.Doom")
        doom_threshold = _number(variables.get("DoomThreshold"))
        if doom_threshold is not None and doom_threshold > 0:
            add(
                "doom",
                "payoff",
                doom_threshold,
                "vars.DoomThreshold",
            )
        elif re.search(
            r"(每当|每次|whenever|when).{0,20}"
            r"(给予|施加|apply).{0,8}(灾厄|厄运|doom)",
            description,
            flags=re.IGNORECASE,
        ):
            add("doom", "payoff", source="description_rule")

        soul_payoff = bool(
            re.search(
                r"(每有|每张|每一张).{0,18}(灵魂|soul)|"
                r"(消耗牌堆|exhaust pile).{0,18}(灵魂|soul)|"
                r"(每当|每次|whenever|when).{0,20}"
                r"(打出|play).{0,8}(灵魂|soul)",
                description,
                flags=re.IGNORECASE,
            )
        )
        if soul_payoff:
            add("soul", "payoff", source="description_rule")
        elif "SOUL" in spawned_cards and re.search(
            r"(加入|添加|放入|将).{0,24}(灵魂|soul)|"
            r"(灵魂|soul).{0,24}(加入|添加|放入)|"
            r"(变为|变换为|transform).{0,8}(灵魂|soul)",
            description,
            flags=re.IGNORECASE,
        ):
            amount = _number(variables.get("Cards")) or 1.0
            add("soul", "provider", amount, "spawns_cards")

    return tags


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
            ("energy_gain", "energy_gain"),
        ):
            value = _number(item.get(field))
            if value is not None and value > 0:
                add(tag, value, field)

        hp_loss = _number(item.get("hp_loss"))
        enemy_hp_loss = bool(re.search(
            r"(敌人|敌方|enemy).{0,16}(失去.{0,8}生命|lose.{0,8}hp)",
            description,
            flags=re.IGNORECASE,
        ))
        if hp_loss is not None and hp_loss > 0 and not enemy_hp_loss:
            add("self_harm", hp_loss, "hp_loss")

        cards_draw = _number(item.get("cards_draw"))
        if cards_draw is not None and cards_draw > 0:
            generated_only = bool(item.get("spawns_cards")) and not re.search(
                r"抽\d*张|抽牌|draw",
                description,
                flags=re.IGNORECASE,
            )
            if not generated_only:
                add("draw", cards_draw, "cards_draw")

        hit_count = _number(item.get("hit_count"))
        if hit_count is not None and hit_count > 1:
            add("multi_hit", hit_count, "hit_count")

        target = str(item.get("target") or "").lower()
        if "all" in target or re.search(r"所有敌人|all enem", identity):
            add("aoe", source="target")
        if item.get("is_x_cost") or item.get("is_x_star_cost"):
            add("x_cost", source="cost")
        generated_card_action = bool(re.search(
            r"(添加|加入|放入|生成|变为|变换为|"
            r"add|create|put|shuffle|transform).{0,28}"
            r"(手牌|牌堆|弃牌堆|card|hand|deck|pile)|"
            r"(牌|card).{0,28}(添加|加入|放入|生成|"
            r"add|create|put|shuffle)",
            description,
            flags=re.IGNORECASE,
        ))
        if item.get("spawns_cards") and generated_card_action:
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
            "scaling": r"本场战斗|永久|this combat|permanent",
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
        triggered_hp_loss = re.search(
            r"(每当|每次|如果|若|whenever|when|if).{0,28}"
            r"(失去.{0,8}生命|lose.{0,8}hp)",
            identity,
        )
        if (
            "self_harm" not in tags
            and not triggered_hp_loss
            and re.search(
                r"失去\d+.{0,5}生命|lose\s+\d+.{0,5}hp",
                identity,
            )
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

        for tag, (magnitude, source) in derive_mechanic_effect_tags(
            item
        ).items():
            add(tag, magnitude, source)

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

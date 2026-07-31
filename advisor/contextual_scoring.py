"""Transparent state factors for real-time card-reward scoring."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

from advisor.character_mechanics import (
    assess_card_mechanics,
    build_mechanic_context,
    supports_character,
)
from storage.relational import RelationalRepository


def _factor(code: str, delta: float, message: str) -> Dict:
    return {
        "code": code,
        "delta": round(float(delta), 4),
        "message": message,
    }


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _cost(card: Dict) -> Optional[int]:
    value = card.get("cost")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _effects(entity: Optional[Dict]) -> Dict[str, float]:
    if not entity:
        return {}
    return {
        str(tag): float(value)
        for tag, value in (entity.get("_effect_tags") or {}).items()
    }


def build_context(
    state: Dict,
    resolved_deck: Iterable[Dict],
    repository: RelationalRepository,
) -> Dict:
    """Build one reusable context for all candidates in an offer."""
    resolved_deck = list(resolved_deck)
    effect_counts: Counter = Counter()
    for row in resolved_deck:
        card = row.get("card")
        if not card:
            continue
        for tag in _effects(card):
            effect_counts[tag] += 1

    relic_effects: Counter = Counter()
    resolved_relics = 0
    relic_ids = list(state.get("relics") or [])
    if not relic_ids:
        relic_ids = [
            relic.get("relic")
            for relic in state.get("relic_states") or []
            if relic.get("relic")
        ]
    for relic_id in dict.fromkeys(relic_ids):
        relic = repository.find_relic(relic_id)
        if relic is None:
            continue
        resolved_relics += 1
        for tag in _effects(relic):
            relic_effects[tag] += 1

    hp_ratio = None
    hp = state.get("hp")
    max_hp = state.get("max_hp")
    if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)):
        if max_hp > 0:
            hp_ratio = _bounded(float(hp) / float(max_hp), 0.0, 1.0)

    character = str(state.get("character") or "").strip().upper()
    preferences = state.get("guide_preferences")
    if preferences is None:
        route_mode = "balanced"
    elif not isinstance(preferences, dict):
        raise ValueError("guide_preferences must be an object")
    else:
        route_mode = preferences.get("route_mode")
        if route_mode not in {"balanced", "survival", "growth"}:
            raise ValueError("invalid guide_preferences.route_mode")

    return {
        "character": character,
        "act": max(1, int(state.get("act") or 1)),
        "floor": max(0, int(state.get("floor") or 0)),
        "hp_ratio": hp_ratio,
        "deck_effect_counts": effect_counts,
        "relic_effects": relic_effects,
        "resolved_relics": resolved_relics,
        "route": _threat_profile(state.get("map_context")),
        "route_mode": route_mode,
        "mechanics": build_mechanic_context(
            character,
            resolved_deck,
        ),
    }


def _threat_profile(map_context: Optional[Dict]) -> Dict:
    """Summarize the next three steps without treating branches as mandatory."""
    empty = {
        "known": False,
        "path_count": 0,
        "danger": 0.0,
        "min_danger": 0.0,
        "max_danger": 0.0,
        "elite_path_ratio": 0.0,
        "boss_path_ratio": 0.0,
        "campfire_path_ratio": 0.0,
        "average_monsters": 0.0,
        "kinds": [],
    }
    if not map_context:
        return empty
    nodes = {
        str(node.get("node_id")): node
        for node in map_context.get("nodes") or []
        if node.get("node_id")
    }
    starts = list(map_context.get("available_next_node_ids") or [])
    paths: List[List[tuple[int, str]]] = []

    def walk(
        node_id: str,
        depth: int,
        path: List[tuple[int, str]],
        visited: set[str],
    ) -> None:
        if len(paths) >= 128 or depth > 3 or node_id in visited:
            if path:
                paths.append(path)
            return
        node = nodes.get(str(node_id))
        if node is None:
            if path:
                paths.append(path)
            return
        kind = str(node.get("kind") or "UNKNOWN").upper()
        next_path = [*path, (depth, kind)]
        edges = [
            str(child)
            for child in node.get("edges") or []
            if str(child) in nodes
        ]
        if depth == 3 or not edges:
            paths.append(next_path)
            return
        next_visited = {*visited, node_id}
        for child in edges:
            walk(child, depth + 1, next_path, next_visited)

    for start in starts:
        walk(str(start), 1, [], set())
    if not paths:
        return empty

    weights = {1: 1.0, 2: 0.65, 3: 0.35}
    danger_values = {
        "MONSTER": 0.55,
        "ELITE": 2.8,
        "BOSS": 4.0,
        "CAMPFIRE": -0.8,
    }
    path_dangers = [
        sum(
            danger_values.get(kind, 0.0) * weights[depth]
            for depth, kind in path
        )
        for path in paths
    ]
    path_kinds = [[kind for _, kind in path] for path in paths]
    path_count = len(paths)
    kinds = sorted({kind for path in path_kinds for kind in path})
    return {
        "known": True,
        "path_count": path_count,
        "danger": round(
            max(0.0, sum(path_dangers) / path_count),
            3,
        ),
        "min_danger": round(max(0.0, min(path_dangers)), 3),
        "max_danger": round(max(0.0, max(path_dangers)), 3),
        "elite_path_ratio": round(
            sum("ELITE" in path for path in path_kinds) / path_count,
            3,
        ),
        "boss_path_ratio": round(
            sum("BOSS" in path for path in path_kinds) / path_count,
            3,
        ),
        "campfire_path_ratio": round(
            sum("CAMPFIRE" in path for path in path_kinds) / path_count,
            3,
        ),
        "average_monsters": round(
            sum(path.count("MONSTER") for path in path_kinds) / path_count,
            3,
        ),
        "kinds": kinds,
    }


def score_state_factors(
    card: Dict,
    option: Dict,
    profile: Dict,
    context: Dict,
    *,
    mechanic_assessment: Optional[Dict] = None,
) -> List[Dict]:
    """Score intrinsic effects plus deck, relic, HP, and route context."""
    factors: List[Dict] = []
    effects = _effects(card)
    cost = _cost(card)
    card_type = str(card.get("type_key") or "").strip().lower()
    deck_size = max(1, int(profile["size"]))
    effect_counts = context["deck_effect_counts"]

    def add(code: str, delta: float, message: str) -> None:
        if abs(delta) >= 0.01:
            factors.append(_factor(code, delta, message))

    card_id = str(card.get("id") or card.get("name") or "").lower()
    existing = int(profile["names"].get(card_id, 0))
    if existing >= 2:
        add(
            "duplicate_pressure",
            -6.0,
            f"牌组中已有 {existing} 张同名牌，继续增加会提高冗余。",
        )
    elif existing == 1:
        add(
            "duplicate_pressure",
            -1.5,
            "牌组中已有一张同名牌，轻度计入牌组膨胀。",
        )

    if profile["high_cost_ratio"] >= 0.35 and cost is not None:
        if cost <= 1:
            add(
                "curve_relief",
                5.0,
                "当前牌组高费牌偏多，这张低费牌能缓解费用曲线。",
            )
        elif cost >= 2:
            add(
                "curve_pressure",
                -5.0,
                "当前牌组高费牌偏多，继续增加高费牌会提高卡手风险。",
            )

    effective_cost = max(1, cost or 1)
    damage = effects.get("damage")
    if damage:
        efficiency = damage / effective_cost
        add(
            "damage_efficiency",
            _bounded((efficiency - 6.0) * 0.55, -3.0, 4.0),
            "根据基础伤害与费用估算即时输出效率。",
        )
    block = effects.get("block")
    if block:
        efficiency = block / effective_cost
        add(
            "block_efficiency",
            _bounded((efficiency - 5.0) * 0.65, -3.0, 4.0),
            "根据基础格挡与费用估算即时防御效率。",
        )
    if effects.get("draw"):
        add(
            "draw_value",
            min(5.0, effects["draw"] * 2.5),
            "抽牌提高了牌组循环和关键牌到手稳定性。",
        )
    if effects.get("energy_gain"):
        add(
            "energy_value",
            min(6.0, effects["energy_gain"] * 3.0),
            "能量收益提高了同回合可执行动作数量。",
        )
    if effects.get("aoe"):
        add("aoe_value", 2.5, "范围伤害提供多目标战斗覆盖。")
    if effects.get("cost_reduction"):
        add(
            "cost_reduction_value",
            3.0,
            "动态降费提高了高费用卡牌的实际可用性。",
        )
    if effects.get("upgrade_hand"):
        add(
            "upgrade_value",
            3.5,
            "升级手牌能为当前回合或后续循环提供成长收益。",
        )
    if effects.get("scaling") or card_type == "power":
        add("scaling_value", 2.0, "该牌提供持续或战斗内成长能力。")
    if effects.get("self_harm"):
        magnitude = effects["self_harm"]
        add(
            "self_harm_cost",
            -min(8.0, max(3.0, magnitude * 0.8)),
            "生命代价会降低当前局面的安全余量。",
        )

    damage_ratio = effect_counts.get("damage", 0) / deck_size
    block_ratio = effect_counts.get("block", 0) / deck_size
    draw_ratio = effect_counts.get("draw", 0) / deck_size
    if damage and damage_ratio < 0.3:
        add(
            "attack_coverage",
            4.5,
            "当前牌组即时输出来源偏少，该牌能补足输出覆盖。",
        )
    if block and block_ratio < 0.25:
        add(
            "defense_coverage",
            5.0,
            "当前牌组直接防御来源偏少，该牌能补足生存能力。",
        )
    if effects.get("draw") and draw_ratio < 0.12:
        add(
            "draw_coverage",
            4.0,
            "当前牌组缺少抽牌来源，该牌能改善循环稳定性。",
        )
    if effects.get("aoe") and effect_counts.get("aoe", 0) == 0:
        add(
            "aoe_coverage",
            3.5,
            "当前牌组没有明确范围伤害，该牌补足多目标能力。",
        )
    if (
        (effects.get("energy_gain") or effects.get("cost_reduction"))
        and profile["high_cost_ratio"] >= 0.3
    ):
        add(
            "energy_curve_synergy",
            3.0,
            "牌组费用压力较高，能量或降费效果更有价值。",
        )

    synergy_tags = {
        "strength",
        "vulnerable",
        "weak",
        "poison",
        "discard",
        "retain",
        "upgrade_hand",
        "cost_reduction",
    }
    supported = [
        tag
        for tag in synergy_tags
        if effects.get(tag) and effect_counts.get(tag, 0) >= 2
    ]
    if supported:
        add(
            "effect_continuity",
            min(7.0, 2.0 * len(supported)),
            "该牌延续了牌组中已有明确证据的效果体系。",
        )
    if damage and effect_counts.get("strength", 0):
        add(
            "strength_attack_synergy",
            min(4.0, 1.5 * effect_counts["strength"]),
            "牌组已有力量来源，攻击牌能利用现有成长。",
        )
    if damage and effect_counts.get("vulnerable", 0):
        add(
            "vulnerable_attack_synergy",
            min(3.0, effect_counts["vulnerable"]),
            "牌组已有易伤来源，攻击牌能利用伤害放大窗口。",
        )

    relic_effects = context["relic_effects"]
    relic_pairs = (
        ("supports_attack", bool(damage), "攻击"),
        ("supports_skill", card_type == "skill", "技能"),
        ("supports_power", card_type == "power", "能力"),
        ("supports_block", bool(block), "格挡"),
        ("supports_draw", bool(effects.get("draw")), "抽牌"),
        (
            "supports_energy",
            bool(effects.get("energy_gain") or effects.get("cost_reduction")),
            "能量",
        ),
        ("supports_strength", bool(effects.get("strength")), "力量"),
        (
            "supports_exhaust",
            bool(effects.get("self_exhaust")),
            "耗竭",
        ),
        ("supports_discard", bool(effects.get("discard")), "弃牌"),
        ("supports_poison", bool(effects.get("poison")), "中毒"),
        (
            "supports_vulnerable",
            bool(damage) and not bool(effects.get("vulnerable")),
            "易伤窗口",
        ),
    )
    matched_relics = [
        label
        for tag, applies, label in relic_pairs
        if applies and relic_effects.get(tag, 0)
    ]
    if matched_relics:
        add(
            "relic_synergy",
            min(6.0, 2.0 * len(matched_relics)),
            "当前遗物支持该牌的"
            + "、".join(matched_relics)
            + "效果。",
        )

    hp_ratio = context.get("hp_ratio")
    if hp_ratio is not None:
        if hp_ratio <= 0.35:
            if block or effects.get("healing"):
                add(
                    "critical_hp_defense",
                    7.0,
                    "当前生命值很低，直接防御或恢复价值显著提高。",
                )
            if cost is not None and cost >= 3:
                add(
                    "critical_hp_slow_card",
                    -4.0,
                    "低生命状态下，高费牌的即时容错较差。",
                )
            if effects.get("self_harm"):
                add(
                    "critical_hp_self_harm",
                    -5.0,
                    "低生命状态会进一步放大生命代价风险。",
                )
        elif hp_ratio <= 0.6 and (block or effects.get("healing")):
            add(
                "low_hp_defense",
                3.0,
                "当前生命值偏低，防御或恢复效果更加重要。",
            )

    # ── One path-aware route factor per candidate ───────────────────
    route = context["route"]
    if route["known"]:
        route_delta = 0.0
        route_reasons: List[str] = []
        route_code = "route_fit"
        boss_ratio = float(route["boss_path_ratio"])
        elite_ratio = float(route["elite_path_ratio"])
        if boss_ratio > 0:
            route_code = "route_boss_fit"
            if effects.get("scaling") or card_type == "power":
                route_delta += 3.0 + 2.0 * boss_ratio
                route_reasons.append("持续成长")
            if damage:
                route_delta += 1.0 + boss_ratio
                route_reasons.append("即时输出")
            if block:
                route_delta += 1.0 + 0.5 * boss_ratio
                route_reasons.append("直接防御")
            if cost is not None and cost >= 3:
                route_delta -= boss_ratio
                route_reasons.append("高费启动较慢")
        elif elite_ratio > 0:
            route_code = "route_elite_fit"
            if damage:
                route_delta += 2.0 + 2.0 * elite_ratio
                route_reasons.append("精英战即时输出")
            if block:
                route_delta += 1.5 + 1.5 * elite_ratio
                route_reasons.append("精英战减伤")
            if cost is not None and cost >= 3:
                route_delta -= 1.5 * elite_ratio
                route_reasons.append("高费启动较慢")
        elif route["danger"] >= 1.5:
            route_code = "route_combat_fit"
            if effects.get("aoe") and route["average_monsters"] >= 1.5:
                route_delta += 3.0
                route_reasons.append("多场小怪范围覆盖")
            elif damage:
                route_delta += 2.0
                route_reasons.append("近期即时输出")
            if block:
                route_delta += 1.5
                route_reasons.append("近期直接防御")
            if cost is not None and cost >= 3:
                route_delta -= 1.0
                route_reasons.append("高费启动较慢")
        if route_reasons:
            add(
                route_code,
                route_delta,
                "按可选路径比例评估：" + "、".join(route_reasons) + "。",
            )

        route_mode = context.get("route_mode", "balanced")
        if route_mode == "survival":
            mode_delta = 0.0
            mode_reasons: List[str] = []
            pressure = (
                route["danger"] >= 1.0
                or elite_ratio > 0
                or boss_ratio > 0
            )
            if pressure and (block or effects.get("healing")):
                mode_delta += 2.0
                mode_reasons.append("提高防御和恢复")
            if pressure and effects.get("self_harm"):
                mode_delta -= 2.0
                mode_reasons.append("压低生命代价")
            if pressure and cost is not None and cost >= 3:
                mode_delta -= 0.75
                mode_reasons.append("降低高费启动风险")
            if mode_reasons:
                add(
                    "route_mode_survival",
                    mode_delta,
                    "稳健生存模式在真实路线压力下"
                    + "、".join(mode_reasons)
                    + "。",
                )
        elif route_mode == "growth":
            hp_ratio = context.get("hp_ratio")
            if hp_ratio is None or hp_ratio <= 0.35:
                factors.append(_factor(
                    "route_mode_growth_safety_floor",
                    0.0,
                    "当前生命信息不足或已触及生存底线，"
                    "激进成长模式不追加正向权重。",
                ))
            else:
                mode_delta = 0.0
                mode_reasons = []
                if effects.get("scaling") or card_type == "power":
                    mode_delta += 1.75
                    mode_reasons.append("持续成长")
                if effects.get("draw") or effects.get("energy_gain"):
                    mode_delta += 0.75
                    mode_reasons.append("资源循环")
                if elite_ratio > 0 and damage:
                    mode_delta += 0.5
                    mode_reasons.append("精英前即时战力")
                if mode_reasons:
                    add(
                        "route_mode_growth",
                        min(2.5, mode_delta),
                        "激进成长模式在生存底线以上提高"
                        + "、".join(mode_reasons)
                        + "价值。",
                    )

    if context["act"] == 1 and context["floor"] <= 5 and damage:
        add(
            "early_act_frontload",
            1.5,
            "第一章前期需要尽快建立基础输出能力。",
        )
    elif context["act"] >= 2 and (
        effects.get("scaling") or effects.get("draw")
    ):
        add(
            "late_act_scaling",
            2.0,
            "中后期更需要循环稳定性和持续成长能力。",
        )

    # ── Conditional effect activation ────────────────────────────────
    cond_tags = [t for t in effects if t.startswith("cond:")]
    for ct in cond_tags:
        if ct == "cond:vulnerable_target" and effect_counts.get(
            "vulnerable", 0
        ):
            add(
                "cond_met:vulnerable",
                3.0,
                "牌组有能力施加易伤，条件效果可以触发。",
            )
        if ct == "cond:weak_target" and effect_counts.get("weak", 0):
            add(
                "cond_met:weak",
                2.5,
                "牌组有能力施加虚弱，条件效果可以触发。",
            )
        if ct == "cond:poison_target" and effect_counts.get("poison", 0):
            add(
                "cond_met:poison",
                3.0,
                "牌组已经拥有效果体系，条件效果可以触发。",
            )
        if ct == "cond:discard_synergy" and effect_counts.get(
            "discard", 0
        ):
            add(
                "cond_met:discard",
                3.0,
                "牌组已有弃牌组件，弃牌协同效果可以触发。",
            )
        if ct == "cond:exhaust_synergy" and effect_counts.get(
            "self_exhaust", 0
        ):
            add(
                "cond_met:exhaust",
                3.0,
                "牌组已有耗竭组件，耗竭协同效果可以触发。",
            )
        if ct == "cond:strength_scaling" and effect_counts.get(
            "strength", 0
        ):
            add(
                "cond_met:strength",
                4.0,
                "牌组已有力量来源，力量倍率效果价值提高。",
            )

    # ── Provider/payoff pairs; mentions alone are not synergy ────────
    engine_matches = []
    # Vanilla characters use the unified MechanicSignal scorer below.  Keep
    # the legacy generic pair only for synthetic/offline compatibility so the
    # same exhaust evidence cannot stack twice in production.
    if not supports_character(context.get("character", "")):
        if (
            effects.get("self_exhaust")
            and effect_counts.get("exhaust_interaction", 0)
        ) or (
            effects.get("exhaust_interaction")
            and effect_counts.get("self_exhaust", 0)
        ):
            engine_matches.append("耗竭")
    if (
        effects.get("ethereal")
        and effect_counts.get("ethereal_interaction", 0)
    ) or (
        effects.get("ethereal_interaction")
        and effect_counts.get("ethereal", 0)
    ):
        engine_matches.append("虚无")
    if engine_matches:
        add(
            "effect_engine_synergy",
            min(4.0, 2.0 * len(engine_matches)),
            "候选与牌组已有的"
            + "、".join(engine_matches)
            + "提供者/收益方形成明确协同。",
        )

    if mechanic_assessment is None:
        mechanic_assessment = assess_card_mechanics(
            context.get("character", ""),
            card,
            context.get("mechanics") or {},
        )
    factors.extend(mechanic_assessment["factors"])

    if int(option.get("upgrades", 0)) > 0:
        add(
            "upgraded_offer",
            4.0,
            "奖励项已经升级，获得即时数值收益。",
        )
    if option.get("affliction"):
        amount = int(option.get("affliction_amount") or 1)
        add(
            "affliction_cost",
            -min(8.0, 2.0 + amount),
            "候选牌带有负面状态，需要计入额外代价。",
        )
    if option.get("enchantment"):
        amount = int(option.get("enchantment_amount") or 1)
        add(
            "enchantment_value",
            min(6.0, 2.0 + amount),
            "候选牌带有正面附魔，提供额外收益。",
        )

    return factors


def score_dynamic_skip(
    profile: Dict,
    recommendations: List[Dict],
    context: Dict,
    can_skip: bool = True,
) -> Dict:
    """Score deck compactness on the same ordinal scale as card options."""
    score = 38.0
    factors: List[Dict] = []

    def add(code: str, delta: float, message: str) -> None:
        nonlocal score
        score += delta
        factors.append(_factor(code, delta, message))

    deck_size = int(profile["size"])
    size_delta = _bounded((deck_size - 12) * 0.9, -5.0, 16.0)
    add(
        "deck_size",
        size_delta,
        (
            "牌组较大，保持精简的边际价值提高。"
            if size_delta > 0
            else "牌组仍小，当前更需要建立基础功能。"
        ),
    )

    known = [row for row in recommendations if row.get("known")]
    strongest_state = max(
        (float(row["state_score"]) for row in known),
        default=0.0,
    )
    if strongest_state >= 62:
        add(
            "strong_candidate",
            -10.0,
            "候选中存在与当前局面高度匹配的牌，不应轻易跳过。",
        )
    elif strongest_state >= 56:
        add(
            "useful_candidate",
            -6.0,
            "候选中存在明确补强项，跳过价值下降。",
        )
    elif strongest_state <= 44 and known:
        add(
            "weak_candidates",
            12.0,
            "所有已知候选对当前局面的适配度都偏低。",
        )
    elif strongest_state < 50 and known:
        add(
            "below_neutral_candidates",
            4.0,
            "候选整体低于中性适配水平，保持精简更有价值。",
        )

    if known and all(int(row.get("existing_copies", 0)) > 0 for row in known):
        add(
            "all_duplicates",
            6.0,
            "所有候选都已在牌组中出现，跳过可避免进一步冗余。",
        )

    effect_counts = context["deck_effect_counts"]
    deck_size_safe = max(1, deck_size)
    missing_functions = sum(
        [
            effect_counts.get("damage", 0) / deck_size_safe < 0.25,
            effect_counts.get("block", 0) / deck_size_safe < 0.2,
            effect_counts.get("draw", 0) / deck_size_safe < 0.08,
        ]
    )
    if missing_functions:
        add(
            "unfilled_deck_needs",
            -2.0 * missing_functions,
            "牌组仍有基础功能缺口，过早跳过会延缓补强。",
        )

    route = context["route"]
    if route["known"] and route["danger"] >= 2.5:
        add(
            "route_pressure",
            -2.5,
            "近期路线战斗压力较高，当前更需要获得有效补强。",
        )

    all_known = len(known) == len(recommendations)
    eligible = (
        can_skip
        and all_known
        and strongest_state <= max(20.0, score)
    )
    score = round(_bounded(score, 15.0, 75.0), 2)
    return {
        "score": score,
        "eligible": eligible,
        "factors": factors,
        "strongest_state_score": strongest_state,
    }

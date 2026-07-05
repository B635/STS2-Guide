"""Transparent baseline for card-reward decisions.

This module intentionally contains no copied tier lists or weights from other
projects. It provides a small, replaceable baseline so state capture,
persistence, API contracts, and evaluation can stabilize before stronger
models are introduced.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional

from advisor.contextual_scoring import (
    build_context,
    score_dynamic_skip,
    score_state_factors,
)
from advisor.data_sources import LocalCardTierSource
from storage.relational import RelationalRepository
from config import (
    ADVISOR_COMMUNITY_SCORE_MIN_PICKS,
    ADVISOR_COMMUNITY_SCORE_WEIGHT,
)


BASELINE_METHOD = "contextual_state_v3"
COMMUNITY_PRIOR_METHOD = "contextual_state_plus_community_v3"
BASE_CARD_SCORE = 50.0
# Independent, bounded ordinal prior for a manually curated local tier file.
# These are intentionally not copied from the reference project.  The local
# prior replaces (rather than stacks with) the Codex prior for the same card.
LOCAL_TIER_DELTAS = {
    "S": 10.0,
    "A": 6.0,
    "B": 2.0,
    "C": -3.0,
    "D": -7.0,
    "F": -10.0,
}


def _factor(code: str, delta: float, message: str) -> Dict:
    return {"code": code, "delta": float(delta), "message": message}


def _numeric_cost(card: Optional[Dict]) -> Optional[int]:
    if not card:
        return None
    value = card.get("cost")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expand_deck(deck: Iterable[Dict]) -> List[Dict]:
    expanded: List[Dict] = []
    for entry in deck:
        count = max(1, int(entry.get("count", 1)))
        expanded.extend([entry] * count)
    return expanded


def _resolve_deck(
    deck: Iterable[Dict],
    repository: RelationalRepository,
) -> List[Dict]:
    resolved = []
    for entry in _expand_deck(deck):
        resolved.append(
            {
                "input": entry,
                "card": repository.find_card(entry["card"]),
            }
        )
    return resolved


def _deck_profile(resolved_deck: List[Dict]) -> Dict:
    known_cards = [row["card"] for row in resolved_deck if row["card"]]
    costs = [
        cost
        for card in known_cards
        if (cost := _numeric_cost(card)) is not None and cost >= 0
    ]
    type_counts = Counter(
        str(card.get("type_key") or "").lower()
        for card in known_cards
        if card.get("type_key")
    )
    block_cards = sum(1 for card in known_cards if (card.get("block") or 0) > 0)
    high_cost_cards = sum(1 for cost in costs if cost >= 2)
    names = Counter(
        str(row["card"].get("id") or row["card"].get("name")).lower()
        for row in resolved_deck
        if row["card"]
    )
    return {
        "size": len(resolved_deck),
        "known": len(known_cards),
        "average_cost": sum(costs) / len(costs) if costs else None,
        "high_cost_ratio": high_cost_cards / len(costs) if costs else 0.0,
        "type_counts": type_counts,
        "block_cards": block_cards,
        "names": names,
    }


def _score_candidate(
    option: Dict,
    repository: RelationalRepository,
    profile: Dict,
    context: Dict,
    character_id: str,
    local_tiers: Optional[LocalCardTierSource],
) -> Dict:
    identifier = option["card"]
    upgrades = int(option.get("upgrades", 0))
    card = repository.find_card(identifier)
    score = BASE_CARD_SCORE
    factors: List[Dict] = []

    if not card:
        delta = -15.0
        score += delta
        factors.append(
            _factor(
                "unknown_card",
                delta,
                "结构化卡牌库中未找到该卡，暂不做高置信度推荐。",
            )
        )
        return {
            "card": identifier,
            "card_id": identifier,
            "upgrades": upgrades,
            "enchantment": option.get("enchantment"),
            "enchantment_amount": option.get("enchantment_amount"),
            "affliction": option.get("affliction"),
            "affliction_amount": option.get("affliction_amount"),
            "score": score,
            "state_score": score,
            "factors": factors,
            "known": False,
        }

    card_id = str(card.get("id") or "").lower()
    existing = int(profile["names"].get(card_id, 0))
    state_factors = score_state_factors(
        card,
        option,
        profile,
        context,
    )
    for factor in state_factors:
        score += float(factor["delta"])
    factors.extend(state_factors)

    state_score = score
    statistics_used = False
    prior_source = None
    local_entry = (
        local_tiers.find(card["id"], character_id)
        if local_tiers is not None
        else None
    )
    if local_entry is not None:
        delta = LOCAL_TIER_DELTAS[local_entry.tier]
        score += delta
        factors.append(
            {
                **_factor(
                    "local_tier_prior",
                    delta,
                    (
                        "本机人工整理的社区 tier 只作为有限基础先验；"
                        "当前牌组和局面规则仍然优先。"
                    ),
                ),
                "source_name": local_tiers.source,
                "source_url": local_tiers.source_url,
                "snapshot_id": local_tiers.captured_at,
                "tier": local_entry.tier,
            }
        )
        statistics_used = True
        prior_source = "local_tier"
    else:
        community = repository.find_latest_entity_stat(
            "cards", card["id"]
        )
        if (
            community
            and isinstance(community.get("score"), (int, float))
            and int(community.get("picks") or 0)
            >= ADVISOR_COMMUNITY_SCORE_MIN_PICKS
        ):
            picks = int(community["picks"])
            reliability = min(1.0, math.log10(picks + 1) / 5.0)
            signal = (float(community["score"]) - 50.0) * reliability
            delta = signal * ADVISOR_COMMUNITY_SCORE_WEIGHT
            score += delta
            factors.append(
                {
                    **_factor(
                        "community_prior",
                        delta,
                        (
                            "Spire Codex 社区聚合分仅作为低权重统计先验；"
                            "它反映相关性，不代表这次局面的因果价值。"
                        ),
                    ),
                    "source_name": community.get("source_name"),
                    "source_url": community.get("source_url"),
                    "snapshot_id": community.get("snapshot_id"),
                    "sample_size": picks,
                    "raw_score": float(community["score"]),
                    "signal": round(signal, 6),
                    "weight": ADVISOR_COMMUNITY_SCORE_WEIGHT,
                }
            )
            statistics_used = True
            prior_source = "spire_codex"

    if not factors:
        factors.append(
            _factor(
                "neutral_baseline",
                0.0,
                "暂未发现明显的费用、覆盖或标签连续性信号。",
            )
        )

    return {
        "card": card.get("name") or identifier,
        "card_id": card.get("id"),
        "upgrades": upgrades,
        "enchantment": option.get("enchantment"),
        "enchantment_amount": option.get("enchantment_amount"),
        "affliction": option.get("affliction"),
        "affliction_amount": option.get("affliction_amount"),
        "score": round(max(0.0, min(100.0, score)), 2),
        "state_score": round(
            max(0.0, min(100.0, state_score)),
            2,
        ),
        "factors": factors,
        "known": True,
        "existing_copies": existing,
        "effect_tags": sorted((card.get("_effect_tags") or {}).keys()),
        "statistics_used": statistics_used,
        "prior_source": prior_source,
    }


def recommend_card_reward(
    state: Dict,
    options: Iterable[Dict],
    repository: RelationalRepository,
    local_tiers: Optional[LocalCardTierSource] = None,
    can_skip: bool = True,
) -> Dict:
    """Rank reward options against the current structured run state."""
    options = list(options)
    if not options:
        raise ValueError("At least one card reward option is required")

    resolved_deck = _resolve_deck(state.get("deck", []), repository)
    profile = _deck_profile(resolved_deck)
    context = build_context(state, resolved_deck, repository)
    recommendations = []
    character_id = str(state.get("character") or "").strip().upper()
    for option_index, option in enumerate(options):
        result = _score_candidate(
            option,
            repository,
            profile,
            context,
            character_id,
            local_tiers,
        )
        result["option_index"] = option_index
        recommendations.append(result)

    recommendations.sort(
        key=lambda row: (-row["score"], row["option_index"])
    )
    for rank, recommendation in enumerate(recommendations, start=1):
        recommendation["rank"] = rank

    best = recommendations[0]
    strongest_state_score = max(
        row["state_score"] for row in recommendations
    )
    skip_result = score_dynamic_skip(
        profile,
        recommendations,
        context,
        can_skip=can_skip,
    )
    skip_score = float(skip_result["score"])
    skip_recommended = (
        bool(skip_result["eligible"])
        and skip_score >= float(best["score"])
        and skip_score >= strongest_state_score
    )
    recommended_option = None if skip_recommended else best["card"]
    recommended_option_index = (
        len(recommendations)
        if skip_recommended
        else int(best["option_index"])
    )
    known_ratio = (
        sum(1 for row in recommendations if row["known"])
        / len(recommendations)
    )
    comparison_scores = [
        float(row["score"])
        for row in recommendations
        if row["option_index"] != best["option_index"]
    ]
    if skip_result["eligible"]:
        comparison_scores.append(skip_score)
    margin = (
        skip_score - float(best["score"])
        if skip_recommended
        else float(best["score"])
        - max(comparison_scores, default=skip_score)
    )
    uncertain = (
        not skip_recommended
        and (
            known_ratio < 1.0
            or (
                best["state_score"] <= 52.0
                and (
                    margin < 3.0
                    or len(recommendations) == 1
                )
            )
        )
    )
    decision_status = (
        "skip"
        if skip_recommended
        else "uncertain"
        if uncertain
        else "recommend"
    )
    confidence = (
        "medium"
        if decision_status == "recommend"
        and known_ratio == 1.0
        and margin >= 8.0
        else "low"
    )
    skip_candidate = {
        "choice": "skip",
        "option_index": len(recommendations),
        "score": skip_score,
        "rank": 1
        + sum(
            1
            for recommendation in recommendations
            if recommendation["score"] > skip_score
        ),
        "eligible": bool(skip_result["eligible"]),
        "factors": skip_result["factors"],
        "reason": (
            "动态精简收益高于所有候选，当前可以考虑跳过。"
            if skip_recommended
            else
            "当前局面或候选质量不足以支持跳过。"
        ),
    }

    statistics_coverage = sum(
        1 for row in recommendations if row.get("statistics_used")
    )
    local_tier_coverage = sum(
        1
        for row in recommendations
        if row.get("prior_source") == "local_tier"
    )
    codex_statistics_coverage = sum(
        1
        for row in recommendations
        if row.get("prior_source") == "spire_codex"
    )
    return {
        "method": (
            COMMUNITY_PRIOR_METHOD
            if statistics_coverage
            else BASELINE_METHOD
        ),
        "recommended_option": recommended_option,
        "recommended_option_index": recommended_option_index,
        "decision_status": decision_status,
        "skip_recommended": skip_recommended,
        "skip_score": skip_score,
        "skip_candidate": skip_candidate,
        "confidence": confidence,
        "recommendations": recommendations,
        "profile": {
            "deck_size": profile["size"],
            "known_deck_cards": profile["known"],
            "average_cost": (
                round(profile["average_cost"], 2)
                if profile["average_cost"] is not None
                else None
            ),
            "high_cost_ratio": round(profile["high_cost_ratio"], 3),
            "block_cards": profile["block_cards"],
            "hp_ratio": (
                round(context["hp_ratio"], 3)
                if context["hp_ratio"] is not None
                else None
            ),
            "resolved_relics": context["resolved_relics"],
            "route": context["route"],
            "effect_tag_counts": dict(
                context["deck_effect_counts"].most_common()
            ),
            "community_statistics_coverage": statistics_coverage,
            "local_tier_coverage": local_tier_coverage,
            "codex_statistics_coverage": codex_statistics_coverage,
            "local_tier_source": (
                local_tiers.diagnostics()
                if local_tiers is not None
                else None
            ),
            "strongest_state_score": strongest_state_score,
            "dynamic_skip_score": skip_score,
        },
        "disclaimer": (
            "这是透明局面 baseline 与低权重社区统计先验的组合，"
            "不代表已训练或已校准的胜率预测。"
            if statistics_coverage
            else
            "这是用于验证状态建模和数据闭环的透明 baseline，"
            "不代表已训练的胜率预测。"
        ),
    }

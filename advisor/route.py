"""Deterministic, bounded route policy for the P1.0 vertical slice.

The route policy deliberately treats the Mod map as the source of graph
truth.  It never guesses an edge, a currently faced monster, or a boss.  The
only persistent inputs are versioned, structured SQLite constants and static
catalog records; the current run graph stays in the active checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from advisor.decision_core import (
    CandidateAssessment,
    DecisionRequest,
    Recommendation,
    ROUTE_CHOICE,
    ROUTE_MODE_BALANCED,
    ROUTE_MODE_GROWTH,
    ROUTE_MODE_SURVIVAL,
)
from advisor.versions import ROUTE_POLICY_VERSION
from advisor.boss_context import build_boss_preparation
from storage.relational import RelationalRepository

# Messages describe existing local risk factors, not future encounters or score-point deltas.
_FACTOR_MESSAGES = {
    "monster_encounter_pool_pressure": "按本幕普通敌人遭遇池估计战斗压力，并非预知下一只敌人。",
    "elite_encounter_pool_pressure": "按本幕精英遭遇池估计压力，并非预知下一场精英。",
    "monster_expected_risk": "路线包含普通战斗，计入基础战斗风险。",
    "elite_expected_risk": "路线包含精英战斗，计入较高的基础风险。",
    "monster_aoe_relief": "当前牌组的群体伤害能力缓解普通战斗压力。",
    "elite_low_hp": "当前生命偏低，经过精英的风险提高。",
    "elite_growth_relief": "当前牌组的成长能力缓解精英压力。",
    "known_boss_risk": "计入本幕已知 Boss 的威胁。",
    "rest_low_hp": "当前生命偏低，路线中的篝火提供休整机会。",
    "boss_preparation_opportunity": "路线中的篝火提供应对本幕 Boss 的准备机会。",
    "shop_affordability": "当前金币达到策略阈值，商店具有补强机会；未假定未来库存。",
    "shop_low_gold": "当前金币较少，商店补强空间有限。",
    "shop_gold_spent_earlier": "金币已计入路线上更早的商店，这间商店不重复计入。",
    "combat_risk_floor": "牌组能力已足以抵消这场战斗的风险；多出的能力不会让战斗变成加分。",
    "potion_buffer_once": "当前药水作为一次战斗缓冲计入，不在沿途重复消耗同一瓶。",
}


def _described_factor(factor: dict[str, Any]) -> dict[str, Any]:
    code = factor["code"]
    message = _FACTOR_MESSAGES.get(code)
    if code.startswith("route_mode_survival_"):
        message = "稳健生存偏好调整沿途节点的风险权重。"
    elif code.startswith("route_mode_growth_safety_floor_"):
        message = "当前生命触及安全底线，激进成长偏好也会优先保命。"
    elif code.startswith("route_mode_growth_"):
        message = "激进成长偏好在生命安全约束内调整沿途收益。"
    return {**factor, "message": message} if message else factor

RISK_PROFILE_KEY = "route_risk_profile_v1"
MAX_ROUTE_SEARCH_EXPANSIONS = 256
_ROUTE_KINDS = frozenset({
    "MONSTER", "ELITE", "CAMPFIRE", "SHOP", "EVENT", "BOSS",
    "TREASURE", "UNKNOWN",
})
_PROFILE_NUMBERS = (
    "hp_low_threshold",
    "hp_critical_threshold",
    "elite_hp_penalty",
    "boss_path_bonus",
    "rest_low_hp_bonus",
    "shop_gold_threshold",
)


@dataclass(frozen=True)
class GraphValidation:
    nodes: Mapping[str, Any]
    origin: str | None
    boss_ids: frozenset[str]
    critical_gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteCapabilities:
    immediate_power: float = 0.0
    survival: float = 0.0
    aoe: float = 0.0
    growth: float = 0.0
    resources: float = 0.0
    relic_support: float = 0.0
    potion_support: float = 0.0
    data_gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchPath:
    node_ids: tuple[str, ...]
    risk: float
    max_combat_risk: float


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float = 0.0) -> float:
    parsed = _as_float(value)
    return default if parsed is None else parsed


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return round(max(low, min(high, value)), 2)


def _route_display_score(effective_risk: float, relic_support: float) -> float:
    """Map route risk to a readable 0-100 suitability score.

    Route risk is the value used by the path search and remains the ranking
    truth.  A hard linear clamp used to turn every sufficiently favourable
    first-act route into ``100``, erasing real differences and making route
    modes fall back to map display order.  The bounded logistic mapping stays
    monotonic without presenting the score as a win probability.
    """

    # Preserve the previous small relic-support contribution in risk units:
    # the former linear display slope was 3.5 points per risk unit and relic
    # support added 0.4 display points.
    adjusted_risk = effective_risk - relic_support * (0.4 / 3.5)
    exponent = max(-60.0, min(60.0, adjusted_risk / 10.0))
    return round(100.0 / (1.0 + math.exp(exponent)), 2)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _upgraded_stat(
    card: Mapping[str, Any],
    tags: Mapping[str, Any],
    field: str,
    upgrades: int,
    *,
    default: float = 0.0,
) -> float:
    """Read a catalog statistic plus its explicit upgrade representation.

    Spire Codex expresses common upgrades as signed deltas (for example
    ``{\"damage\": \"+3\"}``) and occasionally as an absolute upgraded
    value.  We consume only that structured field; unknown upgrade text never
    receives a made-up generic multiplier.
    """
    base = max(_number(tags.get(field), default), _number(card.get(field), default))
    if upgrades <= 0:
        return max(0.0, base)
    upgrade = card.get("upgrade")
    raw = upgrade.get(field) if isinstance(upgrade, Mapping) else None
    if raw is None:
        return max(0.0, base)
    parsed = _as_float(raw)
    if parsed is None:
        return max(0.0, base)
    if isinstance(raw, str) and raw.strip().startswith(("+", "-")):
        return max(0.0, base + parsed * upgrades)
    # A non-signed number is an explicitly catalogued upgraded value.  More
    # than one upgrade is unsupported game data, so do not extrapolate it.
    return max(0.0, parsed)


def validate_route_graph(world_map, candidate_ids: Iterable[str]) -> GraphValidation:
    """Validate the exact Mod graph before a score can be exposed.

    Invalid graphs are a *critical* gap, including malformed edges that a
    permissive renderer might otherwise silently ignore.  This validator does
    not invent a start node when the verified origin is unavailable.
    """
    gaps: list[str] = []
    if world_map is None:
        return GraphValidation({}, None, frozenset(), ("map_context_missing",))
    nodes = list(world_map.nodes)
    if not nodes:
        return GraphValidation({}, None, frozenset(), ("map_nodes_missing",))

    by_id: dict[str, Any] = {}
    for node in nodes:
        if not node.node_id or node.node_id in by_id:
            gaps.append("duplicate_node_id")
            continue
        by_id[node.node_id] = node

    for node in by_id.values():
        seen_edges: set[str] = set()
        for edge in node.edges:
            if edge in seen_edges:
                gaps.append(f"duplicate_edge:{node.node_id}->{edge}")
            seen_edges.add(edge)
            child = by_id.get(edge)
            if child is None:
                gaps.append(f"unknown_edge:{node.node_id}->{edge}")
            elif child.row <= node.row:
                gaps.append(f"non_forward_edge:{node.node_id}->{edge}")

    colours: dict[str, int] = {}

    def visit(node_id: str) -> None:
        colour = colours.get(node_id, 0)
        if colour == 1:
            gaps.append("graph_cycle")
            return
        if colour == 2 or node_id not in by_id:
            return
        colours[node_id] = 1
        for edge in by_id[node_id].edges:
            if edge in by_id:
                visit(edge)
        colours[node_id] = 2

    for node_id in sorted(by_id):
        visit(node_id)

    origin = world_map.origin_node_id
    if not origin:
        gaps.append("origin_missing")
    elif origin not in by_id:
        gaps.append(f"unknown_origin:{origin}")
    if world_map.node_count is not None and world_map.node_count != len(by_id):
        gaps.append("node_count_mismatch")

    candidates = tuple(str(candidate_id) for candidate_id in candidate_ids)
    map_candidates = tuple(str(candidate_id) for candidate_id in world_map.available_next_node_ids)
    if not candidates:
        gaps.append("route_candidates_missing")
    if len(set(candidates)) != len(candidates):
        gaps.append("duplicate_candidate_id")
    if len(set(map_candidates)) != len(map_candidates):
        gaps.append("duplicate_map_candidate_id")
    if candidates != map_candidates:
        gaps.append("candidate_set_mismatch")
    for candidate_id in candidates:
        if candidate_id not in by_id:
            gaps.append(f"unknown_candidate:{candidate_id}")
        elif origin in by_id and candidate_id not in by_id[origin].edges:
            gaps.append(f"candidate_not_direct_child:{candidate_id}")

    boss_ids = frozenset(str(boss_id) for boss_id in world_map.boss_node_ids)
    if not boss_ids:
        gaps.append("boss_nodes_missing")
    for boss_id in sorted(boss_ids):
        if boss_id not in by_id:
            gaps.append(f"unknown_boss_node:{boss_id}")
        elif by_id[boss_id].kind != "BOSS":
            gaps.append(f"boss_kind_mismatch:{boss_id}")
    return GraphValidation(
        MappingProxyType(by_id),
        origin,
        boss_ids,
        _unique(gaps),
    )


def _validate_profile(profile: Mapping[str, Any] | None) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    """Reject malformed risk data instead of borrowing invisible defaults."""
    if not isinstance(profile, Mapping):
        return None, (f"risk_profile_missing:{RISK_PROFILE_KEY}",)
    version = profile.get("version")
    if not isinstance(version, str) or not version.strip():
        return None, ("risk_profile_unversioned",)
    if version != "route-risk-v1":
        return None, (f"risk_profile_unsupported_version:{version}",)
    gaps: list[str] = []
    kind_weights = profile.get("kind_weights")
    if not isinstance(kind_weights, Mapping):
        gaps.append("risk_profile_kind_weights_missing")
    else:
        for kind in sorted(_ROUTE_KINDS):
            if _as_float(kind_weights.get(kind)) is None:
                gaps.append(f"risk_profile_kind_weight_missing:{kind}")
    encounters = profile.get("encounter_expected_risk")
    if not isinstance(encounters, Mapping):
        gaps.append("risk_profile_encounter_expected_risk_missing")
    else:
        for kind in ("MONSTER", "ELITE", "BOSS"):
            value = _as_float(encounters.get(kind))
            if value is None:
                gaps.append(f"risk_profile_encounter_risk_missing:{kind}")
            elif value < 0:
                gaps.append(f"risk_profile_encounter_risk_negative:{kind}")
    capability_weights = profile.get("capability_weights")
    if not isinstance(capability_weights, Mapping):
        gaps.append("risk_profile_capability_weights_missing")
    else:
        for key in ("immediate_power", "survival", "aoe", "growth", "resource"):
            value = _as_float(capability_weights.get(key))
            if value is None:
                gaps.append(f"risk_profile_capability_weight_missing:{key}")
            elif value < 0:
                gaps.append(f"risk_profile_capability_weight_negative:{key}")
    for key in _PROFILE_NUMBERS:
        if _as_float(profile.get(key)) is None:
            gaps.append(f"risk_profile_value_missing:{key}")
    low = _as_float(profile.get("hp_low_threshold"))
    critical = _as_float(profile.get("hp_critical_threshold"))
    shop = _as_float(profile.get("shop_gold_threshold"))
    if low is not None and critical is not None and not (0 < critical < low <= 1):
        gaps.append("risk_profile_hp_thresholds_invalid")
    if shop is not None and shop < 0:
        gaps.append("risk_profile_shop_gold_threshold_invalid")
    relief = profile.get("shop_gold_relief")
    if not isinstance(relief, Mapping):
        gaps.append("risk_profile_shop_gold_relief_missing")
    else:
        per_point, cap = _as_float(relief.get("gold_per_point")), _as_float(relief.get("max"))
        if per_point is None or per_point <= 0 or cap is None or cap < 0:
            gaps.append("risk_profile_shop_gold_relief_invalid")
    for key in (
        "elite_hp_penalty",
        "boss_path_bonus",
        "rest_low_hp_bonus",
    ):
        value = _as_float(profile.get(key))
        if value is not None and value < 0:
            gaps.append(f"risk_profile_{key}_negative")
    if gaps:
        return None, _unique(gaps)
    return profile, ()


class RoutePolicy:
    """Transparent route scoring with one memoized path search per node."""

    decision_type = ROUTE_CHOICE

    def __init__(
        self,
        repository: RelationalRepository,
        *,
        risk_profile: Mapping[str, Any] | None = None,
        max_search_expansions: int = MAX_ROUTE_SEARCH_EXPANSIONS,
    ) -> None:
        self.repository = repository
        self._risk_profile = dict(risk_profile) if risk_profile is not None else None
        self.max_search_expansions = max(1, int(max_search_expansions))
        self._encounter_pool_cache: dict[tuple[tuple[str, ...], int], Mapping[str, Mapping[str, float]]] = {}

    def _profile(self) -> Mapping[str, Any] | None:
        if self._risk_profile is not None:
            return self._risk_profile
        value = self.repository.mechanic_constant(RISK_PROFILE_KEY)
        return value if isinstance(value, Mapping) else None

    def _capabilities(self, request: DecisionRequest) -> RouteCapabilities:
        immediate = survival = aoe = growth = resources = 0.0
        relic_support = potion_support = 0.0
        gaps: list[str] = []

        def effects(entity: Mapping[str, Any]) -> Mapping[str, Any]:
            return entity.get("_effect_tags") or {}

        for card_state in request.world.deck:
            card = self.repository.find_card(card_state.card)
            if card is None:
                gaps.append(f"unknown_card:{card_state.card}")
                continue
            count = max(1, int(card_state.count))
            tags = effects(card)
            damage = _upgraded_stat(
                card,
                tags,
                "damage",
                int(card_state.upgrades),
            )
            block = _upgraded_stat(
                card,
                tags,
                "block",
                int(card_state.upgrades),
            )
            hits = max(
                1.0,
                _upgraded_stat(
                    card,
                    tags,
                    "hit_count",
                    int(card_state.upgrades),
                    default=1.0,
                ),
            )
            draw = _upgraded_stat(
                card,
                tags,
                "cards_draw",
                int(card_state.upgrades),
            )
            if draw <= 0:
                # Catalog variants encode draw either as `cards_draw` or as
                # the effect tag/raw `draw`.  Use the latter only as a
                # fallback: adding both counts the same card text twice.
                draw = _upgraded_stat(
                    card,
                    tags,
                    "draw",
                    int(card_state.upgrades),
                )
            energy_gain = _upgraded_stat(
                card,
                tags,
                "energy_gain",
                int(card_state.upgrades),
            )
            immediate += count * damage * hits
            survival += count * block
            is_aoe = _number(tags.get("aoe")) > 0 or str(
                card.get("target") or ""
            ).replace("_", "").upper() in {"ALLENEMIES", "ALL"}
            if is_aoe:
                # AOE is a capability for multi-enemy rooms, so reflect the
                # actual repeated hit output rather than merely counting one
                # tagged card.
                aoe += count * max(_number(tags.get("aoe")), damage * hits)
            growth += count * sum(
                max(
                    _number(tags.get(tag)),
                    _upgraded_stat(
                        card,
                        {},
                        tag,
                        int(card_state.upgrades),
                    ),
                )
                for tag in ("scaling", "strength", "poison", "power:doom")
            )
            resources += count * (
                draw
                + energy_gain
                + _number(tags.get("cost_reduction"))
            )

        relic_ids = list(request.world.relic_ids)
        relic_ids.extend(relic.relic for relic in request.world.relics)
        for relic_id in _unique(relic_ids):
            relic = self.repository.find_relic(relic_id)
            if relic is None:
                gaps.append(f"unknown_relic:{relic_id}")
                continue
            tags = effects(relic)
            relic_support += sum(
                _number(tags.get(tag))
                for tag in (
                    "supports_attack", "supports_block", "supports_draw",
                    "supports_energy", "supports_strength", "healing",
                )
            )
            survival += _number(tags.get("healing")) * 2.0
            resources += _number(tags.get("supports_draw")) + _number(tags.get("supports_energy"))

        for potion_state in request.world.potions:
            potion = self.repository.find_entity("potions", potion_state.potion)
            if potion is None:
                gaps.append(f"unknown_potion:{potion_state.potion}")
                continue
            tags = effects(potion)
            # The occupied, known potion slot is a real one-combat buffer
            # even when the imported catalog has no structured tag yet.
            potion_support += 1.0
            potion_support += sum(
                _number(tags.get(tag))
                for tag in ("damage", "block", "healing", "aoe", "strength")
            )

        for warning in request.world.capture_warnings:
            gaps.append(f"capture_warning:{warning}")
        return RouteCapabilities(
            immediate_power=round(immediate, 3),
            survival=round(survival, 3),
            aoe=round(aoe, 3),
            growth=round(growth, 3),
            resources=round(resources, 3),
            relic_support=round(relic_support, 3),
            potion_support=round(potion_support, 3),
            data_gaps=_unique(gaps),
        )

    def _boss_threat(
        self,
        request: DecisionRequest,
        profile: Mapping[str, Any],
    ) -> tuple[float, tuple[str, ...]]:
        """Use known boss HP and move records, otherwise disclose the gap."""
        expected = _number(profile["encounter_expected_risk"]["BOSS"])
        facts = build_boss_preparation(
            self.repository,
            request.world.map.boss_encounter_ids if request.world.map else (),
            ascension=request.world.ascension,
        )
        return facts.route_threat(expected), facts.data_gaps

    def _encounter_pools(
        self,
        request: DecisionRequest,
    ) -> tuple[Mapping[str, Mapping[str, float]], tuple[str, ...]]:
        """Resolve static per-act pressure without naming a future enemy."""
        identifiers = _unique(
            request.world.map.boss_encounter_ids if request.world.map else ()
        )
        key = (tuple(sorted(identifiers)), int(request.world.ascension))
        if key not in self._encounter_pool_cache:
            self._encounter_pool_cache[key] = self.repository.encounter_pool_expectations(
                identifiers,
                ascension=request.world.ascension,
            )
        pools = self._encounter_pool_cache[key]
        gaps = tuple(
            f"encounter_pool_missing:{kind}"
            for kind in ("MONSTER", "ELITE")
            if kind not in pools
        )
        return pools, gaps

    @staticmethod
    def _state_gaps(
        request: DecisionRequest,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Separate missing decision-critical state from degradable detail."""
        critical: list[str] = []
        degradable: list[str] = []
        hp = request.world.hp
        max_hp = request.world.max_hp
        if hp is None:
            critical.append("hp_missing")
        if max_hp is None:
            critical.append("max_hp_missing")
        elif max_hp <= 0:
            critical.append("max_hp_invalid")
        if hp is not None and max_hp is not None and hp > max_hp:
            critical.append("hp_exceeds_max_hp")
        if request.world.gold is None:
            # Gold only controls the known Shop opportunity.  Keep a route
            # recommendation conservative and visibly incomplete rather than
            # pretending the player owns zero gold.
            degradable.append("gold_missing")
        return _unique(critical), _unique(degradable)

    @staticmethod
    def _potion_relief(path: SearchPath, capabilities: RouteCapabilities) -> float:
        """Apply one potion buffer to at most one combat on a route.

        Potion inventory is a one-combat resource.  Applying it in every
        monster, elite and boss node creates a false preference for longer
        combat routes, so this bounded relief is calculated once from the
        highest confirmed combat pressure on the selected path.
        """
        if capabilities.potion_support <= 0 or path.max_combat_risk <= 0:
            return 0.0
        return round(min(
            4.0,
            capabilities.potion_support * 0.5,
            path.max_combat_risk * 0.35,
        ), 3)

    def _node_risk(
        self,
        node: Any,
        request: DecisionRequest,
        profile: Mapping[str, Any],
        capabilities: RouteCapabilities,
        boss_threat: float,
        encounter_pools: Mapping[str, Mapping[str, float]],
        gold_available: bool = True,
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        kind = str(node.kind or "UNKNOWN").upper()
        kind_weights = profile["kind_weights"]
        encounter_risk = profile["encounter_expected_risk"]
        weights = profile["capability_weights"]
        hp_ratio = (
            request.world.hp / request.world.max_hp
            if request.world.hp is not None and request.world.max_hp
            else None
        )
        risk = _number(kind_weights[kind])
        factors: list[dict[str, Any]] = []

        if kind in ("MONSTER", "ELITE"):
            risk += _number(encounter_risk[kind])
            pool = encounter_pools.get(kind)
            if pool is not None:
                pressure = (
                    _number(pool.get("average_hp")) / 110.0
                    + _number(pool.get("average_attack")) / 18.0
                    + _number(pool.get("average_enemy_count")) * 0.45
                )
                risk += pressure
                factors.append({
                    "code": f"{kind.lower()}_encounter_pool_pressure",
                    "delta": -round(pressure, 3),
                })
            power_relief = capabilities.immediate_power * _number(weights["immediate_power"])
            defense_relief = capabilities.survival * _number(weights["survival"])
            resource_relief = capabilities.resources * _number(weights["resource"])
            risk -= power_relief + defense_relief + resource_relief
            factors.append({"code": f"{kind.lower()}_expected_risk", "delta": -round(_number(encounter_risk[kind]), 3)})
            if kind == "MONSTER":
                aoe_relief = capabilities.aoe * _number(weights["aoe"])
                risk -= aoe_relief
                if aoe_relief:
                    factors.append({"code": "monster_aoe_relief", "delta": round(aoe_relief, 3)})
            else:
                growth_relief = capabilities.growth * _number(weights["growth"])
                risk -= growth_relief
                if hp_ratio is not None and hp_ratio < _number(profile["hp_low_threshold"]):
                    penalty = _number(profile["elite_hp_penalty"])
                    risk += penalty
                    factors.append({"code": "elite_low_hp", "delta": -round(penalty, 3)})
                if growth_relief:
                    factors.append({"code": "elite_growth_relief", "delta": round(growth_relief, 3)})
            if risk < 0:
                # A deck can make a fight safe, never better than no fight:
                # the surplus is not carried along the path.
                factors.append({"code": "combat_risk_floor", "delta": round(risk, 3)})
                risk = 0.0
        elif kind == "BOSS":
            risk += boss_threat
            risk -= (
                capabilities.immediate_power * _number(weights["immediate_power"])
                + capabilities.survival * _number(weights["survival"])
                + capabilities.growth * _number(weights["growth"])
                + capabilities.resources * _number(weights["resource"])
            )
            factors.append({"code": "known_boss_risk", "delta": -round(boss_threat, 3)})
            if risk < 0:
                factors.append({"code": "combat_risk_floor", "delta": round(risk, 3)})
                risk = 0.0
        elif kind == "CAMPFIRE":
            if hp_ratio is not None and hp_ratio < _number(profile["hp_low_threshold"]):
                relief = _number(profile["rest_low_hp_bonus"])
                risk -= relief
                factors.append({"code": "rest_low_hp", "delta": round(relief, 3)})
            # A campfire is a verified future preparation opportunity.  Its
            # route value rises with the *known* boss pressure, which makes
            # boss identity affect the ranking without pretending to know a
            # future card/relic reward or a combat turn sequence.
            expected_boss = max(1.0, _number(encounter_risk["BOSS"]))
            preparation = min(
                _number(profile["boss_path_bonus"]) * 2.0,
                _number(profile["boss_path_bonus"]) * boss_threat / expected_boss,
            )
            if preparation > 0:
                risk -= preparation
                factors.append({
                    "code": "boss_preparation_opportunity",
                    "delta": round(preparation, 3),
                })
        elif kind == "SHOP":
            if request.world.gold is None:
                factors.append({"code": "shop_gold_unknown", "delta": 0.0})
            elif request.world.gold >= _number(profile["shop_gold_threshold"]) and not gold_available:
                # The gold is spent at the first shop of the path.
                factors.append({"code": "shop_gold_spent_earlier", "delta": 0.0})
            elif request.world.gold >= _number(profile["shop_gold_threshold"]):
                scale = profile["shop_gold_relief"]
                relief = min(
                    _number(scale["max"]),
                    1.0 + request.world.gold / _number(scale["gold_per_point"]),
                )
                risk -= relief
                factors.append({"code": "shop_affordability", "delta": round(relief, 3)})
            else:
                risk += 1.0
                factors.append({"code": "shop_low_gold", "delta": -1.0})
        mode_adjustment, mode_code = self._mode_risk_adjustment(
            kind,
            request,
            profile,
        )
        if mode_adjustment:
            risk += mode_adjustment
            factors.append({
                "code": mode_code,
                # Factors are expressed as recommendation value while this
                # search minimizes risk, hence the inverted sign.
                "delta": round(-mode_adjustment, 3),
            })
        return round(risk, 4), tuple(factors)

    @staticmethod
    def _mode_risk_adjustment(
        kind: str,
        request: DecisionRequest,
        profile: Mapping[str, Any],
    ) -> tuple[float, str]:
        """Return one soft, explainable mode adjustment in risk units.

        Growth never bypasses the shared HP safety model.  Between the
        critical and low-HP thresholds its upside fades continuously; at or
        below the critical threshold combat growth becomes a penalty and
        campfires receive survival relief.
        """
        mode = request.world.route_mode
        hp_ratio = (
            request.world.hp / request.world.max_hp
            if request.world.hp is not None and request.world.max_hp
            else 0.0
        )
        low = _number(profile["hp_low_threshold"])
        critical = _number(profile["hp_critical_threshold"])

        if mode == ROUTE_MODE_BALANCED:
            return 0.0, "route_mode_balanced"
        if mode == ROUTE_MODE_SURVIVAL:
            missing_hp = max(0.0, 1.0 - hp_ratio)
            adjustments = {
                "CAMPFIRE": -(1.2 + 3.0 * missing_hp),
                "ELITE": 2.2 + 2.5 * missing_hp,
                "MONSTER": 0.35 + 0.5 * missing_hp,
                "EVENT": 0.3,
                "UNKNOWN": 0.6,
            }
            return adjustments.get(kind, 0.0), (
                f"route_mode_survival_{kind.lower()}"
            )
        if mode != ROUTE_MODE_GROWTH:
            # WorldState validation makes this unreachable; retaining a
            # neutral branch keeps this helper safe for direct test calls.
            return 0.0, "route_mode_invalid"

        safety = 1.0
        if hp_ratio < low:
            safety = max(
                0.0,
                min(1.0, (hp_ratio - critical) / max(0.001, low - critical)),
            )
        if safety <= 0.0:
            adjustments = {
                "CAMPFIRE": -3.0,
                "ELITE": 3.0,
                "MONSTER": 0.75,
                "EVENT": 0.5,
                "UNKNOWN": 0.8,
            }
            return adjustments.get(kind, 0.0), (
                f"route_mode_growth_safety_floor_{kind.lower()}"
            )

        # Growth incentives are deliberately bounded and taper toward zero
        # near the safety floor.  Shop upside only exists with confirmed
        # spendable gold; otherwise the base policy's data gap remains.
        adjustments = {
            "ELITE": -2.2 * safety,
            "TREASURE": -1.0 * safety,
            "EVENT": -0.35 * safety,
            "SHOP": (
                -1.25 * safety
                if request.world.gold is not None
                and request.world.gold
                >= _number(profile["shop_gold_threshold"])
                else 0.0
            ),
            "CAMPFIRE": 0.35 * safety if hp_ratio >= low else -1.5 * (1.0 - safety),
        }
        return adjustments.get(kind, 0.0), (
            f"route_mode_growth_{kind.lower()}"
        )

    def _best_paths(
        self,
        graph: GraphValidation,
        request: DecisionRequest,
        profile: Mapping[str, Any],
        capabilities: RouteCapabilities,
        boss_threat: float,
        encounter_pools: Mapping[str, Mapping[str, float]],
    ) -> tuple[dict[str, SearchPath], dict[str, tuple[dict[str, Any], ...]], tuple[str, ...]]:
        """Find one minimum-risk path per candidate with bounded memoization."""
        # State: the node and whether the gold is still unspent (no shop yet
        # on the way here); a shop's gold value counts once per path.
        cache: dict[tuple[str, bool], SearchPath | None] = {}
        factors_by_state: dict[tuple[str, bool], tuple[dict[str, Any], ...]] = {}
        factors_by_node: dict[str, tuple[dict[str, Any], ...]] = {}
        expanded = 0
        limit_hit = False

        def best(node_id: str, gold_available: bool = True) -> SearchPath | None:
            nonlocal expanded, limit_hit
            state = (node_id, gold_available)
            if state in cache:
                return cache[state]
            if expanded >= self.max_search_expansions:
                limit_hit = True
                return None
            expanded += 1
            node = graph.nodes[node_id]
            own_risk, factors = self._node_risk(
                node, request, profile, capabilities, boss_threat, encounter_pools,
                gold_available,
            )
            factors_by_state[state] = factors
            gold_after = gold_available and str(node.kind).upper() != "SHOP"
            own_combat_risk = own_risk if str(node.kind).upper() in {
                "MONSTER", "ELITE", "BOSS"
            } else 0.0
            if node_id in graph.boss_ids:
                result = SearchPath((node_id,), own_risk, own_combat_risk)
                cache[state] = result
                return result
            options: list[SearchPath] = []
            for child_id in sorted(node.edges):
                child = best(child_id, gold_after)
                if child is not None:
                    options.append(child)
            if not options:
                cache[state] = None
                return None
            tail = min(options, key=lambda value: (value.risk, value.node_ids))
            result = SearchPath(
                (node_id, *tail.node_ids),
                round(own_risk + tail.risk, 4),
                max(own_combat_risk, tail.max_combat_risk),
            )
            cache[state] = result
            return result

        found = {candidate.candidate_id: best(candidate.candidate_id) for candidate in request.candidates}
        # Factors of each node as it is reached on its candidate's path.
        for path in found.values():
            gold_available = True
            for node_id in path.node_ids if path is not None else ():
                factors_by_node.setdefault(node_id, factors_by_state[(node_id, gold_available)])
                gold_available = gold_available and str(graph.nodes[node_id].kind).upper() != "SHOP"
        gaps: list[str] = []
        if limit_hit:
            gaps.append("route_search_limit_exceeded")
        for candidate_id, path in found.items():
            if path is None:
                gaps.append(f"boss_unreachable_from_candidate:{candidate_id}")
        return (
            {candidate_id: path for candidate_id, path in found.items() if path is not None},
            factors_by_node,
            _unique(gaps),
        )

    def recommend(self, request: DecisionRequest) -> Recommendation:
        candidate_ids = [candidate.candidate_id for candidate in request.candidates]
        graph = validate_route_graph(request.world.map, candidate_ids)
        profile, profile_gaps = _validate_profile(self._profile())
        state_critical_gaps, state_degradable_gaps = self._state_gaps(request)
        critical_gaps = _unique(
            (*graph.critical_gaps, *profile_gaps, *state_critical_gaps)
        )
        if critical_gaps:
            return self._unknown(request, graph.origin, critical_gaps)
        assert profile is not None

        capabilities = self._capabilities(request)
        boss_threat, boss_gaps = self._boss_threat(request, profile)
        encounter_pools, pool_gaps = self._encounter_pools(request)
        paths, path_factors, search_gaps = self._best_paths(
            graph,
            request,
            profile,
            capabilities,
            boss_threat,
            encounter_pools,
        )
        if search_gaps:
            return self._unknown(
                request,
                graph.origin,
                _unique((
                    *search_gaps,
                    *state_degradable_gaps,
                    *capabilities.data_gaps,
                    *boss_gaps,
                    *pool_gaps,
                )),
            )

        assessments: list[CandidateAssessment] = []
        path_rows: list[dict[str, Any]] = []
        ranking_risk: dict[str, float] = {}
        for candidate in request.candidates:
            path = paths[candidate.candidate_id]
            # The capability baseline is shared, while its effect on expected
            # monster/elite/boss risk is path-specific.  It is retained here
            # only as an auditable 0-100 output scale, not a win probability.
            potion_relief = self._potion_relief(path, capabilities)
            effective_risk = path.risk - potion_relief
            score = _route_display_score(
                effective_risk,
                capabilities.relic_support,
            )
            ranking_risk[candidate.candidate_id] = (
                effective_risk
                - capabilities.relic_support * (0.4 / 3.5)
            )
            path_rows.append({
                "candidate_id": candidate.candidate_id,
                "node_ids": list(path.node_ids),
                "score": score if candidate.eligible else None,
            })
            path_factor_rows = tuple(
                factor
                for node_id in path.node_ids
                for factor in path_factors.get(node_id, ())
            )
            if request.world.route_mode == ROUTE_MODE_BALANCED:
                path_factor_rows = (*path_factor_rows, {
                    "code": "route_mode_balanced",
                    "delta": 0.0,
                })
            if potion_relief:
                path_factor_rows = (*path_factor_rows, {
                    "code": "potion_buffer_once",
                    "delta": round(potion_relief, 3),
                })
            all_gaps = _unique((
                *state_degradable_gaps,
                *capabilities.data_gaps,
                *boss_gaps,
                *pool_gaps,
            ))
            path_factor_rows = tuple(
                _described_factor(factor)
                for factor in path_factor_rows
            )
            dimensions = {
                "immediate_power": round(min(10.0, capabilities.immediate_power / 12.0), 3),
                "survival": round(min(10.0, capabilities.survival / 10.0), 3),
                "long_term_growth": round(min(10.0, capabilities.growth), 3),
                "resource_efficiency": round(min(10.0, capabilities.resources), 3),
                "synergy": round(min(10.0, capabilities.relic_support + capabilities.potion_support), 3),
                "route_fit": round(max(-10.0, min(10.0, -effective_risk)), 3),
                "data_completeness": 0.0 if all_gaps else 1.0,
            }
            assessments.append(CandidateAssessment(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                display_index=int(candidate.display_index),
                eligible=candidate.eligible,
                score=score if candidate.eligible else None,
                rank=None,
                factors=path_factor_rows,
                dimensions=dimensions,
                data_gaps=all_gaps,
            ))

        eligible = sorted(
            (assessment for assessment in assessments if assessment.eligible and assessment.score is not None),
            # Rank with the unrounded search value.  Display scores are only a
            # readable projection and may legitimately round to the same
            # number for nearly identical paths.
            key=lambda assessment: (
                ranking_risk[assessment.candidate_id],
                assessment.display_index,
                assessment.candidate_id,
            ),
        )
        rank_by_id = {
            assessment.candidate_id: rank
            for rank, assessment in enumerate(eligible, start=1)
        }
        assessments = [
            replace(assessment, rank=rank_by_id.get(assessment.candidate_id))
            for assessment in assessments
        ]
        selected = eligible[0].candidate_id if eligible else None
        path_by_candidate = {row["candidate_id"]: row["node_ids"] for row in path_rows}
        gaps = _unique((
            *state_degradable_gaps,
            *capabilities.data_gaps,
            *boss_gaps,
            *pool_gaps,
        ))
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload=MappingProxyType({"paths": path_rows}),
            presentation={
                "kind": "route_paths",
                "origin_node_id": graph.origin,
                "primary_path_node_ids": path_by_candidate.get(selected, []),
                # Every candidate path remains in ``paths`` for auditable
                # ranking.  The player-facing map deliberately renders only
                # the current highest-ranked plan.
                "backup_path_node_ids": [],
                "paths": path_rows,
            },
            candidates=tuple(assessments),
            recommended_candidate_id=selected,
            status="recommend" if selected else "uncertain",
            confidence="medium" if not gaps else "low",
            data_gaps=gaps,
            policy_version=ROUTE_POLICY_VERSION,
            world_sequence=request.world.sequence,
            contract_version=2,
        )

    @staticmethod
    def _unknown(
        request: DecisionRequest,
        origin: str | None,
        gaps: tuple[str, ...],
    ) -> Recommendation:
        assessments = tuple(CandidateAssessment(
            candidate_id=candidate.candidate_id,
            label=candidate.label,
            display_index=int(candidate.display_index),
            eligible=candidate.eligible,
            score=None,
            rank=None,
            data_gaps=gaps,
        ) for candidate in request.candidates)
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload=MappingProxyType({"paths": []}),
            presentation={
                "kind": "route_paths",
                "origin_node_id": origin,
                "primary_path_node_ids": [],
                "backup_path_node_ids": [],
                "paths": [],
            },
            candidates=assessments,
            recommended_candidate_id=None,
            status="uncertain",
            confidence="low",
            data_gaps=gaps,
            policy_version="route:unavailable",
            world_sequence=request.world.sequence,
            contract_version=2,
        )

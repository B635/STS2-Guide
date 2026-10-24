"""Card-reward scoring v2: deck fitness (牌组适合度).

Every option, skip included, is scored as the fitness of the deck *after* that
choice. Fitness compares what the deck can do per turn on six capability
dimensions (能力维度) with what the coming fights need (局面需求):

* capabilities come only from assembly-verified card facts. Each card counts
  by how often it is drawn, and the cards drawn in a turn are only played as
  far as energy allows, so a weak or expensive card dilutes the deck by itself;
* needs come from real encounter-pool and boss HP/attack facts, mixed by the
  fight kinds on the current mode's recommended route to this act's boss (the
  act and floor enter through that route); HP and preferences scale how much
  each need matters (its weight), never the amount the enemies require;
* each dimension's value rises at slope 1-2 below its need and at most 0.25
  above it, so a shortfall always costs more than a surplus earns.

No community prior is used. Unknown facts contribute nothing and are reported.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence

from advisor.boss_context import build_boss_preparation
from advisor.encounter_turns import encounter_turns
from advisor.monster_turns import mean
from advisor.deck_mechanics import (
    ORB_TYPES,
    CardProfile,
    RelicProfile,
    Situation,
    adjust_added_card,
    card_profile,
    deck_capabilities,
    enchant,
    relic_profile,
)
from advisor.decision_core import (
    CARD_REWARD,
    ROUTE_CHOICE,
    CandidateAssessment,
    DecisionCandidate,
    DecisionRequest,
    Recommendation,
    WorldMap,
)
from advisor.route import RoutePolicy
from advisor.scoring_preferences import DEFAULT_SCORING_PREFERENCES, ScoringPreferences
# Scoring kernel v2; the ``_1`` suffix versions the deck-fitness parameters.
from advisor.versions import DECK_FITNESS_POLICY_VERSION
from storage.relational import RelationalRepository

PARAMETERS_KEY = "deck_fitness_v1"
DIMENSIONS = ("frontload", "aoe", "defense", "scaling", "cycle", "energy")
DIMENSION_LABELS = {
    "frontload": "前期输出",
    "aoe": "群伤",
    "defense": "防御",
    "scaling": "成长",
    "cycle": "循环",
    "energy": "能量",
}
# Recommendation contract dimensions each capability dimension reports into.
CONTRACT_DIMENSIONS = {
    "frontload": "immediate_power",
    "aoe": "immediate_power",
    "defense": "survival",
    "scaling": "long_term_growth",
    "cycle": "long_term_growth",
    "energy": "resource_efficiency",
}
FIGHT_KINDS = ("MONSTER", "ELITE", "BOSS")


@dataclass(frozen=True)
class Needs:
    values: Mapping[str, float]
    weights: Mapping[str, float]
    fight_turns: float
    extra_enemies: float
    incoming_attack: float
    fight_mix: Mapping[str, float]
    gaps: tuple[str, ...] = ()
    fight_sequence: tuple[str, ...] = ()   # remaining fights of the act, in route order
    # From encounter/monster facts (defaults: one hit per enemy, nothing else).
    enemy_hits: Optional[float] = None     # attack hits aimed at the player per turn
    player_weak: float = 0.0               # share of turns the player is Weak / Frail
    player_frail: float = 0.0
    status_cards: float = 0.0              # cards pushed into the player's piles per turn
    enemy_hp: Optional[float] = None       # total HP of a fight's enemies


@dataclass(frozen=True)
class Fitness:
    score: float
    raw: float
    capabilities: Mapping[str, float]
    terms: Mapping[str, float]
    tempo: float = 1.0   # expected fight length relative to the nominal one


# Every Guide-owned parameter the kernel reads: scalars, and tables with the
# keys they must hold. Nothing has a hidden default: a missing or malformed
# value stops scoring with a parameter gap.
SCALAR_PARAMETERS = (
    "hand_size", "base_energy", "block_share", "low_hp_defense_pressure", "scaling_onset_turns",
    "cycle_need", "vulnerable_damage_bonus", "weak_damage_reduction", "frail_block_reduction",
    "surplus_slope", "satisfaction_cap", "confidence_gap_allowance", "near_tie_points",
)
TABLE_PARAMETERS = {
    "fight_turns": FIGHT_KINDS,
    "fight_importance": FIGHT_KINDS,
    "dimension_weights": DIMENSIONS,
    "fight_tempo": ("min", "max"),
    "lookahead": ("scenarios", "max_rewards"),
    "synergy_notes": ("min_gain", "max_cards"),
    "card_reward_odds": ("ascension_threshold", "rare_offset_initial"),
    "pick_margin_by_act": (),
    "weak_encounters_by_act_index": (),
    "base_orb_slots": (),
}
REQUIRED_PARAMETERS = (*SCALAR_PARAMETERS, *TABLE_PARAMETERS, "preference_dimensions")


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parameter_gaps(params: object) -> tuple[str, ...]:
    if not isinstance(params, Mapping):
        return ("deck_fitness_parameters_missing",)
    missing = [key for key in REQUIRED_PARAMETERS if key not in params]
    if missing:
        return tuple(f"deck_fitness_parameter_missing:{key}" for key in missing)
    invalid = [key for key in SCALAR_PARAMETERS if not _number(params[key])]
    for key, fields in TABLE_PARAMETERS.items():
        table = params[key]
        if not isinstance(table, Mapping) or any(not _number(table.get(field)) for field in fields):
            invalid.append(key)
    if invalid:
        return tuple(f"deck_fitness_parameter_invalid:{key}" for key in invalid)
    odds = params["card_reward_odds"]
    if any(not isinstance(odds.get(level), Mapping) for level in ("base", "ascension")):
        return ("deck_fitness_parameter_invalid:card_reward_odds",)
    rarity_rows = [odds[level].get(kind) for level in ("base", "ascension") for kind in FIGHT_KINDS]
    if any(not isinstance(row, Mapping) or not row or not all(_number(v) for v in row.values()) for row in rarity_rows):
        return ("deck_fitness_parameter_invalid:card_reward_odds",)
    for key in ("pick_margin_by_act", "weak_encounters_by_act_index"):
        table = params[key]
        if not table or any(not str(act).isdigit() or not _number(value) or value < 0 for act, value in table.items()):
            return (f"deck_fitness_parameter_invalid:{key}",)
    if not all(_number(value) for value in params["base_orb_slots"].values()):
        return ("deck_fitness_parameter_invalid:base_orb_slots",)
    preferences = params["preference_dimensions"]
    if not isinstance(preferences, Mapping) or any(
        not hasattr(DEFAULT_SCORING_PREFERENCES, str(name))
        or not isinstance(dimensions, (list, tuple)) or any(d not in DIMENSIONS for d in dimensions)
        for name, dimensions in preferences.items()
    ):
        return ("deck_fitness_parameter_invalid:preference_dimensions",)
    if float(params["hand_size"]) <= 0 or float(params["base_energy"]) <= 0:
        return ("deck_fitness_parameter_invalid:hand_size",)
    lookahead = params["lookahead"]
    if float(lookahead["scenarios"]) < 1 or float(lookahead["max_rewards"]) < 0:
        return ("deck_fitness_parameter_invalid:lookahead",)
    if float(params["surplus_slope"]) < 0:
        return ("deck_fitness_parameter_invalid:surplus_slope",)
    for key in ("confidence_gap_allowance", "near_tie_points"):
        if float(params[key]) < 0:
            return (f"deck_fitness_parameter_invalid:{key}",)
    if any(float(params["fight_turns"][kind]) <= 0 for kind in FIGHT_KINDS):
        return ("deck_fitness_parameter_invalid:fight_turns",)
    if sum(float(params["fight_importance"][kind]) for kind in FIGHT_KINDS) <= 0:
        return ("deck_fitness_parameter_invalid:fight_importance",)
    if sum(float(value) for value in params["dimension_weights"].values()) <= 0:
        return ("deck_fitness_parameter_invalid:dimension_weights",)
    tempo = params["fight_tempo"]
    if not 0 < float(tempo["min"]) <= float(tempo["max"]):
        return ("deck_fitness_parameter_invalid:fight_tempo",)
    if float(params["satisfaction_cap"]) <= 1 or float(params["block_share"]) <= 0:
        return ("deck_fitness_parameter_invalid:satisfaction_cap",)
    return ()


def _weighted(rows: Sequence[tuple[float, float]]) -> float:
    total = sum(weight for _, weight in rows)
    return sum(value * weight for value, weight in rows) / total if total else 0.0


def _reachable_fight_share(
    world_map: WorldMap, starts: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Expected count of each fight kind over all paths from ``starts`` (the
    next nodes by default) to a boss, each path equally likely."""

    by_id = {node.node_id: node for node in world_map.nodes}
    boss_ids = set(world_map.boss_node_ids)
    to_boss: Dict[str, int] = {}

    def paths_to_boss(node_id: str) -> int:
        if node_id not in to_boss:
            node = by_id.get(node_id)
            if node is None:
                to_boss[node_id] = 0
            elif node_id in boss_ids:
                to_boss[node_id] = 1
            else:
                to_boss[node_id] = sum(paths_to_boss(child) for child in node.edges)
        return to_boss[node_id]

    starts = [
        n for n in (world_map.available_next_node_ids if starts is None else starts)
        if paths_to_boss(n) > 0
    ]
    total = sum(to_boss[n] for n in starts)
    if not total:
        return {}
    from_start: Dict[str, int] = {n: 1 for n in starts}
    for node in sorted((by_id[n] for n in to_boss if n in by_id), key=lambda n: n.row):
        for child in node.edges:
            if from_start.get(node.node_id) and to_boss.get(child):
                from_start[child] = from_start.get(child, 0) + from_start[node.node_id]
    share: Dict[str, float] = {}
    for node_id, incoming in from_start.items():
        kind = str(by_id[node_id].kind).upper()
        if kind in FIGHT_KINDS and to_boss.get(node_id):
            share[kind] = share.get(kind, 0.0) + incoming * to_boss[node_id] / total
    return share


def _sequence_from_mix(mix: Mapping[str, float]) -> tuple[str, ...]:
    """Fights in a plausible order when no single path is known."""

    return tuple(
        kind for kind in ("MONSTER", "ELITE", "BOSS")
        for _ in range(int(round(mix.get(kind, 0.0))))
    )


def _fight_mix(
    route_policy: RoutePolicy,
    request: DecisionRequest,
) -> tuple[Dict[str, float], tuple[str, ...], tuple[str, ...]]:
    """Fight kinds ahead (mix, gaps, ordered kinds on the recommended route);
    without a route, the reachable-path share and no order."""

    world_map = request.world.map
    if world_map is None or not world_map.nodes:
        return {}, ("route_context_missing",), ()
    by_id = {node.node_id: node for node in world_map.nodes}
    candidates = [
        DecisionCandidate.create(node_id, by_id[node_id].as_dict(), label=node_id, display_index=i)
        for i, node_id in enumerate(world_map.available_next_node_ids)
        if node_id in by_id
    ]
    if candidates:
        route = route_policy.recommend(DecisionRequest.create(
            decision_id=f"{request.decision_id}:route-context",
            decision_type=ROUTE_CHOICE,
            world=request.world,
            candidates=candidates,
            constraints={
                "origin_node_id": world_map.origin_node_id,
                "map_name": world_map.map_name,
                "route_mode": request.world.route_mode,
            },
        ))
        path = (route.presentation or {}).get("primary_path_node_ids") or ()
        mix: Dict[str, float] = {}
        sequence = []
        for node_id in path:
            kind = str(by_id[node_id].kind).upper() if node_id in by_id else ""
            if kind in FIGHT_KINDS:
                mix[kind] = mix.get(kind, 0.0) + 1.0
                sequence.append(kind)
        if mix:
            return mix, (), tuple(sequence)
    share = _reachable_fight_share(world_map)
    gaps = ("route_path_unavailable",) if share else ("route_path_unavailable", "boss_unreachable")
    return share, gaps, ()


STAT_KEYS = ("hp", "enemies", "attack", "hits", "weak", "frail", "vulnerable", "cards")


def _encounter_row(
    repository: RelationalRepository, encounter_id: str, turns: int, ascension: int,
) -> Optional[tuple]:
    """One reviewed encounter walked for ``turns`` turns: (mean per-turn
    pressure, kinds of what the walk could not follow), or ``None``. Cached
    on the repository: facts are fixed for one release database."""

    cache = repository.__dict__.setdefault("_encounter_row_cache", {})
    key = (str(encounter_id).upper(), int(turns), int(ascension))
    if key in cache:
        return cache[key]

    def monster_fact(monster_id: str) -> Optional[Mapping]:
        fact = repository.find_assembly_fact("monsters", monster_id)
        return fact if fact is not None and fact.get("status") == "converted" else None

    result = None
    fact = repository.find_assembly_fact("encounters", encounter_id)
    if fact is not None and fact.get("status") == "converted":
        walked = encounter_turns(fact, monster_fact, turns, ascension)
        if walked is not None and not any(g.startswith("monster_fact_missing") for g in walked.gaps):
            result = ({
                "hp": walked.hp,
                "enemies": walked.enemies,
                "attack": mean(walked.attack),
                "hits": mean(walked.hits),
                # A debuff stack lasts about a turn: uptime is capped at every turn.
                "weak": mean([min(1.0, v) for v in walked.debuffs["Weak"]]),
                "frail": mean([min(1.0, v) for v in walked.debuffs["Frail"]]),
                "vulnerable": mean([min(1.0, v) for v in walked.debuffs["Vulnerable"]]),
                "cards": mean(walked.cards),
                # What the walk could not follow, by kind (the ids stay in the facts).
            }, frozenset(gap.split(":", 1)[0] for gap in walked.gaps))
    cache[key] = result
    return result


def _encounter_stats(
    repository: RelationalRepository, encounter_ids: Sequence[str], turns: int, ascension: int,
) -> Optional[Dict[str, float]]:
    """Mean per-turn pressure over the reviewed encounters of one pool."""

    rows = []
    kinds: set = set()
    for encounter_id in encounter_ids:
        walked = _encounter_row(repository, encounter_id, turns, ascension)
        if walked is None:
            continue
        rows.append(walked[0])
        kinds.update(walked[1])
    if not rows:
        return None
    stats: Dict = {key: sum(row[key] for row in rows) / len(rows) for key in STAT_KEYS}
    stats["gaps"] = tuple(sorted(kinds))
    return stats


def _fact_pools(
    repository: RelationalRepository, boss_ids: Sequence[str], params: Mapping, ascension: int,
) -> Dict[str, Dict[str, float]]:
    """Per pool (MONSTER, MONSTER_WEAK, ELITE, BOSS) stats from the facts,
    cached on the repository: facts are fixed for one release database."""

    cache = repository.__dict__.setdefault("_fact_pool_cache", {})
    key = (tuple(boss_ids), int(ascension), json.dumps(params["fight_turns"], sort_keys=True))
    if key in cache:
        return cache[key]
    groups: Dict[str, List[str]] = {"MONSTER": [], "MONSTER_WEAK": [], "ELITE": [], "BOSS": list(boss_ids)}
    bosses = {str(b).upper() for b in boss_ids}
    for encounter_id in repository.act_encounter_ids(boss_ids):
        fact = repository.find_assembly_fact("encounters", encounter_id)
        if fact is None or fact.get("status") != "converted" or encounter_id.upper() in bosses:
            continue
        room = str(fact.get("room_type") or "").upper()
        if room == "ELITE":
            groups["ELITE"].append(encounter_id)
        elif room == "MONSTER":
            groups["MONSTER_WEAK" if fact.get("is_weak") else "MONSTER"].append(encounter_id)
    pools: Dict[str, Dict[str, float]] = {}
    for group, ids in groups.items():
        kind = "MONSTER" if group == "MONSTER_WEAK" else group
        stats = _encounter_stats(repository, ids, int(round(float(params["fight_turns"][kind]))), ascension)
        if stats is not None:
            pools[group] = stats
    cache[key] = pools
    return pools


def _monster_fights_done(world_map: WorldMap) -> float:
    """Expected normal fights already fought this act: the mean number of
    MONSTER nodes over all paths from the first row to the current node."""

    by_id = {node.node_id: node for node in world_map.nodes}
    current = world_map.origin_node_id
    if current not in by_id:
        return 0.0
    paths: Dict[str, float] = {}
    fights: Dict[str, float] = {}
    first_row = min(node.row for node in world_map.nodes)
    for node in sorted(world_map.nodes, key=lambda n: n.row):
        if node.row == first_row:
            paths.setdefault(node.node_id, 1.0)
            fights.setdefault(node.node_id, 0.0)
        count = paths.get(node.node_id, 0.0)
        if not count:
            continue
        own = fights.get(node.node_id, 0.0) + (count if str(node.kind).upper() == "MONSTER" else 0.0)
        fights[node.node_id] = own
        for child in node.edges:
            paths[child] = paths.get(child, 0.0) + count
            fights[child] = fights.get(child, 0.0) + own
    return fights.get(current, 0.0) / paths[current] if paths.get(current) else 0.0


def compute_needs(
    repository: RelationalRepository,
    route_policy: RoutePolicy,
    request: DecisionRequest,
    params: Mapping,
    preferences: ScoringPreferences,
) -> Needs:
    world = request.world
    mix, gaps, sequence = _fight_mix(route_policy, request)
    boss_ids = world.map.boss_encounter_ids if world.map else ()
    if not mix and world.map and world.map.nodes and not world.map.available_next_node_ids:
        # Reward after the act's boss: the next act's map is not generated
        # yet. Its encounter pools and possible bosses are catalog facts; the
        # fight mix is taken from the act map just finished (same structure).
        next_bosses = repository.next_act_boss_ids(boss_ids)
        first_row = min(node.row for node in world.map.nodes)
        mix = _reachable_fight_share(
            world.map, [n.node_id for n in world.map.nodes if n.row == first_row]
        )
        if next_bosses and mix:
            boss_ids = next_bosses
            gaps = ("next_act_needs_from_pools", "next_act_fight_mix_from_finished_map")
    pools = repository.encounter_pool_expectations(boss_ids, ascension=world.ascension)
    fact_pools = _fact_pools(repository, boss_ids, params, int(world.ascension or 0)) if boss_ids else {}
    if not (pools or fact_pools) or not mix:
        # Without the act's encounter pools and the fights ahead there is no
        # factual need to compare against; nothing may be scored.
        return Needs({}, {}, 0.0, 0.0, 0.0, mix,
                     tuple(dict.fromkeys((*gaps, "situation_needs_unavailable"))))
    gaps = [*gaps, *(
        f"encounter_pool_missing:{k}" for k in mix if k not in pools and k not in fact_pools
    )]
    boss = build_boss_preparation(repository, boss_ids, ascension=world.ascension)
    gaps.extend(boss.data_gaps)

    if world.hp is None or not world.max_hp:
        # Unknown HP is scored as full HP (no extra weight on defense) and said so.
        gaps = [*gaps, "hp_unknown"]
    hp_ratio = world.hp / world.max_hp if world.hp is not None and world.max_hp else 1.0
    rows: Dict[str, List[tuple[float, float]]] = {}
    # The act's first normal fights come from its weak pool
    # (ActModel.NumberOfWeakEncounters); the rest from the regular pool.
    weak_total = float((params.get("weak_encounters_by_act_index") or {}).get(
        str(repository.act_index(boss_ids)), 0.0
    ))
    done = _monster_fights_done(world.map) if world.map and world.map.nodes and "next_act_needs_from_pools" not in gaps else 0.0
    weak_left = max(0.0, weak_total - done)
    parts: List[tuple[str, str, float]] = []
    for kind, count in mix.items():
        if kind == "MONSTER" and "MONSTER_WEAK" in fact_pools:
            weak = min(float(count), weak_left)
            parts.append(("MONSTER", "MONSTER_WEAK", weak))
            parts.append(("MONSTER", "MONSTER", float(count) - weak))
        else:
            parts.append((kind, kind, float(count)))
    vulnerable_bonus = float(params["vulnerable_damage_bonus"])
    for kind, group, count in parts:
        if count <= 0:
            continue
        stats = fact_pools.get(group)
        pool = pools.get(kind)
        if stats is not None:
            gaps.extend(f"encounter_pool_partial:{group}:{gap}" for gap in stats.get("gaps") or ())
            hp, enemies = stats["hp"], max(1.0, stats["enemies"])
            # Vulnerable on the player raises the attack damage taken.
            attack = stats["attack"] * (1.0 + vulnerable_bonus * stats["vulnerable"])
            hits, weak_share, frail_share, cards = stats["hits"], stats["weak"], stats["frail"], stats["cards"]
        elif pool is not None:
            gaps.append(f"encounter_facts_missing:{group}")
            hp = float(pool["average_hp"])
            enemies = max(1.0, float(pool["average_enemy_count"]))
            if kind == "BOSS" and boss.encounters:
                hp = sum(row.total_hp for row in boss.encounters) / len(boss.encounters)
                enemies = sum(row.member_count for row in boss.encounters) / len(boss.encounters)
            attack = float(pool["average_attack"])
            hits, weak_share, frail_share, cards = enemies, 0.0, 0.0, 0.0
        else:
            continue
        turns = float(params["fight_turns"][kind])
        weight = count * float(params["fight_importance"][kind])
        damage = hp / turns
        for key, value in (
            ("frontload", damage),
            ("aoe", damage * (enemies - 1.0) / enemies),
            ("defense", attack * float(params["block_share"])),
            ("scaling", damage * max(0.0, turns - float(params["scaling_onset_turns"])) / turns),
            ("turns", turns),
            ("hp", hp),
            ("extra", enemies - 1.0),
            ("attack", attack),
            ("hits", hits),
            ("weak", weak_share),
            ("frail", frail_share),
            ("cards", cards),
        ):
            rows.setdefault(key, []).append((value, weight))

    if not rows:
        return Needs({}, {}, 0.0, 0.0, 0.0, mix,
                     tuple(dict.fromkeys((*gaps, "situation_needs_unavailable"))))
    weights = {d: float(params["dimension_weights"][d]) for d in DIMENSIONS}
    for preference, dimensions in params["preference_dimensions"].items():
        for dimension in dimensions:
            weights[dimension] *= float(getattr(preferences, preference))
    # How much block a fight needs is set by the enemies; low HP changes how
    # much meeting that need matters, so it scales the weight, not the amount.
    weights["defense"] *= 1.0 + float(params["low_hp_defense_pressure"]) * (1.0 - hp_ratio)
    sequence = sequence or _sequence_from_mix(mix)
    return Needs(
        fight_sequence=tuple(sequence),
        values={
            "frontload": _weighted(rows["frontload"]),
            "aoe": _weighted(rows["aoe"]),
            "defense": _weighted(rows["defense"]),
            "scaling": _weighted(rows["scaling"]),
            "cycle": float(params["cycle_need"]),
            "energy": 1.0,
        },
        weights=weights,
        fight_turns=_weighted(rows["turns"]),
        extra_enemies=_weighted(rows["extra"]),
        incoming_attack=_weighted(rows["attack"]),
        fight_mix=mix,
        gaps=tuple(dict.fromkeys(gaps)),
        enemy_hits=_weighted(rows["hits"]),
        player_weak=_weighted(rows["weak"]),
        player_frail=_weighted(rows["frail"]),
        status_cards=_weighted(rows["cards"]),
        enemy_hp=_weighted(rows["hp"]),
    )


def fight_tempo(capabilities: Mapping[str, float], needs: Needs, params: Mapping) -> float:
    """Expected fight length relative to the nominal one.

    The damage need is enemy HP over the nominal turns, so need / damage per
    turn is how many nominal fights the deck takes to kill. Bounded by the
    ``fight_tempo`` parameters; 1.0 (no effect) when they are absent.
    """

    bounds = params.get("fight_tempo")
    if not bounds or needs.values["frontload"] <= 0:
        return 1.0
    low, high = float(bounds["min"]), float(bounds["max"])
    damage = capabilities["frontload"] + capabilities["scaling"]
    if damage <= 0:
        return high
    return max(low, min(high, needs.values["frontload"] / damage))


def fitness(capabilities: Mapping[str, float], needs: Needs, params: Mapping) -> Fitness:
    cap = float(params["satisfaction_cap"])
    slope = float(params["surplus_slope"])

    def gain(satisfaction: float) -> float:
        # Below the need: s - (1-s)^2/2, slope 2 at s=0 falling to 1 at s=1.
        # Above it: logarithmic, slope at most ``slope`` (0.25). Shifted by
        # 0.5 so an empty dimension is worth 0. Only defense can go below 0
        # (a slow fight takes more than the nominal fight's damage); the same
        # curve continues there, so more block or a shorter fight always counts.
        s = max(floor, min(cap, satisfaction))
        value = s - 0.5 * (1.0 - s) ** 2 if s <= 1.0 else 1.0 + slope * math.log(s)
        return value + 0.5

    satisfaction = {
        d: capabilities[d] / needs.values[d] if needs.values[d] > 0 else cap
        for d in DIMENSIONS
    }
    tempo = fight_tempo(capabilities, needs, params)
    floor = min(0.0, 1.0 - float((params.get("fight_tempo") or {}).get("max", 1.0)))
    if satisfaction["defense"] < 1.0:
        # Unblocked damage is taken every turn the fight lasts: a deck that
        # kills faster than the nominal fight length takes less of it, a
        # slower one more.
        satisfaction["defense"] = 1.0 - (1.0 - max(0.0, satisfaction["defense"])) * tempo
    norm = sum(needs.weights[d] for d in DIMENSIONS) * gain(cap)
    terms = {
        d: 100.0 * needs.weights[d]
        * gain(satisfaction[d] if d == "defense" else max(0.0, satisfaction[d])) / norm
        for d in DIMENSIONS
    }
    raw = max(0.0, sum(terms.values()))
    return Fitness(score=round(raw, 1), raw=raw, capabilities=capabilities, terms=terms, tempo=tempo)


def _contract_dimensions(before: Fitness, after: Fitness, known: bool) -> Dict[str, float]:
    """Fitness change versus skip, summed into the contract's dimensions."""

    result: Dict[str, float] = {}
    for dimension in DIMENSIONS:
        key = CONTRACT_DIMENSIONS[dimension]
        result[key] = result.get(key, 0.0) + after.terms[dimension] - before.terms[dimension]
    result = {key: round(value, 4) for key, value in result.items()}
    result["data_completeness"] = 1.0 if known else 0.0
    return result


def _dimension_factors(
    label: str, before: Fitness, after: Fitness, needs: Needs, deck_size: int,
) -> list[dict]:
    rows = []
    for dimension in DIMENSIONS:
        delta = after.terms[dimension] - before.terms[dimension]
        if abs(delta) < 0.05:
            continue
        if (dimension == "defense" and abs(after.tempo - before.tempo) >= 0.005
                and abs(after.capabilities[dimension] - before.capabilities[dimension]) < 0.05):
            # Same block, different fight length: the change is damage taken.
            rows.append({
                "code": "deck_fitness:defense",
                "delta": round(delta, 2),
                "message": f"拿{label}后战斗长度变化，挡不住的伤害随之增减："
                           f"预计战斗长度为标准的 {before.tempo:.2f} → {after.tempo:.2f} 倍",
            })
            continue
        detail = (
            f"每回合额外抽牌 {before.capabilities[dimension]:.2f} → {after.capabilities[dimension]:.2f}"
            f"（需求 {needs.values[dimension]:.1f}）"
            if dimension == "cycle"
            else f"每回合 {before.capabilities[dimension]:.1f} → "
                 f"{after.capabilities[dimension]:.1f}（需求 {needs.values[dimension]:.1f}）"
        )
        rows.append({
            "code": f"deck_fitness:{dimension}",
            "delta": round(delta, 2),
            "message": f"拿{label}后{DIMENSION_LABELS[dimension]}：{detail}",
        })
    return sorted(rows, key=lambda row: -abs(row["delta"]))


def _gap_notes(
    unknown_cards: Sequence[str],
    unmodeled_relics: Sequence[str],
    partial_cards: Sequence[str] = (),
    card_name=lambda c: c,
    relic_name=lambda r: r,
) -> list[dict]:
    """The detailed disclosures (kept in the advice for diagnosis)."""

    def names(ids, name_of):
        return "、".join(sorted({name_of(item) for item in ids}))

    notes = []
    if unknown_cards:
        notes.append({
            "code": "unmodeled_deck_cards",
            "delta": 0.0,
            "message": f"牌组中 {len(unknown_cards)} 张牌（{names(unknown_cards, card_name)}）"
                       "尚无经核对的机制事实，只计入稀释。",
        })
    if partial_cards:
        notes.append({
            "code": "partially_modeled_deck_cards",
            "delta": 0.0,
            "message": f"牌组中 {len(partial_cards)} 张牌（{names(partial_cards, card_name)}）"
                       "有部分效果尚未计入评分。",
        })
    if unmodeled_relics:
        notes.append({
            "code": "unmodeled_relics",
            "delta": 0.0,
            "message": f"遗物 {names(unmodeled_relics, relic_name)} 尚未计入评分。",
        })
    return notes


def _scope_note(unknown_cards: Sequence[str], unmodeled_relics: Sequence[str], partial_cards: Sequence[str]) -> str:
    """One short sentence for the drawer: how much of the deck and relics the
    score leaves out. Empty when everything is counted."""

    parts = []
    cards = len(unknown_cards) + len(partial_cards)
    if cards:
        parts.append(f"牌组中 {cards} 张牌")
    if unmodeled_relics:
        parts.append(f"{len(unmodeled_relics)} 件遗物")
    return "、".join(parts) + "未完整计入评分" if parts else ""


def _legacy_payload(
    rows: Sequence[tuple[DecisionCandidate, Optional[float]]],
    skip_score: Optional[float],
    best: Optional[str],
    status: str,
    extra: Mapping,
) -> dict:
    """The schema-bound legacy ``advice`` object. The Mod reads the generic
    recommendation; ``known`` and ``decision_status`` mark every number here
    that is not a real score or choice."""

    cards = [
        {
            "card_id": str(candidate.payload.get("card") or candidate.candidate_id),
            "option_index": index,
            "score": score if score is not None else 0.0,
            "known": score is not None,
        }
        for index, (candidate, score) in enumerate(rows)
    ] or [{"card_id": "none", "option_index": 0, "score": 0.0, "known": False}]
    return {
        "method": "deck_fitness_2",
        "decision_status": status,
        "recommendations": cards,
        "skip_candidate": {
            "choice": "skip",
            "score": skip_score if skip_score is not None else 0.0,
            "known": skip_score is not None,
        },
        "recommended_option_index": next(
            (i for i, (candidate, _) in enumerate(rows) if candidate.candidate_id == best),
            len(rows),
        ),
        **extra,
    }


BASIC_CARD = re.compile(r"^(STRIKE|DEFEND)_[A-Z]+$")


def _synergy_notes(card, label, deck, score_deck, params, name_of=lambda c: c) -> list[dict]:
    """Owned cards whose removal lowers this card's gain the most.

    Basic Strikes/Defends are left out: removing one only shifts the deck's
    mix, which is not a synergy worth naming.
    """

    settings = params.get("synergy_notes") or {}
    threshold = float(settings.get("min_gain", 0.0))
    if not threshold:
        return []
    base_gain = score_deck([*deck, card]) - score_deck(deck)
    seen, rows = set(), []
    for owned in deck:
        if owned is None or owned.card_id in seen or BASIC_CARD.match(owned.card_id):
            continue
        seen.add(owned.card_id)
        rest = list(deck)
        rest.remove(owned)
        boost = base_gain - (score_deck([*rest, card]) - score_deck(rest))
        if boost >= threshold:
            rows.append((boost, owned.card_id))
    rows.sort(reverse=True)
    if not rows:
        return []
    names = "、".join(name_of(card_id) for _, card_id in rows[: int(settings.get("max_cards", 2))])
    return [{
        "code": "deck_fitness:synergy",
        "delta": 0.0,
        "message": f"牌组里的 {names} 让{label}更有价值（说明，不另加分）",
    }]


def _pick_margin(params: Mapping, act: Optional[int]) -> float:
    """Points a card must beat skipping by; the last act listed covers later ones."""

    table = params.get("pick_margin_by_act") or {}
    if not table or act is None:
        return 0.0
    known = sorted(int(key) for key in table)
    chosen = max((key for key in known if key <= int(act)), default=known[0])
    return float(table[str(chosen)])


def _recommendation_confidence(
    shown: Mapping[str, float],
    ranked: Sequence[str],
    partly_scored: Sequence[str],
    unscored_cards: bool,
    capture_warnings: bool,
    params: Mapping,
) -> str:
    """"low" only when what is not scored could change the recommendation.

    ``shown`` is each scored option's displayed score (one decimal, skip's
    pick margin included) and ``ranked`` the options best first. Missing
    effects only ever add value, so a gap on the recommended option itself
    cannot overturn it; a gap on a rival can. The rule:

    * "low" when the state cannot be trusted or judged: no recommendation,
      a capture warning, or an eligible card that could not be scored at all
      (it could be worth anything);
    * "low" when the lead over the runner-up is below ``near_tie_points``
      (a near tie is not a firm call whatever the data);
    * "low" when any rival that is only partly scored (its own data gaps,
      ``partly_scored``) could catch up, i.e. its displayed score plus
      ``confidence_gap_allowance`` reaches the recommended option's;
    * "medium" otherwise. "high" is never given.

    Gaps in the owned deck, relics and fight needs are left out of the
    bound: they are part of every option, skip included, so they shift all
    scores alike (their interaction with one candidate is the known blind
    spot of this rule).
    """

    if not ranked or unscored_cards or capture_warnings:
        return "low"
    best = shown[ranked[0]]
    if len(ranked) > 1 and best - shown[ranked[1]] < float(params["near_tie_points"]):
        return "low"
    allowance = float(params["confidence_gap_allowance"])
    if any(
        shown[cid] + allowance >= best for cid in ranked[1:] if cid in partly_scored
    ):
        return "low"
    return "medium"


def _scenario_seed(decision_id: str) -> int:
    return int(hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:12], 16)


def _roll_rarity(rng: random.Random, odds: Mapping[str, float], rare_offset: float) -> str:
    """CardRarityOdds.RollWithoutChangingFutureOdds: rare, then uncommon, else common."""

    roll = rng.random()
    rare = float(odds.get("rare", 0.0)) + (rare_offset if odds.get("rare", 0.0) < 1.0 else 0.0)
    if roll < rare:
        return "rare"
    if roll < rare + float(odds.get("uncommon", 0.0)):
        return "uncommon"
    return "common"


@dataclass
class DeckContext:
    """A situation ready for scoring decks (see ``DeckFitnessPolicy.deck_context``)."""

    params: Mapping
    deck: List[Optional[CardProfile]]
    relics: List[RelicProfile]
    needs: Needs
    situation: Situation
    gaps: List[str]
    unknown_cards: List[str]
    partial_cards: List[str]
    unmodeled_relics: List[str]

    def score(
        self, cards: Sequence[Optional[CardProfile]], relics: Optional[Sequence[RelicProfile]] = None,
    ) -> tuple["Fitness", tuple[str, ...]]:
        """Fitness of ``cards`` (with ``relics``, by default the current ones)
        here, and the engine gaps met on the way."""

        capabilities, engine_gaps = deck_capabilities(
            list(cards), list(self.relics if relics is None else relics), self.situation, self.params,
        )
        return fitness(capabilities, self.needs, self.params), tuple(engine_gaps)


class DeckFitnessPolicy:
    decision_type = CARD_REWARD

    def __init__(
        self,
        repository: RelationalRepository,
        preferences: ScoringPreferences = DEFAULT_SCORING_PREFERENCES,
    ):
        self.repository = repository
        self.preferences = preferences
        self.route_policy = RoutePolicy(repository)
        self._pools: Dict[str, Dict[str, List[CardProfile]]] = {}
        self._names: Dict[tuple, str] = {}
        # Facts are fixed for one release database: resolve each card, relic
        # and orb once per policy instead of once per request.
        self._cards: Dict[tuple, tuple] = {}
        self._relics: Dict[str, tuple] = {}
        self._orbs: Optional[Dict[str, Mapping]] = None

    def _card(self, card_id: str, upgrades: int) -> tuple:
        key = (card_id, upgrades)
        if key not in self._cards:
            self._cards[key] = card_profile(self.repository, card_id, upgrades)
        return self._cards[key]

    def _name(self, entity_type: str, entity_id: str) -> str:
        # Cached like the profiles: the release data does not change.
        key = (entity_type, str(entity_id).upper())
        if key not in self._names:
            entity = self.repository.find_entity(entity_type, key[1])
            self._names[key] = str((entity or {}).get("name") or entity_id)
        return self._names[key]

    def _card_name(self, card_id: str) -> str:
        return self._name("cards", card_id)

    def _relic_name(self, relic_id: str) -> str:
        return self._name("relics", relic_id)

    def warm_up(self) -> None:
        """Build what the first decision would otherwise pay for: every
        character's reward pool and every encounter's walk (ascension 0)."""

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        if _parameter_gaps(params):
            return
        with self.repository.connect() as connection:
            characters = [
                str(row[0]).split(":", 1)[1] for row in connection.execute(
                    "SELECT entity_key FROM catalog_entities WHERE entity_type = 'characters'"
                )
            ]
            encounters = [
                str(row[0]).split(":", 1)[1] for row in connection.execute(
                    "SELECT entity_key FROM assembly_facts WHERE entity_key LIKE 'encounters:%'"
                )
            ]
        for character in characters:
            self._pool(character)
        turns = sorted({int(round(float(params["fight_turns"][kind]))) for kind in FIGHT_KINDS})
        for encounter_id in encounters:
            for count in turns:
                _encounter_row(self.repository, encounter_id, count, 0)
        self._orb_facts()

    def _relic(self, relic_id: str) -> tuple:
        if relic_id not in self._relics:
            self._relics[relic_id] = relic_profile(self.repository, relic_id)
        return self._relics[relic_id]

    def _orb_facts(self) -> Dict[str, Mapping]:
        if self._orbs is None:
            self._orbs = {
                orb: fact for orb in ORB_TYPES.values()
                if (fact := self.repository.find_assembly_fact("orbs", orb)) is not None
                and fact.get("status") == "converted"
            }
        return self._orbs

    def _pool(self, character: str) -> Dict[str, List[CardProfile]]:
        """Reward pool by rarity: the character's common/uncommon/rare cards
        that have a scorable fact, excluding multiplayer-only cards."""

        key = character.upper()
        if key not in self._pools:
            pool: Dict[str, List[CardProfile]] = {"common": [], "uncommon": [], "rare": []}
            for card_id, rarity in self.repository.card_reward_pool(key):
                fact = self.repository.find_assembly_fact("cards", card_id)
                if fact is None or fact.get("multiplayer_only"):
                    continue
                profile, _ = self._card(card_id, 0)
                if profile is not None:
                    pool[rarity].append(profile)
            self._pools[key] = pool
        return self._pools[key]

    def _lookahead(
        self,
        request: DecisionRequest,
        options: Dict[str, List[Optional[CardProfile]]],
        score_deck,
        needs: Needs,
        params: Mapping,
    ) -> tuple[Dict[str, float], int]:
        """Each option's fitness averaged over the act's remaining fights,
        with the rewards between them sampled and picked greedily."""

        settings = params.get("lookahead") or {}
        sequence = needs.fight_sequence
        rewards = min(int(settings.get("max_rewards", 0)), max(0, len(sequence) - 1))
        if not rewards or not options:
            return {}, 0
        pool = self._pool(str(request.world.character or ""))
        odds_table = params["card_reward_odds"]
        high = int(request.world.ascension or 0) >= int(odds_table.get("ascension_threshold", 99))
        odds_by_kind = odds_table["ascension" if high else "base"]
        offset = float(odds_table.get("rare_offset_initial", 0.0))
        importance = params["fight_importance"]
        rng = random.Random(_scenario_seed(request.decision_id))
        scenarios = []
        for _ in range(int(settings.get("scenarios", 1))):
            offers = []
            for index in range(rewards):
                odds = odds_by_kind[sequence[index]]
                chosen: List[CardProfile] = []
                for _ in range(64):
                    if len(chosen) == 3:
                        break
                    cards = pool[_roll_rarity(rng, odds, offset)]
                    if cards:
                        card = cards[rng.randrange(len(cards))]
                        if card not in chosen:
                            chosen.append(card)
                offers.append(chosen)
            scenarios.append(offers)
        weights = [float(importance[kind]) for kind in sequence]
        total_weight = sum(weights)
        values: Dict[str, float] = {}
        for option, start in options.items():
            totals = 0.0
            for offers in scenarios:
                deck = list(start)
                current = score_deck(deck)
                value = current * weights[0]
                for index, offer in enumerate(offers):
                    best_deck, best_value = deck, current
                    for card in offer:
                        trial = score_deck([*deck, card])
                        if trial > best_value:
                            best_deck, best_value = [*deck, card], trial
                    deck, current = best_deck, best_value
                    value += current * weights[index + 1]
                # Fights past the horizon keep the last deck.
                value += current * sum(weights[rewards + 1:])
                totals += value / total_weight
            values[option] = totals / len(scenarios)
        return values, rewards

    def deck_context(self, request: DecisionRequest) -> tuple[Optional["DeckContext"], tuple[str, ...]]:
        """Everything a deck is scored against in this situation: the deck
        and relics as profiles, the fight needs and the engine situation.
        ``(None, gaps)`` when the situation cannot be scored."""

        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        parameter_gaps = _parameter_gaps(params)
        if parameter_gaps:
            return None, parameter_gaps

        state = request.world.scoring_state()
        deck: List[Optional[CardProfile]] = []
        unknown_cards: List[str] = []
        partial_cards: List[str] = []
        gaps: List[str] = []
        for entry in state.get("deck") or ():
            count = int(entry.get("count") or 1)
            profile, profile_gaps = self._card(
                str(entry["card"]).upper(), int(entry.get("upgrades") or 0)
            )
            if entry.get("affliction"):
                gaps.append(f"card_instance_modifier_unmodeled:{entry['card']}")
            if entry.get("enchantment") and profile is not None:
                profile, enchant_gaps = enchant(
                    profile,
                    self.repository.find_assembly_fact("enchantments", str(entry["enchantment"]).upper()),
                    float(entry.get("enchantment_amount") or 1.0),
                )
                profile_gaps = (*profile_gaps, *enchant_gaps)
            gaps.extend(profile_gaps)
            if profile is None:
                unknown_cards.extend([str(entry["card"])] * count)
            elif profile_gaps:
                partial_cards.extend([str(entry["card"])] * count)
            deck.extend([profile] * count)
        relics: List[RelicProfile] = []
        unmodeled_relics = []
        for relic in state.get("relics") or ():
            profile, relic_gaps = self._relic(str(relic).upper())
            gaps.extend(relic_gaps)
            if profile is not None:
                relics.append(profile)
            if profile is None or relic_gaps:
                unmodeled_relics.append(str(relic))

        needs = compute_needs(
            self.repository, self.route_policy, request, params, self.preferences
        )
        gaps.extend(needs.gaps)
        if not needs.values:
            return None, tuple(gaps)
        situation = Situation(
            fight_turns=needs.fight_turns,
            extra_enemies=needs.extra_enemies,
            incoming_attack=needs.incoming_attack,
            fight_mix=needs.fight_mix,
            hp_ratio=(
                request.world.hp / request.world.max_hp
                if request.world.hp is not None and request.world.max_hp else 1.0
            ),
            potion_count=(
                len(state["potions"]) if isinstance(state.get("potions"), list) else None
            ),
            orb_slots=float((params.get("base_orb_slots") or {}).get(
                str(request.world.character or "").upper(), 0.0
            )),
            orbs=self._orb_facts(),
            enemy_hits=needs.enemy_hits,
            player_weak=needs.player_weak,
            player_frail=needs.player_frail,
            status_cards=needs.status_cards,
            enemy_hp=needs.enemy_hp,
            sovereign_blade=self._card("SOVEREIGN_BLADE", 0)[0],
        )

        return DeckContext(
            params=params, deck=deck, relics=relics, needs=needs, situation=situation, gaps=gaps,
            unknown_cards=unknown_cards, partial_cards=partial_cards, unmodeled_relics=unmodeled_relics,
        ), ()

    def recommend(self, request: DecisionRequest) -> Recommendation:
        context, context_gaps = self.deck_context(request)
        if context is None:
            return self._uncertain(request, context_gaps)
        params, deck, relics, needs, situation = (
            context.params, context.deck, context.relics, context.needs, context.situation,
        )
        gaps = context.gaps
        unknown_cards, partial_cards, unmodeled_relics = (
            context.unknown_cards, context.partial_cards, context.unmodeled_relics,
        )
        state = request.world.scoring_state()

        def evaluate(cards: List[Optional[CardProfile]]) -> Fitness:
            capabilities, engine_gaps = deck_capabilities(cards, relics, situation, params)
            gaps.extend(engine_gaps)
            return fitness(capabilities, needs, params)

        engine_gap_count = len(gaps)
        baseline = evaluate(deck)
        baseline_engine_gaps = set(gaps[engine_gap_count:])
        # Options scored only in part: their own fact gaps or an engine gap
        # that adding them brings (the confidence bound counts these).
        partly_scored: List[str] = []
        notes = _gap_notes(
            unknown_cards, unmodeled_relics, partial_cards, self._card_name, self._relic_name,
        )
        # The drawer shows three reasons a row: the whole disclosure goes in
        # one line (the code is one the Mod always shows first).
        scope = _scope_note(unknown_cards, unmodeled_relics, partial_cards)
        scope_notes = ({
            "code": "estimate_scope",
            "delta": 0.0,
            "message": scope + "，分数只反映已计入的部分。",
        },) if scope else ()
        # Engine gaps (e.g. an unsupported power) also make the data incomplete.
        complete = (
            not unknown_cards and not partial_cards and not unmodeled_relics
            and len(gaps) == engine_gap_count
        )

        assessments: List[CandidateAssessment] = []
        legacy_rows: List[tuple[DecisionCandidate, Optional[float]]] = []
        # Rank on unrounded fitness; the displayed score keeps one decimal.
        raw_by_id: Dict[str, float] = {}
        added_by_id: Dict[str, CardProfile] = {}
        for candidate in request.candidates:
            if candidate.candidate_id == "skip":
                if candidate.eligible:
                    raw_by_id["skip"] = baseline.raw
                assessments.append(CandidateAssessment(
                    candidate_id="skip",
                    label="跳过",
                    display_index=candidate.display_index,
                    eligible=candidate.eligible,
                    score=baseline.score if candidate.eligible else None,
                    rank=None,
                    factors=(*scope_notes, *notes),
                    dimensions=_contract_dimensions(baseline, baseline, complete),
                ))
                continue
            card_id = str(candidate.payload.get("card") or "").upper()
            profile, profile_gaps = self._card(card_id, int(candidate.payload.get("upgrades") or 0))
            if profile is None or not candidate.eligible:
                assessments.append(CandidateAssessment(
                    candidate_id=candidate.candidate_id,
                    label=candidate.label,
                    display_index=candidate.display_index,
                    eligible=candidate.eligible,
                    score=None,
                    rank=None,
                    factors=({
                        "code": "insufficient_data",
                        "delta": 0.0,
                        "message": f"数据不足：{self._card_name(card_id)} 尚无经程序集核对的机制事实，不给分。",
                    },) if profile is None else (),
                    dimensions={"data_completeness": 0.0},
                    data_gaps=profile_gaps,
                ))
                legacy_rows.append((candidate, None))
                continue
            profile = adjust_added_card(profile, relics, self.repository)
            before_gaps = len(gaps)
            after = evaluate([*deck, profile])
            if profile_gaps or set(gaps[before_gaps:]) - baseline_engine_gaps:
                partly_scored.append(candidate.candidate_id)
            raw_by_id[candidate.candidate_id] = after.raw
            added_by_id[candidate.candidate_id] = profile
            name = self._card_name(card_id)
            # The reasons stay readable: the exact gap codes are in data_gaps.
            own_notes = ({
                "code": "unmodeled_mechanic",
                "delta": 0.0,
                "message": f"{name}的部分效果尚未计入评分" + (f"；{scope}。" if scope else "。"),
            },) if profile_gaps else scope_notes
            assessments.append(CandidateAssessment(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                display_index=candidate.display_index,
                eligible=True,
                score=after.score,
                rank=None,
                # Unmodeled notes first, then dimensions by absolute effect.
                factors=(*own_notes, *notes, *_dimension_factors(
                    name, baseline, after, needs, len(deck)
                )),
                dimensions=_contract_dimensions(baseline, after, complete and not profile_gaps),
                data_gaps=profile_gaps,
            ))
            legacy_rows.append((candidate, after.score))

        # The engine's result depends only on which cards are in the deck,
        # not their order, and the lookahead meets the same deck many times.
        scored: Dict[tuple, float] = {}

        def score_deck(cards: List[Optional[CardProfile]]) -> float:
            key = tuple(sorted(map(id, cards)))
            if key not in scored:
                capabilities, _ = deck_capabilities(cards, relics, situation, params)
                scored[key] = fitness(capabilities, needs, params).raw
            return scored[key]

        # Explanation only (spec 2.8 addendum): which owned cards make each
        # candidate stronger. The engine already scores these interactions,
        # so this never changes a score.
        assessments = [
            replace(a, factors=(*a.factors, *_synergy_notes(
                added_by_id[a.candidate_id], self._card_name(added_by_id[a.candidate_id].card_id),
                deck, score_deck, params, self._card_name,
            ))) if a.candidate_id in added_by_id else a
            for a in assessments
        ]
        options = {cid: list(deck) for cid in raw_by_id if cid == "skip"}
        options.update({cid: [*deck, card] for cid, card in added_by_id.items()})
        ahead, rewards = self._lookahead(request, options, score_deck, needs, params)
        if ahead:
            assessments = [
                replace(
                    a,
                    score=round(ahead[a.candidate_id], 1),
                    factors=(*a.factors, {
                        "code": "deck_fitness:lookahead",
                        # What looking ahead changes for this candidate beyond what it
                        # changes for skipping (0 for skip itself: nothing to explain).
                        "delta": round(
                            (ahead[a.candidate_id] - raw_by_id[a.candidate_id])
                            - ((ahead["skip"] - raw_by_id["skip"]) if "skip" in ahead else 0.0), 2
                        ),
                        "message": f"计入后续 {rewards} 次选牌（按卡池稀有度抽样）后的牌组适合度",
                    }),
                ) if a.candidate_id in ahead else a
                for a in assessments
            ]
            legacy_rows = [
                (c, round(ahead[c.candidate_id], 1) if c.candidate_id in ahead else s)
                for c, s in legacy_rows
            ]
            raw_by_id = dict(ahead)
        # A card is worth taking only when it beats skipping by a clear
        # margin, larger in later acts (spec 2.8 addendum 2026-10-13). The
        # margin is added to skip's score and listed as its own factor.
        margin = _pick_margin(params, request.world.act)
        if margin > 0 and "skip" in raw_by_id and len(raw_by_id) > 1:
            raw_by_id["skip"] += margin
            assessments = [
                replace(
                    a,
                    score=round(raw_by_id["skip"], 1),
                    factors=(*a.factors, {
                        "code": "deck_fitness:pick_margin",
                        "delta": round(margin, 2),
                        "message": f"第 {request.world.act} 幕：候选要比跳过高出 {margin:g} 分以上才值得拿"
                                   "（门槛按玩家历史对局中的取舍设定）",
                    }),
                ) if a.candidate_id == "skip" else a
                for a in assessments
            ]
        # Order by the score as shown (one decimal): a card level with skip
        # on screen has not beaten it, so skip comes first; the unrounded
        # value only separates cards that show the same score.
        ranked = sorted(raw_by_id, key=lambda cid: (
            -round(raw_by_id[cid], 1),
            cid != "skip",
            -raw_by_id[cid],
            next(a.display_index for a in assessments if a.candidate_id == cid),
        ))
        rank_by_id = {cid: index for index, cid in enumerate(ranked, start=1)}
        assessments = [replace(a, rank=rank_by_id.get(a.candidate_id)) for a in assessments]
        # With no scored card, "skip wins" would be an artefact of missing
        # facts, not a judgement: the decision stays uncertain.
        any_card_scored = any(cid != "skip" for cid in raw_by_id)
        best = ranked[0] if ranked and any_card_scored else None
        status = "uncertain" if best is None else ("skip" if best == "skip" else "recommend")
        shown = {cid: round(value, 1) for cid, value in raw_by_id.items()}
        # Explanation only: when the top two are within a fraction of a point
        # both rows say so (first), so the highlight is not read as firm.
        if best is not None and len(ranked) > 1:
            runner = ranked[1]
            gap = round(shown[best] - shown[runner], 1)
            if gap < float(params["near_tie_points"]):
                def option_name(cid: str) -> str:
                    return "跳过" if cid == "skip" else self._card_name(added_by_id[cid].card_id)
                def tie_note(other: str) -> dict:
                    return {
                        "code": "deck_fitness:near_tie",
                        "delta": 0.0,
                        "message": (
                            f"与{option_name(other)}" + (f"只差 {gap:.1f} 分" if gap > 0 else "同分")
                            + "，两者差不多，按自己的打法选择即可。"
                        ),
                    }
                notes_by_id = {best: tie_note(runner), runner: tie_note(best)}
                assessments = [
                    replace(a, factors=(notes_by_id[a.candidate_id], *a.factors))
                    if a.candidate_id in notes_by_id else a
                    for a in assessments
                ]
        confidence = "low" if best is None else _recommendation_confidence(
            shown, ranked, partly_scored,
            unscored_cards=any(
                a.eligible and a.score is None and a.candidate_id != "skip" for a in assessments
            ),
            capture_warnings=bool(request.world.capture_warnings),
            params=params,
        )
        data_gaps = tuple(dict.fromkeys((
            *gaps,
            *(gap for row in assessments for gap in row.data_gaps),
            *(f"capture_warning:{w}" for w in request.world.capture_warnings),
        )))
        skip = next((a for a in assessments if a.candidate_id == "skip"), None)
        payload = _legacy_payload(
            legacy_rows,
            skip.score if skip else None,
            best,
            status,
            {
                "act": request.world.act,
                "floor": request.world.floor,
                "needs": dict(needs.values),
                "weights": dict(needs.weights),
                "fight_mix": dict(needs.fight_mix),
                "baseline_capabilities": dict(baseline.capabilities),
            },
        )
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload=MappingProxyType(payload),
            world_sequence=request.world.sequence,
            policy_version=DECK_FITNESS_POLICY_VERSION,
            candidates=tuple(assessments),
            recommended_candidate_id=best,
            status=status,
            confidence=confidence,
            data_gaps=data_gaps,
        )

    @staticmethod
    def _uncertain(request: DecisionRequest, gaps: Sequence[str]) -> Recommendation:
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload=MappingProxyType(_legacy_payload(
                [(c, None) for c in request.candidates if c.candidate_id != "skip"],
                None, None, "uncertain", {},
            )),
            world_sequence=request.world.sequence,
            policy_version=DECK_FITNESS_POLICY_VERSION,
            candidates=tuple(
                CandidateAssessment(
                    candidate_id=c.candidate_id, label=c.label, display_index=c.display_index,
                    eligible=c.eligible, score=None, rank=None,
                )
                for c in request.candidates
            ),
            recommended_candidate_id=None,
            status="uncertain",
            confidence="low",
            data_gaps=tuple(dict.fromkeys(gaps)),
        )

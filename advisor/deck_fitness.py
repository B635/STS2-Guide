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

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence

from advisor.boss_context import build_boss_preparation
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
from storage.relational import RelationalRepository

# Scoring kernel v2; the ``_1`` suffix versions the deck-fitness parameters.
DECK_FITNESS_POLICY_VERSION = "card_reward:deck_fitness_1"
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
# Powers a card may apply, with the side that benefits.
POWER_EFFECTS = {
    ("Vulnerable", "enemy"): "vulnerable_turns",
    ("Weak", "enemy"): "weak_turns",
    ("Strength", "self"): "strength",
}
REQUIRED_PARAMETERS = (
    "hand_size", "base_energy", "fight_turns", "fight_importance", "block_share",
    "low_hp_defense_pressure", "scaling_onset_turns", "cycle_need",
    "vulnerable_damage_bonus", "weak_damage_reduction", "dimension_weights",
    "preference_dimensions", "surplus_slope", "satisfaction_cap",
)


@dataclass(frozen=True)
class CardProfile:
    """Per-play effects of one card instance, from a valid assembly fact."""

    card_id: str
    cost: Optional[float]
    single_damage: float = 0.0
    all_damage: float = 0.0
    attack_hits: float = 0.0
    block: float = 0.0
    draw: float = 0.0
    energy: float = 0.0
    vulnerable_turns: float = 0.0
    weak_turns: float = 0.0
    strength: float = 0.0
    hp_loss: float = 0.0
    exhausts_self: bool = False


@dataclass(frozen=True)
class Needs:
    values: Mapping[str, float]
    weights: Mapping[str, float]
    fight_turns: float
    extra_enemies: float
    incoming_attack: float
    fight_mix: Mapping[str, float]
    gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fitness:
    score: float
    raw: float
    capabilities: Mapping[str, float]
    terms: Mapping[str, float]


def _parameter_gaps(params: object) -> tuple[str, ...]:
    if not isinstance(params, Mapping):
        return ("deck_fitness_parameters_missing",)
    missing = [key for key in REQUIRED_PARAMETERS if key not in params]
    if missing:
        return tuple(f"deck_fitness_parameter_missing:{key}" for key in missing)
    if any(float(params["fight_turns"].get(kind) or 0) <= 0 for kind in FIGHT_KINDS):
        return ("deck_fitness_parameter_invalid:fight_turns",)
    if sum(float(value) for value in params["dimension_weights"].values()) <= 0:
        return ("deck_fitness_parameter_invalid:dimension_weights",)
    return ()


def _amount(effect: Mapping, upgraded: bool) -> float:
    value = float(effect.get("amount") or 0.0)
    if upgraded:
        value += float(effect.get("upgrade") or 0.0)
    return value


def card_profile(
    repository: RelationalRepository,
    card_id: str,
    upgrades: int,
) -> tuple[Optional[CardProfile], tuple[str, ...]]:
    """Resolve one card instance to its per-play profile, or report why not."""

    fact = repository.find_assembly_fact("cards", card_id)
    if fact is None or fact.get("status") != "converted":
        return None, (f"card_fact_missing:{card_id}",)
    if upgrades > 1:
        return None, (f"card_upgrade_count_unmodeled:{card_id}",)
    upgraded = upgrades == 1
    cost = (fact.get("cost") or {}).get("upgraded" if upgraded else "base")
    values: Dict[str, float] = {}
    exhausts_self = False
    for effect in fact.get("effects") or ():
        kind = effect.get("kind")
        if kind == "damage":
            hits = float(effect.get("hits") or 1)
            key = "all_damage" if effect.get("target") == "all" else "single_damage"
            values[key] = values.get(key, 0.0) + _amount(effect, upgraded) * hits
            values["attack_hits"] = values.get("attack_hits", 0.0) + hits
        elif kind in {"block", "draw", "energy", "hp_loss"}:
            values[kind] = values.get(kind, 0.0) + _amount(effect, upgraded)
        elif kind == "apply_power":
            key = POWER_EFFECTS.get((effect.get("power"), effect.get("target")))
            if key is None:
                return None, (
                    f"card_power_unmodeled:{card_id}:{effect.get('power')}:{effect.get('target')}",
                )
            values[key] = values.get(key, 0.0) + _amount(effect, upgraded)
        elif kind == "exhaust_self":
            exhausts_self = True
        else:
            return None, (f"card_effect_unmodeled:{card_id}:{kind}",)
    return CardProfile(
        card_id=card_id,
        cost=None if cost is None else float(cost),
        exhausts_self=exhausts_self,
        **values,
    ), ()


def _weighted(rows: Sequence[tuple[float, float]]) -> float:
    total = sum(weight for _, weight in rows)
    return sum(value * weight for value, weight in rows) / total if total else 0.0


def _reachable_fight_share(world_map: WorldMap) -> Dict[str, float]:
    """Expected count of each fight kind over all paths from the next nodes
    to a boss, each path equally likely (the no-recommendation fallback)."""

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

    starts = [n for n in world_map.available_next_node_ids if paths_to_boss(n) > 0]
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


def _fight_mix(
    route_policy: RoutePolicy,
    request: DecisionRequest,
) -> tuple[Dict[str, float], tuple[str, ...]]:
    """Fight kinds ahead: the recommended route, else reachable-path share."""

    world_map = request.world.map
    if world_map is None or not world_map.nodes:
        return {}, ("route_context_missing",)
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
        for node_id in path:
            kind = str(by_id[node_id].kind).upper() if node_id in by_id else ""
            if kind in FIGHT_KINDS:
                mix[kind] = mix.get(kind, 0.0) + 1.0
        if mix:
            return mix, ()
    share = _reachable_fight_share(world_map)
    return share, ("route_path_unavailable",) if share else ("route_path_unavailable", "boss_unreachable")


def compute_needs(
    repository: RelationalRepository,
    route_policy: RoutePolicy,
    request: DecisionRequest,
    params: Mapping,
    preferences: ScoringPreferences,
) -> Needs:
    world = request.world
    mix, gaps = _fight_mix(route_policy, request)
    boss_ids = world.map.boss_encounter_ids if world.map else ()
    pools = repository.encounter_pool_expectations(boss_ids, ascension=world.ascension)
    if not pools or not mix:
        # Without the act's encounter pools and the fights ahead there is no
        # factual need to compare against; nothing may be scored.
        return Needs({}, {}, 0.0, 0.0, 0.0, mix,
                     tuple(dict.fromkeys((*gaps, "situation_needs_unavailable"))))
    gaps = [*gaps, *(f"encounter_pool_missing:{k}" for k in mix if k not in pools)]
    boss = build_boss_preparation(repository, boss_ids, ascension=world.ascension)
    gaps.extend(boss.data_gaps)

    hp_ratio = world.hp / world.max_hp if world.hp is not None and world.max_hp else 1.0
    rows: Dict[str, List[tuple[float, float]]] = {}
    for kind, count in mix.items():
        pool = pools.get(kind)
        if pool is None:
            continue
        hp = float(pool["average_hp"])
        enemies = max(1.0, float(pool["average_enemy_count"]))
        if kind == "BOSS" and boss.encounters:
            # The act's boss is a known fact: use its exact HP and size. Attack
            # keeps the pool mean so every fight uses the same statistic.
            hp = sum(row.total_hp for row in boss.encounters) / len(boss.encounters)
            enemies = sum(row.member_count for row in boss.encounters) / len(boss.encounters)
        turns = float(params["fight_turns"][kind])
        weight = count * float(params["fight_importance"][kind])
        damage = hp / turns
        for key, value in (
            ("frontload", damage),
            ("aoe", damage * (enemies - 1.0) / enemies),
            ("defense", float(pool["average_attack"]) * float(params["block_share"])),
            ("scaling", damage * max(0.0, turns - float(params["scaling_onset_turns"])) / turns),
            ("turns", turns),
            ("extra", enemies - 1.0),
            ("attack", float(pool["average_attack"])),
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
    return Needs(
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
    )


def deck_capabilities(
    cards: Sequence[Optional[CardProfile]],
    needs: Needs,
    params: Mapping,
) -> Dict[str, float]:
    """Expected per-turn capabilities; unknown cards only take a deck slot."""

    size = max(1, len(cards))
    known = [card for card in cards if card is not None]
    hand = float(params["hand_size"])
    base = min(1.0, hand / size)
    drawn = min(float(size), hand + sum(base * card.draw for card in known))
    per_turn = min(1.0, drawn / size)

    def drawn_rate(card: CardProfile) -> float:
        # A self-exhausting card is played at most once per fight.
        return min(per_turn, 1.0 / needs.fight_turns) if card.exhausts_self else per_turn

    demand = sum(drawn_rate(c) * c.cost for c in known if c.cost is not None)
    supply = float(params["base_energy"]) + sum(drawn_rate(c) * c.energy for c in known)
    # Cards drawn beyond the energy available are not played this turn.
    playable = min(1.0, supply / demand) if demand > 0 else 1.0

    def total(attribute: str) -> float:
        return sum(drawn_rate(c) * playable * getattr(c, attribute) for c in known)

    vulnerable = min(1.0, total("vulnerable_turns"))
    amplify = 1.0 + float(params["vulnerable_damage_bonus"]) * vulnerable
    average_strength = total("strength") * max(0.0, needs.fight_turns - 1.0) / 2.0
    # Vulnerable sits on one enemy: it amplifies damage to that target only.
    return {
        "frontload": (total("single_damage") + total("all_damage")) * amplify,
        "aoe": total("all_damage") * needs.extra_enemies,
        "defense": (
            total("block")
            + float(params["weak_damage_reduction"]) * needs.incoming_attack
            * min(1.0, total("weak_turns"))
            - total("hp_loss")
        ),
        "scaling": average_strength * total("attack_hits") * amplify,
        "cycle": drawn / size,
        "energy": supply / demand if demand > 0 else float(params["satisfaction_cap"]),
    }


def fitness(capabilities: Mapping[str, float], needs: Needs, params: Mapping) -> Fitness:
    cap = float(params["satisfaction_cap"])
    slope = float(params["surplus_slope"])

    def gain(satisfaction: float) -> float:
        # Below the need: s - (1-s)^2/2, slope 2 at s=0 falling to 1 at s=1.
        # Above it: logarithmic, slope at most ``slope`` (0.25). Shifted by
        # 0.5 so an empty dimension is worth 0.
        s = max(0.0, min(cap, satisfaction))
        value = s - 0.5 * (1.0 - s) ** 2 if s <= 1.0 else 1.0 + slope * math.log(s)
        return value + 0.5

    norm = sum(needs.weights[d] for d in DIMENSIONS) * gain(cap)
    terms = {
        d: 100.0 * needs.weights[d]
        * gain(capabilities[d] / needs.values[d] if needs.values[d] > 0 else cap) / norm
        for d in DIMENSIONS
    }
    raw = sum(terms.values())
    return Fitness(score=round(raw, 1), raw=raw, capabilities=capabilities, terms=terms)


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
        detail = (
            f"牌组 {deck_size} → {deck_size + 1} 张，每回合看到的比例 "
            f"{before.capabilities[dimension]:.2f} → {after.capabilities[dimension]:.2f}"
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


def _gap_notes(unknown_cards: Sequence[str], unmodeled_relics: Sequence[str]) -> list[dict]:
    notes = []
    if unknown_cards:
        notes.append({
            "code": "unmodeled_deck_cards",
            "delta": 0.0,
            "message": f"牌组中 {len(unknown_cards)} 张牌（{'、'.join(sorted(set(unknown_cards)))}）"
                       "尚无经核对的机制事实，只计入稀释。",
        })
    if unmodeled_relics:
        notes.append({
            "code": "unmodeled_relics",
            "delta": 0.0,
            "message": f"遗物 {'、'.join(unmodeled_relics)} 尚未计入评分。",
        })
    return notes


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
        "method": "deck_fitness_1",
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

    def recommend(self, request: DecisionRequest) -> Recommendation:
        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        parameter_gaps = _parameter_gaps(params)
        if parameter_gaps:
            return self._uncertain(request, parameter_gaps)

        state = request.world.scoring_state()
        deck: List[Optional[CardProfile]] = []
        unknown_cards: List[str] = []
        gaps: List[str] = []
        for entry in state.get("deck") or ():
            count = int(entry.get("count") or 1)
            if entry.get("enchantment") or entry.get("affliction"):
                gaps.append(f"card_instance_modifier_unmodeled:{entry['card']}")
            profile, profile_gaps = card_profile(
                self.repository, str(entry["card"]).upper(), int(entry.get("upgrades") or 0)
            )
            gaps.extend(profile_gaps)
            if profile is None:
                unknown_cards.extend([str(entry["card"])] * count)
            deck.extend([profile] * count)
        unmodeled_relics = []
        for relic in state.get("relics") or ():
            fact = self.repository.find_assembly_fact("relics", str(relic).upper())
            if fact is None or fact.get("category") != "irrelevant":
                unmodeled_relics.append(str(relic))
                gaps.append(
                    f"relic_fact_missing:{relic}" if fact is None
                    else f"relic_effect_unmodeled:{relic}"
                )

        needs = compute_needs(
            self.repository, self.route_policy, request, params, self.preferences
        )
        gaps.extend(needs.gaps)
        if not needs.values:
            return self._uncertain(request, gaps)
        baseline = fitness(deck_capabilities(deck, needs, params), needs, params)
        notes = _gap_notes(unknown_cards, unmodeled_relics)
        complete = not unknown_cards and not unmodeled_relics

        assessments: List[CandidateAssessment] = []
        legacy_rows: List[tuple[DecisionCandidate, Optional[float]]] = []
        # Rank on unrounded fitness; the displayed score keeps one decimal.
        raw_by_id: Dict[str, float] = {}
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
                    factors=tuple(notes),
                    dimensions=_contract_dimensions(baseline, baseline, complete),
                ))
                continue
            card_id = str(candidate.payload.get("card") or "").upper()
            profile, profile_gaps = card_profile(
                self.repository, card_id, int(candidate.payload.get("upgrades") or 0)
            )
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
                        "message": f"数据不足：{candidate.label} 尚无经程序集核对的机制事实，不给分。",
                    },) if profile is None else (),
                    dimensions={"data_completeness": 0.0},
                    data_gaps=profile_gaps,
                ))
                legacy_rows.append((candidate, None))
                continue
            after = fitness(deck_capabilities([*deck, profile], needs, params), needs, params)
            raw_by_id[candidate.candidate_id] = after.raw
            assessments.append(CandidateAssessment(
                candidate_id=candidate.candidate_id,
                label=candidate.label,
                display_index=candidate.display_index,
                eligible=True,
                score=after.score,
                rank=None,
                # Unmodeled notes first, then dimensions by absolute effect.
                factors=(*notes, *_dimension_factors(
                    candidate.label, baseline, after, needs, len(deck)
                )),
                dimensions=_contract_dimensions(baseline, after, complete),
            ))
            legacy_rows.append((candidate, after.score))

        ranked = sorted(raw_by_id, key=lambda cid: (
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
            confidence="low" if data_gaps else "medium",
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

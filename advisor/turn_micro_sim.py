"""Common-Random-Numbers (CRN) discrete turn micro-simulator for card rewards.

Stage-2 refinement complementing the mean-field analytical engine
(``advisor.deck_mechanics``). Captures discrete effects that mean-field
averages smooth away:

- Discrete 5-card hand draw, reshuffle cycling, and chain draw mid-turn;
- Integer energy knapsack play order (2-3 cost hand clumping, 0-cost/draw
  enablers first, dynamic ``cost_rule``, ``X``-cost, unplayable Status/Curse
  hand clogging);
- ``Retain`` cross-turn holding waiting for spike incoming-attack turns or
  sufficient energy;
- Per-turn unblocked HP loss ``max(0, incoming_t - block_t)`` and worst-25%
  tail risk ``CVaR_0.25(Delta HP)``.

Pure Python standard library (Python 3.9+ compatible) for sub-25ms Stage-2
execution on close top candidates.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from advisor.deck_mechanics import (
    MELODY_TAGS,
    CardProfile,
    RelicProfile,
    Situation,
    _has_zero_x_effect,
)

DEFAULT_CRN_TRAJECTORIES = 24
DEFAULT_TAIL_ALPHA = 0.25
MAX_HAND_SIZE = 10
MAX_SIM_CYCLES = 8


class _SimCard(NamedTuple):
    card_id: str
    base_cost: Optional[float]
    x_cost: bool
    zero_x_effect: bool
    card_type: str
    exhausts: bool
    ethereal: bool
    retain: bool
    innate: bool
    returns: Optional[str]
    cost_rule: Optional[tuple]
    melody_tags: Tuple[str, ...]
    is_osty_attack: bool
    damage_single: float
    hits: float
    hits_x: bool
    damage_all: float
    damage_per_x: bool
    is_arrow: bool
    block: float
    block_per_x: bool
    block_instances: float
    draw: int
    energy_gain: float
    double_energy: bool
    energy_next_turn: float
    strength_gain: float
    dexterity_gain: float
    vigor_gain: float
    aim_gain: float
    stretch_gain: float
    vulnerable: float
    weak: float
    poison: float
    doom: float
    hp_loss: float
    heal: float
    orbs_lightning: float
    orbs_frost: float
    orbs_plasma: float
    orb_evokes: int
    generates_status: int
    generates_hand_attacks: int
    hand_attack_damage: float
    star_cost: float
    star_gain: float


class _CardInst:
    """Mutable per-combat instance of a card (tracks per-draw/per-play cost rules)."""

    __slots__ = ("card", "slot_id", "times_drawn", "times_played")

    def __init__(self, card: _SimCard, slot_id: int) -> None:
        self.card = card
        self.slot_id = slot_id
        self.times_drawn = 0
        self.times_played = 0


class _RelicSpec(NamedTuple):
    max_energy_bonus: int
    hand_draw_bonus: int
    turn_energy: Mapping[int, float]
    every_turn_energy: Tuple[Tuple[int, float], ...]
    start_strength: float
    start_dexterity: float
    start_block: float
    start_frost: int
    start_lightning: int
    no_block_turn_end_block: float
    keep_block_cap: Optional[float]
    shuriken_str: float
    kunai_dex: float
    fan_block: float
    play_limit: Optional[int]


@dataclass(frozen=True)
class CRNScenarioPack:
    """Shared Common-Random-Numbers scenarios reused across all candidates."""

    seed: int
    trajectories: int
    max_turns: int
    base_deck_size: int
    enemy_hps: Tuple[float, ...]
    # incoming_attacks[k][t]: incoming damage aimed at player on turn t (0-indexed)
    incoming_attacks: Tuple[Tuple[float, ...], ...]
    # base_keys[k][cycle][slot]: uniform sort key for base deck card slot
    base_keys: Tuple[Tuple[Tuple[float, ...], ...], ...]
    # cand_keys[k][cycle]: uniform sort key for any single added candidate card
    cand_keys: Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class MicroSimResult:
    """Aggregated CRN discrete turn micro-simulation metrics for one deck."""

    trajectories: int
    mean_hp_loss: float
    cvar25_hp_loss: float
    p75_hp_loss: float
    mean_turns: float
    clump_rate: float
    retain_saves: float
    losses: Tuple[float, ...]


_SIM_CARD_CACHE: Dict[int, _SimCard] = {}
_UNKNOWN_SIM_CARD = _SimCard(
    card_id="UNKNOWN", base_cost=None, x_cost=False, zero_x_effect=False,
    card_type="Status", exhausts=False, ethereal=False, retain=False,
    innate=False, returns=None, cost_rule=None, melody_tags=(),
    is_osty_attack=False, damage_single=0.0, hits=0.0, hits_x=False,
    damage_all=0.0, damage_per_x=False, is_arrow=False, block=0.0,
    block_per_x=False, block_instances=0.0, draw=0, energy_gain=0.0,
    double_energy=False, energy_next_turn=0.0, strength_gain=0.0,
    dexterity_gain=0.0, vigor_gain=0.0, aim_gain=0.0, stretch_gain=0.0,
    vulnerable=0.0, weak=0.0, poison=0.0, doom=0.0, hp_loss=0.0,
    heal=0.0, orbs_lightning=0.0, orbs_frost=0.0, orbs_plasma=0.0,
    orb_evokes=0, generates_status=0, generates_hand_attacks=0,
    hand_attack_damage=0.0, star_cost=0.0, star_gain=0.0,
)


def compile_sim_card(profile: Optional[CardProfile]) -> _SimCard:
    """Compile a ``CardProfile`` into a flat fast-path simulation tuple."""

    if profile is None:
        return _UNKNOWN_SIM_CARD
    cached = _SIM_CARD_CACHE.get(id(profile))
    if cached is not None and cached.card_id == profile.card_id:
        return cached

    damage_single = 0.0
    damage_all = 0.0
    hits = 1.0
    hits_x = False
    damage_per_x = False
    is_arrow = False
    block = 0.0
    block_per_x = False
    block_instances = 0.0
    draw = 0
    energy_gain = 0.0
    double_energy = False
    energy_next_turn = 0.0
    strength_gain = 0.0
    dexterity_gain = 0.0
    vigor_gain = 0.0
    aim_gain = 0.0
    stretch_gain = 0.0
    vulnerable = 0.0
    weak = 0.0
    poison = 0.0
    doom = 0.0
    hp_loss = 0.0
    heal = 0.0
    orbs_lightning = 0.0
    orbs_frost = 0.0
    orbs_plasma = 0.0
    orb_evokes = 0
    star_gain = 0.0

    for eff in profile.effects:
        kind = eff.get("kind")
        amt = float(eff.get("amount") or 0.0)
        if kind == "damage":
            h = float(eff.get("hits") or 1.0)
            if eff.get("hits_x"):
                hits_x = True
            else:
                hits = max(hits, h)
            if eff.get("per_x") and not eff.get("hits_x"):
                damage_per_x = True
            if eff.get("source") == "arrow":
                is_arrow = True
            if eff.get("target") == "all":
                damage_all += amt * (1.0 if hits_x else h)
            else:
                damage_single += amt * (1.0 if hits_x else h)
        elif kind == "block":
            reps = float(eff.get("times") or 1.0) if isinstance(eff.get("times"), (int, float)) else 1.0
            block += amt * reps
            block_instances += reps
            if eff.get("per_x"):
                block_per_x = True
        elif kind == "draw":
            draw += int(round(amt))
        elif kind == "energy":
            scale = eff.get("scale")
            if isinstance(scale, Mapping) and scale.get("per") == "energy_after_cost":
                double_energy = True
            else:
                energy_gain += amt
        elif kind == "hp_loss":
            hp_loss += amt
        elif kind == "heal":
            heal += amt
        elif kind == "gain_stars":
            star_gain += amt
        elif kind == "channel_orb":
            orb = str(eff.get("orb") or "")
            if orb == "Lightning":
                orbs_lightning += amt
            elif orb == "Frost":
                orbs_frost += amt
            elif orb == "Plasma":
                orbs_plasma += amt
        elif kind == "evoke_orb":
            orb_evokes += int(round(float(eff.get("count") or 1.0)))
        elif kind == "apply_power":
            power = str(eff.get("power") or "")
            target = str(eff.get("target") or "")
            if target == "self":
                if power == "Strength":
                    strength_gain += amt
                elif power == "Dexterity":
                    dexterity_gain += amt
                elif power == "Vigor":
                    vigor_gain += amt
                elif power == "Aim":
                    aim_gain += amt
                elif power == "Stretch":
                    stretch_gain += amt
            else:
                if power == "Vulnerable":
                    vulnerable += amt
                elif power == "Weak":
                    weak += amt
                elif power == "Poison":
                    poison += amt
                elif power == "Doom":
                    doom += amt

    # Simple power contributions (next-turn energy, turn-start strength/block).
    for use in profile.powers:
        for mech in use.mechanics:
            eff = mech.get("effect") or {}
            if mech.get("when") == "turn_start" and eff.get("kind") == "energy":
                if mech.get("turns") == "next_own_turn" or mech.get("limit") == "once":
                    energy_next_turn += float(eff.get("amount") or 0.0)
                else:
                    energy_gain += float(eff.get("amount") or 0.0) * 0.5
            elif mech.get("when") == "turn_start" and eff.get("kind") == "apply_power":
                if eff.get("target") == "self" and eff.get("power") == "Strength":
                    strength_gain += float(eff.get("amount") or 0.0) * 1.5
            elif mech.get("when") == "turn_end" and eff.get("kind") == "block":
                block += float(eff.get("amount") or 0.0) * 1.5

    generates_status = 0
    generates_hand_attacks = 0
    hand_attack_damage = 0.0
    for gen in profile.generates:
        cnt = int(round(float(gen.count or 0.0)))
        if cnt <= 0:
            continue
        if gen.profile.card_type == "Status" and gen.destination != "hand":
            generates_status += cnt
        elif gen.destination == "hand" and gen.profile.card_type == "Attack":
            dmg = sum(
                float(e.get("amount") or 0.0) * float(e.get("hits") or 1.0)
                for e in gen.profile.effects if e.get("kind") == "damage"
            )
            if dmg > 0.0 and (gen.profile.cost or 0.0) <= 0.0:
                generates_hand_attacks += cnt
                hand_attack_damage = max(hand_attack_damage, dmg)

    # For hits_x cards, damage_single/damage_all store per-hit base damage.
    compiled = _SimCard(
        card_id=profile.card_id,
        base_cost=profile.cost,
        x_cost=profile.x_cost,
        zero_x_effect=_has_zero_x_effect(profile) if profile.x_cost else False,
        card_type=profile.card_type,
        exhausts=profile.exhausts_self,
        ethereal=profile.ethereal,
        retain=profile.retain,
        innate=profile.innate,
        returns=profile.returns,
        cost_rule=profile.cost_rule,
        melody_tags=tuple(t for t in MELODY_TAGS if t in profile.tags),
        is_osty_attack="OstyAttack" in profile.tags,
        damage_single=damage_single,
        hits=hits,
        hits_x=hits_x,
        damage_all=damage_all,
        damage_per_x=damage_per_x,
        is_arrow=is_arrow,
        block=block,
        block_per_x=block_per_x,
        block_instances=block_instances,
        draw=max(0, draw),
        energy_gain=energy_gain,
        double_energy=double_energy,
        energy_next_turn=energy_next_turn,
        strength_gain=strength_gain,
        dexterity_gain=dexterity_gain,
        vigor_gain=vigor_gain,
        aim_gain=aim_gain,
        stretch_gain=stretch_gain,
        vulnerable=vulnerable,
        weak=weak,
        poison=poison,
        doom=doom,
        hp_loss=hp_loss,
        heal=heal,
        orbs_lightning=orbs_lightning,
        orbs_frost=orbs_frost,
        orbs_plasma=orbs_plasma,
        orb_evokes=orb_evokes,
        generates_status=generates_status,
        generates_hand_attacks=generates_hand_attacks,
        hand_attack_damage=hand_attack_damage,
        star_cost=float(profile.star_cost or 0.0),
        star_gain=star_gain,
    )
    if len(_SIM_CARD_CACHE) < 4096:
        _SIM_CARD_CACHE[id(profile)] = compiled
    return compiled


def compile_relics(relics: Sequence[RelicProfile]) -> _RelicSpec:
    max_energy = 0
    hand_draw = 0
    turn_energy: Dict[int, float] = {}
    every_turn_energy: List[Tuple[int, float]] = []
    start_str = 0.0
    start_dex = 0.0
    start_blk = 0.0
    start_frost = 0
    start_lightning = 0
    no_block_blk = 0.0
    keep_block_cap: Optional[float] = None
    shuriken_str = 0.0
    kunai_dex = 0.0
    fan_block = 0.0
    play_limit: Optional[int] = None

    for relic in relics:
        for mech in relic.mechanics:
            mod = mech.get("modifier")
            if mod == "max_energy" and not mech.get("condition") and not mech.get("turns_from"):
                max_energy += int(round(float(mech.get("add") or 0.0)))
            elif mod == "hand_draw" and not mech.get("turns"):
                hand_draw += int(round(float(mech.get("add") or 0.0)))
            elif mod == "cards_per_turn_limit":
                lim = int(round(float(mech.get("max") or 99)))
                play_limit = lim if play_limit is None else min(play_limit, lim)
            elif mod == "keep_block":
                cap = mech.get("max")
                if cap is None:
                    keep_block_cap = 9999.0
                else:
                    c = float(cap)
                    keep_block_cap = c if keep_block_cap is None else max(keep_block_cap, c)
            elif "when" in mech:
                when = mech["when"]
                eff = mech.get("effect") or {}
                kind = eff.get("kind")
                amt = float(eff.get("amount") or 0.0)
                if when == "combat_start":
                    if kind == "energy":
                        turn_energy[0] = turn_energy.get(0, 0.0) + amt
                    elif kind == "block":
                        start_blk += amt
                    elif kind == "apply_power" and eff.get("target") == "self":
                        if eff.get("power") == "Strength":
                            start_str += amt
                        elif eff.get("power") == "Dexterity":
                            start_dex += amt
                    elif kind == "channel_orb":
                        if eff.get("orb") == "Frost":
                            start_frost += int(round(amt))
                        elif eff.get("orb") == "Lightning":
                            start_lightning += int(round(amt))
                elif when == "turn_start":
                    fixed = mech.get("turns")
                    if kind == "energy" and isinstance(fixed, list):
                        for t in fixed:
                            idx = max(0, int(t) - 1)
                            turn_energy[idx] = turn_energy.get(idx, 0.0) + amt
                    elif kind == "energy" and mech.get("every_turns"):
                        every_turn_energy.append((max(1, int(mech["every_turns"])), amt))
                elif when == "turn_end" and kind == "block":
                    if (mech.get("condition") or {}).get("no_block"):
                        no_block_blk += amt
                elif when == "card_played" and mech.get("every") == 3:
                    flt = mech.get("filter") or {}
                    if flt.get("card_type") == "Attack":
                        if kind == "apply_power" and eff.get("target") == "self":
                            if eff.get("power") == "Strength":
                                shuriken_str += amt
                            elif eff.get("power") == "Dexterity":
                                kunai_dex += amt
                        elif kind == "block":
                            fan_block += amt

    return _RelicSpec(
        max_energy_bonus=max_energy,
        hand_draw_bonus=hand_draw,
        turn_energy=turn_energy,
        every_turn_energy=tuple(every_turn_energy),
        start_strength=start_str,
        start_dexterity=start_dex,
        start_block=start_blk,
        start_frost=start_frost,
        start_lightning=start_lightning,
        no_block_turn_end_block=no_block_blk,
        keep_block_cap=keep_block_cap,
        shuriken_str=shuriken_str,
        kunai_dex=kunai_dex,
        fan_block=fan_block,
        play_limit=play_limit,
    )


def _deterministic_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_crn_scenarios(
    decision_key: str,
    base_deck_size: int,
    situation: Situation,
    *,
    trajectories: int = DEFAULT_CRN_TRAJECTORIES,
    attack_templates: Optional[Sequence[Sequence[float]]] = None,
) -> CRNScenarioPack:
    """Build deterministic Common-Random-Numbers enemy schedules and draw keys.

    All candidates in the same decision reuse the exact same ``CRNScenarioPack``:
    - Enemy HP & per-turn incoming attack sequences are identical per trajectory;
    - Base deck card draw keys preserve exact relative draw order when 1 card is
      added, reducing Monte Carlo comparison variance by >10x.
    """

    seed = _deterministic_seed(f"sts2-crn-v4:{decision_key}")
    rng = random.Random(seed)
    k_count = max(4, int(trajectories))
    nominal = max(2.0, float(situation.fight_turns or 5.0))
    max_turns = min(14, max(6, int(round(nominal * 1.8))))
    base_hp = float(situation.enemy_hp or 0.0)
    if base_hp <= 0.0:
        base_hp = max(30.0, nominal * 14.0)
    mean_atk = max(0.0, float(situation.incoming_attack or 0.0))

    # Realistic STS2 intent patterns (mean = 1.0): alternating setup/normal/spike turns
    # so Retain cards and consistent defense face genuine burst-turn variance.
    patterns = (
        (0.0, 1.35, 1.65, 0.45, 1.55, 1.0),
        (1.25, 0.0, 1.75, 1.15, 0.45, 1.4),
        (0.75, 1.45, 0.0, 1.80, 1.15, 0.85),
        (1.0, 1.1, 1.5, 0.0, 1.4, 1.0),
    )

    enemy_hps: List[float] = []
    incoming_attacks: List[Tuple[float, ...]] = []
    base_keys: List[Tuple[Tuple[float, ...], ...]] = []
    cand_keys: List[Tuple[float, ...]] = []

    for k in range(k_count):
        # Mild encounter-pool HP spread (±12%), symmetric across trajectories.
        hp_mult = 0.88 + 0.24 * (k / max(1, k_count - 1))
        enemy_hps.append(base_hp * hp_mult)

        if attack_templates:
            tpl = attack_templates[k % len(attack_templates)]
            seq = tuple(
                max(0.0, float(tpl[t % len(tpl)]) * (0.90 + 0.20 * rng.random()))
                for t in range(max_turns)
            )
        else:
            pat = patterns[k % len(patterns)]
            phase = rng.randrange(len(pat))
            scale_jitter = 0.90 + 0.20 * rng.random()
            raw = [mean_atk * pat[(t + phase) % len(pat)] * scale_jitter for t in range(max_turns)]
            # Re-center mean over nominal turns to keep exact first-moment alignment.
            nom_n = max(1, min(max_turns, int(round(nominal))))
            avg_nom = sum(raw[:nom_n]) / nom_n
            if avg_nom > 1e-6 and mean_atk > 0.0:
                # Preserve encounter-to-encounter severity spread while anchoring mean.
                sev = 0.85 + 0.30 * (k / max(1, k_count - 1))
                factor = (mean_atk * sev) / avg_nom
                raw = [v * factor for v in raw]
            seq = tuple(max(0.0, v) for v in raw)
        incoming_attacks.append(seq)

        # Coupled shuffle keys per reshuffle cycle.
        cycles_b = []
        cycles_c = []
        for _ in range(MAX_SIM_CYCLES):
            cycles_b.append(tuple(rng.random() for _ in range(max(1, base_deck_size))))
            cycles_c.append(rng.random())
        base_keys.append(tuple(cycles_b))
        cand_keys.append(tuple(cycles_c))

    return CRNScenarioPack(
        seed=seed,
        trajectories=k_count,
        max_turns=max_turns,
        base_deck_size=base_deck_size,
        enemy_hps=tuple(enemy_hps),
        incoming_attacks=tuple(incoming_attacks),
        base_keys=tuple(base_keys),
        cand_keys=tuple(cand_keys),
    )


def _effective_cost(
    inst: _CardInst,
    *,
    energy_now: int,
    skills_this_turn: int,
    attacks_this_turn: int,
    melodies_this_turn: set,
    osty_attacked: bool,
    status_generated: int,
) -> Optional[int]:
    """Discrete integer energy cost of ``inst`` at the current moment in the turn."""

    card = inst.card
    if card.x_cost:
        if energy_now <= 0 and not card.zero_x_effect:
            return None
        return energy_now
    if card.base_cost is None:
        return None
    cost = float(card.base_cost)
    rule = card.cost_rule
    if rule:
        kind = rule[0]
        if kind == "after_first":
            if inst.times_played > 0:
                cost = max(0.0, float(rule[1]))
        elif kind == "per_own_play":
            cost = max(0.0, cost + float(rule[1]) * inst.times_played)
        elif kind == "per_own_draw":
            cost = max(0.0, cost + float(rule[1]) * inst.times_drawn)
        elif kind == "free_if_osty_attacked":
            if osty_attacked:
                cost = 0.0
        elif kind == "per_status_generated":
            cost = max(0.0, cost + float(rule[1]) * status_generated)
        elif kind == "per_type_this_turn":
            prior = skills_this_turn if rule[2] == "Skill" else attacks_this_turn
            cost = max(0.0, cost + float(rule[1]) * prior)
        elif kind == "per_melodies_this_turn":
            cost = max(0.0, cost + float(rule[1]) * len(melodies_this_turn))
        elif kind == "if_melody_played_this_turn":
            if any(t in melodies_this_turn for t in rule[2]):
                cost = max(0.0, cost + float(rule[1]))
    return max(0, int(round(cost)))


def simulate_deck_crn(
    base_cards: Sequence[_SimCard],
    added_card: Optional[_SimCard],
    relic_spec: _RelicSpec,
    situation: Situation,
    params: Mapping,
    scenarios: CRNScenarioPack,
) -> MicroSimResult:
    """Run ``scenarios.trajectories`` CRN combat rollouts for ``base_cards (+ added_card)``."""

    base_energy = int(round(float(params.get("base_energy", 3.0)))) + relic_spec.max_energy_bonus
    hand_size = max(1, int(round(float(params.get("hand_size", 5.0)))) + relic_spec.hand_draw_bonus)
    vuln_bonus = float(params.get("vulnerable_damage_bonus", 0.5))
    weak_red = float(params.get("weak_damage_reduction", 0.25))
    frail_red = float(params.get("frail_block_reduction", 0.25))
    extra_enemies = max(0.0, float(situation.extra_enemies or 0.0))
    player_weak_rate = float(situation.player_weak or 0.0)
    player_frail_rate = float(situation.player_frail or 0.0)
    enemy_status_rate = float(situation.status_cards or 0.0)
    orb_slots = int(round(float(situation.orb_slots or 0.0)))
    has_arrows = float(situation.arrow_slots or 0.0) > 0.0

    # Dummy status card for enemy-injected status cards.
    enemy_status = compile_sim_card(None)

    losses: List[float] = []
    turns_list: List[float] = []
    clump_rates: List[float] = []
    retain_saves_list: List[float] = []

    deck_template: List[Tuple[_SimCard, int]] = [(c, idx) for idx, c in enumerate(base_cards)]
    if added_card is not None:
        deck_template.append((added_card, -1))

    for k in range(scenarios.trajectories):
        enemy_hp = scenarios.enemy_hps[k]
        incoming_seq = scenarios.incoming_attacks[k]
        b_keys = scenarios.base_keys[k]
        c_keys = scenarios.cand_keys[k]

        cycle = 0

        def shuffle_pile(pile: List[_CardInst], cyc: int, initial: bool = False) -> None:
            ci = cyc % MAX_SIM_CYCLES
            bk = b_keys[ci]
            ck = c_keys[ci]
            if initial:
                # Innate cards go to top of draw pile (drawn first on turn 1).
                pile.sort(
                    key=lambda inst: (
                        0 if inst.card.innate else 1,
                        ck if inst.slot_id < 0 else (bk[inst.slot_id] if inst.slot_id < len(bk) else 0.5),
                    )
                )
            else:
                pile.sort(
                    key=lambda inst: (
                        ck if inst.slot_id < 0 else (bk[inst.slot_id] if inst.slot_id < len(bk) else 0.5),
                    )
                )

        draw_pile: List[_CardInst] = [_CardInst(c, sid) for c, sid in deck_template]
        shuffle_pile(draw_pile, 0, initial=True)
        discard_pile: List[_CardInst] = []
        hand: List[_CardInst] = []

        strength = relic_spec.start_strength
        dexterity = relic_spec.start_dexterity
        vigor = 0.0
        aim = 0.0
        stretch = 0.0
        enemy_vuln = 0.0
        enemy_weak = 0.0
        enemy_poison = 0.0
        enemy_doom = 0.0
        stars = 0.0
        pending_energy = 0.0
        kept_block = relic_spec.start_block
        status_generated = 0
        status_accum = 0.0
        arrows_ready = 1 if has_arrows else 0
        arrow_hit_counter = 0

        # Simple FIFO orb queue: "L"=Lightning(3/8), "F"=Frost(2/5), "P"=Plasma(1/2)
        orbs: List[str] = []
        for _ in range(relic_spec.start_frost):
            if orb_slots > 0:
                orbs.append("F")
        for _ in range(relic_spec.start_lightning):
            if orb_slots > 0 and len(orbs) < orb_slots:
                orbs.append("L")

        def draw_cards(count: int) -> None:
            nonlocal cycle
            for _ in range(count):
                if len(hand) >= MAX_HAND_SIZE:
                    return
                if not draw_pile:
                    if not discard_pile:
                        return
                    draw_pile.extend(discard_pile)
                    discard_pile.clear()
                    cycle += 1
                    shuffle_pile(draw_pile, cycle, initial=False)
                inst = draw_pile.pop(0)
                inst.times_drawn += 1
                hand.append(inst)

        hp_lost = 0.0
        turns_taken = 0
        clump_turns = 0
        retain_saves = 0
        retained_last_turn: set = set()

        for t in range(scenarios.max_turns):
            turns_taken = t + 1
            incoming_raw = incoming_seq[t]
            incoming = incoming_raw * (1.0 - weak_red) if enemy_weak > 0.0 else incoming_raw

            # Start-of-turn Poison tick + Plasma passive energy.
            if enemy_poison > 0.0:
                enemy_hp -= enemy_poison
                enemy_poison = max(0.0, enemy_poison - 1.0)
            if enemy_hp <= enemy_doom:
                break

            turn_bonus = relic_spec.turn_energy.get(t, 0.0)
            for period, amt in relic_spec.every_turn_energy:
                if (t + 1) % period == 0:
                    turn_bonus += amt
            plasma_passive = sum(1 for o in orbs if o == "P")
            energy = max(0, int(round(base_energy + turn_bonus + pending_energy + plasma_passive)))
            pending_energy = 0.0

            block = kept_block
            kept_block = 0.0
            # Enemy-injected status cards accumulate into discard.
            if enemy_status_rate > 0.0:
                status_accum += enemy_status_rate
                while status_accum >= 1.0:
                    status_accum -= 1.0
                    discard_pile.append(_CardInst(enemy_status, 999))
                    status_generated += 1

            draw_cards(hand_size)

            attacks_this_turn = 0
            skills_this_turn = 0
            plays_this_turn = 0
            melodies_this_turn: set = set()
            osty_attacked = False
            high_cost_played = 0
            p_weak_mult = 1.0 - weak_red * player_weak_rate
            p_frail_mult = 1.0 - frail_red * player_frail_rate
            rem_turns = max(1.0, float(scenarios.max_turns - t))

            # Greedy integer energy knapsack play loop.
            while hand and (relic_spec.play_limit is None or plays_this_turn < relic_spec.play_limit):
                best_idx = -1
                best_score = -1e9
                best_cost = 0
                best_is_lethal = False

                need_block = max(0.0, incoming - block)

                for idx, inst in enumerate(hand):
                    card = inst.card
                    if card.star_cost > stars + 1e-9:
                        continue
                    cost = _effective_cost(
                        inst,
                        energy_now=energy,
                        skills_this_turn=skills_this_turn,
                        attacks_this_turn=attacks_this_turn,
                        melodies_this_turn=melodies_this_turn,
                        osty_attacked=osty_attacked,
                        status_generated=status_generated,
                    )
                    if cost is None or cost > energy:
                        continue

                    x_val = float(cost) if card.x_cost else 1.0
                    # Estimate immediate damage & block of playing this card now.
                    amp = (1.0 + vuln_bonus) if enemy_vuln > 0.0 else 1.0
                    eff_hits = x_val if card.hits_x else card.hits
                    if card.is_arrow and arrows_ready <= 0:
                        dmg_est = 0.0
                    else:
                        base_s = card.damage_single * (x_val if (card.damage_per_x or card.hits_x) else 1.0)
                        base_a = card.damage_all * (x_val if (card.damage_per_x or card.hits_x) else 1.0)
                        has_dmg = (base_s + base_a) > 0.0
                        str_add = (strength * eff_hits) if has_dmg else 0.0
                        vig_add = vigor if has_dmg else 0.0
                        aim_add = (aim * eff_hits) if (card.is_arrow and has_dmg) else 0.0
                        dmg_est = (base_s + str_add + vig_add + aim_add + base_a * (1.0 + extra_enemies * 0.7)) * amp * p_weak_mult
                    if card.generates_hand_attacks > 0:
                        dmg_est += card.generates_hand_attacks * (card.hand_attack_damage + strength) * amp * p_weak_mult

                    blk_raw = card.block * (x_val if card.block_per_x else 1.0)
                    if blk_raw > 0.0:
                        blk_est = max(0.0, (blk_raw + dexterity * card.block_instances + stretch) * p_frail_mult)
                    else:
                        blk_est = 0.0

                    # Smart Retain hold: keep pure defense Retain cards for spike turns when no block is needed.
                    if (
                        card.retain
                        and need_block <= 0.5
                        and blk_est > 0.0
                        and dmg_est <= 0.0
                        and card.draw <= 0
                        and card.energy_gain <= 0.0
                        and not card.double_energy
                        and card.strength_gain <= 0.0
                        and card.dexterity_gain <= 0.0
                    ):
                        continue

                    is_lethal = dmg_est >= enemy_hp - enemy_doom and dmg_est > 0.0
                    if is_lethal:
                        best_idx, best_cost, best_is_lethal = idx, cost, True
                        break

                    # 0-cost draw / energy enablers play first to expand options this turn.
                    if cost == 0 and (card.energy_gain > 0.0 or card.draw > 0 or card.double_energy):
                        val = 500.0 + card.energy_gain * 40.0 + card.draw * 30.0
                    else:
                        blk_saved = min(need_block, blk_est)
                        e_gain = (max(0.0, float(energy - cost)) if card.double_energy else card.energy_gain)
                        net_e = e_gain - cost
                        val = (
                            blk_saved * 1.45
                            + dmg_est * 0.90
                            + card.poison * min(4.0, rem_turns) * 0.85
                            + card.doom * 0.75
                            + (card.strength_gain + card.dexterity_gain) * rem_turns * 2.2
                            + card.vigor_gain * 0.9 + card.aim_gain * 0.9 + card.stretch_gain * 1.1
                            + card.weak * min(incoming * weak_red, 8.0) * 0.9
                            + card.vulnerable * 4.0
                            + card.draw * 4.5
                            + max(0.0, net_e) * 7.0
                            + card.energy_next_turn * 4.5
                            + card.orbs_frost * min(3.0, rem_turns) * 2.2
                            + card.orbs_lightning * min(3.0, rem_turns) * 2.5
                            + card.orbs_plasma * min(4.0, rem_turns) * 4.0
                            + card.heal * 1.1
                            - card.hp_loss * 1.2
                            - card.generates_status * 1.5
                        )
                        if cost > 0:
                            val /= max(0.65, float(cost) ** 0.75)
                        # Avoid spending energy on X-cost before playing higher-value cards if energy allows both.
                        if card.x_cost:
                            val *= 0.82
                        if val <= 0.0 and cost > 0:
                            continue

                    if val > best_score:
                        best_score = val
                        best_idx = idx
                        best_cost = cost

                if best_idx < 0:
                    break

                inst = hand.pop(best_idx)
                card = inst.card
                x_val = float(best_cost) if card.x_cost else 1.0
                energy -= best_cost
                stars = max(0.0, stars - card.star_cost + card.star_gain)
                inst.times_played += 1
                plays_this_turn += 1
                if best_cost >= 2:
                    high_cost_played += 1
                if inst.slot_id in retained_last_turn and incoming > 0.0 and card.block > 0.0:
                    retain_saves += 1

                if card.card_type == "Attack":
                    attacks_this_turn += 1
                    if attacks_this_turn % 3 == 0:
                        strength += relic_spec.shuriken_str
                        dexterity += relic_spec.kunai_dex
                        block += relic_spec.fan_block
                elif card.card_type == "Skill":
                    skills_this_turn += 1
                if card.is_osty_attack:
                    osty_attacked = True
                for tag in card.melody_tags:
                    melodies_this_turn.add(tag)

                # Apply damage.
                amp = (1.0 + vuln_bonus) if enemy_vuln > 0.0 else 1.0
                eff_hits = x_val if card.hits_x else card.hits
                if not (card.is_arrow and arrows_ready <= 0):
                    base_s = card.damage_single * (x_val if (card.damage_per_x or card.hits_x) else 1.0)
                    base_a = card.damage_all * (x_val if (card.damage_per_x or card.hits_x) else 1.0)
                    if (base_s + base_a) > 0.0:
                        dmg = (
                            base_s + strength * eff_hits + vigor
                            + (aim * eff_hits if card.is_arrow else 0.0)
                            + base_a * (1.0 + extra_enemies * 0.7)
                        ) * amp * p_weak_mult
                        enemy_hp -= max(0.0, dmg)
                        vigor = 0.0
                        if card.is_arrow:
                            aim = 0.0
                            arrows_ready = max(0, arrows_ready - 1)
                        elif has_arrows:
                            arrow_hit_counter += int(max(1, round(eff_hits)))
                            if arrow_hit_counter >= 3:
                                arrows_ready = 1
                                arrow_hit_counter = 0
                if card.generates_hand_attacks > 0:
                    for _ in range(card.generates_hand_attacks):
                        enemy_hp -= max(0.0, (card.hand_attack_damage + strength) * amp * p_weak_mult)
                        attacks_this_turn += 1
                        if attacks_this_turn % 3 == 0:
                            strength += relic_spec.shuriken_str
                            dexterity += relic_spec.kunai_dex
                            block += relic_spec.fan_block

                # Apply block, buffs, debuffs, orbs, energy, draw.
                blk_raw = card.block * (x_val if card.block_per_x else 1.0)
                if blk_raw > 0.0:
                    block += max(0.0, (blk_raw + dexterity * card.block_instances + stretch) * p_frail_mult)
                    stretch = 0.0
                strength += card.strength_gain
                dexterity += card.dexterity_gain
                vigor += card.vigor_gain
                aim += card.aim_gain
                stretch += card.stretch_gain
                enemy_vuln += card.vulnerable
                enemy_weak += card.weak
                enemy_poison += card.poison
                enemy_doom += card.doom
                hp_lost += card.hp_loss
                if card.heal > 0.0:
                    hp_lost = max(0.0, hp_lost - card.heal)

                if card.double_energy:
                    energy += energy
                elif card.energy_gain != 0.0:
                    energy = max(0, int(round(energy + card.energy_gain)))
                pending_energy += card.energy_next_turn

                # Orbs channel/evoke.
                if orb_slots > 0:
                    def channel(code: str, cnt: int) -> None:
                        nonlocal enemy_hp, block, energy
                        for _ in range(cnt):
                            if len(orbs) >= orb_slots:
                                evicted = orbs.pop(0)
                                if evicted == "L":
                                    enemy_hp -= 8.0
                                elif evicted == "F":
                                    block += 5.0
                                elif evicted == "P":
                                    energy += 2
                            orbs.append(code)

                    if card.orbs_lightning > 0:
                        channel("L", int(round(card.orbs_lightning * (x_val if card.x_cost else 1.0))))
                    if card.orbs_frost > 0:
                        channel("F", int(round(card.orbs_frost)))
                    if card.orbs_plasma > 0:
                        channel("P", int(round(card.orbs_plasma)))
                    for _ in range(card.orb_evokes):
                        if orbs:
                            ev = orbs.pop(0)
                            if ev == "L":
                                enemy_hp -= 8.0
                            elif ev == "F":
                                block += 5.0
                            elif ev == "P":
                                energy += 2

                for _ in range(card.generates_status):
                    discard_pile.append(_CardInst(enemy_status, 999))
                    status_generated += 1

                if card.draw > 0:
                    draw_cards(card.draw)

                # Destination pile.
                if card.exhausts:
                    pass
                elif card.returns == "hand" and len(hand) < MAX_HAND_SIZE:
                    hand.append(inst)
                elif card.returns == "draw_top":
                    draw_pile.insert(0, inst)
                else:
                    discard_pile.append(inst)

                if best_is_lethal or enemy_hp <= enemy_doom:
                    break

            # End-of-turn orb passives (Lightning 3, Frost 2).
            for o in orbs:
                if o == "L":
                    enemy_hp -= 3.0
                elif o == "F":
                    block += 2.0

            if enemy_hp <= enemy_doom:
                break

            if block <= 0.0 and relic_spec.no_block_turn_end_block > 0.0:
                block += relic_spec.no_block_turn_end_block

            # Detect hand clumping (unplayed >=2-cost card left when another >=2-cost was played
            # or unspent energy couldn't afford it).
            if any(
                (inst.card.base_cost is not None and inst.card.base_cost >= 2.0)
                for inst in hand
            ) and (high_cost_played > 0 or energy > 0):
                clump_turns += 1

            unblocked = max(0.0, incoming - block)
            hp_lost += unblocked

            if relic_spec.keep_block_cap is not None:
                surplus = max(0.0, block - incoming)
                kept_block = min(surplus, relic_spec.keep_block_cap)
            else:
                kept_block = 0.0

            enemy_vuln = max(0.0, enemy_vuln - 1.0)
            enemy_weak = max(0.0, enemy_weak - 1.0)

            # End-of-turn hand cleanup: Retain stays, Ethereal exhausts, others discard.
            next_hand: List[_CardInst] = []
            retained_last_turn = set()
            for inst in hand:
                if inst.card.retain:
                    next_hand.append(inst)
                    retained_last_turn.add(inst.slot_id)
                elif inst.card.ethereal:
                    pass
                else:
                    discard_pile.append(inst)
            hand = next_hand

        losses.append(hp_lost)
        turns_list.append(float(turns_taken))
        clump_rates.append(clump_turns / max(1, turns_taken))
        retain_saves_list.append(float(retain_saves))

    ordered = sorted(losses)
    n = len(ordered)
    tail_n = max(1, int(math.ceil(n * DEFAULT_TAIL_ALPHA)))
    cvar25 = sum(ordered[n - tail_n:]) / tail_n
    p75_idx = min(n - 1, max(0, int(round(0.75 * (n - 1)))))

    return MicroSimResult(
        trajectories=n,
        mean_hp_loss=sum(ordered) / n,
        cvar25_hp_loss=cvar25,
        p75_hp_loss=ordered[p75_idx],
        mean_turns=sum(turns_list) / n,
        clump_rate=sum(clump_rates) / n,
        retain_saves=sum(retain_saves_list) / n,
        losses=tuple(losses),
    )


def compare_candidates_crn(
    decision_key: str,
    base_deck: Sequence[Optional[CardProfile]],
    candidates: Mapping[str, Optional[CardProfile]],
    relics: Sequence[RelicProfile],
    situation: Situation,
    params: Mapping,
    *,
    trajectories: int = DEFAULT_CRN_TRAJECTORIES,
    attack_templates: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, MicroSimResult]:
    """Evaluate ``candidates`` (e.g. ``{"skip": None, "0:CARD": profile, ...}``)
    under identical CRN scenarios for low-variance head-to-head comparison."""

    compiled_base = [compile_sim_card(c) for c in base_deck]
    relic_spec = compile_relics(relics)
    scenarios = build_crn_scenarios(
        decision_key,
        len(compiled_base),
        situation,
        trajectories=trajectories,
        attack_templates=attack_templates,
    )
    results: Dict[str, MicroSimResult] = {}
    for cid, added in candidates.items():
        sim_added = compile_sim_card(added) if added is not None else None
        results[cid] = simulate_deck_crn(
            compiled_base, sim_added, relic_spec, situation, params, scenarios
        )
    return results

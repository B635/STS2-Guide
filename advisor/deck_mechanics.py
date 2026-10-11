"""Per-turn capability engine for deck fitness (scoring v2).

Turns assembly-verified card, relic and power facts (docs/dev/mechanic-facts.md)
into expected per-turn capabilities. Everything is an analytic expectation:

* each card is drawn at ``drawn / deck size`` per turn and played as far as
  energy allows; self-exhausting and Power cards at most once per fight; an
  X-cost card spends the energy the rest of the turn leaves over;
* an event-triggered mechanic fires at the event's per-turn rate, reduced by
  its ``every``/``counter``/``limit`` rules (Poisson counts) and multiplied by
  the probability of its condition;
* draw and energy produced by effects feed back into what is played, solved
  by a fixed number of passes.

Anything the engine cannot convert is returned as an explicit gap and adds
nothing. Unknown keys, values or filters make the whole mechanic unsupported:
an ignored restriction would credit cards it does not apply to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from storage.assembly_facts import class_to_entity_id
from storage.relational import RelationalRepository

FEEDBACK_PASSES = 3
POISSON_TERMS = 12
SUPPORTED_EFFECTS = frozenset({
    "damage", "block", "draw", "energy", "apply_power", "hp_loss", "heal",
    "exhaust_self", "exhaust_other", "discard", "gain_stars",
    "channel_orb", "evoke_orb", "orb_slots", "orb_passive", "summon", "kill_osty",
    "doom_kill", "enemy_strength_down", "forge", "generate",
})
SELF_POWERS = {"Strength": "strength", "Dexterity": "dexterity", "Vigor": "vigor", "Focus": "focus"}
# Orb names on effects -> orb fact ids (orbs:<ID>).
ORB_TYPES = {"Lightning": "LIGHTNING_ORB", "Frost": "FROST_ORB", "Dark": "DARK_ORB",
             "Plasma": "PLASMA_ORB", "Glass": "GLASS_ORB"}
ENEMY_POWERS = {"Vulnerable": "vulnerable", "Weak": "weak", "Poison": "poison", "Doom": "doom"}
DAMAGE_TARGETS = frozenset({"single", "all", "random", "attacker"})
EFFECT_KEYS = frozenset({
    "kind", "amount", "upgrade", "hits", "target", "power", "count", "choice", "per_x", "orb", "scope",
    # Osty attacks: dealt by the pet (``source: osty``), only while it lives.
    "source",
    # enemy_strength_down: until the enemy's turn ends, or for the combat.
    "lasting",
    # Power facts scale an effect by the stacks applied; ``unpowered`` only
    # says Strength/Dexterity do not apply, which the engine never adds to
    # mechanic effects anyway.
    "amount_per_stack", "unpowered", "hits_upgrade", "hits_x",
    # generate: a named card a relic/power mechanic creates (resolved profile).
    "profile", "destination",
    # State-dependent amounts (see ``_Engine._resolve``); *_text keeps the
    # reviewer's wording of the same rule.
    "scale", "hits_scale", "count_scale", "condition", "amount_text", "hits_text",
    # Timing notes that do not change per-turn expectations.
    "deferred_to", "timing",
})
# Game-state quantities an amount may scale with, estimated per play.
SCALE_SOURCES = frozenset({
    "exhaust_pile", "block", "enemy_vulnerable", "strength", "strike_cards",
    "hand_size", "hp_loss_events", "non_attack_in_hand", "attack_draw_chain",
    "own_previous_plays", "condition", "osty_current_hp", "osty_max_hp",
    # Estimated from the previous pass and the deck (see ``_estimate``).
    "orb_count", "distinct_orb_types", "ethereal_cards_played_this_combat",
    "stars_gained_this_turn", "cards_drawn_this_turn_outside_hand_draw", "target_doom",
    "target_doom_div_10", "status_cards_in_combat", "discard_pile",
    "cards_generated_this_combat", "prior_hits_on_target_this_turn",
    # Per card (see ``_resolve``).
    "attacks_played_this_turn", "skills_played_this_turn", "osty_attacks_this_turn",
    "other_osty_attack_cards", "times_drawn_this_combat", "star_cost_cards", "x_value",
})
EFFECT_CONDITIONS = frozenset({
    "target_has_power", "exhaust_pile_at_least", "lost_hp_this_turn",
    "card_exhausted_this_turn", "target_alive", "osty_alive",
})
NUMERIC_EFFECT_KEYS = ("amount", "upgrade", "hits", "count", "amount_per_stack", "hits_upgrade")
POWER_DURATIONS = frozenset({"combat", "this_turn", "turns_decrement"})
# One-shot powers that fire once at the owner's next turn start.
NEXT_TURN_MARKS = frozenset({"next_own_turn", "next own turn only"})
NEXT_TURN_DROP = ("turns", "limit", "timing", "turns_relative_to")
SUPPORTED_WHEN = frozenset({
    "combat_start", "turn_start", "turn_end", "card_played", "card_exhausted",
    "card_discarded", "shuffle", "damage_received", "enemy_died", "block_gained",
    "card_drawn", "damage_dealt",
})
SUPPORTED_CONDITIONS = frozenset({
    "no_block", "per_card_in_hand", "hand_empty", "block_at_least",
    "hp_ratio_at_most", "no_card_type_played_this_turn",
    "cards_played_this_turn_at_most", "no_potions", "turn_at_least",
    "unblocked_damage", "during_own_turn", "caused_by_ethereal", "block_gained_above",
    # Always true for an attack that has a dealer.
    "has_dealer", "attacker_exists",
})
MECHANIC_KEYS = frozenset({
    "when", "filter", "every", "counter", "limit", "condition", "effect",
    "turns", "every_turns", "turns_from",
    # Order within a turn (before/after the hand draw, before/after the card
    # resolves); per-turn expectations do not depend on it.
    "timing",
})
SCHEDULE_KEYS = frozenset({"turns", "every_turns", "turns_from"})
# Fields each supported modifier understands; anything else is unsupported.
MODIFIER_FIELDS = {
    "hand_draw": {"add", "filter"} | SCHEDULE_KEYS,
    "max_energy": {"add", "filter"} | SCHEDULE_KEYS,
    "damage": {"add", "multiply", "filter"} | SCHEDULE_KEYS,
    "block": {"multiply", "filter", "limit"},
    "energy_cost": {"add", "filter"},
    "cards_per_turn_limit": {"max"},
    "no_draw_during_turn": set(),
    "keep_block": {"max"},
    "strength": {"add"},  # temporary Strength on the owner
    "focus": {"add"},     # temporary Focus on the owner
    # Extra free plays: ``uses`` matching plays per turn, or every match.
    "card_play_count": {"add", "filter", "uses", "counter", "applies_to"},
    "hp_loss_negate": {"uses"},
    "card_added_to_deck": {"filter", "upgrade"},
}
CARD_FILTER_KEYS = frozenset({"card_type", "tag", "min_cost", "upgraded", "x_cost", "owner_card", "card"})
# Filters on an attack: the hit that is dealt or received.
ATTACK_FILTER_KEYS = frozenset({"powered_attack", "powered_attack_or_card", "dealer"})
COUNTERS = frozenset({None, "turn", "combat", "run"})
LIMITS = frozenset({None, "once_per_turn", "once_per_combat"})
# Generated cards: a named card sent to a pile. Pools, chosen copies and
# dynamic counts stay gaps until their distribution is modeled.
ADD_CARD_KEYS = frozenset({
    "kind", "card", "count", "count_upgrade", "destination", "free", "position",
    "copy_of_self", "upgraded_if_self_upgraded", "upgraded_if_this_upgraded",
    "upgraded_generated",
})
ADD_CARD_DESTINATIONS = frozenset({"hand", "draw", "discard"})
# Choosing cards from a pile into the hand: at least worth drawing as many.
MOVE_TO_HAND_KEYS = frozenset({
    "kind", "choice", "count", "count_upgrade", "destination", "to", "from", "source", "position",
})


@dataclass(frozen=True)
class PowerUse:
    """A non-generic power a card applies to its owner, resolved to mechanics."""

    power_id: str
    duration: str                   # combat | this_turn | turns_decrement
    stacks: float
    mechanics: tuple


@dataclass(frozen=True)
class Generated:
    """Copies of a named card one play of the source card creates."""

    profile: "CardProfile"
    count: float
    destination: str                # hand | draw | discard


@dataclass(frozen=True)
class CardProfile:
    """One card instance with its effects resolved for its upgrade state."""

    card_id: str
    cost: Optional[float]           # None: unplayable
    x_cost: bool
    card_type: str
    tags: frozenset
    upgraded: bool
    exhausts_self: bool
    effects: tuple
    powers: tuple = ()
    ethereal: bool = False
    generates: tuple = ()           # Generated
    star_cost: Optional[float] = None
    cost_rule: Optional[tuple] = None   # (kind, value[, card_type]) see ``_cost_rule``
    sly: bool = False               # played for free when discarded by an effect

    @property
    def once_per_fight(self) -> bool:
        # Played Power cards and self-exhausting cards leave the fight.
        return self.exhausts_self or self.card_type == "Power"


@dataclass(frozen=True)
class RelicProfile:
    relic_id: str
    category: str
    mechanics: tuple


@dataclass(frozen=True)
class Situation:
    """Fight context the engine needs; derived from needs and run state."""

    fight_turns: float
    extra_enemies: float
    incoming_attack: float
    fight_mix: Mapping[str, float]
    hp_ratio: float
    potion_count: Optional[int]
    orb_slots: float = 0.0            # character base slots (0: one is added on first channel)
    orbs: Mapping[str, Mapping] = field(default_factory=dict)   # reviewed orb facts by ORB_TYPES id
    sovereign_blade: Optional["CardProfile"] = None   # the card a first forge creates
    # From encounter/monster facts; None/0 when they are not available.
    enemy_hits: Optional[float] = None    # attack hits aimed at the player per turn
    player_weak: float = 0.0              # share of turns the player is Weak
    player_frail: float = 0.0             # share of turns the player is Frail
    status_cards: float = 0.0             # cards enemies add to the player's piles per turn


@dataclass
class _Totals:
    single: float = 0.0
    all: float = 0.0
    random: float = 0.0
    block: float = 0.0
    draw: float = 0.0
    energy: float = 0.0
    stars: float = 0.0
    heal: float = 0.0
    hp_loss: float = 0.0
    strength_flat: float = 0.0
    strength_rate: float = 0.0
    dexterity_flat: float = 0.0
    dexterity_rate: float = 0.0
    vigor: float = 0.0
    vulnerable: float = 0.0
    weak: float = 0.0
    poison_single: float = 0.0
    poison_all: float = 0.0
    exhausts: float = 0.0
    ethereal_exhausts: float = 0.0
    discards: float = 0.0
    chosen_discards: float = 0.0    # the player picks which card (Sly first)
    block_events: float = 0.0
    self_damage_events: float = 0.0
    temp_strength: float = 0.0
    focus_flat: float = 0.0
    focus_rate: float = 0.0
    channels: Dict[str, float] = field(default_factory=dict)
    orb_passives: Dict[str, float] = field(default_factory=dict)
    evokes: float = 0.0
    slot_gain: float = 0.0
    orb_single: float = 0.0
    orb_all: float = 0.0
    orb_random: float = 0.0
    osty_single: float = 0.0
    osty_all: float = 0.0
    osty_random: float = 0.0
    summoned: float = 0.0          # Osty HP summoned per turn
    summon_events: float = 0.0
    osty_kills: float = 0.0
    doom_single: float = 0.0
    doom_all: float = 0.0
    weaken_turn: float = 0.0       # enemy attack instances softened this turn
    forged: float = 0.0            # Sovereign Blade damage forged per turn
    generated: Dict[int, float] = field(default_factory=dict)   # mechanic-made copies per turn
    forge_events: float = 0.0
    orbs_held: float = 0.0
    orb_kinds: float = 0.0
    weaken_rate: float = 0.0       # permanent enemy Strength loss gained per turn
    gaps: List[str] = field(default_factory=list)


def effect_unsupported(effect: object) -> Optional[str]:
    """Why the engine cannot convert one effect, or ``None``."""

    if not isinstance(effect, Mapping):
        return "effect:missing"
    kind = effect.get("kind")
    if kind not in SUPPORTED_EFFECTS:
        return f"effect:{kind}"
    # ``*_text`` keys keep the reviewer's wording next to the structured
    # base case; a value given only as text stays non-numeric and is rejected.
    extra = {key for key in set(effect) - EFFECT_KEYS if not key.endswith("_text")}
    if extra:
        return f"effect_key:{sorted(extra)[0]}"
    for key in NUMERIC_EFFECT_KEYS:
        value = effect.get(key)
        if key == "hits" and value == "X" and effect.get("per_x"):
            continue  # one hit per energy spent (Whirlwind, Skewer)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"effect_value:{key}"
    for key in ("scale", "hits_scale", "count_scale"):
        scale = effect.get(key)
        if scale is not None and (not isinstance(scale, Mapping) or scale.get("per") not in SCALE_SOURCES):
            return f"{key}:unsupported"
    for condition in (effect.get("condition") or {}, (effect.get("hits_scale") or {}).get("condition") or {}):
        for key, value in condition.items():
            if key not in EFFECT_CONDITIONS:
                return f"condition:{key}"
            if key == "target_has_power" and value not in {"Vulnerable", "Weak"}:
                return f"condition:{key}:{value}"
    if kind == "damage" and (effect.get("target") or "single") not in DAMAGE_TARGETS:
        return f"effect_target:{effect.get('target')}"
    if kind in {"exhaust_other", "discard"} and effect.get("choice") not in {None, "random", "choose"}:
        return f"effect_choice:{effect.get('choice')}"
    if kind in {"channel_orb", "orb_passive"}:
        if effect.get("orb") not in ORB_TYPES:
            return f"orb:{effect.get('orb')}"
        if kind == "channel_orb" and effect.get("target") not in {None, "self"}:
            return f"orb_target:{effect.get('target')}"
        if kind == "orb_passive" and effect.get("scope") != "all_of_type":
            return f"orb_scope:{effect.get('scope')}"
    if effect.get("source") not in {None, "osty"}:
        return f"effect_source:{effect.get('source')}"
    if kind == "summon" and effect.get("target") not in {None, "self"}:
        return f"summon_target:{effect.get('target')}"
    if kind == "evoke_orb" and (effect.get("orb") != "next" or not (
        isinstance(effect.get("count"), (int, float)) or effect.get("count_scale")
    )):
        return "evoke_orb:unsupported"
    if kind == "apply_power":
        power, target = effect.get("power"), effect.get("target")
        if target in {"damaged_creature", "damaged enemy"}:
            target = "enemy"
        if not ((target == "self" and power in SELF_POWERS)
                or (target in {"enemy", "all_enemies"} and power in ENEMY_POWERS)):
            return f"power:{power}:{target}"
    return None


def _mechanic_cards(repository: RelationalRepository, mechanic: Mapping) -> tuple[dict, tuple]:
    """Resolve a mechanic's ``add_card`` of a named card to a ``generate``
    effect carrying the card's profile; pools and copies stay unsupported."""

    effect = mechanic.get("effect") if isinstance(mechanic, Mapping) else None
    if not isinstance(effect, Mapping) or effect.get("kind") != "add_card":
        return dict(mechanic), ()
    name = effect.get("card")
    extra = {k for k in effect if not k.endswith("_text")} - {
        "kind", "card", "count", "count_per_stack", "amount_per_stack", "destination", "free", "position",
    }
    if not name or extra or str(name).lower().startswith("copy") or effect.get("destination") not in ADD_CARD_DESTINATIONS:
        return dict(mechanic), ()
    card_id = class_to_entity_id(str(name)) if not str(name).isupper() else str(name)
    profile, gaps = card_profile(repository, card_id, 0, _depth=1)
    if profile is None:
        return dict(mechanic), (f"generated_card_fact_missing:{card_id}",)
    if effect.get("free") in (True, "this_turn") and profile.cost is not None:
        profile = replace(profile, cost=0.0)
    generated = {"kind": "generate", "profile": profile, "destination": effect["destination"]}
    for key in ("count", "count_per_stack", "amount_per_stack"):
        if key in effect:
            generated["amount_per_stack" if key != "count" else "amount"] = effect[key]
    return {**mechanic, "effect": generated}, tuple(f"generated:{card_id}:{g}" for g in gaps)


def _resolve_stacks(mechanic: Mapping, stacks: float) -> dict:
    """Replace ``amount_per_stack`` by the amount for ``stacks`` stacks."""

    resolved = {k: v for k, v in mechanic.items() if v is not None}
    effect = dict(resolved.get("effect") or {})
    if "amount_per_stack" in effect:
        effect["amount"] = float(effect.pop("amount_per_stack")) * stacks
    if effect:
        resolved["effect"] = effect
    for key in ("amount_per_stack", "add_per_stack"):
        if key in resolved:
            resolved["add"] = float(resolved.pop(key)) * stacks
    for key in ("limit_per_stack", "count_per_stack"):
        if key in resolved and resolved.get("modifier") in {"card_play_count", "hp_loss_negate"}:
            resolved["uses"] = float(resolved.pop(key)) * stacks
    return resolved


def _next_turn_once(mechanic: Mapping) -> Optional[dict]:
    """A mechanic that fires once at the owner's next turn start, as the same
    mechanic firing on the turn the card is played (one turn early)."""

    turns, limit = mechanic.get("turns"), mechanic.get("limit")
    next_turn = isinstance(turns, str) and turns in NEXT_TURN_MARKS
    if not (next_turn or (isinstance(limit, str) and limit in {"once", "once_per_combat"})):
        return None
    if mechanic.get("when") == "turn_start" or (
        mechanic.get("modifier") in {"hand_draw", "max_energy"} and next_turn
    ):
        return {k: v for k, v in mechanic.items() if k not in NEXT_TURN_DROP}
    return None


def power_use(
    repository: RelationalRepository,
    power: str,
    stacks: float,
) -> tuple[Optional[PowerUse], tuple[str, ...]]:
    """Resolve a self-applied non-generic power through its reviewed fact."""

    power_id = class_to_entity_id(power)
    fact = repository.find_assembly_fact("powers", power_id)
    if fact is None or fact.get("status") != "converted":
        return None, (f"power_fact_missing:{power_id}",)
    duration = (fact.get("duration") or {}).get("kind")
    next_turn = duration == "until_triggered" and all(
        _next_turn_once(m) is not None for m in fact.get("mechanics") or ()
    ) and bool(fact.get("mechanics"))
    consumable = duration == "until_triggered" and bool(fact.get("mechanics")) and all(
        m.get("modifier") == "hp_loss_negate" for m in fact.get("mechanics") or ()
    )
    if next_turn:
        # Active once, on the turn after each play: weighted like "this turn".
        duration = "this_turn"
    elif consumable:
        duration = "combat"  # held until its uses are spent
    elif duration not in POWER_DURATIONS:
        return None, (f"power_duration_unsupported:{power_id}:{duration}",)
    mechanics, gaps = [], []
    for raw in fact.get("mechanics") or ():
        raw, generated_gaps = _mechanic_cards(repository, raw)
        gaps.extend(generated_gaps)
        mechanic = _resolve_stacks(_next_turn_once(raw) if next_turn else raw, stacks)
        reason = unsupported_reason(mechanic)
        if reason:
            gaps.append(f"power_mechanic_unsupported:{power_id}:{reason}")
        else:
            mechanics.append(mechanic)
    return PowerUse(power_id, duration, stacks, tuple(mechanics)), tuple(gaps)


def _filter_unsupported(
    flt: object, card_filter_allowed: bool, attack_filter_allowed: bool = False,
) -> Optional[str]:
    if flt is None:
        return None
    if not isinstance(flt, Mapping):
        return "filter:invalid"
    allowed = (CARD_FILTER_KEYS if card_filter_allowed else frozenset()) | {"fight_kind"}
    if attack_filter_allowed:
        allowed = allowed | ATTACK_FILTER_KEYS
        if flt.get("dealer") not in {None, "owner", "owner_or_owner_pet"}:
            return f"filter:dealer:{flt.get('dealer')}"
        if flt.get("powered_attack") is False:
            return "filter:powered_attack"
    for key, value in flt.items():
        if key not in allowed:
            return f"filter:{key}"
        if isinstance(value, (list, dict)):
            return f"filter_value:{key}"
    return None


def unsupported_reason(mechanic: Mapping) -> Optional[str]:
    """Why the engine cannot convert one relic/power mechanic, or ``None``."""

    if "modifier" in mechanic:
        name = mechanic["modifier"]
        if name not in MODIFIER_FIELDS:
            return str(name)
        extra = set(mechanic) - MODIFIER_FIELDS[name] - {"modifier"}
        if mechanic.get("applies_to") == "self":
            extra.discard("applies_to")  # the power's owner, the player
        if extra:
            return f"key:{sorted(extra)[0]}"
        card_filter = name not in {"hand_draw", "max_energy"}
        reason = _filter_unsupported(mechanic.get("filter"), card_filter)
        if reason:
            return reason
        if name == "card_added_to_deck":
            if mechanic.get("upgrade") is not True:
                return "card_added_to_deck:without_upgrade"
            extra_filter = set(mechanic.get("filter") or {}) - {"card_type", "tag"}
            if extra_filter:
                return f"filter:{sorted(extra_filter)[0]}"
        if name == "energy_cost" and "min_cost" in (mechanic.get("filter") or {}):
            return "filter:min_cost"
        if name == "block" and mechanic.get("limit") not in {None, "once_per_combat"}:
            return f"limit:{mechanic.get('limit')}"
        return None
    extra = set(mechanic) - MECHANIC_KEYS
    if extra:
        return f"key:{sorted(extra)[0]}"
    when = mechanic.get("when")
    if when not in SUPPORTED_WHEN:
        return str(when)
    reason = _filter_unsupported(
        mechanic.get("filter"), when in {"card_played", "card_drawn"},
        when in {"damage_dealt", "damage_received"},
    )
    if reason:
        return reason
    if mechanic.get("counter") not in COUNTERS:
        return f"counter:{mechanic.get('counter')}"
    if mechanic.get("limit") not in LIMITS:
        return f"limit:{mechanic.get('limit')}"
    for key in (mechanic.get("condition") or {}):
        if key not in SUPPORTED_CONDITIONS:
            return f"condition:{key}"
    return effect_unsupported(mechanic.get("effect"))


def _amount(effect: Mapping, upgraded: bool) -> float:
    value = float(effect.get("amount") or 0.0)
    if upgraded:
        value += float(effect.get("upgrade") or 0.0)
    return value


def _cost_rule(modifier: object) -> Optional[tuple]:
    """A cost change driven by the card's own plays/draws or same-turn plays.

    ``("after_first", v)``: costs ``v`` after its first play this combat;
    ``("per_own_play", a)`` / ``("per_own_draw", a)``: ``a`` per earlier play
    or draw this combat; ``("per_type_this_turn", a, type)``: ``a`` per card
    of that type already played this turn. Anything else stays a gap.
    """

    if not isinstance(modifier, Mapping):
        return None
    m = {k: v for k, v in modifier.items() if not str(k).endswith("text")}
    if m.get("kind") == "set_this_combat" and m.get("after") == "first_play" and isinstance(m.get("value"), (int, float)):
        return ("after_first", float(m["value"]))
    flt = m.get("filter") or {}
    add = m.get("add")
    if isinstance(add, (int, float)) and m.get("duration") == "combat" and flt == {"self": True}:
        if m.get("when") == "card_played":
            return ("per_own_play", float(add))
        if m.get("when") == "card_drawn":
            return ("per_own_draw", float(add))
    if (isinstance(add, (int, float)) and m.get("duration") == "this_turn" and m.get("counter") == "turn"
            and m.get("when") == "card_played" and set(flt) == {"card_type"}):
        return ("per_type_this_turn", float(add), str(flt["card_type"]))
    if (m.get("duration") == "combat" and isinstance(m.get("amount"), (int, float))
            and str(m.get("per") or "").startswith("each play of this card (EnergyCost.AddThisCombat(1)")):
        return ("per_own_play", float(m["amount"]))
    return None


def _enemy_strength_down(
    repository: RelationalRepository, effect: Mapping, upgraded: bool,
) -> Optional[dict]:
    """Strength an enemy loses: a negative ``Strength`` (for the combat) or a
    reviewed power that is only a temporary negative Strength modifier on its
    owner until that enemy's turn ends (Piercing Wail, Dark Shackles...)."""

    target = effect.get("target")
    if effect.get("kind") != "apply_power" or target not in {"enemy", "all_enemies"}:
        return None
    if set(k for k in effect if not k.endswith("_text")) - {"kind", "power", "amount", "upgrade", "target"}:
        return None
    amount = _amount(effect, upgraded)
    power = effect.get("power")
    if power == "Strength":
        if amount >= 0:
            return None
        return {"kind": "enemy_strength_down", "amount": -amount, "target": target, "lasting": "combat"}
    fact = repository.find_assembly_fact("powers", class_to_entity_id(str(power)))
    if fact is None or fact.get("status") != "converted":
        return None
    if (fact.get("duration") or {}).get("kind") != "this_turn":
        return None
    mechanics = fact.get("mechanics") or []
    if len(mechanics) != 1:
        return None
    m = mechanics[0]
    per_stack = m.get("add_per_stack")
    if m.get("modifier") != "strength" or not isinstance(per_stack, (int, float)) or per_stack >= 0:
        return None
    if set(k for k in m if not k.endswith("_text")) - {"modifier", "add_per_stack", "applies_to"}:
        return None
    return {"kind": "enemy_strength_down", "amount": -per_stack * amount, "target": target, "lasting": "turn"}


def _move_to_hand(effect: Mapping, upgraded: bool) -> Optional[dict]:
    """A chosen move from the draw or discard pile into the hand, as a draw.

    The choice makes it worth at least a random draw of the same count, so
    this is a lower bound. Moves onto the draw pile, of the card itself, or
    of filtered/unbounded sets are not converted.
    """

    if set(k for k in effect if not k.endswith("_text")) - MOVE_TO_HAND_KEYS:
        return None
    destination = effect.get("destination") or effect.get("to")
    origin = effect.get("from") or effect.get("source")
    count = effect.get("count")
    if destination != "hand" or origin not in {"draw", "discard"}:
        return None
    if effect.get("choice") not in {None, "choose"}:
        return None
    if not isinstance(count, (int, float)) or isinstance(count, bool):
        return None
    amount = float(count) + (float(effect.get("count_upgrade") or 0.0) if upgraded else 0.0)
    return {"kind": "draw", "amount": amount}


def _generated(
    repository: RelationalRepository,
    effect: Mapping,
    upgraded: bool,
    card_id: str,
    depth: int,
) -> tuple[Optional[Generated], tuple[str, ...]]:
    """Resolve one ``add_card`` effect to the card it creates."""

    extra = {key for key in set(effect) - ADD_CARD_KEYS if not key.endswith("_text")}
    if extra:
        return None, (f"card_effect_unsupported:{card_id}:add_card:{sorted(extra)[0]}",)
    target = effect.get("card")
    count = effect.get("count")
    if not target or not isinstance(count, (int, float)) or isinstance(count, bool):
        return None, (f"card_effect_unsupported:{card_id}:add_card:unnamed_or_dynamic",)
    if effect.get("destination") not in ADD_CARD_DESTINATIONS:
        return None, (f"card_effect_unsupported:{card_id}:add_card:destination",)
    if depth > 0:
        # A generated card that generates again (e.g. Anger's copies) is
        # followed one level only.
        return None, (f"card_effect_unsupported:{card_id}:add_card:generated_chain_truncated",)
    follows_upgrade = any(effect.get(k) for k in (
        "upgraded_if_self_upgraded", "upgraded_if_this_upgraded", "copy_of_self",
    ))
    upgrade = 1 if effect.get("upgraded_generated") or (follows_upgrade and upgraded) else 0
    profile, gaps = card_profile(repository, str(target).upper(), upgrade, _depth=depth + 1)
    if profile is None:
        return None, (f"generated_card_fact_missing:{card_id}:{target}", *gaps)
    if effect.get("free") in (True, "this_turn"):
        profile = replace(profile, cost=0.0) if profile.cost is not None else profile
    amount = float(count) + (float(effect.get("count_upgrade") or 0.0) if upgraded else 0.0)
    return Generated(profile, amount, str(effect["destination"])), tuple(
        f"generated:{card_id}:{gap}" for gap in gaps
    )


def card_profile(
    repository: RelationalRepository,
    card_id: str,
    upgrades: int,
    _depth: int = 0,
) -> tuple[Optional[CardProfile], tuple[str, ...]]:
    """Resolve one card instance.

    Without a reviewed fact there is nothing to score (``None``). With one,
    effects the engine cannot convert yet are left out and returned as gaps,
    so the rest of the card is still scored and the omission is disclosed.
    """

    fact = repository.find_assembly_fact("cards", card_id)
    if fact is None or fact.get("status") != "converted":
        return None, (f"card_fact_missing:{card_id}",)
    if upgrades > 1:
        return None, (f"card_upgrade_count_unmodeled:{card_id}",)
    upgraded = upgrades == 1
    effects, powers, generates, gaps = [], [], [], []
    for effect in fact.get("effects") or ():
        weakened = _enemy_strength_down(repository, effect, upgraded) if isinstance(effect, Mapping) else None
        if weakened is not None:
            effects.append(weakened)
            continue
        if isinstance(effect, Mapping) and effect.get("kind") == "move_card":
            moved = _move_to_hand(effect, upgraded)
            if moved is not None:
                effects.append(moved)
                continue
        if isinstance(effect, Mapping) and effect.get("kind") == "add_card":
            generated, generated_gaps = _generated(repository, effect, upgraded, card_id, _depth)
            gaps.extend(generated_gaps)
            if generated is not None:
                generates.append(generated)
            continue
        power = effect.get("power")
        if (
            effect.get("kind") == "apply_power" and effect.get("target") == "self"
            and power not in SELF_POWERS
        ):
            use, use_gaps = power_use(repository, str(power), _amount(effect, upgraded))
            gaps.extend(use_gaps)
            if use is not None:
                powers.append(use)
            continue
        reason = effect_unsupported({k: v for k, v in effect.items() if k != "count_upgrade"})
        if reason:
            gaps.append(f"card_effect_unsupported:{card_id}:{reason}")
            continue
        resolved = {**effect, "amount": _amount(effect, upgraded)}
        if effect.get("hits") == "X":
            # X counts hits, not damage: the engine sets hits to the X spent.
            resolved.pop("per_x", None)
            resolved["hits"] = 1.0
            resolved["hits_x"] = True
        if "count_upgrade" in effect:
            resolved.pop("count_upgrade")
            if upgraded and isinstance(effect.get("count"), (int, float)):
                resolved["count"] = float(effect["count"]) + float(effect["count_upgrade"] or 0.0)
        if effect.get("kind") == "evoke_orb" and upgraded:
            resolved["count"] = float(effect.get("count") or 0.0) + float(effect.get("upgrade") or 0.0)
        if upgraded and "scale" in effect:
            scale = dict(effect["scale"])
            scale["factor"] = float(scale.get("factor") or 0.0) + float(scale.get("factor_upgrade") or 0.0)
            resolved["scale"] = scale
        if upgraded and effect.get("hits_upgrade"):
            # Some upgrades add hits rather than damage (e.g. Sword Boomerang).
            resolved["hits"] = float(effect.get("hits") or 1) + float(effect["hits_upgrade"])
        effects.append(resolved)
    cost_rule = _cost_rule(fact.get("cost_modifier")) if fact.get("cost_modifier") else None
    if fact.get("cost_modifier") and cost_rule is None:
        gaps.append(f"card_effect_unsupported:{card_id}:cost_modifier")
    if fact.get("mechanics"):
        gaps.append(f"card_effect_unsupported:{card_id}:mechanics")
    raw_stars = fact.get("star_cost")
    if raw_stars == "X":
        gaps.append(f"card_effect_unsupported:{card_id}:star_cost:X")
        star_cost = None
    elif isinstance(raw_stars, dict):
        star_cost = float(raw_stars.get("upgraded" if upgraded else "base") or 0.0)
    else:
        star_cost = None
    raw_cost = fact.get("cost")
    # X-cost cards record their cost as "X" rather than a base/upgraded pair.
    cost = raw_cost if not isinstance(raw_cost, dict) else raw_cost.get("upgraded" if upgraded else "base")
    keywords = set(fact.get("keywords") or ())
    return CardProfile(
        card_id=card_id,
        cost=None if cost is None or cost == "X" else float(cost),
        x_cost=cost == "X",
        card_type=str(fact.get("card_type") or ""),
        tags=frozenset(fact.get("tags") or ()),
        upgraded=upgraded,
        exhausts_self="Exhaust" in keywords or any(e.get("kind") == "exhaust_self" for e in effects),
        ethereal="Ethereal" in keywords,
        sly="Sly" in keywords,
        effects=tuple(effects),
        powers=tuple(powers),
        generates=tuple(generates),
        star_cost=star_cost,
        cost_rule=cost_rule,
    ), tuple(dict.fromkeys(gaps))  # a card may repeat one unsupported effect


def relic_profile(
    repository: RelationalRepository,
    relic_id: str,
) -> tuple[Optional[RelicProfile], tuple[str, ...]]:
    fact = repository.find_assembly_fact("relics", relic_id)
    if fact is None or fact.get("status") != "converted":
        return None, (f"relic_fact_missing:{relic_id}",)
    supported, gaps = [], []
    for mechanic in fact.get("mechanics") or ():
        mechanic, generated_gaps = _mechanic_cards(repository, mechanic)
        gaps.extend(generated_gaps)
        effect = mechanic.get("effect") or {}
        if (
            mechanic.get("when") == "combat_start" and set(mechanic) <= {"when", "effect"}
            and effect.get("kind") == "apply_power" and effect.get("target") == "self"
            and effect.get("power") not in SELF_POWERS
        ):
            # A power held from the start of every fight (e.g. Bronze Scales'
            # Thorns): its own mechanics apply for the whole combat.
            use, use_gaps = power_use(repository, str(effect["power"]), float(effect.get("amount") or 0.0))
            if use is not None and use.duration == "combat":
                supported.extend(use.mechanics)
                gaps.extend(use_gaps)
                continue
        reason = unsupported_reason(mechanic)
        if reason:
            gaps.append(f"relic_mechanic_unsupported:{relic_id}:{reason}")
        else:
            supported.append(mechanic)
    return RelicProfile(relic_id, str(fact.get("category") or ""), tuple(supported)), tuple(gaps)


def _poisson_at_least(rate: float, k: int) -> float:
    if k <= 0:
        return 1.0
    term = math.exp(-rate)
    below = term
    for n in range(1, k):
        term *= rate / n
        below += term
    return max(0.0, 1.0 - below)


def _expected_completions(rate: float, every: int) -> float:
    """E[floor(X / every)] for X ~ Poisson(rate)."""

    return sum(_poisson_at_least(rate, every * k) for k in range(1, POISSON_TERMS))


def _triggers_per_turn(mechanic: Mapping, rate: float, turns: float) -> float:
    """Expected firings per turn of an event mechanic seeing ``rate`` events."""

    every = max(1, int(mechanic.get("every") or 1))
    limit = mechanic.get("limit")
    if limit == "once_per_turn":
        return _poisson_at_least(rate, every)
    if limit == "once_per_combat":
        return _poisson_at_least(rate * turns, every) / turns
    if every == 1:
        return rate
    counter = mechanic.get("counter")
    if counter == "turn":
        return _expected_completions(rate, every)
    if counter == "run":
        # A count carried across fights completes at the long-run average.
        return rate / every
    return _expected_completions(rate * turns, every) / turns


def _schedule(mechanic: Mapping, turns: float) -> tuple[float, Optional[float]]:
    """(share of turns it fires on, persisting share of a one-off gain).

    The second value is set when the mechanic fires on fixed turns: a gain on
    turn t is held for the remaining ``turns - t + 1`` turns of the fight.
    """

    if mechanic.get("turns"):
        fired = [t for t in mechanic["turns"] if t <= turns]
        return len(fired) / turns, sum((turns - t + 1) / turns for t in fired)
    if mechanic.get("every_turns"):
        return 1.0 / float(mechanic["every_turns"]), None
    first = mechanic.get("turns_from") or (mechanic.get("condition") or {}).get("turn_at_least")
    if first:
        return max(0.0, turns - float(first) + 1.0) / turns, None
    return 1.0, None


def _poison_dealt(amount: float, turns: float) -> float:
    """Damage of one Poison application decaying by 1 per turn over a fight."""

    if amount <= turns:
        return amount * (amount + 1.0) / 2.0
    return turns * amount - turns * (turns - 1.0) / 2.0


_MISSING = object()


class _Engine:
    def __init__(
        self,
        cards: Sequence[Optional[CardProfile]],
        relics: Sequence[RelicProfile],
        situation: Situation,
        params: Mapping,
    ):
        self.known = [card for card in cards if card is not None]
        self.size = max(1, len(cards))
        # Generated cards are extra card instances whose play rate follows
        # their source's plays (previous pass); kept stable for ``id`` keys.
        self.sources = [(c, g) for c in self.known for g in c.generates]
        self.virtual = list({id(g.profile): g.profile for _, g in self.sources}.values())
        # ForgeCmd: with no Sovereign Blade in combat the first forge creates
        # one in hand; it has Retain, so once made it is there every turn.
        blade = situation.sovereign_blade
        self.blade = blade if blade is not None and not any(
            c.card_id == blade.card_id for c in self.known
        ) else None
        if self.blade is not None:
            self.virtual.append(self.blade)
        self.forge_bonus = 0.0
        self.mechanic_made: Dict[int, tuple] = {}
        mechanic_sources = [m for relic in relics for m in relic.mechanics] + [
            m for card in self.known for use in card.powers for m in use.mechanics
        ]
        for mechanic in mechanic_sources:
            effect = mechanic.get("effect") or {}
            if effect.get("kind") == "generate":
                profile = effect["profile"]
                if id(profile) not in self.mechanic_made:
                    self.mechanic_made[id(profile)] = (profile, effect["destination"])
                    self.virtual.append(profile)
        self.cards = [*self.known, *self.virtual]
        self.situation = situation
        self.params = params
        # (mechanic, weight): relics always apply; powers by how much of the
        # fight they are active, recomputed each pass (see ``_activate``).
        self.relic_mechanics = [(m, 1.0) for relic in relics for m in relic.mechanics]
        self.estimates: Dict[str, float] = {}
        self.rates: Dict[int, float] = {}
        self.x_value = 0.0
        self.play_totals: Dict[str, float] = {"Attack": 0.0, "Skill": 0.0, "OstyAttack": 0.0}
        self.known_set = {id(c) for c in self.known}
        self.deck_counts = {
            "OstyAttack": float(sum(1 for c in self.known if "OstyAttack" in c.tags)),
            "star_cost": float(sum(1 for c in self.known if c.star_cost is not None)),
        }
        self._cost_cache: Dict[int, Optional[float]] = {}
        self.last_plays: Dict[int, float] = {}
        self._set_mechanics(self.relic_mechanics)

    def _set_mechanics(self, weighted: Sequence[tuple[Mapping, float]]) -> None:
        self.modifiers = [(m, w) for m, w in weighted if "modifier" in m]
        self.events = [(m, w) for m, w in weighted if "modifier" not in m]

    def _activate(self, plays: Mapping[int, float], drawn: float) -> None:
        """Weight each played card's powers by their expected active share."""

        turns = self.situation.fight_turns
        first_turn = min(turns, (self.size / max(drawn, 1e-9) + 1.0) / 2.0)
        weighted = list(self.relic_mechanics)
        for card in self.cards:
            for use in card.powers:
                rate = plays[id(card)]
                if use.duration == "this_turn":
                    weight = rate  # active on the turns the card is played
                else:
                    played = min(1.0, rate * turns)  # chance it is played this fight
                    weight = played * max(0.0, turns - first_turn + 1.0) / turns
                for mechanic in use.mechanics:
                    if use.duration == "turns_decrement":
                        # Stacks drop by one per turn: spread the total over the fight.
                        mechanic = dict(mechanic)
                        effect = dict(mechanic.get("effect") or {})
                        effect["amount"] = _poison_dealt(float(effect.get("amount") or 0.0), turns) / turns
                        mechanic["effect"] = effect
                        weight = min(1.0, rate * turns)
                    weighted.append((mechanic, weight))
        self._set_mechanics(weighted)

    def _fight_share(self, flt: Optional[Mapping]) -> float:
        kind = (flt or {}).get("fight_kind")
        if kind is None:
            return 1.0
        mix = self.situation.fight_mix
        total = sum(mix.values())
        return mix.get(kind, 0.0) / total if total else 0.0

    def _modifier_total(self, name: str) -> float:
        turns = self.situation.fight_turns
        return sum(
            float(m.get("add") or 0.0) * w * _schedule(m, turns)[0] * self._fight_share(m.get("filter"))
            for m, w in self.modifiers if m["modifier"] == name
        )

    def _cost(self, card: CardProfile) -> Optional[float]:
        """Printed cost after cost modifiers; X cards are priced separately.

        Constant within a pass (modifiers, rates and last plays are fixed
        until the pass ends), so it is cached per pass.
        """

        cached = self._cost_cache.get(id(card), _MISSING)
        if cached is not _MISSING:
            return cached
        cost = self._uncached_cost(card)
        self._cost_cache[id(card)] = cost
        return cost

    def _uncached_cost(self, card: CardProfile) -> Optional[float]:
        if card.x_cost or card.cost is None:
            return None
        delta = sum(
            float(m.get("add") or 0.0) * w for m, w in self.modifiers
            if m["modifier"] == "energy_cost" and self._matches(card, m.get("filter"), None)
        )
        cost = max(0.0, card.cost + delta)
        rule = card.cost_rule
        if rule:
            turns = self.situation.fight_turns
            rate = self.rates.get(id(card), 0.0)
            plays = max(1.0, self.last_plays.get(id(card), rate) * turns)
            if rule[0] == "after_first":
                cost = (cost + (plays - 1.0) * rule[1]) / plays
            elif rule[0] == "per_own_play":
                cost = max(0.0, cost + rule[1] * (plays - 1.0) / 2.0)
            elif rule[0] == "per_own_draw":
                cost = max(0.0, cost + rule[1] * rate * turns / 2.0)
            elif rule[0] == "per_type_this_turn":
                # Cards of that type played earlier the same turn, on average half.
                others = sum(
                    self.last_plays.get(id(c), 0.0) for c in self.cards
                    if c is not card and c.card_type == rule[2]
                )
                cost = max(0.0, cost + rule[1] * others / 2.0)
        return cost

    def _matches(self, card: CardProfile, flt: Optional[Mapping], x_value: Optional[float]) -> bool:
        if not flt:
            return True
        if flt.get("owner_card") is False:
            return False  # another player's card: never in a single-player deck
        if "card_type" in flt and card.card_type != flt["card_type"]:
            return False
        if "card" in flt and card.card_id != str(flt["card"]).upper():
            return False
        if "tag" in flt and flt["tag"] not in card.tags:
            return False
        if flt.get("upgraded") and not card.upgraded:
            return False
        if flt.get("x_cost") and not card.x_cost:
            return False
        if "min_cost" in flt:
            # Energy actually paid: the modified cost, or the X spend. Cost
            # modifiers never filter on min_cost (rejected as unsupported).
            paid = x_value if card.x_cost else self._cost(card)
            if paid is None or paid < float(flt["min_cost"]):
                return False
        return True

    def run(self) -> tuple[Dict[str, float], tuple[str, ...]]:
        params, turns = self.params, self.situation.fight_turns
        extra_draw = extra_energy = block_estimate = star_supply = 0.0
        previous = _Totals()
        previous_plays: Dict[int, float] = {}
        previous_discards = previous_chosen = 0.0
        for _ in range(FEEDBACK_PASSES):
            totals = _Totals()
            hand = float(params["hand_size"]) + self._modifier_total("hand_draw")
            base_energy = float(params["base_energy"]) + self._modifier_total("max_energy")
            no_draw = any(m["modifier"] == "no_draw_during_turn" and w >= 1.0 for m, w in self.modifiers)
            play_limit = min(
                (float(m["max"]) for m, w in self.modifiers
                 if m["modifier"] == "cards_per_turn_limit" and w >= 1.0),
                default=None,
            )
            elapsed = max(0.0, turns - 1.0) / 2.0
            # Copies sent to a pile stay for the rest of the fight: on average
            # half built up, they dilute every other card's draws.
            piled = sum(
                previous_plays.get(id(c), 0.0) * g.count * elapsed
                for c, g in self.sources if g.destination != "hand"
            ) + sum(
                previous.generated.get(key, 0.0) * elapsed
                for key, (_, destination) in self.mechanic_made.items() if destination != "hand"
            ) + self.situation.status_cards * elapsed
            size = self.size + piled
            drawn = min(size, hand + (0.0 if no_draw else extra_draw))
            per_turn = min(1.0, drawn / size)
            supply = base_energy + extra_energy
            rates = {
                id(c): min(per_turn, 1.0 / turns) if c.once_per_fight else per_turn
                for c in self.known
            }
            for card in self.virtual:
                rates[id(card)] = 0.0
            if self.blade is not None and previous.forge_events > 0:
                # Chance a forge has happened by a random turn of the fight.
                rates[id(self.blade)] = 1.0 - math.exp(-previous.forge_events * max(1.0, elapsed))
            self.forge_bonus = previous.forged * max(1.0, elapsed)
            self.rates = rates
            self._cost_cache = {}
            for key, (profile, destination) in self.mechanic_made.items():
                created = previous.generated.get(key, 0.0)
                if destination == "hand":
                    rates[key] += created
                else:
                    held = created * elapsed * per_turn
                    rates[key] += min(created, held) if profile.once_per_fight else held
            for source, g in self.sources:
                created = previous_plays.get(id(source), 0.0) * g.count
                if g.destination == "hand":
                    rates[id(g.profile)] += created
                else:
                    held = created * elapsed * per_turn
                    rates[id(g.profile)] += min(created, held) if g.profile.once_per_fight else held
            demand = sum(rates[id(c)] * cost for c in self.cards if (cost := self._cost(c)) is not None)
            playable = min(1.0, supply / demand) if demand > 0 else 1.0
            x_rate = sum(rates[id(c)] for c in self.cards if c.x_cost)
            # X cards spend what the other plays leave, shared between them.
            x_value = max(0.0, supply - demand) / max(1.0, x_rate) if x_rate else 0.0
            self.x_value = x_value
            plays = {
                id(c): (
                    rates[id(c)] * (1.0 if c.x_cost else playable)
                    if c.x_cost or self._cost(c) is not None else 0.0
                )
                for c in self.cards
            }
            # Stars carry over between turns (PlayerCombatState keeps them;
            # only death resets them): star cards share the fight's stars.
            star_demand = sum(plays[id(c)] * c.star_cost for c in self.cards if c.star_cost)
            if star_demand > 0:
                star_playable = min(1.0, star_supply / star_demand)
                for c in self.cards:
                    if c.star_cost:
                        plays[id(c)] *= star_playable
            # Replays (Burst, Echo Form...): extra plays that cost nothing,
            # shared between the matching cards in proportion to their plays.
            for modifier, weight in ((m, w) for m, w in self.modifiers if m["modifier"] == "card_play_count"):
                matching = [c for c in self.cards if self._matches(c, modifier.get("filter"), x_value)]
                base_plays = sum(plays[id(c)] for c in matching)
                if base_plays <= 0:
                    continue
                add = float(modifier.get("add") or 0.0) * weight
                uses = modifier.get("uses")
                factor = add if uses is None else add * min(1.0, float(uses) / base_plays)
                for c in set(matching):
                    plays[id(c)] *= 1.0 + factor
            # Sly (CardCmd.Discard -> DiscardAndDraw): a Sly card discarded by
            # an effect is played for free. A chosen discard takes a Sly card
            # in hand when there is one; a random one hits a copy 1/size of the
            # time. End-of-turn discards do not trigger it.
            sly = [c for c in self.cards if c.sly]
            if sly and previous_discards > 0:
                in_hand = len(sly) * per_turn
                chosen = min(1.0, previous_chosen / in_hand) if in_hand > 0 else 0.0
                random_share = max(0.0, previous_discards - previous_chosen) / size
                for c in sly:
                    plays[id(c)] += min(per_turn, per_turn * chosen + random_share)
            played = sum(plays.values())
            if play_limit is not None and played > play_limit:
                scale = play_limit / played
                plays = {k: v * scale for k, v in plays.items()}
                played = play_limit
            totals.ethereal_exhausts = sum(
                max(0.0, rates[id(c)] - plays[id(c)]) for c in self.cards if c.ethereal
            )
            self._estimate(previous, plays, drawn)
            for card in self.cards:
                for effect in card.effects:
                    resolved, chance = self._resolve(effect, card, plays[id(card)])
                    self._apply(totals, resolved, plays[id(card)] * chance, x_value if card.x_cost else None)
            self._apply_damage_modifiers(totals, plays, x_value)
            self._apply_block_modifiers(totals, plays)
            self._apply_events(totals, plays, drawn, played, block_estimate, x_value)
            self._apply_orbs(totals)
            extra_draw, extra_energy, block_estimate = totals.draw, totals.energy, totals.block
            star_supply = totals.stars
            previous = totals
            previous_plays = plays
            previous_discards = totals.discards
            previous_chosen = totals.chosen_discards
            self.last_plays = plays
            self._activate(plays, drawn)
        return (
            self._capabilities(totals, drawn, size, supply, demand, plays),
            tuple(dict.fromkeys(totals.gaps)),
        )

    def _estimate(self, previous: "_Totals", plays: Mapping[int, float], drawn: float) -> None:
        """Per-play estimates of the game state amounts may scale with.

        Uses the previous pass's totals; quantities that build up during a
        fight are taken at its average elapsed turn.
        """

        turns = self.situation.fight_turns
        elapsed = max(0.0, turns - 1.0) / 2.0
        exhausts = previous.exhausts + sum(plays[id(c)] for c in self.cards if c.exhausts_self)
        known = max(1, len(self.known))
        attacks = sum(1 for c in self.known if c.card_type == "Attack") / known
        hand = max(0.0, drawn - 1.0) / 2.0  # other cards still in hand at a random play
        incoming = self.situation.incoming_attack
        hit = max(0.0, min(1.0, (incoming - previous.block) / incoming)) if incoming > 0 else 0.0
        revive = 1.0 - math.exp(-previous.summon_events)
        death = min(1.0, hit + previous.osty_kills)
        alive = revive / (revive + death) if revive + death > 0 else (1.0 if previous.summoned > 0 else 0.0)
        last = self.last_plays
        # Per-type play totals of the previous pass, shared by every card.
        self.play_totals = {
            "Attack": sum(last.get(id(c), 0.0) for c in self.cards if c.card_type == "Attack"),
            "Skill": sum(last.get(id(c), 0.0) for c in self.cards if c.card_type == "Skill"),
            "OstyAttack": sum(last.get(id(c), 0.0) for c in self.cards if "OstyAttack" in c.tags),
        }
        doom = (previous.doom_single + previous.doom_all) * max(1.0, elapsed)
        generated = sum(last.get(id(c), 0.0) * g.count for c, g in self.sources) + sum(
            previous.generated.values()
        )
        self.estimates = {
            "orb_count": previous.orbs_held,
            "distinct_orb_types": previous.orb_kinds,
            "ethereal_cards_played_this_combat": sum(
                last.get(id(c), 0.0) for c in self.cards if c.ethereal
            ) * elapsed,
            "stars_gained_this_turn": previous.stars,
            "cards_drawn_this_turn_outside_hand_draw": previous.draw,
            "target_doom": doom,
            "target_doom_div_10": float(int(doom // 10)),
            "status_cards_in_combat": float(sum(1 for c in self.known if c.card_type == "Status"))
            + sum(
                last.get(id(c), 0.0) * g.count * elapsed
                for c, g in self.sources if g.profile.card_type == "Status"
            ),
            # Half the cards outside the hand sit in the discard pile on average.
            "discard_pile": max(0.0, self.size - drawn) / 2.0,
            "cards_generated_this_combat": generated * elapsed,
            "prior_hits_on_target_this_turn": sum(
                last.get(id(c), 0.0) * self._hits(e)
                for c in self.cards for e in c.effects if e.get("kind") == "damage"
            ) / 2.0,
            "osty_alive": alive,
            "osty_current_hp": alive * previous.summoned * max(1.0, elapsed),
            "osty_max_hp": previous.summoned * max(1.0, elapsed),
            "exhaust_pile": exhausts * elapsed,
            "exhausts_per_turn": exhausts,
            "block": previous.block / 2.0,
            "enemy_vulnerable": previous.vulnerable,
            "vulnerable_uptime": min(1.0, previous.vulnerable),
            "weak_uptime": min(1.0, previous.weak),
            "strength": previous.strength_flat + previous.strength_rate * elapsed,
            "strike_cards": float(sum(1 for c in self.known if "Strike" in c.tags)),
            "hand_size": hand,
            "hp_loss_events": hit * elapsed,
            "hp_lost_this_turn": hit,
            "non_attack_in_hand": hand * (1.0 - attacks),
            "attack_draw_chain": min(10.0, 1.0 / max(1e-9, 1.0 - attacks)),
        }

    def _condition_chance(self, condition: Mapping) -> float:
        chance = 1.0
        for key, value in condition.items():
            if key == "target_has_power":
                chance *= self.estimates["vulnerable_uptime" if value == "Vulnerable" else "weak_uptime"]
            elif key == "exhaust_pile_at_least":
                chance *= _poisson_at_least(self.estimates["exhaust_pile"], int(value))
            elif key == "lost_hp_this_turn":
                chance *= self.estimates["hp_lost_this_turn"]
            elif key == "card_exhausted_this_turn":
                chance *= 1.0 - math.exp(-self.estimates["exhausts_per_turn"])
            elif key == "osty_alive":
                chance *= self.estimates.get("osty_alive", 0.0)
            # target_alive: the card is played on a living target.
        return chance

    def _resolve(self, effect: Mapping, card: CardProfile, rate: float) -> tuple:
        """Fill a state-dependent effect with per-play estimates.

        Returns the resolved effect and the chance its condition holds.
        """

        if (
            "scale" not in effect and "hits_scale" not in effect and "count_scale" not in effect
            and not effect.get("condition") and card.card_id != "SOVEREIGN_BLADE"
        ):
            return effect, 1.0  # nothing depends on the fight's state
        resolved = dict(effect)
        if card.card_id == "SOVEREIGN_BLADE" and effect.get("kind") == "damage":
            resolved["amount"] = float(effect.get("amount") or 0.0) + self.forge_bonus
        estimates = dict(self.estimates)
        # Plays of this card earlier in the same fight, on average.
        estimates["own_previous_plays"] = max(0.0, rate * self.situation.fight_turns - 1.0) / 2.0
        # A card counting this turn's plays is played after the rest: the
        # pass totals (``_estimate``) minus this card's own share.
        own = self.last_plays.get(id(card), 0.0)
        osty = "OstyAttack" in card.tags
        estimates["attacks_played_this_turn"] = self.play_totals["Attack"] - (own if card.card_type == "Attack" else 0.0)
        estimates["skills_played_this_turn"] = self.play_totals["Skill"] - (own if card.card_type == "Skill" else 0.0)
        estimates["osty_attacks_this_turn"] = estimates.get("osty_alive", 0.0) * (
            self.play_totals["OstyAttack"] - (own if osty else 0.0)
        )
        estimates["other_osty_attack_cards"] = self.deck_counts["OstyAttack"] - (
            1.0 if osty and id(card) in self.known_set else 0.0
        )
        estimates["times_drawn_this_combat"] = self.rates.get(id(card), rate) * self.situation.fight_turns / 2.0
        estimates["star_cost_cards"] = self.deck_counts["star_cost"]
        estimates["x_value"] = self.x_value
        scale = effect.get("scale")
        if scale:
            value = estimates[scale["per"]]
            if "min" in scale:
                value = max(float(scale["min"]), value)
            resolved["amount"] = float(effect.get("amount") or 0.0) + float(scale.get("factor") or 0.0) * value
        hits_scale = effect.get("hits_scale")
        if hits_scale:
            base = float(hits_scale.get("base") or 0.0)
            if hits_scale["per"] == "condition":
                chance = self._condition_chance(hits_scale.get("condition") or {})
                resolved["hits"] = base + (float(hits_scale["hits_if"]) - base) * chance
            else:
                resolved["hits"] = base + estimates[hits_scale["per"]]
        count_scale = effect.get("count_scale")
        if count_scale:
            base = float(count_scale.get("base") or 0.0) + (
                float(count_scale.get("base_upgrade") or 0.0) if card.upgraded else 0.0
            )
            resolved["count"] = base + estimates[count_scale["per"]]
        return resolved, self._condition_chance(effect.get("condition") or {})

    def _apply(
        self,
        totals: _Totals,
        effect: Mapping,
        times: float,
        x_value: Optional[float],
        persist_share: Optional[float] = None,
    ) -> None:
        """Add ``times`` (per turn) applications of one effect.

        ``persist_share`` marks a one-off gain held for that share of the
        fight (combat start: 1). Self Strength/Dexterity then count as flat.
        """

        amount = float(effect.get("amount") or 0.0)
        if x_value is not None and effect.get("per_x"):
            amount *= x_value
        kind = effect.get("kind")
        if kind == "apply_power" and effect.get("target") == "self" and persist_share is not None:
            name = SELF_POWERS[effect["power"]]
            if name != "vigor":
                setattr(totals, f"{name}_flat", getattr(totals, f"{name}_flat") + amount * persist_share)
                return
        if times <= 0:
            return
        turns = self.situation.fight_turns
        if kind == "damage":
            hits = self._hits(effect)
            target = effect.get("target") or "single"
            if target == "attacker":
                target = "single"
            if effect.get("source") == "osty":
                target = f"osty_{target}"
            setattr(totals, target, getattr(totals, target) + amount * hits * times)
        elif kind == "summon":
            totals.summoned += amount * times
            totals.summon_events += times
        elif kind == "kill_osty":
            totals.osty_kills += times
        elif kind == "generate":
            key = id(effect["profile"])
            totals.generated[key] = totals.generated.get(key, 0.0) + amount * times
        elif kind == "forge":
            totals.forged += amount * times
            totals.forge_events += times
        elif kind == "enemy_strength_down":
            # One attack per affected enemy per turn (multi-hit attacks would
            # lose more: a conservative count).
            total = self._enemy_hits()
            enemies = total if effect.get("target") == "all_enemies" else total / (1.0 + self.situation.extra_enemies)
            if effect.get("lasting") == "combat":
                totals.weaken_rate += amount * enemies * times
            else:
                totals.weaken_turn += amount * enemies * times
        elif kind == "doom_kill":
            pass  # the same HP <= Doom check Doom itself makes, one turn early
        elif kind == "gain_stars":
            totals.stars += amount * times
        elif kind == "channel_orb":
            orb = ORB_TYPES[effect["orb"]]
            totals.channels[orb] = totals.channels.get(orb, 0.0) + amount * times
        elif kind == "orb_passive":
            orb = ORB_TYPES[effect["orb"]]
            totals.orb_passives[orb] = totals.orb_passives.get(orb, 0.0) + amount * times
        elif kind == "evoke_orb":
            totals.evokes += float(effect.get("count") or 0.0) * times
        elif kind == "orb_slots":
            totals.slot_gain += amount * times
        elif kind in {"block", "draw", "energy", "heal", "hp_loss"}:
            setattr(totals, kind, getattr(totals, kind) + amount * times)
            if kind == "block":
                totals.block_events += times
            elif kind == "hp_loss":
                totals.self_damage_events += times
        elif kind == "exhaust_other":
            totals.exhausts += float(effect.get("count") or 1) * times
        elif kind == "discard":
            totals.discards += float(effect.get("count") or 1) * times
            if effect.get("choice") == "choose":
                totals.chosen_discards += float(effect.get("count") or 1) * times
        elif kind == "apply_power":
            power, target = effect["power"], effect["target"]
            if target in {"damaged_creature", "damaged enemy"}:
                target = "enemy"
            if target == "self":
                name = SELF_POWERS[power]
                if name == "vigor":
                    totals.vigor += amount * times
                else:
                    setattr(totals, f"{name}_rate", getattr(totals, f"{name}_rate") + amount * times)
            elif power == "Doom":
                # Doom never decays and kills at HP <= Doom: each stack
                # stands in for one point of damage (not amplified).
                if target == "all_enemies":
                    totals.doom_all += amount * times
                else:
                    totals.doom_single += amount * times
            elif power == "Poison":
                dealt = _poison_dealt(amount, turns) * times
                if target == "all_enemies":
                    totals.poison_all += dealt
                else:
                    totals.poison_single += dealt
            else:
                name = ENEMY_POWERS[power]
                setattr(totals, name, getattr(totals, name) + amount * times)

    def _apply_damage_modifiers(self, totals: _Totals, plays: Mapping[int, float], x_value: float) -> None:
        turns = self.situation.fight_turns
        for modifier, weight in ((m, w) for m, w in self.modifiers if m["modifier"] == "damage"):
            share = _schedule(modifier, turns)[0] * weight
            for card in self.cards:
                if not self._matches(card, modifier.get("filter"), x_value):
                    continue
                for effect in card.effects:
                    if effect.get("kind") != "damage":
                        continue
                    hits = self._hits(effect)
                    base = float(effect.get("amount") or 0.0)
                    if card.x_cost and effect.get("per_x"):
                        base *= x_value
                    extra = float(modifier.get("add") or 0.0) * hits + (
                        float(modifier.get("multiply") or 1.0) - 1.0
                    ) * base * hits
                    target = effect.get("target") or "single"
                    setattr(totals, target, getattr(totals, target) + extra * plays[id(card)] * share)

    def _apply_block_modifiers(self, totals: _Totals, plays: Mapping[int, float]) -> None:
        turns = self.situation.fight_turns
        for modifier, weight in ((m, w) for m, w in self.modifiers if m["modifier"] == "block"):
            factor = (float(modifier.get("multiply") or 1.0) - 1.0) * weight
            matching = [
                (card, effect) for card in self.cards
                if self._matches(card, modifier.get("filter"), None)
                for effect in card.effects if effect.get("kind") == "block"
            ]
            gained = sum(plays[id(c)] * float(e.get("amount") or 0.0) for c, e in matching)
            if modifier.get("limit") == "once_per_combat":
                rate = sum(plays[id(c)] for c, _ in matching)
                average = gained / rate if rate else 0.0
                totals.block += factor * average * _poisson_at_least(rate * turns, 1) / turns
            else:
                totals.block += factor * gained

    def _block_plays(self, plays: Mapping[int, float]) -> float:
        return sum(
            plays[id(c)] for c in self.cards
            if any(e.get("kind") == "block" for e in c.effects)
        )

    def _event_rate(
        self, mechanic: Mapping, plays: Mapping[int, float], drawn: float,
        block_estimate: float, totals: _Totals, x_value: float,
    ) -> float:
        situation = self.situation
        when = mechanic["when"]
        if when == "card_played":
            return sum(
                plays[id(c)] for c in self.cards
                if self._matches(c, mechanic.get("filter"), x_value)
            )
        if when == "card_exhausted":
            own = totals.exhausts + sum(plays[id(c)] for c in self.cards if c.exhausts_self)
            ethereal = (mechanic.get("condition") or {}).get("caused_by_ethereal")
            if ethereal is True:
                return totals.ethereal_exhausts
            if ethereal is False:
                return own
            return own + totals.ethereal_exhausts
        if when == "block_gained":
            return totals.block_events
        if when == "card_drawn":
            # Every drawn card is the owner's; the filter keeps the deck share.
            matching = sum(1 for c in self.known if self._matches(c, mechanic.get("filter"), x_value))
            return drawn * matching / self.size
        if when == "card_discarded":
            return totals.discards
        if when == "shuffle":
            return drawn / self.size
        if when == "damage_dealt":
            # Powered attack hits the owner (and its pet) lands per turn.
            hits = sum(
                plays[id(c)] * self._hits(e)
                for c in self.cards for e in c.effects
                if e.get("kind") == "damage" and (
                    e.get("source") != "osty" or (mechanic.get("filter") or {}).get("dealer") == "owner_or_owner_pet"
                )
            )
            return hits
        if when == "damage_received" and (
            (mechanic.get("filter") or {}).get("powered_attack")
            or "powered_attack_or_card" in (mechanic.get("filter") or {})
        ):
            # Enemy attack hits aimed at the owner, blocked or not.
            return self._enemy_hits()
        if when == "damage_received" and (mechanic.get("condition") or {}).get("during_own_turn"):
            # HP lost during the owner's own turn comes from its own cards.
            return totals.self_damage_events
        if when == "damage_received":
            incoming = situation.incoming_attack
            return max(0.0, min(1.0, (incoming - block_estimate) / incoming)) if incoming > 0 else 0.0
        if when == "enemy_died":
            return (1.0 + situation.extra_enemies) / situation.fight_turns
        return 1.0  # turn_start / turn_end: once per turn

    def _condition(
        self, condition: Mapping, plays: Mapping[int, float], drawn: float, played: float,
        totals: _Totals,
    ) -> float:
        probability = 1.0
        for key, value in condition.items():
            if key == "no_block":
                probability *= math.exp(-self._block_plays(plays))
            elif key == "per_card_in_hand":
                probability *= max(0.0, drawn - played)
            elif key == "hand_empty":
                probability *= math.exp(-max(0.0, drawn - played))
            elif key == "block_at_least":
                # Block from cards this turn (events are applied after cards).
                probability *= min(1.0, totals.block / float(value)) if value else 1.0
            elif key == "hp_ratio_at_most":
                probability *= 1.0 if self.situation.hp_ratio <= float(value) else 0.0
            elif key == "no_card_type_played_this_turn":
                rate = sum(plays[id(c)] for c in self.cards if c.card_type == value)
                probability *= math.exp(-rate)
            elif key == "cards_played_this_turn_at_most":
                probability *= 1.0 - _poisson_at_least(played, int(value) + 1)
            elif key == "no_potions":
                count = self.situation.potion_count
                if count is None:
                    totals.gaps.append("potion_state_unknown")
                    probability = 0.0
                else:
                    probability *= 1.0 if count == 0 else 0.0
            # turn_at_least: handled by the schedule; unblocked_damage,
            # during_own_turn and caused_by_ethereal: what the event rate
            # already measures; block_gained_above 0: every block event.
        return probability

    def _apply_events(
        self, totals: _Totals, plays: Mapping[int, float], drawn: float, played: float,
        block_estimate: float, x_value: float,
    ) -> None:
        turns = self.situation.fight_turns
        for mechanic, weight in self.events:
            share = weight * self._fight_share(mechanic.get("filter"))
            share *= self._condition(mechanic.get("condition") or {}, plays, drawn, played, totals)
            effect = mechanic["effect"]
            if mechanic["when"] == "combat_start":
                self._apply(totals, effect, share / turns, None, persist_share=share)
                continue
            rate = self._event_rate(mechanic, plays, drawn, block_estimate, totals, x_value)
            fired_share, persist = _schedule(mechanic, turns)
            times = _triggers_per_turn(mechanic, rate, turns) * fired_share * share
            if mechanic["when"] == "turn_start" and persist is not None:
                self._apply(totals, effect, times, None, persist_share=persist * share)
            else:
                self._apply(totals, effect, times, None)

    def _enemy_hits(self) -> float:
        """Attack hits the enemies aim at the player per turn: from the
        encounter facts, else one per enemy."""

        hits = self.situation.enemy_hits
        return float(hits) if hits is not None else 1.0 + self.situation.extra_enemies

    def _hits(self, effect: Mapping) -> float:
        if effect.get("hits_x"):
            return self.x_value
        return float(effect.get("hits") or 1)

    def _focus(self, totals: _Totals) -> float:
        growth = max(0.0, self.situation.fight_turns - 1.0) / 2.0
        temporary = sum(
            float(m.get("add") or 0.0) * w for m, w in self.modifiers if m["modifier"] == "focus"
        )
        return totals.focus_flat + totals.focus_rate * growth + temporary

    def _apply_orbs(self, totals: _Totals) -> None:
        """Channeled orbs over a fight (OrbCmd.Channel, OrbModel facts).

        Orbs fill the slots at the channel rate; each held orb triggers its
        passive once per turn; channeling into full slots evokes the oldest,
        plus explicit evokes while an orb is held. Values come from the
        reviewed orb facts, Focus added where the fact says so (floor 0).
        """

        channel = sum(totals.channels.values())
        if channel <= 0:
            return
        turns = self.situation.fight_turns
        elapsed = max(0.0, turns - 1.0) / 2.0
        base = self.situation.orb_slots or 1.0  # OrbCmd.Channel adds 1 slot when there are none
        slots = max(0.0, base + totals.slot_gain * elapsed)
        if slots <= 0:
            return
        # Average orbs held: min(slots, channel * t) over the fight.
        if channel * turns <= slots:
            held = channel * turns / 2.0
        else:
            full_at = slots / channel
            held = (slots * full_at / 2.0 + slots * (turns - full_at)) / turns
        totals.orbs_held = held
        totals.orb_kinds = min(slots, float(sum(1 for rate in totals.channels.values() if rate > 0)))
        evokes = max(0.0, channel * turns - slots) / turns + totals.evokes * min(1.0, held)
        # Passives before an orb is evicted (Dark grows, Glass decays).
        age = min(turns, slots / channel)
        focus = self._focus(totals)
        for orb, rate in totals.channels.items():
            fact = self.situation.orbs.get(orb)
            if not fact:
                totals.gaps.append(f"orb_fact_missing:{orb}")
                continue
            share = rate / channel
            passives = held * share * (1.0 + totals.orb_passives.get(orb, 0.0))
            self._orb_effect(totals, fact, "passive", passives, focus, age)
            self._orb_effect(totals, fact, "evoke", evokes * share, focus, age)

    def _orb_effect(
        self, totals: _Totals, fact: Mapping, part: str, times: float, focus: float, age: float,
    ) -> None:
        spec = fact.get(part) or {}
        effect = spec.get("effect") or {}
        state = fact.get("state") or {}
        base = float(effect.get("amount") or 0.0)
        adds_focus = str(spec.get("focus") or "").startswith("add")
        value = max(0.0, base + focus) if adds_focus else base
        passive_value = max(0.0, float((fact.get("passive") or {}).get("effect", {}).get("amount") or 0.0) + focus)
        if "evoke_value" in state:
            # Dark-like: passive feeds the stored evoke value, evoke deals it.
            if part == "passive":
                return
            value = float(state["evoke_value"].get("initial") or base) + age * passive_value
        elif "passive_value" in state:
            # Glass-like: the stored value drops by one per passive (floor 0).
            initial = float(state["passive_value"].get("initial") or base)
            steps = max(1, int(round(age)))
            if part == "passive":
                value = sum(max(0.0, initial - i + focus) for i in range(steps)) / steps
            else:
                value = 2.0 * max(0.0, initial - steps + focus)
        kind = effect.get("kind")
        if kind == "damage":
            target = {"all": "all", "random": "random"}.get(effect.get("target"), "single")
            setattr(totals, f"orb_{target}", getattr(totals, f"orb_{target}") + value * times)
        elif kind == "block":
            totals.block += value * times
            totals.block_events += times
        elif kind == "energy":
            totals.energy += value * times
        elif kind not in {None, "orb_evoke_gain"}:
            totals.gaps.append(f"orb_effect_unsupported:{kind}")

    def _capabilities(
        self, totals: _Totals, drawn: float, size: float, supply: float, demand: float,
        plays: Mapping[int, float],
    ) -> Dict[str, float]:
        situation, params = self.situation, self.params
        turns = situation.fight_turns
        amplify = 1.0 + float(params["vulnerable_damage_bonus"]) * min(1.0, totals.vulnerable)
        attack_hits = sum(
            plays[id(c)] * self._hits(e)
            for c in self.cards for e in c.effects if e.get("kind") == "damage"
        )
        # Strength or Dexterity gained during a fight is on average half built up.
        growth = max(0.0, turns - 1.0) / 2.0
        vigor = totals.vigor if attack_hits > 0 else 0.0  # next attack only
        temp_strength = sum(
            float(m.get("add") or 0.0) * w for m, w in self.modifiers if m["modifier"] == "strength"
        )
        incoming = situation.incoming_attack
        kept_block = 0.0
        for m, w in self.modifiers:
            if m["modifier"] == "keep_block":
                surplus = max(0.0, totals.block - incoming)
                cap = m.get("max")
                kept_block += w * (min(surplus, float(cap)) if cap else surplus)
        return {
            # Orb damage is unpowered: no Strength or Vulnerable amplification.
            "frontload": (
                totals.single + totals.all + totals.random + vigor
                + (totals.strength_flat + temp_strength) * attack_hits
            ) * amplify * (1.0 - float(params["weak_damage_reduction"]) * situation.player_weak)
            + totals.poison_single + totals.poison_all
            + totals.orb_single + totals.orb_all + totals.orb_random
            # Osty attacks are powered (Vulnerable applies) but use Osty's
            # Strength, not the player's; Doom stands in for damage.
            + (totals.osty_single + totals.osty_all + totals.osty_random) * amplify
            + totals.doom_single + totals.doom_all,
            "aoe": (
                totals.all + totals.poison_all + totals.orb_all + totals.osty_all + totals.doom_all
            ) * situation.extra_enemies,
            "defense": (
                totals.block * (1.0 - float(params.get("frail_block_reduction", 0.25)) * situation.player_frail)
                + kept_block + totals.heal - totals.hp_loss
                # Die For You: Osty takes the unblocked attack damage, up to
                # the HP summoned for it.
                + min(totals.summoned, max(0.0, incoming - totals.block))
                # Buffer: each use negates one hit's unblocked damage.
                + max(0.0, incoming - totals.block) * min(turns, sum(
                    float(m.get("uses") or 0.0) * w for m, w in self.modifiers
                    if m["modifier"] == "hp_loss_negate"
                )) / turns
                # Enemy Strength lost: less attack damage, never below none.
                + min(incoming, totals.weaken_turn + totals.weaken_rate * growth)
                + (totals.dexterity_flat + totals.dexterity_rate * growth) * self._block_plays(plays)
                + float(params["weak_damage_reduction"]) * situation.incoming_attack
                * min(1.0, totals.weak)
            ),
            "scaling": totals.strength_rate * growth * attack_hits * amplify,
            # Extra cards seen per turn beyond the base hand. Deck size is
            # not counted here: it already lowers every card's per-turn rate.
            "cycle": max(0.0, drawn - float(params["hand_size"])),
            "energy": supply / demand if demand > 0 else float(params["satisfaction_cap"]),
        }


def deck_capabilities(
    cards: Sequence[Optional[CardProfile]],
    relics: Sequence[RelicProfile],
    situation: Situation,
    params: Mapping,
) -> tuple[Dict[str, float], tuple[str, ...]]:
    """Expected per-turn capabilities; unknown cards only take a deck slot."""

    return _Engine(cards, relics, situation, params).run()


def adjust_added_card(
    card: CardProfile,
    relics: Iterable[RelicProfile],
    repository: RelationalRepository,
) -> CardProfile:
    """Apply ``card_added_to_deck`` upgrade modifiers to a newly taken card."""

    for relic in relics:
        for mechanic in relic.mechanics:
            if mechanic.get("modifier") != "card_added_to_deck" or card.upgraded:
                continue
            flt = mechanic.get("filter") or {}
            if "card_type" in flt and card.card_type != flt["card_type"]:
                continue
            if "tag" in flt and flt["tag"] not in card.tags:
                continue
            upgraded, _ = card_profile(repository, card.card_id, 1)
            if upgraded is not None:
                card = upgraded
    return card

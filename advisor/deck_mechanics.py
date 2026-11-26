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
  by a fixed number of passes;
* a card upgraded during a fight plays as its upgraded profile for the
  share of its draws that come after the upgrade.

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
    "doom_kill", "enemy_strength_down", "forge", "generate", "upgrade_cards",
    "arrow_slots", "dislodge_arrow",
})
SELF_POWERS = {
    "Strength": "strength", "Dexterity": "dexterity", "Vigor": "vigor", "Focus": "focus",
    "Aim": "aim", "Stretch": "stretch",
}
# Orb names on effects -> orb fact ids (orbs:<ID>).
ORB_TYPES = {"Lightning": "LIGHTNING_ORB", "Frost": "FROST_ORB", "Dark": "DARK_ORB",
             "Plasma": "PLASMA_ORB", "Glass": "GLASS_ORB"}
ENEMY_POWERS = {
    "Vulnerable": "vulnerable", "Weak": "weak", "Poison": "poison", "Doom": "doom",
    "DislikeYou": "dislike_you",
}
DAMAGE_TARGETS = frozenset({"single", "all", "random", "attacker", "evoke_targets"})
EFFECT_KEYS = frozenset({
    "kind", "amount", "upgrade", "hits", "target", "power", "count", "choice", "per_x", "orb", "scope",
    # With ``per_x``: the upgraded card counts X + this (Tempest, Malaise).
    "bonus_if_upgraded",
    # Osty attacks (``source: osty``) and Fool arrow shots (``source: arrow``).
    "source", "aim_multiplier", "owner_only", "times", "multiply_if",
    # enemy_strength_down: until the enemy's turn ends, or for the combat.
    "lasting",
    # Power facts scale an effect by the stacks applied; ``unpowered`` only
    # says Strength/Dexterity do not apply, which the engine never adds to
    # mechanic effects anyway.
    "amount_per_stack", "unpowered", "hits_upgrade", "hits_x",
    # generate: a named card a relic/power mechanic creates (resolved profile).
    "profile", "destination",
    # Enemy block is not modeled, so "cannot be blocked" changes nothing here.
    "unblockable",
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
    "enemies_killed_by_this_card", "energy_after_cost", "own_exhausted_damage",
    # Fool melodies.
    "melodies_this_turn", "melody_cards_in_combat",
    # From the previous pass at an average moment (see ``_estimate``).
    "cards_drawn_this_combat", "draw_pile", "target_debuff_types", "discards_this_turn",
    "hand_space", "enemy_count", "orbs_channeled_this_combat", "lightning_channeled_this_combat",
    "cards_played_this_combat",
    "energy_spent_this_turn", "skills_in_hand",
})
EFFECT_CONDITIONS = frozenset({
    "target_has_power", "exhaust_pile_at_least", "lost_hp_this_turn",
    "card_exhausted_this_turn", "target_alive", "osty_alive",
    "target_impaled", "target_impaled_at_play_start", "has_arrow", "shot_succeeded",
    "melody_played_this_turn", "melodies_this_turn_at_least",
})
NUMERIC_EFFECT_KEYS = ("amount", "upgrade", "hits", "count", "amount_per_stack", "hits_upgrade", "aim_multiplier", "times")
# Effects that act on a set of cards and may restrict it with ``filter``
# (see ``_card_set_unsupported``); they have no amount, only a count.
CARD_SET_EFFECTS = frozenset({"exhaust_other", "discard", "upgrade_cards"})
# Count sources that only count cards of one type: a filter on that type is
# already part of the count (Flak Cannon: every Status card in combat).
TYPED_COUNT_SOURCES = {"status_cards_in_combat": "Status"}
# get_MaxUpgradeLevel is 0 for every reviewed Status, Curse and Quest card;
# every other card keeps CardModel's default of 1, and IsUpgradable is
# CurrentUpgradeLevel < MaxUpgradeLevel.
NOT_UPGRADABLE_TYPES = frozenset({"Status", "Curse", "Quest"})
# Piles an in-fight upgrade picks from; ``all_combat`` is every pile.
UPGRADE_SCOPES = frozenset({"hand", "draw", "discard", "all_combat"})
# A value written as text needs its structured scale to be counted in full.
VALUE_TEXT_KEYS = {"amount_text": "scale", "hits_text": "hits_scale", "count_text": "count_scale"}
# Effects that only ever act on the card's owner.
SELF_ONLY_EFFECTS = frozenset({"block", "draw", "energy", "heal", "hp_loss"})
POWER_EFFECT_KEYS = frozenset({"kind", "power", "amount", "upgrade", "target"})
# Keywords the engine accounts for. Eternal (cannot leave the deck) and
# Unplayable (no cost) need nothing more inside a fight.
MODELED_KEYWORDS = frozenset({"Exhaust", "Ethereal", "Sly", "Retain", "Innate", "Eternal", "Unplayable"})
POWER_DURATIONS = frozenset({"combat", "this_turn", "turns_decrement"})
# One-shot powers that fire once at the owner's next turn start.
NEXT_TURN_MARKS = frozenset({"next_own_turn", "next own turn only"})
NEXT_TURN_DROP = ("turns", "limit", "timing", "turns_relative_to")
SUPPORTED_WHEN = frozenset({
    "combat_start", "turn_start", "turn_end", "card_played", "card_exhausted",
    "card_discarded", "shuffle", "damage_received", "enemy_died", "block_gained",
    "card_drawn", "damage_dealt", "orb_evoked",
    "arrow_shot", "arrow_impaled", "arrow_dislodged",
})
SUPPORTED_CONDITIONS = frozenset({
    "no_block", "per_card_in_hand", "hand_empty", "block_at_least",
    "hp_ratio_at_most", "no_card_type_played_this_turn",
    "cards_played_this_turn_at_most", "no_potions", "turn_at_least",
    "unblocked_damage", "during_own_turn", "caused_by_ethereal", "block_gained_above",
    # Always true for an attack that has a dealer.
    "has_dealer", "attacker_exists",
    # Fool melody and arrow trigger conditions.
    "melodies_played_this_turn", "target_impaled",
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
    "hp_loss_negate": {"uses", "filter"},
    "card_added_to_deck": {"filter", "upgrade"},
}
# Card-level triggers that fire between fights (quest and curse counters).
RUN_ONLY_WHEN = frozenset({"combat_end", "room_entered", "rest_site", "map_generated", "quest_complete"})
CARD_FILTER_KEYS = frozenset({
    "card_type", "tag", "min_cost", "upgraded", "x_cost", "owner_card", "card",
    "owner",
})
# Filters on an attack: the hit that is dealt or received.
ATTACK_FILTER_KEYS = frozenset({
    "powered_attack", "powered_attack_or_card", "dealer", "from_arrow", "target_has_power",
})
MELODY_TAGS = ("Wah", "Cho", "Ki")
COUNTERS = frozenset({None, "turn", "combat", "run"})
LIMITS = frozenset({None, "once_per_turn", "once_per_combat"})
# Generated cards: a named card sent to a pile. Pools, chosen copies and
# dynamic counts stay gaps until their distribution is modeled.
ADD_CARD_KEYS = frozenset({
    "kind", "card", "count", "count_upgrade", "upgrade", "destination", "free", "position",
    "copy_of_self", "upgraded_if_self_upgraded", "upgraded_if_this_upgraded",
    "upgraded_generated",
})
ADD_CARD_DESTINATIONS = frozenset({"hand", "draw", "discard"})
# Choosing cards from a pile into the hand: at least worth drawing as many.
MOVE_TO_HAND_KEYS = frozenset({
    "kind", "choice", "count", "count_upgrade", "destination", "to", "from", "source", "position",
})
# Transforms (CardCmd.Transform): cards of the hand or draw pile replaced by
# a named card for the rest of the fight (see ``_transformed``).
TRANSFORM_KEYS = frozenset({
    "kind", "choice", "count", "filter", "into", "scope",
    "upgraded_if_self_upgraded", "upgraded_result_if_upgraded",
})


@dataclass(frozen=True)
class PowerUse:
    """A non-generic power a card applies to its owner, resolved to mechanics."""

    power_id: str
    duration: str                   # combat | this_turn | turns_decrement
    stacks: float
    mechanics: tuple
    per_stack: tuple = ()           # per mechanic: does its amount follow the stacks?


@dataclass(frozen=True)
class Generated:
    """Copies of a named card one play of the source card creates.

    A transform (``targets`` set) replaces cards instead of adding them: the
    deck keeps its size, and the copies made per play are the cards found to
    replace, estimated by the engine under the ``targets`` key, at most
    ``count`` (``None``: all of them).
    """

    profile: "CardProfile"
    count: Optional[float]
    destination: str                # hand | draw | discard
    targets: Optional[str] = None


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
    retain: bool = False            # stays in hand until played
    # After a play the card goes back to the hand ("hand") or onto the draw
    # pile ("top") instead of the discard pile: once drawn, it is there again
    # every turn (Particle Wall, Shining Strike).
    returns: Optional[str] = None
    innate: bool = False            # starts every fight in the opening hand
    # The same card upgraded, for upgrades made during a fight (Drain Power,
    # Armaments...); None when it cannot be upgraded (already upgraded, or a
    # Status, Curse or Quest card) or is a generated card.
    upgraded_profile: Optional["CardProfile"] = None

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
    arrow_slots: float = 0.0          # Fool base arrow slots (1 for Fool, 0 otherwise)
    orbs: Mapping[str, Mapping] = field(default_factory=dict)   # reviewed orb facts by ORB_TYPES id
    sovereign_blade: Optional["CardProfile"] = None   # the card a first forge creates
    # From encounter/monster facts; None/0 when they are not available.
    enemy_hits: Optional[float] = None    # attack hits aimed at the player per turn
    enemy_hp: Optional[float] = None      # total HP of a fight's enemies
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
    aim: float = 0.0               # Fool Aim (next powered arrow attack, per hit)
    stretch: float = 0.0           # Fool Stretch (next powered block card, per block instance)
    dislike_you: float = 0.0       # Fool DislikeYou enemy debuff (+1%/stack on arrow damage)
    arrow_slots: float = 0.0       # extra combat arrow slots (Quiver)
    dislodges: float = 0.0         # explicit dislodge_arrow returns per turn
    arrow_damage: float = 0.0      # powered arrow base damage per turn (for DislikeYou)
    arrow_aim_hits: float = 0.0    # resolved arrow hits * aim_multiplier per turn (for Aim)
    max_turn_extra: float = 0.0    # flexible single-turn peak energy gain (support bound)
    max_turn_loss: float = 0.0     # flexible single-turn peak energy loss
    fixed_turn_energy: Dict[int, float] = field(default_factory=dict)  # net energy on fixed turns (1, 2, 3...)
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
    status_exhausts: float = 0.0   # Status cards exhausted per turn (out of the piles)
    # In-fight upgrades, in upgradable card instances: held for the whole
    # fight (combat start, opening hand), per turn usable from the next turn,
    # per turn sent to the discard pile (usable after it is reshuffled).
    upgraded_held: float = 0.0
    upgrades: float = 0.0
    upgrades_discard: float = 0.0
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
    allowed = EFFECT_KEYS | ({"filter"} if kind in CARD_SET_EFFECTS else frozenset())
    extra = {key for key in set(effect) - allowed if not key.endswith("_text")}
    if extra:
        return f"effect_key:{sorted(extra)[0]}"
    if kind in SELF_ONLY_EFFECTS and effect.get("target") not in (None, "self"):
        return f"effect_target:{kind}:{effect.get('target')}"
    if kind in CARD_SET_EFFECTS:
        reason = _card_set_unsupported(effect)
        if reason:
            return reason
    for key in NUMERIC_EFFECT_KEYS:
        value = effect.get(key)
        if key == "hits" and value == "X" and effect.get("per_x"):
            continue  # one hit per energy spent (Whirlwind, Skewer)
        if key == "times" and value == "X" and effect.get("per_x"):
            continue  # repeated X times (Brace)
        if key == "count" and value == "all" and kind in {"upgrade_cards", "dislodge_arrow"}:
            continue  # every upgradable card in scope / every owner arrow on target
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            return f"effect_value:{key}"
    if "owner_only" in effect and kind != "dislodge_arrow":
        return "effect_key:owner_only"
    if "aim_multiplier" in effect and (kind != "damage" or effect.get("source") != "arrow"):
        return "effect_key:aim_multiplier"
    if "times" in effect and not (
        (kind == "block" and isinstance(effect.get("times"), (int, float)) and not isinstance(effect.get("times"), bool))
        or (effect.get("times") == "X" and effect.get("per_x"))
    ):
        return "effect_key:times"
    multiply_if = effect.get("multiply_if")
    if multiply_if is not None:
        if (
            not isinstance(multiply_if, Mapping)
            or set(multiply_if) != {"condition", "multiply"}
            or not isinstance(multiply_if.get("condition"), Mapping)
            or isinstance(multiply_if.get("multiply"), bool)
            or not isinstance(multiply_if.get("multiply"), (int, float))
        ):
            return "effect_key:multiply_if"
    for key in ("scale", "hits_scale", "count_scale"):
        scale = effect.get(key)
        if scale is not None and (not isinstance(scale, Mapping) or scale.get("per") not in SCALE_SOURCES):
            return f"{key}:unsupported"
    for condition in (
        effect.get("condition") or {},
        (effect.get("hits_scale") or {}).get("condition") or {},
        (multiply_if.get("condition") if isinstance(multiply_if, Mapping) else None) or {},
    ):
        for key, value in condition.items():
            if key not in EFFECT_CONDITIONS:
                return f"condition:{key}"
            if key == "target_has_power" and value not in {"Vulnerable", "Weak"}:
                return f"condition:{key}:{value}"
            if key == "melody_played_this_turn":
                tags = value if isinstance(value, list) else [value]
                if not tags or any(t not in MELODY_TAGS for t in tags):
                    return f"condition:{key}"
    if kind == "damage" and (effect.get("target") or "single") not in DAMAGE_TARGETS:
        return f"effect_target:{effect.get('target')}"
    if kind in {"exhaust_other", "discard"} and effect.get("choice") not in {None, "random", "choose"} and not (
        effect.get("choice") == "all" and effect.get("count_scale")
    ):
        # ``all`` needs the count from its source (all cards in hand, every
        # Status card); without one the count would be taken as one card.
        return f"effect_choice:{effect.get('choice')}"
    if kind in {"channel_orb", "orb_passive"}:
        if effect.get("orb") not in ORB_TYPES:
            return f"orb:{effect.get('orb')}"
        if kind == "channel_orb" and effect.get("target") not in {None, "self"}:
            return f"orb_target:{effect.get('target')}"
        if kind == "orb_passive" and effect.get("scope") != "all_of_type":
            return f"orb_scope:{effect.get('scope')}"
    if effect.get("source") not in {None, "osty", "arrow"}:
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


def _card_set_unsupported(effect: Mapping) -> Optional[str]:
    """Why an effect on a set of cards (exhaust, discard, upgrade) is not
    convertible, or ``None``.

    exhaust_other / discard: ``filter`` may only restrict the card type. A
    fixed count then picks from the matching cards in hand (``random`` or
    ``choose``); a count from a source must be a source that counts that
    type only, so the filter is already in the count.

    upgrade_cards: a pile in ``UPGRADE_SCOPES``, a fixed count or ``all``,
    and at most the ``upgradable`` filter every in-fight upgrade applies
    anyway (IsUpgradable). Counts that follow the fight state, conditions
    and other scopes (the played card, a generated card) stay gaps.
    """

    kind = effect.get("kind")
    flt = effect.get("filter")
    if kind == "upgrade_cards":
        if flt is not None and flt != {"upgradable": True}:
            return "upgrade_cards:filter"
        if effect.get("scope") not in UPGRADE_SCOPES:
            return f"upgrade_cards:scope:{effect.get('scope')}"
        if effect.get("choice") not in {None, "random", "choose", "all"}:
            return f"upgrade_cards:choice:{effect.get('choice')}"
        if any(effect.get(key) for key in ("scale", "hits_scale", "count_scale", "condition")):
            return "upgrade_cards:dynamic"
        count = effect.get("count")
        if effect.get("scope") == "all_combat" and count != "all":
            return "upgrade_cards:count"
        if count != "all" and (isinstance(count, bool) or not isinstance(count, (int, float))):
            return "upgrade_cards:count"
        return None
    if flt is None:
        return None
    if not isinstance(flt, Mapping) or set(flt) != {"card_type"} or not isinstance(flt["card_type"], str):
        return "effect_filter:unsupported"
    scale = effect.get("count_scale")
    if scale:
        if not isinstance(scale, Mapping) or TYPED_COUNT_SOURCES.get(scale.get("per")) != flt["card_type"]:
            return "effect_filter:count_scale"
        return None
    count = effect.get("count")
    if effect.get("choice") not in {"random", "choose"} or isinstance(count, bool) or not isinstance(count, (int, float)):
        return "effect_filter:count"
    return None


def _unknowable(fact: Optional[Mapping], entity_key: str) -> tuple[str, ...]:
    """Gaps for the ``unknowable`` notes a reviewed fact carries."""

    if not fact or not fact.get("unknowable"):
        return ()
    return (f"fact_unknowable:{entity_key}",)


def effect_text_only(effect: Mapping) -> Optional[str]:
    """A value written only as text: the structured number is a base case.

    ``amount_text`` / ``hits_text`` / ``count_text`` say the value depends on
    something the structured fields do not carry, unless the matching scale
    is there (a plain remark is written ``note_text``). The effect is still
    scored at its structured value; the caller reports the rest as a gap.

    An effect on a set of cards has no amount: its ``amount_text`` words the
    count, given in full by ``count_scale`` or by ``count: "all"``.
    """

    for text_key, scale_key in VALUE_TEXT_KEYS.items():
        if text_key == "amount_text" and effect.get("kind") in (CARD_SET_EFFECTS | {"dislodge_arrow"}) and "amount" not in effect:
            if effect.get("count") == "all":
                continue
            scale_key = "count_scale"
        if text_key == "amount_text" and (
            effect.get("multiply_if")
            or (effect.get("per_x") and effect.get("times") == "X")
        ):
            continue
        if text_key in effect and not effect.get(scale_key) and effect.get("hits") != "X":
            return f"effect_text_only:{text_key}"
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
    if "multiply_percent_per_stack" in resolved:
        resolved["multiply"] = 1.0 + float(resolved.pop("multiply_percent_per_stack")) * stacks / 100.0
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
    mechanics, gaps, per_stack = [], [], []
    for raw in fact.get("mechanics") or ():
        raw, generated_gaps = _mechanic_cards(repository, raw)
        gaps.extend(generated_gaps)
        source = _next_turn_once(raw) if next_turn else raw
        mechanic = _resolve_stacks(source, stacks)
        if (mechanic.get("effect") or {}).get("kind") == "move_card":
            as_draw, rider_gaps = _move_as_draw(mechanic["effect"])
            if as_draw is not None:
                mechanic = {**mechanic, "effect": as_draw}
                gaps.extend(f"power_mechanic_unsupported:{power_id}:{gap}" for gap in rider_gaps)
        reason = unsupported_reason(mechanic)
        if reason:
            gaps.append(f"power_mechanic_unsupported:{power_id}:{reason}")
        else:
            mechanics.append(mechanic)
            per_stack.append(
                "amount_per_stack" in (source.get("effect") or {})
                or any(key in source for key in ("amount_per_stack", "add_per_stack", "multiply_percent_per_stack"))
            )
    return PowerUse(power_id, duration, stacks, tuple(mechanics), tuple(per_stack)), tuple(gaps)


def _filter_unsupported(
    flt: object,
    card_filter_allowed: bool,
    attack_filter_allowed: bool = False,
    first_of_tag_allowed: bool = False,
) -> Optional[str]:
    if flt is None:
        return None
    if not isinstance(flt, Mapping):
        return "filter:invalid"
    allowed = (CARD_FILTER_KEYS if card_filter_allowed else frozenset()) | {"fight_kind"}
    if first_of_tag_allowed:
        allowed = allowed | {"first_of_tag_this_turn"}
    if attack_filter_allowed:
        allowed = allowed | ATTACK_FILTER_KEYS
        if flt.get("dealer") not in {None, "owner", "owner_or_owner_pet"}:
            return f"filter:dealer:{flt.get('dealer')}"
        if flt.get("powered_attack") is False:
            return "filter:powered_attack"
        if flt.get("from_arrow") not in {None, True}:
            return f"filter:from_arrow:{flt.get('from_arrow')}"
        if flt.get("target_has_power") not in {None, "Impaled"}:
            return f"filter:target_has_power:{flt.get('target_has_power')}"
    if card_filter_allowed:
        if flt.get("owner") not in {None, "self"}:
            return f"filter:owner:{flt.get('owner')}"
    if first_of_tag_allowed:
        if flt.get("first_of_tag_this_turn") not in {None, True}:
            return "filter:first_of_tag_this_turn"
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
        if name == "hp_loss_negate":
            flt = mechanic.get("filter")
            if flt is not None and (not isinstance(flt, Mapping) or set(flt) - {"source_card_not"}):
                return f"filter:{sorted(set(flt) - {'source_card_not'})[0]}" if isinstance(flt, Mapping) else "filter:invalid"
            return None
        card_filter = name not in {"hand_draw", "max_energy"}
        reason = _filter_unsupported(mechanic.get("filter"), card_filter, attack_filter_allowed=(name == "damage"))
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
    if when == "orb_evoked":
        orb_filter = mechanic.get("filter") or {}
        if set(orb_filter) - {"orb"} or orb_filter.get("orb") not in {None, *ORB_TYPES}:
            return "filter:orb"
        reason = None
    elif when in {"arrow_shot", "arrow_impaled", "arrow_dislodged"}:
        arrow_filter = mechanic.get("filter") or {}
        if not isinstance(arrow_filter, Mapping) or set(arrow_filter) - {"owner_only", "from_arrow"}:
            return "filter:arrow"
        reason = None
    else:
        reason = _filter_unsupported(
            mechanic.get("filter"), when in {"card_played", "card_drawn"},
            when in {"damage_dealt", "damage_received"},
            first_of_tag_allowed=(when == "card_played"),
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
    effect = mechanic.get("effect")
    if isinstance(effect, Mapping) and effect.get("kind") in {"exhaust_other", "discard"} and effect.get("filter"):
        # Mechanic effects are not resolved per event (``_apply_events``): a
        # filtered pick from hand would count as always finding a card.
        return "effect_filter:mechanic"
    if isinstance(effect, Mapping):
        for dynamic_key in ("scale", "hits_scale", "count_scale", "multiply_if", "condition"):
            if effect.get(dynamic_key):
                return f"effect_{dynamic_key}:mechanic"
    return effect_unsupported(effect) or effect_text_only(effect)


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
    per = str(m.get("per") or "")
    if m.get("duration") == "combat" and isinstance(m.get("amount"), (int, float)) and per.startswith("each play of this card"):
        # The recorded IL call carries the sign (AddThisCombat(1) / (-1)).
        if "AddThisCombat(1)" in per:
            return ("per_own_play", abs(float(m["amount"])))
        if "AddThisCombat(-1)" in per:
            return ("per_own_play", -abs(float(m["amount"])))
    if m.get("kind") == "set_this_turn" and m.get("when") == "osty_attacked_this_turn" and m.get("value") == 0:
        return ("free_if_osty_attacked",)
    if (m.get("duration") == "until_played" and isinstance(m.get("amount"), (int, float))
            and per.startswith("Status card generated")):
        return ("per_status_generated", float(m["amount"]))
    if isinstance(add, (int, float)) and m.get("when") == "in_combat" and per == "melodies_this_turn":
        return ("per_melodies_this_turn", float(add))
    cond = m.get("condition")
    if (isinstance(add, (int, float)) and m.get("when") == "in_combat"
            and isinstance(cond, Mapping) and set(cond) == {"melody_played_this_turn"}):
        raw_tags = cond["melody_played_this_turn"]
        tags = tuple(raw_tags if isinstance(raw_tags, list) else [raw_tags])
        if tags and all(t in MELODY_TAGS for t in tags):
            return ("if_melody_played_this_turn", float(add), tags)
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


def _returns_after_play(effect: Mapping) -> Optional[str]:
    """``hand`` / ``top`` for a card whose play result goes back to the hand
    or onto the top of the draw pile, else ``None``.

    The fact's text explains the redirection (``GetResultLocationForCardPlay``);
    the structure (the card itself, from play, the destination) is complete.
    A random position or a move to another player is not this case.
    """

    if effect.get("card") != "self" or effect.get("from") != "play":
        return None
    if set(k for k in effect if not k.endswith("_text")) - {"kind", "card", "from", "destination", "position"}:
        return None
    destination = effect.get("destination")
    if destination == "hand" and effect.get("position") in (None, "top"):
        return "hand"
    if destination == "draw" and effect.get("position") == "top":
        return "top"
    return None


def _move_as_draw(effect: Mapping) -> tuple[Optional[dict], tuple[str, ...]]:
    """A power's move of cards from the draw or discard pile into the hand,
    as a draw of that many cards (Aggression: a random Attack from the
    discard pile each turn). Riders the engine does not value are gaps."""

    destination = effect.get("destination") or effect.get("to")
    origin = effect.get("from") or effect.get("source")
    amount = effect.get("amount", effect.get("count"))
    if destination != "hand" or origin not in {"draw", "discard"}:
        return None, ()
    if effect.get("choice") not in {None, "choose", "random"}:
        return None, ()
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return None, ()
    known = {"kind", "from", "source", "to", "destination", "amount", "count", "choice", "filter", "upgrade_moved"}
    if set(k for k in effect if not k.endswith("_text")) - known:
        return None, ()
    if effect.get("filter"):
        # Only some cards can be moved (Aggression: an Attack from the discard
        # pile, empty on turn 1): not a plain draw.
        return None, ()
    gaps = ("move_rider:upgrade_moved",) if effect.get("upgrade_moved") else ()
    return {"kind": "draw", "amount": float(amount)}, gaps


def _move_to_hand(effect: Mapping, upgraded: bool) -> Optional[dict]:
    """A chosen move from the draw or discard pile into the hand, as a draw.

    The choice makes it worth at least a random draw of the same count, so
    this is a lower bound. Moves onto the draw pile, of the card itself, or
    of filtered/unbounded sets are not converted.
    """

    if set(k for k in effect if not k.endswith("_text")) - MOVE_TO_HAND_KEYS:
        return None
    if effect_text_only(effect):
        return None  # the count depends on something the structure does not carry
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
    amount = float(count) + (
        float(effect.get("count_upgrade") or 0.0) + float(effect.get("upgrade") or 0.0) if upgraded else 0.0
    )
    return Generated(profile, amount, str(effect["destination"])), tuple(
        f"generated:{card_id}:{gap}" for gap in gaps
    )


def _transformed(
    repository: RelationalRepository,
    effect: Mapping,
    upgraded: bool,
    card_id: str,
    depth: int,
) -> tuple[Optional[Generated], tuple[str, ...]]:
    """Resolve one ``transform_cards`` effect to the card it turns cards into.

    CardCmd.Transform puts a newly created card in the place of the chosen
    one for the rest of the fight. It is scored as copies of the result card
    made where the replaced cards sit (the hand: playable this turn; the
    draw pile: drawn later), like ``add_card``.

    Approximation: the replaced cards are not taken out of the deck; the
    results that stay in the fight dilute every card's draws instead, which
    amounts to each result replacing an average card. The player gives up
    the card they need least (a Status, a Curse, a card the turn cannot pay
    for), so this leans low. A result that exhausts (all the Minion cards)
    leaves the fight after its play and is not diluting; its replaced card is
    simply gone, and the thinner deck that follows is not counted, nor are
    the removed cards taken out of other estimates (Status cards in combat).
    Every card is taken to be transformable (``IsTransformable`` is not
    recorded in the facts).

    How many cards one play replaces (estimates of the engine, ``_estimate``):

    * choose N in hand: N, at most the other cards of the turn's hand;
    * choose N in the draw pile: N, at most the average draw pile;
    * all Status cards in hand (Compact): the Status cards drawn per turn;
    * any number in hand (Guards): the cards the turn leaves unplayed. Each
      of them would do nothing this turn, so a player gives them up for a
      free result card at least; playable cards worse than the result are
      not counted, which leans low.

    The result is upgraded when this card is and the fact says it follows.
    Random results (Entropy), whole card types the turn would have played
    (Primal Force: every Attack in hand) and other shapes stay gaps.
    """

    def gap(reason: str) -> tuple[None, tuple[str, ...]]:
        return None, (f"card_effect_unsupported:{card_id}:transform_cards:{reason}",)

    extra = {key for key in set(effect) - TRANSFORM_KEYS if not key.endswith("_text")}
    if extra:
        return gap(f"key:{sorted(extra)[0]}")
    into, scope = str(effect.get("into") or ""), effect.get("scope")
    choice, count, flt = effect.get("choice"), effect.get("count"), effect.get("filter") or {}
    fixed = isinstance(count, (int, float)) and not isinstance(count, bool)
    if not into or into == "random":
        return gap(f"into:{into or 'missing'}")
    if choice == "choose" and fixed and not flt and scope in {"hand", "draw"}:
        targets = "other_cards_in_hand" if scope == "hand" else "draw_pile"
    elif choice == "choose" and count == "any" and not flt and scope == "hand":
        targets = "unplayed_cards_in_hand"
    elif choice == "all" and count is None and flt == {"card_type": "Status"} and scope == "hand":
        targets = "status_cards_in_hand"
    else:
        shape = f"filter:{flt.get('card_type')}" if flt else f"{choice or count}"
        return gap(f"{scope}:{shape}")
    if depth > 0:
        return gap("generated_chain_truncated")
    follows = effect.get("upgraded_if_self_upgraded") or effect.get("upgraded_result_if_upgraded")
    result_id = into if into.isupper() else class_to_entity_id(into)
    profile, gaps = card_profile(repository, result_id, 1 if follows and upgraded else 0, _depth=depth + 1)
    if profile is None:
        return None, (f"generated_card_fact_missing:{card_id}:{result_id}", *gaps)
    return Generated(profile, float(count) if fixed else None, str(scope), targets), tuple(
        f"generated:{card_id}:{g}" for g in gaps
    )


def _keeps_exhausted_damage(effect: Mapping) -> bool:
    """The reviewed ``then`` of an exhaust says this card's own Damage grows
    by the exhausted card's damage for its later plays.

    Thrash (OnPlay): after its attack it exhausts a random Attack from the
    hand and adds that card's damage (through Hook.ModifyDamage) to its own
    Damage.BaseValue for the rest of the fight. Any other ``then`` stays a
    gap (``effect_key:then``).
    """

    then = str(effect.get("then") or "")
    return "Damage base value increases by the exhausted card's damage" in then and "later plays" in then


def _upgrade_branch(effect: Mapping, upgraded: bool) -> dict:
    """An upgrade effect whose upgraded card takes another branch, recorded
    as ``upgraded`` overrides (Armaments: ``if IsUpgraded`` every upgradable
    card in hand, else one chosen card)."""

    branch = {k: v for k, v in effect.items() if k != "upgraded"}
    if upgraded:
        branch.update(effect["upgraded"])
    return branch


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
    returns: Optional[str] = None
    growth: Optional[Mapping] = None    # an exhaust whose card's damage this card keeps
    # What the review registered as impossible to know is disclosed with the
    # card, for itself and for every power it applies.
    gaps.extend(_unknowable(fact, f"cards:{card_id}"))
    for effect in fact.get("effects") or ():
        if isinstance(effect, Mapping) and effect.get("kind") == "apply_power" and effect.get("power"):
            power_id = class_to_entity_id(str(effect["power"]))
            gaps.extend(_unknowable(repository.find_assembly_fact("powers", power_id), f"powers:{power_id}"))
    for effect in fact.get("effects") or ():
        weakened = _enemy_strength_down(repository, effect, upgraded) if isinstance(effect, Mapping) else None
        if weakened is not None:
            effects.append(weakened)
            continue
        if isinstance(effect, Mapping) and effect.get("kind") == "move_card":
            back = _returns_after_play(effect)
            if back == "hand":
                # Back in the hand it is still discarded at the end of the
                # turn: the gain is replays within the turn, not modeled.
                gaps.append(f"card_effect_unsupported:{card_id}:returns_to_hand_replays")
                continue
            if back is not None:
                returns = back
                continue
            moved = _move_to_hand(effect, upgraded)
            if moved is not None:
                effects.append(moved)
                continue
        if isinstance(effect, Mapping) and effect.get("kind") in {"add_card", "transform_cards"}:
            resolve = _generated if effect["kind"] == "add_card" else _transformed
            generated, generated_gaps = resolve(repository, effect, upgraded, card_id, _depth)
            gaps.extend(generated_gaps)
            if generated is not None:
                generates.append(generated)
            continue
        power = effect.get("power")
        if (
            effect.get("kind") == "apply_power" and effect.get("target") == "self"
            and power not in SELF_POWERS
        ):
            extra = sorted(
                key for key in effect
                if key not in POWER_EFFECT_KEYS and not key.endswith("_text") or key in VALUE_TEXT_KEYS
            )
            if extra:
                gaps.append(f"card_effect_unsupported:{card_id}:power_effect_key:{extra[0]}")
                continue
            use, use_gaps = power_use(repository, str(power), _amount(effect, upgraded))
            gaps.extend(use_gaps)
            if use is not None:
                powers.append(use)
            continue
        if effect.get("kind") == "exhaust_other" and _keeps_exhausted_damage(effect):
            # Thrash: the ``then`` is valued on the damage below.
            effect = growth = {k: v for k, v in effect.items() if k != "then"}
        if effect.get("kind") == "upgrade_cards" and isinstance(effect.get("upgraded"), Mapping):
            effect = _upgrade_branch(effect, upgraded)
        reason = effect_unsupported({k: v for k, v in effect.items() if k != "count_upgrade"})
        if reason:
            gaps.append(f"card_effect_unsupported:{card_id}:{reason}")
            continue
        partial = effect_text_only(effect)
        if partial:
            # Counted at its structured base value; the rest is disclosed.
            gaps.append(f"card_effect_unsupported:{card_id}:{partial}")
        resolved = {**effect, "amount": _amount(effect, upgraded)}
        if "bonus_if_upgraded" in effect:
            bonus = resolved.pop("bonus_if_upgraded")
            if not effect.get("per_x") or isinstance(bonus, bool) or not isinstance(bonus, (int, float)):
                gaps.append(f"card_effect_unsupported:{card_id}:effect_value:bonus_if_upgraded")
                continue
            resolved["x_bonus"] = float(bonus) if upgraded else 0.0
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
        if effect is growth:
            resolved["grows_own_damage"] = True
        effects.append(resolved)
    if any(e.get("grows_own_damage") for e in effects):
        damage = [e for e in effects if e.get("kind") == "damage"]
        if len(damage) == 1 and "scale" not in damage[0]:
            # The kept damage is added to Damage.BaseValue: every hit of the
            # card's later plays (see ``_Engine._exhausted_damage``).
            damage[0]["scale"] = {"per": "own_exhausted_damage", "factor": 1.0}
        else:
            gaps.append(f"card_effect_unsupported:{card_id}:exhaust_growth")
    cost_rule = _cost_rule(fact.get("cost_modifier")) if fact.get("cost_modifier") else None
    if fact.get("cost_modifier") and cost_rule is None:
        gaps.append(f"card_effect_unsupported:{card_id}:cost_modifier")
    if any(m.get("when") not in RUN_ONLY_WHEN for m in fact.get("mechanics") or ()):
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
    if upgraded:
        changes = fact.get("upgrade_keywords") or {}
        keywords -= set(changes.get("remove") or ())
        keywords |= set(changes.get("add") or ())
    for keyword in sorted(keywords - MODELED_KEYWORDS):
        gaps.append(f"card_keyword_unmodeled:{card_id}:{keyword}")
    upgraded_profile = None
    if not upgraded and _depth == 0 and str(fact.get("card_type") or "") not in NOT_UPGRADABLE_TYPES:
        # Used when an in-fight upgrade picks this card. What only the
        # upgraded card cannot convert is disclosed with it.
        upgraded_profile, upgraded_gaps = card_profile(repository, card_id, 1)
        gaps.extend(f"upgraded:{gap}" for gap in upgraded_gaps if gap not in gaps)
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
        retain="Retain" in keywords,
        innate="Innate" in keywords,
        returns=None if "Exhaust" in keywords else returns,
        effects=tuple(effects),
        powers=tuple(powers),
        generates=tuple(generates),
        star_cost=star_cost,
        cost_rule=cost_rule,
        upgraded_profile=upgraded_profile,
    ), tuple(dict.fromkeys(gaps))  # a card may repeat one unsupported effect


def enchant(profile: CardProfile, fact: Optional[Mapping], amount: float) -> tuple[CardProfile, tuple[str, ...]]:
    """Apply an enchantment that changes this card's own numbers.

    Supported: flat or multiplied damage/block on the enchanted card and
    effects added to each of its plays. Once-per-combat riders, replays and
    keyword changes are returned as a gap.
    """

    if fact is None or fact.get("status") != "converted":
        return profile, (f"enchantment_fact_missing:{profile.card_id}",)
    effects = [dict(e) for e in profile.effects]
    gaps = []
    if fact.get("adds_keywords") or fact.get("removes_keywords"):
        gaps.append(f"enchantment_unsupported:{profile.card_id}:keywords")
    for mechanic in fact.get("mechanics") or ():
        m = {k: v for k, v in mechanic.items() if v is not None and k not in {"scope", "applies_to"}}
        if mechanic.get("scope") != "enchanted_card" or m.get("limit"):
            gaps.append(f"enchantment_unsupported:{profile.card_id}:{m.get('modifier') or m.get('when')}")
            continue
        name = m.get("modifier")
        if name in {"damage", "block"} and set(m) <= {"modifier", "add", "add_per_stack", "multiply", "filter"} \
                and set(m.get("filter") or {}) <= {"powered_attack"}:
            add = float(m.get("add") or 0.0) + float(m.get("add_per_stack") or 0.0) * amount
            for effect in effects:
                if effect.get("kind") == name and effect.get("source") != "osty":
                    effect["amount"] = (float(effect.get("amount") or 0.0) + add) * float(m.get("multiply") or 1.0)
            continue
        effect = m.get("effect") or {}
        if m.get("when") == "card_played" and (m.get("filter") or {}) == {"self": True} \
                and set(m) <= {"when", "filter", "effect"}:
            added = {k: v for k, v in effect.items() if k not in {"amount_per_stack", "upgrade"}}
            added["amount"] = float(effect.get("amount") or 0.0) + float(effect.get("amount_per_stack") or 0.0) * amount
            if effect_unsupported(added) is None:
                effects.append(added)
                continue
        gaps.append(f"enchantment_unsupported:{profile.card_id}:{name or m.get('when')}")
    # An in-fight upgrade keeps the enchantment on the card.
    upgraded = enchant(profile.upgraded_profile, fact, amount)[0] if profile.upgraded_profile else None
    return replace(profile, effects=tuple(effects), upgraded_profile=upgraded), tuple(gaps)


def relic_profile(
    repository: RelationalRepository,
    relic_id: str,
) -> tuple[Optional[RelicProfile], tuple[str, ...]]:
    fact = repository.find_assembly_fact("relics", relic_id)
    if fact is None or fact.get("status") != "converted":
        return None, (f"relic_fact_missing:{relic_id}",)
    supported, gaps = [], list(_unknowable(fact, f"relics:{relic_id}"))
    for mechanic in fact.get("mechanics") or ():
        # SongBow (starter relic) uses AfterRoomEntered gated on CombatRoom
        # (all three combat room types), firing once on entering every fight.
        if isinstance(mechanic, Mapping) and mechanic.get("when") == "room_entered":
            flt = mechanic.get("filter")
            rt = flt.get("room_type") if isinstance(flt, Mapping) and set(flt) == {"room_type"} else None
            if rt == "CombatRoom" or (isinstance(rt, list) and set(rt) == {"Monster", "Elite", "Boss"}):
                mechanic = {k: v for k, v in mechanic.items() if k != "filter"}
                mechanic["when"] = "combat_start"
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
    for mechanic in fact.get("mechanics") or ():
        effect = mechanic.get("effect") if isinstance(mechanic, Mapping) else None
        if isinstance(effect, Mapping) and effect.get("kind") == "apply_power" and effect.get("power"):
            power_id = class_to_entity_id(str(effect["power"]))
            gaps.extend(_unknowable(repository.find_assembly_fact("powers", power_id), f"powers:{power_id}"))
    return RelicProfile(relic_id, str(fact.get("category") or ""), tuple(supported)), tuple(dict.fromkeys(gaps))


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


def _upgrades_in_fight(cards: Sequence[CardProfile], relics: Sequence[RelicProfile]) -> bool:
    """Whether a card, a card it creates, a power it applies or a relic
    upgrades cards during a fight."""

    effects = [e for c in cards for e in c.effects] + [
        e for c in cards for g in c.generates for e in g.profile.effects
    ] + [
        m.get("effect") or {} for c in cards for use in c.powers for m in use.mechanics
    ] + [m.get("effect") or {} for relic in relics for m in relic.mechanics]
    return any(e.get("kind") == "upgrade_cards" for e in effects)


def _expected_taken(count: float, mean: float) -> float:
    """E[min(count, X)] for X ~ Poisson(mean): cards an effect taking up to
    ``count`` matching cards finds, when ``mean`` match on average."""

    return sum(_poisson_at_least(mean, k) for k in range(1, int(count) + 1))


def _held_share(rate: float, turns: float, delay: float = 0.0) -> float:
    """Fight-average share of a pool converted at ``rate`` (share per turn).

    What is converted on one turn pays off from the next turn on, ``delay``
    turns later still: on turn t the share is min(1, rate * (t - 1 - delay)).
    Averaged over the fight (continuous in t); with no delay and no cap this
    is the usual half-built-up ``rate * (turns - 1) / 2``.
    """

    span = max(0.0, turns - 1.0)
    active = span - delay
    if rate <= 0 or active <= 0:
        return 0.0
    if rate * active <= 1.0:
        return rate * active * active / 2.0 / span
    return (active - 0.5 / rate) / span


_MISSING = object()
_DUP_OWNER = object()


def _has_zero_x_effect(card: CardProfile) -> bool:
    """Whether an X-cost card still yields value when played at X=0."""

    if card.powers:
        return True
    for generated in card.generates:
        if generated.count is not None and generated.count > 0:
            return True
    for effect in card.effects:
        if effect.get("hits_x"):
            continue
        if effect.get("per_x"):
            if float(effect.get("x_bonus") or 0.0) != 0.0:
                return True
            continue
        count_scale = effect.get("count_scale")
        if isinstance(count_scale, Mapping) and count_scale.get("per") == "x_value":
            base = float(count_scale.get("base") or 0.0) + (
                float(count_scale.get("base_upgrade") or 0.0) if card.upgraded else 0.0
            )
            if base > 0.0:
                return True
            continue
        return True
    return False


def _poisson_cost_states(base_c: float, delta: float, lam: float) -> list[tuple[float, float]]:
    """Discrete (probability, cost) distribution for cost = max(0, base_c + delta * K), K ~ Poisson(lam)."""

    if lam <= 1e-9 or delta == 0.0:
        return [(1.0, base_c)]
    if delta < 0.0 and base_c > 0.0:
        k_max = min(20, max(1, int(math.ceil(base_c / (-delta)))))
    else:
        k_max = min(20, max(6, int(math.ceil(lam + 4.0 * math.sqrt(lam) + 2.0))))
    states: list[tuple[float, float]] = []
    prob = math.exp(-lam)
    covered = 0.0
    for k in range(k_max):
        if prob > 0.0:
            states.append((prob, max(0.0, base_c + delta * k)))
            covered += prob
        prob *= lam / float(k + 1)
    tail = max(0.0, 1.0 - covered)
    if tail > 0.0:
        states.append((tail, max(0.0, base_c + delta * k_max)))
    return states or [(1.0, base_c)]


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
        # In-fight upgrades: an upgradable deck card's upgraded profile is a
        # virtual card taking the upgraded share of its draws (``run``).
        self.upgradable = [
            c for c in self.known if c.upgraded_profile is not None
        ] if _upgrades_in_fight(self.known, relics) else []
        # Unknown cards take a deck slot and may take a random upgrade too:
        # counted as targets that gain nothing from it.
        self.upgrade_pool = float(len(self.upgradable) + len(cards) - len(self.known))
        self.upgraded_base = {id(c.upgraded_profile): c for c in self.upgradable}
        upgraded = list({id(c.upgraded_profile): c.upgraded_profile for c in self.upgradable}.values())
        owners = [*self.known, *upgraded]
        # Generated cards are extra card instances whose play rate follows
        # their source's plays (previous pass); kept stable for ``id`` keys.
        self.sources = [(c, g) for c in owners for g in c.generates]
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
            m for card in owners for use in card.powers for m in use.mechanics
        ]
        for mechanic in mechanic_sources:
            effect = mechanic.get("effect") or {}
            if effect.get("kind") == "generate":
                profile = effect["profile"]
                if id(profile) not in self.mechanic_made:
                    self.mechanic_made[id(profile)] = (profile, effect["destination"])
                    self.virtual.append(profile)
        self.virtual.extend(upgraded)
        self.cards = [*self.known, *self.virtual]
        # Copies share one profile: (profile, number of copies), for sums
        # that are linear in the copies.
        grouped: Dict[int, list] = {}
        for card in self.cards:
            grouped.setdefault(id(card), [card, 0])[1] += 1
        self.copies = [(card, float(n)) for card, n in grouped.values()]
        # Cards that exhaust every Status card in combat (Flak Cannon).
        self.status_clearers = {
            id(c) for c in self.cards if any(
                e.get("kind") == "exhaust_other" and e.get("choice") == "all"
                and (e.get("filter") or {}).get("card_type") == "Status"
                for e in c.effects
            )
        }
        self.status_pool = 0.0          # Status cards of a whole fight (``_estimate``)
        self.status_clear_plays = 0.0   # plays per turn of the cards above
        self.pass_size = float(self.size)   # deck plus piled copies (this pass)
        self.pass_drawn = 0.0               # cards drawn per turn (this pass)
        self.situation = situation
        self.params = params
        # (mechanic, weight, owner): relics have owner None; powers carry their
        # source card (or _DUP_OWNER for extra identical copies) so single-turn
        # energy peaks deduplicate copies and in-fight upgrades.
        self.relic_mechanics = [(m, 1.0, None) for relic in relics for m in relic.mechanics]
        self.estimates: Dict[str, float] = {}
        self.rates: Dict[int, float] = {}
        self.x_value = 0.0
        self.card_x: Dict[int, float] = {}
        self.play_totals: Dict[str, float] = {"Attack": 0.0, "Skill": 0.0, "OstyAttack": 0.0}
        self.status_made = 0.0   # Status cards created for the player per turn
        self.damage_dealt = 0.0  # damage per turn over all enemies (previous pass)
        self.supply = 0.0        # energy per turn (this pass)
        self.per_turn = 0.0      # chance a deck card is drawn on a turn (this pass)
        self.energy_spent = 0.0  # energy the turn's plays use (this pass)
        self.orb_evokes: Dict[str, float] = {}   # evokes per turn by orb (previous pass)
        self.arrow_avail = 1.0 if situation.arrow_slots > 0 else 0.0
        self.impaled_prob = 0.0
        self.arrow_shots = 0.0
        self.arrow_dislodges = 0.0
        self.arrow_slots_now = situation.arrow_slots
        self.melody_plays: Dict[str, float] = {t: 0.0 for t in MELODY_TAGS}
        self.melody_source_plays: Mapping[int, float] = {}
        self.known_set = {id(c) for c in self.known}
        self.deck_counts = {
            "OstyAttack": float(sum(1 for c in self.known if "OstyAttack" in c.tags)),
            "star_cost": float(sum(1 for c in self.known if c.star_cost is not None)),
            "melody_cards": float(sum(1 for c in self.known if c.tags & set(MELODY_TAGS))),
        }
        self._cost_cache: Dict[int, Optional[float]] = {}
        self.last_plays: Dict[int, float] = {}
        self._set_mechanics(self.relic_mechanics)

    def _set_mechanics(self, weighted: Sequence[tuple]) -> None:
        self.modifiers = [(m, w) for m, w, _ in weighted if "modifier" in m]
        self.events = [(m, w, owner) for m, w, owner in weighted if "modifier" not in m]

    def _activate(self, plays: Mapping[int, float], drawn: float) -> None:
        """Weight each played card's powers by their expected active share."""

        turns = self.situation.fight_turns
        first_turn = min(turns, (self.size / max(drawn, 1e-9) + 1.0) / 2.0)
        weighted = list(self.relic_mechanics)
        seen_cards: set[int] = set()
        for card in self.cards:
            first_copy = id(card) not in seen_cards
            seen_cards.add(id(card))
            owner = card if first_copy else _DUP_OWNER
            for use in card.powers:
                rate = plays[id(card)]
                if use.duration == "this_turn":
                    weight = rate  # active on the turns the card is played
                else:
                    played = min(1.0, rate * turns)  # chance it is played this fight
                    # An Innate card is in the opening hand: in play from turn 1.
                    start = 1.0 if card.innate else first_turn
                    weight = played * max(0.0, turns - start + 1.0) / turns
                for index, mechanic in enumerate(use.mechanics):
                    follows_stacks = index < len(use.per_stack) and use.per_stack[index]
                    if use.duration == "turns_decrement" and follows_stacks:
                        # Stacks drop by one per turn and the amount follows
                        # them: spread the total over the fight.
                        mechanic = dict(mechanic)
                        effect = dict(mechanic.get("effect") or {})
                        effect["amount"] = _poison_dealt(float(effect.get("amount") or 0.0), turns) / turns
                        mechanic["effect"] = effect
                        weight = min(1.0, rate * turns)
                    elif use.duration == "turns_decrement":
                        # A fixed effect or a modifier: it is simply there for
                        # as many turns as it has stacks.
                        weight = min(1.0, rate * min(max(use.stacks, 1.0), turns))
                    weighted.append((mechanic, weight, owner))
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

    def _base_cost(self, card: CardProfile) -> Optional[float]:
        if card.x_cost or card.cost is None:
            return None
        delta = sum(
            float(m.get("add") or 0.0) * w for m, w in self.modifiers
            if m["modifier"] == "energy_cost" and self._matches(card, m.get("filter"), None)
        )
        return max(0.0, card.cost + delta)

    @staticmethod
    def _afford_for_cost(
        cost: float,
        base_energy: float,
        extra_energy: float,
        max_turn_extra: float,
        deficit: float,
        max_turn_loss: float,
    ) -> float:
        if cost <= base_energy:
            if deficit > 0.0 and cost > base_energy - max_turn_loss + 1e-9:
                need_loss = max(1.0, base_energy - cost + 1.0)
                return max(0.0, 1.0 - deficit / need_loss)
            return 1.0
        if extra_energy <= 0.0 or cost > base_energy + max_turn_extra + 1e-9:
            return 0.0
        return min(1.0, extra_energy / (cost - base_energy))

    def _cost_states(
        self,
        card: CardProfile,
        afford_fn,
    ) -> list[tuple[float, float]]:
        """Discrete (probability, cost) states for a non-X playable card."""

        base_c = self._base_cost(card)
        if base_c is None:
            return []
        rule = card.cost_rule
        if not rule:
            return [(1.0, base_c)]
        turns = self.situation.fight_turns
        rate = self.rates.get(id(card), 0.0)
        draws = max(1.0, rate * turns)
        if rule[0] == "after_first":
            if afford_fn(base_c) <= 0.0:
                return [(1.0, base_c)]
            if draws <= 1.0:
                return [(1.0, base_c)]
            return [(1.0 / draws, base_c), ((draws - 1.0) / draws, max(0.0, rule[1]))]
        if rule[0] == "per_own_play":
            # Cost changes at the end of OnPlay, so once a play cost is unplayable
            # the remaining draws stay stuck at that unplayable cost.
            full = int(math.floor(draws))
            rem = draws - full
            states: list[tuple[float, float]] = []
            remaining = 1.0
            for k in range(full + (1 if rem > 1e-9 else 0)):
                w = (1.0 / draws) if k < full else (rem / draws)
                c_k = max(0.0, base_c + rule[1] * k)
                if afford_fn(c_k) <= 0.0:
                    states.append((remaining, c_k))
                    remaining = 0.0
                    break
                states.append((w, c_k))
                remaining = max(0.0, remaining - w)
            if remaining > 1e-9 and states:
                states.append((remaining, states[-1][1]))
            return states or [(1.0, base_c)]
        if rule[0] == "per_own_draw":
            # AfterCardDrawn fires when the card is drawn, before it is played
            # (draw #1 already gets 1 step of delta, draw #2 gets 2, etc.).
            full = int(math.floor(draws))
            rem = draws - full
            states = [
                (1.0 / draws, max(0.0, base_c + rule[1] * (k + 1)))
                for k in range(full)
            ]
            if rem > 1e-9:
                states.append((rem / draws, max(0.0, base_c + rule[1] * (full + 1))))
            return states or [(1.0, max(0.0, base_c + rule[1]))]
        if rule[0] == "free_if_osty_attacked":
            others = self.play_totals["OstyAttack"] - (
                self.last_plays.get(id(card), 0.0) if "OstyAttack" in card.tags else 0.0
            )
            p_free = self.estimates.get("osty_alive", 0.0) * (1.0 - math.exp(-max(0.0, others)))
            return [(w, c) for w, c in ((p_free, 0.0), (1.0 - p_free, base_c)) if w > 0.0]
        if rule[0] == "per_status_generated":
            between = self.status_made / max(rate, 1.0 / turns)
            return _poisson_cost_states(base_c, rule[1], max(0.0, between))
        if rule[0] == "per_type_this_turn":
            others = sum(
                n * self.last_plays.get(id(c), 0.0) for c, n in self.copies
                if c.card_type == rule[2]
            ) - (self.last_plays.get(id(card), 0.0) if card.card_type == rule[2] else 0.0)
            return _poisson_cost_states(base_c, rule[1], max(0.0, others / 2.0))
        if rule[0] == "per_melodies_this_turn":
            probs = [1.0 - math.exp(-self._other_melody_plays(t, card) / 2.0) for t in MELODY_TAGS]
            dist = [1.0]
            for p in probs:
                nxt = [0.0] * (len(dist) + 1)
                for idx, w in enumerate(dist):
                    nxt[idx] += w * (1.0 - p)
                    nxt[idx + 1] += w * p
                dist = nxt
            return [(w, max(0.0, base_c + rule[1] * m)) for m, w in enumerate(dist) if w > 0.0]
        if rule[0] == "if_melody_played_this_turn":
            prob = self._melody_any_prob(rule[2], card)
            reduced = max(0.0, base_c + rule[1])
            return [(w, c) for w, c in ((prob, reduced), (1.0 - prob, base_c)) if w > 0.0]
        return [(1.0, base_c)]

    def _cost_profile(
        self,
        card: CardProfile,
        base_energy: float,
        extra_energy: float,
        max_turn_extra: float,
        deficit: float,
        max_turn_loss: float,
    ) -> tuple[float, float, float, float, Optional[float]]:
        """Return (free_share, paid_afford, paid_energy_per_draw, brick_cost, effective_cost)."""

        if card.x_cost or card.cost is None:
            return 0.0, 0.0, 0.0, 0.0, None
        afford_fn = lambda c: self._afford_for_cost(
            c, base_energy, extra_energy, max_turn_extra, deficit, max_turn_loss
        )
        states = self._cost_states(card, afford_fn)
        free_share = paid_afford = paid_energy = brick_cost = mean_cost = 0.0
        for weight, state_cost in states:
            mean_cost += weight * state_cost
            if state_cost <= 0.0:
                free_share += weight
                continue
            a = afford_fn(state_cost)
            paid_afford += weight * a
            paid_energy += weight * a * state_cost
            brick_cost += weight * (1.0 - a) * state_cost
        total_playable = free_share + paid_afford
        eff_cost = (paid_energy / total_playable) if total_playable > 1e-9 else mean_cost
        return free_share, paid_afford, paid_energy, brick_cost, eff_cost

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
        states = self._cost_states(card, lambda _c: 1.0)
        if not states:
            return None
        return sum(w * c for w, c in states)

    def _other_melody_plays(self, tag: str, card: Optional[CardProfile]) -> float:
        own = self.melody_source_plays.get(id(card), 0.0) if (card is not None and tag in card.tags) else 0.0
        return max(0.0, self.melody_plays.get(tag, 0.0) - own)

    def _melodies_for_card(self, card: Optional[CardProfile]) -> float:
        return sum(1.0 - math.exp(-self._other_melody_plays(t, card) / 2.0) for t in MELODY_TAGS)

    def _melody_any_prob(self, tags: Sequence[str], card: Optional[CardProfile]) -> float:
        p_none = 1.0
        for tag in tags:
            p_none *= math.exp(-self._other_melody_plays(tag, card) / 2.0)
        return 1.0 - p_none

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
            paid = self.card_x.get(id(card), x_value) if card.x_cost else self._cost(card)
            if paid is None or paid < float(flt["min_cost"]):
                return False
        return True

    def _made(self, generated: Generated) -> float:
        """Copies one play of the source makes: the printed count, or for a
        transform the cards it finds to replace (previous pass), at most its
        count."""

        if generated.targets is None:
            return float(generated.count or 0.0)
        found = self.estimates.get(generated.targets, 0.0)
        return found if generated.count is None else min(generated.count, found)

    def run(self) -> tuple[Dict[str, float], tuple[str, ...]]:
        params, turns = self.params, self.situation.fight_turns
        extra_draw = extra_energy = block_estimate = star_supply = 0.0
        max_turn_extra = max_turn_loss = 0.0
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
            # half built up, they dilute every other card's draws. A
            # transform's results stay too (wherever they are made, unless
            # they exhaust); the replaced cards are still counted at full
            # value, so the dilution is what stands for losing them: each
            # result takes the place of an average card.
            piled = sum(
                previous_plays.get(id(c), 0.0) * g.count * elapsed
                for c, g in self.sources if g.destination != "hand" and g.targets is None
            ) + sum(
                previous_plays.get(id(c), 0.0) * self._made(g) * elapsed
                for c, g in self.sources if g.targets is not None and not g.profile.once_per_fight
            ) + sum(
                previous.generated.get(key, 0.0) * elapsed
                for key, (_, destination) in self.mechanic_made.items() if destination != "hand"
            ) + self.situation.status_cards * elapsed
            if previous.status_exhausts > 0:
                piled -= self._status_removed(previous, previous_plays, elapsed)
            size = self.size + piled
            drawn = min(size, hand + (0.0 if no_draw else extra_draw))
            per_turn = min(1.0, drawn / size)
            self.per_turn = per_turn
            self.pass_size, self.pass_drawn = size, drawn
            supply = max(0.0, base_energy + extra_energy)
            self.supply = supply
            rates = {
                id(c): min(per_turn, 1.0 / turns) if c.once_per_fight else per_turn
                for c in self.known
            }
            for c in self.known:
                if c.innate:
                    # Always in the opening hand: a one-off card is certain to
                    # be seen, any other gains the first-turn draws it lacked.
                    rates[id(c)] = (
                        1.0 / turns if c.once_per_fight
                        else min(1.0, per_turn + (1.0 - per_turn) / turns)
                    )
            for c in {id(card): card for card in self.known}.values():   # copies share one rate
                if c.returns and not c.once_per_fight:
                    # Seen by turn t with chance 1-(1-p)^t, then every turn.
                    # (A card put back on top also takes one of the turn's
                    # draws; that small cost to the other cards is left out.)
                    whole = max(1, int(round(turns)))
                    seen = sum(1.0 - (1.0 - rates[id(c)]) ** t for t in range(1, whole + 1)) / whole
                    rates[id(c)] = 1.0 if c.innate else max(rates[id(c)], seen)
            for card in self.virtual:
                rates[id(card)] = 0.0
            if self.blade is not None and previous.forge_events > 0:
                # Chance a forge has happened by a random turn of the fight.
                rates[id(self.blade)] = 1.0 - math.exp(-previous.forge_events * max(1.0, elapsed))
            upgraded_share = self._upgraded_share(previous, drawn, size)
            if upgraded_share > 0:
                # The upgraded copy plays that share of each upgradable card's
                # draws (copies sharing one profile each move their own).
                moved = {id(c): rates[id(c)] * upgraded_share for c in self.upgradable}
                for c in self.upgradable:
                    rates[id(c.upgraded_profile)] += moved[id(c)]
                for key, value in moved.items():
                    rates[key] -= value
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
                created = previous_plays.get(id(source), 0.0) * self._made(g)
                if g.destination == "hand":
                    rates[id(g.profile)] += created
                else:
                    held = created * elapsed * per_turn
                    rates[id(g.profile)] += min(created, held) if g.profile.once_per_fight else held
            # Cards above base_energy need both positive mean extra_energy and a
            # reachable single-turn peak (cost <= base_energy + max_turn_extra);
            # negative net energy (deficit > 0) also reduces playability of cards
            # near base_energy. Cost-rule cards evaluate affordability across
            # their discrete cost states rather than collapsing to a scalar mean.
            deficit = max(0.0, -extra_energy)
            free_share: Dict[int, float] = {}
            paid_afford: Dict[int, float] = {}
            paid_energy: Dict[int, float] = {}
            brick_cost: Dict[int, float] = {}
            for c, _ in self.copies:
                fs, pa, pe, bc, eff_c = self._cost_profile(
                    c, base_energy, extra_energy, max_turn_extra, deficit, max_turn_loss
                )
                free_share[id(c)] = fs
                paid_afford[id(c)] = pa
                paid_energy[id(c)] = pe
                brick_cost[id(c)] = bc
                if not c.x_cost:
                    self._cost_cache[id(c)] = eff_c
            demand = sum(
                n * rates[id(c)] * paid_energy[id(c)]
                for c, n in self.copies if c.cost is not None and not c.x_cost
            )
            # Unplayable high-cost cards keep a size-invariant brick penalty so
            # adding unrelated 0-cost cards cannot inflate energy by diluting a
            # dead card's draw rate. In-fight upgraded profiles blend with their
            # base deck card instead of double-counting.
            brick_demand = 0.0
            for c, n in self.copies:
                if c.cost is None or c.x_cost:
                    continue
                if id(c) in self.known_set:
                    eff_brick = brick_cost[id(c)]
                    if c.upgraded_profile is not None and id(c.upgraded_profile) in brick_cost:
                        eff_brick = (
                            (1.0 - upgraded_share) * eff_brick
                            + upgraded_share * brick_cost[id(c.upgraded_profile)]
                        )
                    brick_demand += n * eff_brick
                elif id(c) not in self.upgraded_base:
                    brick_demand += n * rates[id(c)] * brick_cost[id(c)]
            energy_demand = demand + brick_demand
            playable = min(1.0, supply / demand) if demand > 0 else 1.0
            # A retained card waits in hand until it can be paid for, so it is
            # paid first; the other cards share what is left.
            kept_demand = sum(
                n * rates[id(c)] * paid_energy[id(c)] for c, n in self.copies
                if c.retain and c.cost is not None and not c.x_cost
            )
            kept_share = min(1.0, supply / kept_demand) if kept_demand > 0 else 1.0
            if kept_demand > 0 and demand > supply:
                rest = demand - kept_demand
                playable = max(0.0, supply - kept_demand * kept_share) / rest if rest > 0 else 1.0
            x_rate = sum(n * rates[id(c)] for c, n in self.copies if c.x_cost)
            # X cards spend what the other plays leave, shared between them.
            x_value = max(0.0, supply - demand) / max(1.0, x_rate) if x_rate else 0.0
            self.x_value = x_value
            self.card_x = {
                id(c): (
                    x_value if _has_zero_x_effect(c)
                    else (max(1.0, x_value) if x_value > 0.0 else 0.0)
                )
                for c, _ in self.copies if c.x_cost
            }
            x_spent = x_rate * x_value if x_rate > 0.0 else 0.0
            self.energy_spent = min(supply, demand + x_spent)
            plays = {
                id(c): (
                    rates[id(c)] * (
                        1.0 if _has_zero_x_effect(c) else min(1.0, x_value)
                    )
                    if c.x_cost else (
                        rates[id(c)] * (
                            free_share[id(c)] + paid_afford[id(c)] * (kept_share if c.retain else playable)
                        )
                        if c.cost is not None else 0.0
                    )
                )
                for c, _ in self.copies
            }
            # Stars carry over between turns (PlayerCombatState keeps them;
            # only death resets them): star cards share the fight's stars.
            star_demand = sum(n * plays[id(c)] * c.star_cost for c, n in self.copies if c.star_cost)
            if star_demand > 0:
                star_playable = min(1.0, star_supply / star_demand)
                # Copies share one profile (and one entry in ``plays``).
                for key in {id(c) for c in self.cards if c.star_cost}:
                    plays[key] *= star_playable
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
                # Copies share one profile (and one entry in ``plays``).
                for key in {id(c) for c in matching}:
                    plays[key] *= 1.0 + factor
            # Sly (CardCmd.Discard -> DiscardAndDraw): a Sly card discarded by
            # an effect is played for free. A chosen discard takes a Sly card
            # in hand when there is one; a random one hits a copy 1/size of the
            # time. End-of-turn discards do not trigger it.
            sly = [c for c in self.cards if c.sly]
            if sly and previous_discards > 0:
                in_hand = len(sly) * per_turn
                chosen = min(1.0, previous_chosen / in_hand) if in_hand > 0 else 0.0
                random_share = max(0.0, previous_discards - previous_chosen) / size
                for key in {id(c) for c in sly}:
                    plays[key] += min(per_turn, per_turn * chosen + random_share)
            played = sum(plays.values())
            if play_limit is not None and played > play_limit:
                scale = play_limit / played
                plays = {k: v * scale for k, v in plays.items()}
                played = play_limit
            totals.ethereal_exhausts = sum(
                n * max(0.0, rates[id(c)] - plays[id(c)]) for c, n in self.copies if c.ethereal
            )
            # Fool arrows: flow conservation across starting slots + Quiver slots,
            # Impaled passive hit-counter returns (every 3 powered hits), and
            # explicit dislodge_arrow plays. Without arrow slots no shot lands.
            arrow_slots = self.situation.arrow_slots + previous.arrow_slots * elapsed
            self.arrow_slots_now = arrow_slots
            shot_demand = sum(
                n * plays[id(c)] for c, n in self.copies
                if any(e.get("kind") == "damage" and e.get("source") == "arrow" for e in c.effects)
            )
            non_arrow_hits = sum(
                n * plays[id(c)] * self._hits(e, self.card_x.get(id(c)) if c.x_cost else None)
                for c, n in self.copies for e in c.effects
                if e.get("kind") == "damage" and e.get("source") not in {"arrow", "osty"}
            )
            if arrow_slots > 0.0:
                arrow_return_rate = non_arrow_hits / 3.0 + previous.dislodges
                arrow_supply = arrow_slots / turns + arrow_return_rate
                self.arrow_avail = min(1.0, arrow_supply / shot_demand) if shot_demand > 0.0 else 1.0
                self.arrow_shots = shot_demand * self.arrow_avail
                stay = min(turns / 2.0, 1.0 / max(1e-6, arrow_return_rate + 1.0 / turns))
                self.impaled_prob = (1.0 - math.exp(-self.arrow_shots * stay)) if self.arrow_shots > 0.0 else 0.0
                self.arrow_dislodges = min(
                    self.arrow_shots,
                    self.impaled_prob * (non_arrow_hits / 3.0) + previous.dislodges + self.arrow_shots / turns,
                )
            else:
                self.arrow_avail = 0.0
                self.arrow_shots = 0.0
                self.impaled_prob = 0.0
                self.arrow_dislodges = 0.0
            self.melody_plays = {
                tag: sum(n * plays[id(c)] for c, n in self.copies if tag in c.tags)
                for tag in MELODY_TAGS
            }
            self.melody_source_plays = plays
            self._estimate(previous, plays, drawn)
            # Copies share one profile: resolved once, applied for each copy
            # (``_apply`` is linear in the number of applications).
            card_peak_energy: Dict[int, float] = {}
            card_peak_loss: Dict[int, float] = {}
            for card, copies in self.copies:
                card_x = self.card_x.get(id(card)) if card.x_cost else None
                pos_gain = neg_loss = 0.0
                for effect in card.effects:
                    resolved, chance = self._resolve(effect, card, plays[id(card)])
                    if (
                        resolved.get("kind") == "damage"
                        and resolved.get("source") == "arrow"
                        and not ({"has_arrow", "shot_succeeded"} & set(effect.get("condition") or ()))
                    ):
                        chance *= self.arrow_avail
                    applied_times = plays[id(card)] * chance * copies
                    self._apply(totals, resolved, applied_times, card_x)
                    if resolved.get("kind") == "energy" and applied_times > 0.0:
                        amt = float(resolved.get("amount") or 0.0)
                        if card_x is not None and resolved.get("per_x"):
                            amt *= card_x + float(resolved.get("x_bonus") or 0.0)
                        scale = effect.get("scale")
                        if isinstance(scale, Mapping) and scale.get("per") == "energy_after_cost":
                            c_cost = self._cost(card) or 0.0
                            peak_amt = float(effect.get("amount") or 0.0) + (
                                float(scale.get("factor") or 0.0) * max(0.0, base_energy - c_cost)
                            )
                            amt = max(amt, peak_amt)
                        if amt > 0.0:
                            pos_gain += amt
                        elif amt < 0.0:
                            neg_loss += -amt
                if pos_gain > 0.0:
                    c_cost = 0.0 if (card.x_cost or (card.sly and previous_discards > 0)) else (self._cost(card) or 0.0)
                    net_gain = max(0.0, pos_gain - c_cost)
                    if net_gain > 0.0:
                        card_peak_energy[id(card)] = net_gain
                if neg_loss > 0.0:
                    card_peak_loss[id(card)] = neg_loss
            # Deduplicate in-fight upgraded virtual profiles against their deck base copy.
            for card, copies in self.copies:
                if id(card) in self.known_set:
                    peak = card_peak_energy.get(id(card), 0.0)
                    loss = card_peak_loss.get(id(card), 0.0)
                    if card.upgraded_profile is not None:
                        peak = max(peak, card_peak_energy.get(id(card.upgraded_profile), 0.0))
                        loss = max(loss, card_peak_loss.get(id(card.upgraded_profile), 0.0))
                    totals.max_turn_extra += copies * peak
                    totals.max_turn_loss += copies * loss
                elif id(card) not in self.upgraded_base:
                    if id(card) in card_peak_energy:
                        totals.max_turn_extra += max(1.0, math.ceil(plays[id(card)])) * card_peak_energy[id(card)]
                    if id(card) in card_peak_loss:
                        totals.max_turn_loss += max(1.0, math.ceil(plays[id(card)])) * card_peak_loss[id(card)]
            self._apply_damage_modifiers(totals, plays, x_value)
            self._apply_block_modifiers(totals, plays)
            self._apply_events(totals, plays, drawn, played, block_estimate, x_value)
            self._apply_orbs(totals)
            extra_draw, extra_energy, block_estimate = totals.draw, totals.energy, totals.block
            peak_fixed_gain = max((v for v in totals.fixed_turn_energy.values() if v > 0.0), default=0.0)
            peak_fixed_loss = max((-v for v in totals.fixed_turn_energy.values() if v < 0.0), default=0.0)
            max_turn_extra = totals.max_turn_extra + peak_fixed_gain
            max_turn_loss = totals.max_turn_loss + peak_fixed_loss
            star_supply = totals.stars
            previous = totals
            previous_plays = plays
            previous_discards = totals.discards
            previous_chosen = totals.chosen_discards
            self.last_plays = plays
            self._activate(plays, drawn)
        return (
            self._capabilities(totals, drawn, size, supply, energy_demand, plays),
            tuple(dict.fromkeys(totals.gaps)),
        )

    def _status_removed(self, previous: "_Totals", previous_plays: Mapping[int, float], elapsed: float) -> float:
        """Status cards exhausted from the piles by an average moment.

        Exhausted Status cards (Flak Cannon) stop diluting the other cards'
        draws: as many as were exhausted by then, at most the Status cards
        that sit in the piles (the deck's own and those piled during the
        fight; Status cards made in hand were never counted as piled).
        """

        held = float(sum(1 for c in self.known if c.card_type == "Status")) + (
            self.situation.status_cards * elapsed
        ) + sum(
            previous_plays.get(id(c), 0.0) * self._made(g) * elapsed
            for c, g in self.sources if g.destination != "hand" and g.profile.card_type == "Status"
        ) + sum(
            previous.generated.get(key, 0.0) * elapsed
            for key, (profile, destination) in self.mechanic_made.items()
            if destination != "hand" and profile.card_type == "Status"
        )
        return min(held, previous.status_exhausts * elapsed)

    def _upgraded_share(self, previous: "_Totals", drawn: float, size: float) -> float:
        """Fight-average share of upgradable card instances already upgraded
        when one of them is played (CardCmd.Upgrade lasts the whole fight).

        Upgrades are counted in instances out of ``upgrade_pool`` (every
        upgradable deck card, plus unknown cards that may take a random
        upgrade but gain nothing here): those held from the start of the
        fight count in full; those made per turn build up (``_held_share``),
        usable from the next turn, or for cards upgraded in the discard pile
        (Drain Power) only once it is reshuffled, on average half a pass
        through the deck later. Capped at every upgradable card.
        """

        pool = self.upgrade_pool
        if not self.upgradable or pool <= 0:
            return 0.0
        turns = self.situation.fight_turns
        cycle = size / max(drawn, 1e-9)   # turns to draw through the deck
        return min(1.0, previous.upgraded_held / pool
                   + _held_share(previous.upgrades / pool, turns)
                   + _held_share(previous.upgrades_discard / pool, turns, cycle / 2.0))

    def _upgrade_instances(self, effect: Mapping, moment: str, card: Optional[CardProfile] = None) -> float:
        """Upgradable instances one ``upgrade_cards`` effect upgrades.

        Its pile holds upgradable cards in proportion to the deck: at combat
        start every card is in the draw pile; on the opening turn the hand
        holds the cards drawn and the discard pile is empty; at other turn
        starts the hand is full; at a card play the hand holds the other
        cards still there (``hand_size``) and the draw and discard piles half
        of the rest each. ``all`` upgrades every one of them, a count picks
        up to that many (``choose`` is valued as a random pick, a lower
        bound), ``all_combat`` upgrades every upgradable instance.

        A played card is in the play pile while it resolves, so it is not
        one of its own targets. (It still takes the common upgraded share
        that other copies or sources give it: a small overestimate.)
        """

        own = 1.0 if (
            card is not None and id(card) in self.known_set and card.upgraded_profile is not None
        ) else 0.0
        pool = self.upgrade_pool - own
        if pool <= 0:
            return 0.0
        scope = effect.get("scope")
        if scope == "all_combat":
            return pool
        size, drawn = self.pass_size - own, self.pass_drawn
        estimates = self.estimates
        if moment == "combat_start":
            held = {"hand": 0.0, "draw": size, "discard": 0.0}
        elif moment == "opening_turn":
            held = {"hand": drawn, "draw": max(0.0, size - drawn), "discard": 0.0}
        else:
            held = {
                "hand": drawn if moment == "turn_start" else estimates["hand_size"],
                "draw": estimates["draw_pile"], "discard": estimates["discard_pile"],
            }
        mean = held[scope] * pool / max(size, 1.0)
        count = effect.get("count")
        return mean if count == "all" else _expected_taken(float(count), mean)

    def _hand_takes(self, effect: Mapping, card: CardProfile) -> float:
        """Cards a filtered exhaust/discard of a fixed count takes from hand.

        The other cards in hand at a random play (``hand_size``) match the
        card-type filter in proportion to the rest of the deck; it takes up
        to ``count`` of them (Thrash: one random Attack, if there is one).
        """

        card_type = effect["filter"]["card_type"]
        matching = sum(1 for c in self.known if c.card_type == card_type)
        if card.card_type == card_type and id(card) in self.known_set:
            matching -= 1   # the card itself is in the play pile
        mean = self.estimates["hand_size"] * max(0, matching) / max(1.0, self.pass_size - 1.0)
        return _expected_taken(float(effect.get("count") or 0.0), mean)

    def _exhausted_damage(self, card: CardProfile) -> float:
        """Damage per hit one play adds to the card's later plays by keeping
        the damage of the card it exhausts (Thrash).

        The exhausted card is a random matching card from hand: the expected
        number taken (``_hand_takes``) times the average per-hit damage of the
        deck's other matching cards (first damage effect; 0 without one). The
        game adds that card's damage through Hook.ModifyDamage, so the owner's
        Strength is in it too; that part is left out (a lower bound). Losing
        the exhausted card for the rest of the fight is not counted, as for
        every other exhaust of a card from hand.
        """

        exhaust = next((e for e in card.effects if e.get("grows_own_damage")), None)
        if exhaust is None:
            return 0.0
        card_type = exhaust["filter"]["card_type"]

        def damage(c: CardProfile) -> float:
            return next((float(e.get("amount") or 0.0) for e in c.effects if e.get("kind") == "damage"), 0.0)

        pool = [damage(c) for c in self.known if c.card_type == card_type]
        total, count = sum(pool), len(pool)
        if card.card_type == card_type and id(card) in self.known_set:
            total, count = total - damage(card), count - 1
        if count <= 0:
            return 0.0
        return self._hand_takes(exhaust, card) * total / count

    def _estimate(self, previous: "_Totals", plays: Mapping[int, float], drawn: float) -> None:
        """Per-play estimates of the game state amounts may scale with.

        Uses the previous pass's totals; quantities that build up during a
        fight are taken at its average elapsed turn.
        """

        turns = self.situation.fight_turns
        elapsed = max(0.0, turns - 1.0) / 2.0
        exhausts = previous.exhausts + sum(n * plays[id(c)] for c, n in self.copies if c.exhausts_self)
        known = max(1, len(self.known))
        attacks = sum(1 for c in self.known if c.card_type == "Attack") / known
        hand = max(0.0, drawn - 1.0) / 2.0  # other cards still in hand at a random play
        incoming = self.situation.incoming_attack
        hit = max(0.0, min(1.0, (incoming - previous.block) / incoming)) if incoming > 0 else 0.0
        revive = 1.0 - math.exp(-previous.summon_events)
        death = min(1.0, hit + previous.osty_kills)
        alive = revive / (revive + death) if revive + death > 0 else (1.0 if previous.summoned > 0 else 0.0)
        last = self.last_plays
        # Damage dealt per turn summed over the enemies (previous pass).
        self.damage_dealt = (
            previous.single + previous.random + previous.osty_single + previous.osty_random
            + (previous.all + previous.osty_all) * (1.0 + self.situation.extra_enemies)
        )
        # Per-type play totals of the previous pass, shared by every card.
        self.play_totals = {
            "Attack": sum(n * last.get(id(c), 0.0) for c, n in self.copies if c.card_type == "Attack"),
            "Skill": sum(n * last.get(id(c), 0.0) for c, n in self.copies if c.card_type == "Skill"),
            "OstyAttack": sum(n * last.get(id(c), 0.0) for c, n in self.copies if "OstyAttack" in c.tags),
        }
        # A transform is not counted as a generated card (no IL evidence
        # that CardCmd.Transform records a generation).
        self.status_made = self.situation.status_cards + sum(
            last.get(id(c), 0.0) * g.count for c, g in self.sources
            if g.profile.card_type == "Status" and g.targets is None
        ) + sum(
            previous.generated.get(key, 0.0) for key, (profile, _) in self.mechanic_made.items()
            if profile.card_type == "Status"
        )
        status_deck = float(sum(1 for c in self.known if c.card_type == "Status"))
        self.status_pool = status_deck + self.status_made * turns
        self.status_clear_plays = sum(n * last.get(id(c), 0.0) for c, n in self.copies if id(c) in self.status_clearers)
        doom = (previous.doom_single + previous.doom_all) * max(1.0, elapsed)
        generated = sum(
            last.get(id(c), 0.0) * g.count for c, g in self.sources if g.targets is None
        ) + sum(previous.generated.values())
        # Cards a transform can find (see ``_transformed``). Status cards
        # enemies push into the piles are drawn like the deck's own, half
        # built up; a card drawn this turn is taken to be in hand when the
        # transform is played.
        rates = self.rates
        enemy_status = self.situation.status_cards * elapsed * self.per_turn
        others = max(0.0, drawn - 1.0)
        status_in_hand = enemy_status + sum(
            n * rates.get(id(c), 0.0) for c, n in self.copies if c.card_type == "Status"
        )
        unplayed = enemy_status + sum(
            n * max(0.0, rates.get(id(c), 0.0) - plays[id(c)]) for c, n in self.copies
        )
        self.estimates = {
            "orb_count": previous.orbs_held,
            "distinct_orb_types": previous.orb_kinds,
            "ethereal_cards_played_this_combat": sum(
                n * last.get(id(c), 0.0) for c, n in self.copies if c.ethereal
            ) * elapsed,
            "stars_gained_this_turn": previous.stars,
            "cards_drawn_this_turn_outside_hand_draw": previous.draw,
            "target_doom": doom,
            "target_doom_div_10": float(int(doom // 10)),
            # The deck's Status cards plus those made so far (by enemies, the
            # deck's cards and powers); a card exhausting them all sees fewer
            # (see ``_resolve``). Status cards that exhaust themselves
            # (Ethereal Dazed, a played Slimed) are not taken off: the
            # enemies' cards are only known as a count.
            "status_cards_in_combat": status_deck + self.status_made * elapsed,
            # Other cards in hand when the turn starts; Status cards drawn this
            # turn; drawn cards the turn leaves unplayed (all within the hand).
            "other_cards_in_hand": others,
            "status_cards_in_hand": min(others, status_in_hand),
            "unplayed_cards_in_hand": min(others, unplayed),
            # Half the cards outside the hand sit in the discard pile on average.
            "discard_pile": max(0.0, self.size - drawn) / 2.0,
            "cards_generated_this_combat": generated * elapsed,
            "prior_hits_on_target_this_turn": sum(
                n * last.get(id(c), 0.0) * self._hits(e, self.card_x.get(id(c)) if c.x_cost else None)
                for c, n in self.copies for e in c.effects if e.get("kind") == "damage"
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
            # Cards drawn so far this fight: the earlier turns and this hand.
            "cards_drawn_this_combat": drawn * (elapsed + 1.0),
            # Half of what is not in hand sits in the draw pile on average.
            "draw_pile": max(0.0, self.size - drawn) / 2.0,
            # Distinct debuffs on the target: Vulnerable, Weak, Poison, Doom.
            "target_debuff_types": (
                min(1.0, previous.vulnerable) + min(1.0, previous.weak)
                + (1.0 if previous.poison_single + previous.poison_all > 0 else 0.0)
                + (1.0 if previous.doom_single + previous.doom_all > 0 else 0.0)
            ),
            "discards_this_turn": previous.discards / 2.0,
            # CardPile.MaxCardsInHand is 10.
            "hand_space": max(0.0, 10.0 - hand),
            "enemy_count": 1.0 + self.situation.extra_enemies,
            "orbs_channeled_this_combat": sum(previous.channels.values()) * elapsed,
            "lightning_channeled_this_combat": previous.channels.get(ORB_TYPES["Lightning"], 0.0) * elapsed,
            "cards_played_this_combat": sum(n * last.get(id(c), 0.0) for c, n in self.copies) * (elapsed + 0.5),
            "energy_spent_this_turn": self.energy_spent / 2.0,
            "skills_in_hand": hand * sum(1 for c in self.known if c.card_type == "Skill") / known,
        }

    def _condition_chance(self, condition: Mapping, card: Optional[CardProfile] = None) -> float:
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
            elif key in {"target_impaled", "target_impaled_at_play_start"}:
                chance *= self.impaled_prob
            elif key in {"has_arrow", "shot_succeeded"}:
                chance *= self.arrow_avail
            elif key == "melody_played_this_turn":
                tags = value if isinstance(value, list) else [value]
                chance *= self._melody_any_prob(tags, card)
            elif key == "melodies_this_turn_at_least":
                k = int(value)
                if k <= 0:
                    pass
                elif k >= len(MELODY_TAGS):
                    for tag in MELODY_TAGS:
                        chance *= 1.0 - math.exp(-self._other_melody_plays(tag, card) / 2.0)
                else:
                    chance *= _poisson_at_least(self._melodies_for_card(card), k)
            # target_alive: the card is played on a living target.
        return chance

    def _resolve(self, effect: Mapping, card: CardProfile, rate: float) -> tuple:
        """Fill a state-dependent effect with per-play estimates.

        Returns the resolved effect and the chance its condition holds.
        """

        if (
            "scale" not in effect and "hits_scale" not in effect and "count_scale" not in effect
            and "multiply_if" not in effect
            and "filter" not in effect and effect.get("kind") != "upgrade_cards"
            and not effect.get("condition") and card.card_id != "SOVEREIGN_BLADE"
        ):
            return effect, 1.0  # nothing depends on the fight's state
        resolved = dict(effect)
        if effect.get("kind") == "upgrade_cards":
            resolved["instances"] = self._upgrade_instances(effect, "play", card)
        elif effect.get("filter") and "count_scale" not in effect:
            resolved["count"] = self._hand_takes(effect, card)
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
        estimates["melodies_this_turn"] = self._melodies_for_card(card)
        estimates["melody_cards_in_combat"] = self.deck_counts["melody_cards"]
        estimates["x_value"] = self.card_x.get(id(card), self.x_value)
        # Energy left after paying for this card, at an average moment.
        estimates["energy_after_cost"] = max(0.0, self.supply - (self._cost(card) or 0.0)) / 2.0
        # Damage kept from the cards its earlier plays exhausted (Thrash).
        estimates["own_exhausted_damage"] = estimates["own_previous_plays"] * self._exhausted_damage(card)
        if id(card) in self.status_clearers:
            # It exhausts every Status card it counts: its plays share the
            # fight's Status cards, at most the count at an average moment.
            plays = max(1.0, self.status_clear_plays * self.situation.fight_turns)
            estimates["status_cards_in_combat"] = min(
                estimates["status_cards_in_combat"], self.status_pool / plays
            )
        scale = effect.get("scale")
        if scale:
            value = estimates[scale["per"]]
            if "min" in scale:
                value = max(float(scale["min"]), value)
            resolved["amount"] = float(effect.get("amount") or 0.0) + float(scale.get("factor") or 0.0) * value
        hits_scale = effect.get("hits_scale")
        if hits_scale and hits_scale.get("per") == "enemies_killed_by_this_card":
            estimates["enemies_killed_by_this_card"] = self._repeats_on_kill(resolved)
        if hits_scale:
            base = float(hits_scale.get("base") or 0.0)
            if hits_scale["per"] == "condition":
                chance = self._condition_chance(hits_scale.get("condition") or {}, card)
                resolved["hits"] = base + (float(hits_scale["hits_if"]) - base) * chance
            else:
                resolved["hits"] = base + estimates[hits_scale["per"]]
        multiply_if = effect.get("multiply_if")
        if multiply_if:
            m_chance = self._condition_chance(multiply_if.get("condition") or {}, card)
            resolved["amount"] = float(resolved.get("amount") or 0.0) * (
                1.0 + (float(multiply_if["multiply"]) - 1.0) * m_chance
            )
        count_scale = effect.get("count_scale")
        if count_scale:
            base = float(count_scale.get("base") or 0.0) + (
                float(count_scale.get("base_upgrade") or 0.0) if card.upgraded else 0.0
            )
            resolved["count"] = base + estimates[count_scale["per"]]
        return resolved, self._condition_chance(effect.get("condition") or {}, card)

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
            amount *= x_value + float(effect.get("x_bonus") or 0.0)
        kind = effect.get("kind")
        if kind == "upgrade_cards":
            # Upgraded instances (resolved by ``_resolve`` / ``_apply_events``).
            instances = float(effect.get("instances") or 0.0)
            if persist_share is not None:
                totals.upgraded_held += instances * persist_share
            elif times > 0 and effect.get("scope") == "discard":
                totals.upgrades_discard += instances * times
            elif times > 0:
                totals.upgrades += instances * times
            return
        if kind == "apply_power" and effect.get("target") == "self" and persist_share is not None:
            name = SELF_POWERS[effect["power"]]
            if name not in {"vigor", "aim", "stretch"}:
                setattr(totals, f"{name}_flat", getattr(totals, f"{name}_flat") + amount * persist_share)
                return
        if times <= 0:
            return
        turns = self.situation.fight_turns
        if kind == "damage":
            hits = self._hits(effect, x_value)
            target = effect.get("target") or "single"
            if target == "attacker":
                target = "single"
            elif target == "evoke_targets":
                target = "random"
            if effect.get("source") == "osty":
                target = f"osty_{target}"
            dealt = amount * hits * times
            setattr(totals, target, getattr(totals, target) + dealt)
            if effect.get("source") == "arrow":
                totals.arrow_damage += dealt
                totals.arrow_aim_hits += hits * float(effect.get("aim_multiplier") or 1.0) * times
        elif kind == "arrow_slots":
            totals.arrow_slots += amount * times
        elif kind == "dislodge_arrow":
            cnt = self.arrow_slots_now if effect.get("count") == "all" else float(effect.get("count") or 1.0)
            totals.dislodges += cnt * times
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
            reps = float(effect.get("times") or 1.0) if (kind == "block" and isinstance(effect.get("times"), (int, float))) else 1.0
            setattr(totals, kind, getattr(totals, kind) + amount * reps * times)
            if kind == "block":
                totals.block_events += reps * times
            elif kind == "hp_loss":
                totals.self_damage_events += times
        elif kind == "exhaust_other":
            # A resolved count may be 0 (no matching card in hand).
            count = float(effect.get("count") if effect.get("count") is not None else 1)
            totals.exhausts += count * times
            if (effect.get("filter") or {}).get("card_type") == "Status":
                totals.status_exhausts += count * times
        elif kind == "discard":
            count = float(effect.get("count") if effect.get("count") is not None else 1)
            totals.discards += count * times
            if effect.get("choice") == "choose":
                totals.chosen_discards += count * times
        elif kind == "apply_power":
            power, target = effect["power"], effect["target"]
            if target in {"damaged_creature", "damaged enemy"}:
                target = "enemy"
            if target == "self":
                name = SELF_POWERS[power]
                if name in {"vigor", "aim", "stretch"}:
                    setattr(totals, name, getattr(totals, name) + amount * times)
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
            flt = modifier.get("filter") or {}
            share = _schedule(modifier, turns)[0] * weight
            if flt.get("target_has_power") == "Impaled":
                share *= self.impaled_prob
            from_arrow_only = bool(flt.get("from_arrow"))
            for card in self.cards:
                if not self._matches(card, flt, x_value):
                    continue
                for effect in card.effects:
                    if effect.get("kind") != "damage":
                        continue
                    is_arrow = effect.get("source") == "arrow"
                    if from_arrow_only and not is_arrow:
                        continue
                    cx = self.card_x.get(id(card), x_value) if card.x_cost else None
                    hits = self._hits(effect, cx)
                    base = float(effect.get("amount") or 0.0)
                    if card.x_cost and effect.get("per_x"):
                        base *= cx
                    extra = float(modifier.get("add") or 0.0) * hits + (
                        float(modifier.get("multiply") or 1.0) - 1.0
                    ) * base * hits
                    eff_share = share * (self.arrow_avail if is_arrow else 1.0)
                    target = effect.get("target") or "single"
                    if effect.get("source") == "osty":
                        target = f"osty_{target}"
                    setattr(totals, target, getattr(totals, target) + extra * plays[id(card)] * eff_share)

    def _apply_block_modifiers(self, totals: _Totals, plays: Mapping[int, float]) -> None:
        turns = self.situation.fight_turns
        for modifier, weight in ((m, w) for m, w in self.modifiers if m["modifier"] == "block"):
            factor = (float(modifier.get("multiply") or 1.0) - 1.0) * weight
            matching = [
                (card, effect) for card in self.cards
                if self._matches(card, modifier.get("filter"), None)
                for effect in card.effects if effect.get("kind") == "block"
            ]
            gained = sum(
                plays[id(c)] * float(e.get("amount") or 0.0)
                * (float(e.get("times") or 1.0) if isinstance(e.get("times"), (int, float)) else 1.0)
                * self._condition_chance(e.get("condition") or {}, c)
                for c, e in matching
            )
            if modifier.get("limit") == "once_per_combat":
                rate = sum(plays[id(c)] for c, _ in matching)
                average = gained / rate if rate else 0.0
                totals.block += factor * average * _poisson_at_least(rate * turns, 1) / turns
            else:
                totals.block += factor * gained

    def _block_plays(self, plays: Mapping[int, float]) -> float:
        return sum(
            plays[id(c)] * sum(
                (float(e.get("times") or 1.0) if isinstance(e.get("times"), (int, float)) else 1.0)
                * self._condition_chance(e.get("condition") or {}, c)
                for e in c.effects if e.get("kind") == "block"
            )
            for c in self.cards
            if any(e.get("kind") == "block" for e in c.effects)
        )

    def _event_rate(
        self, mechanic: Mapping, plays: Mapping[int, float], drawn: float,
        block_estimate: float, totals: _Totals, x_value: float,
    ) -> float:
        situation = self.situation
        when = mechanic["when"]
        if when == "card_played":
            flt = mechanic.get("filter") or {}
            rate = sum(
                plays[id(c)] for c in self.cards
                if self._matches(c, flt, x_value)
            )
            if flt.get("first_of_tag_this_turn"):
                return 1.0 - math.exp(-rate)
            return rate
        if when in {"arrow_shot", "arrow_impaled"}:
            return self.arrow_shots
        if when == "arrow_dislodged":
            return self.arrow_dislodges
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
        if when == "orb_evoked":
            orb = (mechanic.get("filter") or {}).get("orb")
            return self.orb_evokes.get(ORB_TYPES[orb], 0.0) if orb else sum(self.orb_evokes.values())
        if when == "damage_dealt":
            # Powered attack hits the owner (and its pet) lands per turn.
            hits = sum(
                plays[id(c)] * self._hits(e, self.card_x.get(id(c)) if c.x_cost else None)
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
            elif key == "melodies_played_this_turn":
                tags = value if isinstance(value, list) else [value]
                for tag in tags:
                    probability *= 1.0 - math.exp(-self.melody_plays.get(tag, 0.0))
            elif key == "target_impaled":
                probability *= self.impaled_prob
            # turn_at_least: handled by the schedule; unblocked_damage,
            # during_own_turn and caused_by_ethereal: what the event rate
            # already measures; block_gained_above 0: every block event.
        return probability

    def _apply_events(
        self, totals: _Totals, plays: Mapping[int, float], drawn: float, played: float,
        block_estimate: float, x_value: float,
    ) -> None:
        turns = self.situation.fight_turns
        power_peak_gain: Dict[int, float] = {}
        power_peak_loss: Dict[int, float] = {}
        for mechanic, weight, owner in self.events:
            share = weight * self._fight_share(mechanic.get("filter"))
            share *= self._condition(mechanic.get("condition") or {}, plays, drawn, played, totals)
            effect = mechanic["effect"]
            if effect.get("kind") == "upgrade_cards":
                when = mechanic["when"]
                moment = "play" if when not in {"combat_start", "turn_start"} else (
                    "opening_turn" if when == "turn_start" and mechanic.get("turns") == [1] else when
                )
                effect = {**effect, "instances": self._upgrade_instances(effect, moment)}
            if mechanic["when"] == "combat_start":
                self._apply(totals, effect, share / turns, None, persist_share=share)
                if effect.get("kind") == "energy" and share > 0.0 and owner is not _DUP_OWNER:
                    amt = float(effect.get("amount") or 0.0)
                    if owner is None:
                        totals.fixed_turn_energy[1] = totals.fixed_turn_energy.get(1, 0.0) + amt
                    elif amt > 0.0:
                        power_peak_gain[id(owner)] = power_peak_gain.get(id(owner), 0.0) + amt
                    elif amt < 0.0:
                        power_peak_loss[id(owner)] = power_peak_loss.get(id(owner), 0.0) - amt
                continue
            rate = self._event_rate(mechanic, plays, drawn, block_estimate, totals, x_value)
            fired_share, persist = _schedule(mechanic, turns)
            times = _triggers_per_turn(mechanic, rate, turns) * fired_share * share
            if mechanic["when"] == "turn_start" and persist is not None:
                self._apply(totals, effect, times, None, persist_share=persist * share)
            else:
                self._apply(totals, effect, times, None)
            if effect.get("kind") == "energy" and times > 0.0 and owner is not _DUP_OWNER:
                amt = float(effect.get("amount") or 0.0)
                if amt > 0.0 and mechanic["when"] == "card_played":
                    min_cost = float((mechanic.get("filter") or {}).get("min_cost") or 0.0)
                    amt = max(0.0, amt - min_cost)
                if owner is None:
                    fixed_turns = mechanic.get("turns")
                    if isinstance(fixed_turns, list) and fixed_turns:
                        for t in fixed_turns:
                            totals.fixed_turn_energy[int(t)] = totals.fixed_turn_energy.get(int(t), 0.0) + amt
                    elif amt > 0.0:
                        totals.max_turn_extra += amt
                    elif amt < 0.0:
                        totals.max_turn_loss += -amt
                elif amt > 0.0:
                    power_peak_gain[id(owner)] = power_peak_gain.get(id(owner), 0.0) + amt
                elif amt < 0.0:
                    power_peak_loss[id(owner)] = power_peak_loss.get(id(owner), 0.0) - amt
        if power_peak_gain or power_peak_loss:
            for card, copies in self.copies:
                if id(card) in self.known_set:
                    peak = power_peak_gain.get(id(card), 0.0)
                    loss = power_peak_loss.get(id(card), 0.0)
                    if card.upgraded_profile is not None:
                        peak = max(peak, power_peak_gain.get(id(card.upgraded_profile), 0.0))
                        loss = max(loss, power_peak_loss.get(id(card.upgraded_profile), 0.0))
                    totals.max_turn_extra += copies * peak
                    totals.max_turn_loss += copies * loss
                elif id(card) not in self.upgraded_base:
                    if id(card) in power_peak_gain:
                        totals.max_turn_extra += max(1.0, math.ceil(plays[id(card)])) * power_peak_gain[id(card)]
                    if id(card) in power_peak_loss:
                        totals.max_turn_loss += max(1.0, math.ceil(plays[id(card)])) * power_peak_loss[id(card)]

    def _repeats_on_kill(self, effect: Mapping) -> float:
        """Extra rounds of a hit that repeats once per enemy it kills.

        Every point of damage is an equal share of a kill, so a round kills
        (its damage to one enemy) / (one enemy's HP) of each enemy it hits.
        Each kill adds a round against the enemies still alive. Without the
        encounter HP, the HP is what the deck deals over the fight.
        """

        enemies = 1.0 + self.situation.extra_enemies
        amount = float(effect.get("amount") or 0.0)
        if amount <= 0:
            return 0.0
        # Enemies still alive at an average moment of the fight.
        alive = (enemies + 1.0) / 2.0
        targets = alive if effect.get("target") == "all" else 1.0
        total_hp = self.situation.enemy_hp or max(self.damage_dealt, amount * targets) * self.situation.fight_turns
        chance = min(1.0, amount * enemies / total_hp)
        repeats, rounds = 0.0, 1.0
        for _ in range(int(math.ceil(enemies))):
            kills = min(alive, rounds * chance * (alive if effect.get("target") == "all" else min(1.0, alive)))
            repeats += kills
            alive -= kills
            rounds = kills
        return repeats

    def _enemy_hits(self) -> float:
        """Attack hits the enemies aim at the player per turn: from the
        encounter facts, else one per enemy."""

        hits = self.situation.enemy_hits
        return float(hits) if hits is not None else 1.0 + self.situation.extra_enemies

    def _hits(self, effect: Mapping, x_value: Optional[float] = None) -> float:
        if effect.get("hits_x"):
            return self.x_value if x_value is None else x_value
        hits = effect.get("hits")
        return 1.0 if hits is None else float(hits)

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
        self.orb_evokes = {orb: evokes * rate / channel for orb, rate in totals.channels.items()}
        for orb, rate in totals.channels.items():
            fact = self.situation.orbs.get(orb)
            if not fact:
                totals.gaps.append(f"orb_fact_missing:{orb}")
                continue
            share = rate / channel
            extra_passive = totals.orb_passives.get(orb, 0.0)
            passives = held * share * (1.0 + extra_passive)
            passive_peak = (
                min(slots, float(math.ceil(rate * turns))) + (1.0 if extra_passive > 1e-9 else 0.0)
                if rate * turns > 1e-9 else 0.0
            )
            evoke_peak = (
                min(slots, float(math.ceil(evokes * share * turns)))
                if evokes * share * turns > 1e-9 else 0.0
            )
            self._orb_effect(totals, fact, "passive", passives, focus, age, passive_peak)
            self._orb_effect(totals, fact, "evoke", evokes * share, focus, age, evoke_peak)

    def _orb_effect(
        self, totals: _Totals, fact: Mapping, part: str, times: float, focus: float, age: float,
        peak_instances: float = 0.0,
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
            if times > 0.0 and value > 0.0 and peak_instances > 0.0:
                totals.max_turn_extra += value * peak_instances
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
            plays[id(c)] * self._hits(e, self.card_x.get(id(c)) if c.x_cost else None)
            * (self.arrow_avail if e.get("source") == "arrow" else 1.0)
            for c in self.cards for e in c.effects
            if e.get("kind") == "damage" and e.get("source") != "osty"
        )
        # Strength or Dexterity gained during a fight is on average half built up.
        growth = max(0.0, turns - 1.0) / 2.0
        vigor = totals.vigor if attack_hits > 0 else 0.0  # next attack only
        # Aim: next powered arrow attack only, multiplied by resolved hits and card aim_multiplier.
        aim_bonus = (
            totals.aim * (totals.arrow_aim_hits / self.arrow_shots)
            if totals.aim > 0.0 and self.arrow_shots > 0.0 else 0.0
        )
        dislike_bonus = totals.arrow_damage * (totals.dislike_you / 100.0) if totals.dislike_you > 0.0 else 0.0
        # Stretch: next powered block-gaining card only, once per block instance on that card.
        block_card_plays = sum(
            plays[id(c)] for c in self.cards if any(e.get("kind") == "block" for e in c.effects)
        )
        block_instances = self._block_plays(plays)
        stretch_bonus = (
            totals.stretch * (block_instances / block_card_plays)
            if totals.stretch > 0.0 and block_card_plays > 0.0 else 0.0
        )
        effective_block = totals.block + stretch_bonus
        temp_strength = sum(
            float(m.get("add") or 0.0) * w for m, w in self.modifiers if m["modifier"] == "strength"
        )
        incoming = situation.incoming_attack
        kept_block = 0.0
        for m, w in self.modifiers:
            if m["modifier"] == "keep_block":
                surplus = max(0.0, effective_block - incoming)
                cap = m.get("max")
                kept_block += w * (min(surplus, float(cap)) if cap else surplus)
        return {
            # Orb damage is unpowered: no Strength or Vulnerable amplification.
            "frontload": (
                totals.single + totals.all + totals.random + vigor + aim_bonus + dislike_bonus
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
                effective_block * (1.0 - float(params["frail_block_reduction"]) * situation.player_frail)
                + kept_block + totals.heal - totals.hp_loss
                # Die For You: Osty takes the unblocked attack damage, up to
                # the HP summoned for it.
                + min(totals.summoned, max(0.0, incoming - effective_block))
                # Buffer: each use negates one hit's unblocked damage.
                + max(0.0, incoming - effective_block) * min(turns, sum(
                    float(m.get("uses") or 0.0) * w for m, w in self.modifiers
                    if m["modifier"] == "hp_loss_negate"
                )) / turns
                # Enemy Strength lost: less attack damage, never below none.
                + min(incoming, totals.weaken_turn + totals.weaken_rate * growth)
                + (totals.dexterity_flat + totals.dexterity_rate * growth) * block_instances
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

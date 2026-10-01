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
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from storage.assembly_facts import class_to_entity_id
from storage.relational import RelationalRepository

FEEDBACK_PASSES = 3
POISSON_TERMS = 12
SUPPORTED_EFFECTS = frozenset({
    "damage", "block", "draw", "energy", "apply_power", "hp_loss", "heal",
    "exhaust_self", "exhaust_other", "discard",
})
SELF_POWERS = {"Strength": "strength", "Dexterity": "dexterity", "Vigor": "vigor"}
ENEMY_POWERS = {"Vulnerable": "vulnerable", "Weak": "weak", "Poison": "poison"}
DAMAGE_TARGETS = frozenset({"single", "all", "random"})
EFFECT_KEYS = frozenset({
    "kind", "amount", "upgrade", "hits", "target", "power", "count", "choice", "per_x",
    # Power facts scale an effect by the stacks applied; ``unpowered`` only
    # says Strength/Dexterity do not apply, which the engine never adds to
    # mechanic effects anyway.
    "amount_per_stack", "unpowered", "hits_upgrade",
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
    "own_previous_plays", "condition",
})
EFFECT_CONDITIONS = frozenset({
    "target_has_power", "exhaust_pile_at_least", "lost_hp_this_turn",
    "card_exhausted_this_turn", "target_alive",
})
NUMERIC_EFFECT_KEYS = ("amount", "upgrade", "hits", "count", "amount_per_stack", "hits_upgrade")
POWER_DURATIONS = frozenset({"combat", "this_turn", "turns_decrement"})
SUPPORTED_WHEN = frozenset({
    "combat_start", "turn_start", "turn_end", "card_played", "card_exhausted",
    "card_discarded", "shuffle", "damage_received", "enemy_died", "block_gained",
})
SUPPORTED_CONDITIONS = frozenset({
    "no_block", "per_card_in_hand", "hand_empty", "block_at_least",
    "hp_ratio_at_most", "no_card_type_played_this_turn",
    "cards_played_this_turn_at_most", "no_potions", "turn_at_least",
    "unblocked_damage", "during_own_turn", "caused_by_ethereal", "block_gained_above",
})
MECHANIC_KEYS = frozenset({
    "when", "filter", "every", "counter", "limit", "condition", "effect",
    "turns", "every_turns", "turns_from",
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
    "card_added_to_deck": {"filter", "upgrade"},
}
CARD_FILTER_KEYS = frozenset({"card_type", "tag", "min_cost", "upgraded", "x_cost"})
COUNTERS = frozenset({None, "turn", "combat", "run"})
LIMITS = frozenset({None, "once_per_turn", "once_per_combat"})


@dataclass(frozen=True)
class PowerUse:
    """A non-generic power a card applies to its owner, resolved to mechanics."""

    power_id: str
    duration: str                   # combat | this_turn | turns_decrement
    stacks: float
    mechanics: tuple


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


@dataclass
class _Totals:
    single: float = 0.0
    all: float = 0.0
    random: float = 0.0
    block: float = 0.0
    draw: float = 0.0
    energy: float = 0.0
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
    block_events: float = 0.0
    self_damage_events: float = 0.0
    temp_strength: float = 0.0
    gaps: List[str] = field(default_factory=list)


def effect_unsupported(effect: object) -> Optional[str]:
    """Why the engine cannot convert one effect, or ``None``."""

    if not isinstance(effect, Mapping):
        return "effect:missing"
    kind = effect.get("kind")
    if kind not in SUPPORTED_EFFECTS:
        return f"effect:{kind}"
    extra = set(effect) - EFFECT_KEYS
    if extra:
        return f"effect_key:{sorted(extra)[0]}"
    for key in NUMERIC_EFFECT_KEYS:
        value = effect.get(key)
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
    if kind == "apply_power":
        power, target = effect.get("power"), effect.get("target")
        if not ((target == "self" and power in SELF_POWERS)
                or (target in {"enemy", "all_enemies"} and power in ENEMY_POWERS)):
            return f"power:{power}:{target}"
    return None


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
    return resolved


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
    if duration not in POWER_DURATIONS:
        return None, (f"power_duration_unsupported:{power_id}:{duration}",)
    mechanics, gaps = [], []
    for raw in fact.get("mechanics") or ():
        mechanic = _resolve_stacks(raw, stacks)
        reason = unsupported_reason(mechanic)
        if reason:
            gaps.append(f"power_mechanic_unsupported:{power_id}:{reason}")
        else:
            mechanics.append(mechanic)
    return PowerUse(power_id, duration, stacks, tuple(mechanics)), tuple(gaps)


def _filter_unsupported(flt: object, card_filter_allowed: bool) -> Optional[str]:
    if flt is None:
        return None
    if not isinstance(flt, Mapping):
        return "filter:invalid"
    allowed = (CARD_FILTER_KEYS if card_filter_allowed else frozenset()) | {"fight_kind"}
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
    reason = _filter_unsupported(mechanic.get("filter"), when == "card_played")
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


def card_profile(
    repository: RelationalRepository,
    card_id: str,
    upgrades: int,
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
    effects, powers, gaps = [], [], []
    for effect in fact.get("effects") or ():
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
        reason = effect_unsupported(effect)
        if reason:
            gaps.append(f"card_effect_unsupported:{card_id}:{reason}")
            continue
        resolved = {**effect, "amount": _amount(effect, upgraded)}
        if upgraded and "scale" in effect:
            scale = dict(effect["scale"])
            scale["factor"] = float(scale.get("factor") or 0.0) + float(scale.get("factor_upgrade") or 0.0)
            resolved["scale"] = scale
        if upgraded and effect.get("hits_upgrade"):
            # Some upgrades add hits rather than damage (e.g. Sword Boomerang).
            resolved["hits"] = float(effect.get("hits") or 1) + float(effect["hits_upgrade"])
        effects.append(resolved)
    for field_name in ("mechanics", "cost_modifier"):
        if fact.get(field_name):
            gaps.append(f"card_effect_unsupported:{card_id}:{field_name}")
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
        effects=tuple(effects),
        powers=tuple(powers),
    ), tuple(gaps)


def relic_profile(
    repository: RelationalRepository,
    relic_id: str,
) -> tuple[Optional[RelicProfile], tuple[str, ...]]:
    fact = repository.find_assembly_fact("relics", relic_id)
    if fact is None or fact.get("status") != "converted":
        return None, (f"relic_fact_missing:{relic_id}",)
    supported, gaps = [], []
    for mechanic in fact.get("mechanics") or ():
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
        self.situation = situation
        self.params = params
        # (mechanic, weight): relics always apply; powers by how much of the
        # fight they are active, recomputed each pass (see ``_activate``).
        self.relic_mechanics = [(m, 1.0) for relic in relics for m in relic.mechanics]
        self.estimates: Dict[str, float] = {}
        self._set_mechanics(self.relic_mechanics)

    def _set_mechanics(self, weighted: Sequence[tuple[Mapping, float]]) -> None:
        self.modifiers = [(m, w) for m, w in weighted if "modifier" in m]
        self.events = [(m, w) for m, w in weighted if "modifier" not in m]

    def _activate(self, plays: Mapping[int, float], drawn: float) -> None:
        """Weight each played card's powers by their expected active share."""

        turns = self.situation.fight_turns
        first_turn = min(turns, (self.size / max(drawn, 1e-9) + 1.0) / 2.0)
        weighted = list(self.relic_mechanics)
        for card in self.known:
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
        """Printed cost after cost modifiers; X cards are priced separately."""

        if card.x_cost or card.cost is None:
            return None
        delta = sum(
            float(m.get("add") or 0.0) * w for m, w in self.modifiers
            if m["modifier"] == "energy_cost" and self._matches(card, m.get("filter"), None)
        )
        return max(0.0, card.cost + delta)

    def _matches(self, card: CardProfile, flt: Optional[Mapping], x_value: Optional[float]) -> bool:
        if not flt:
            return True
        if "card_type" in flt and card.card_type != flt["card_type"]:
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
        extra_draw = extra_energy = block_estimate = 0.0
        previous = _Totals()
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
            drawn = min(float(self.size), hand + (0.0 if no_draw else extra_draw))
            per_turn = min(1.0, drawn / self.size)
            supply = base_energy + extra_energy
            rates = {
                id(c): min(per_turn, 1.0 / turns) if c.once_per_fight else per_turn
                for c in self.known
            }
            demand = sum(rates[id(c)] * cost for c in self.known if (cost := self._cost(c)) is not None)
            playable = min(1.0, supply / demand) if demand > 0 else 1.0
            x_rate = sum(rates[id(c)] for c in self.known if c.x_cost)
            # X cards spend what the other plays leave, shared between them.
            x_value = max(0.0, supply - demand) / max(1.0, x_rate) if x_rate else 0.0
            plays = {
                id(c): (
                    rates[id(c)] * (1.0 if c.x_cost else playable)
                    if c.x_cost or self._cost(c) is not None else 0.0
                )
                for c in self.known
            }
            played = sum(plays.values())
            if play_limit is not None and played > play_limit:
                scale = play_limit / played
                plays = {k: v * scale for k, v in plays.items()}
                played = play_limit
            totals.ethereal_exhausts = sum(
                max(0.0, rates[id(c)] - plays[id(c)]) for c in self.known if c.ethereal
            )
            self._estimate(previous, plays, drawn)
            for card in self.known:
                for effect in card.effects:
                    resolved, chance = self._resolve(effect, card, plays[id(card)])
                    self._apply(totals, resolved, plays[id(card)] * chance, x_value if card.x_cost else None)
            self._apply_damage_modifiers(totals, plays, x_value)
            self._apply_block_modifiers(totals, plays)
            self._apply_events(totals, plays, drawn, played, block_estimate, x_value)
            extra_draw, extra_energy, block_estimate = totals.draw, totals.energy, totals.block
            previous = totals
            self._activate(plays, drawn)
        return (
            self._capabilities(totals, drawn, supply, demand, plays),
            tuple(dict.fromkeys(totals.gaps)),
        )

    def _estimate(self, previous: "_Totals", plays: Mapping[int, float], drawn: float) -> None:
        """Per-play estimates of the game state amounts may scale with.

        Uses the previous pass's totals; quantities that build up during a
        fight are taken at its average elapsed turn.
        """

        turns = self.situation.fight_turns
        elapsed = max(0.0, turns - 1.0) / 2.0
        exhausts = previous.exhausts + sum(plays[id(c)] for c in self.known if c.exhausts_self)
        known = max(1, len(self.known))
        attacks = sum(1 for c in self.known if c.card_type == "Attack") / known
        hand = max(0.0, drawn - 1.0) / 2.0  # other cards still in hand at a random play
        incoming = self.situation.incoming_attack
        hit = max(0.0, min(1.0, (incoming - previous.block) / incoming)) if incoming > 0 else 0.0
        self.estimates = {
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
            # target_alive: the card is played on a living target.
        return chance

    def _resolve(self, effect: Mapping, card: CardProfile, rate: float) -> tuple:
        """Fill a state-dependent effect with per-play estimates.

        Returns the resolved effect and the chance its condition holds.
        """

        resolved = dict(effect)
        estimates = dict(self.estimates)
        # Plays of this card earlier in the same fight, on average.
        estimates["own_previous_plays"] = max(0.0, rate * self.situation.fight_turns - 1.0) / 2.0
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
            resolved["count"] = estimates[count_scale["per"]]
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
            hits = float(effect.get("hits") or 1)
            target = effect.get("target") or "single"
            setattr(totals, target, getattr(totals, target) + amount * hits * times)
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
        elif kind == "apply_power":
            power, target = effect["power"], effect["target"]
            if target == "self":
                name = SELF_POWERS[power]
                if name == "vigor":
                    totals.vigor += amount * times
                else:
                    setattr(totals, f"{name}_rate", getattr(totals, f"{name}_rate") + amount * times)
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
            for card in self.known:
                if not self._matches(card, modifier.get("filter"), x_value):
                    continue
                for effect in card.effects:
                    if effect.get("kind") != "damage":
                        continue
                    hits = float(effect.get("hits") or 1)
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
                (card, effect) for card in self.known
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
            plays[id(c)] for c in self.known
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
                plays[id(c)] for c in self.known
                if self._matches(c, mechanic.get("filter"), x_value)
            )
        if when == "card_exhausted":
            own = totals.exhausts + sum(plays[id(c)] for c in self.known if c.exhausts_self)
            ethereal = (mechanic.get("condition") or {}).get("caused_by_ethereal")
            if ethereal is True:
                return totals.ethereal_exhausts
            if ethereal is False:
                return own
            return own + totals.ethereal_exhausts
        if when == "block_gained":
            return totals.block_events
        if when == "card_discarded":
            return totals.discards
        if when == "shuffle":
            return drawn / self.size
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
                rate = sum(plays[id(c)] for c in self.known if c.card_type == value)
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

    def _capabilities(
        self, totals: _Totals, drawn: float, supply: float, demand: float,
        plays: Mapping[int, float],
    ) -> Dict[str, float]:
        situation, params = self.situation, self.params
        turns = situation.fight_turns
        amplify = 1.0 + float(params["vulnerable_damage_bonus"]) * min(1.0, totals.vulnerable)
        attack_hits = sum(
            plays[id(c)] * float(e.get("hits") or 1)
            for c in self.known for e in c.effects if e.get("kind") == "damage"
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
            "frontload": (
                totals.single + totals.all + totals.random + vigor
                + (totals.strength_flat + temp_strength) * attack_hits
            ) * amplify + totals.poison_single + totals.poison_all,
            "aoe": (totals.all + totals.poison_all) * situation.extra_enemies,
            "defense": (
                totals.block + kept_block + totals.heal - totals.hp_loss
                + (totals.dexterity_flat + totals.dexterity_rate * growth) * self._block_plays(plays)
                + float(params["weak_damage_reduction"]) * situation.incoming_attack
                * min(1.0, totals.weak)
            ),
            "scaling": totals.strength_rate * growth * attack_hits * amplify,
            "cycle": drawn / self.size,
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

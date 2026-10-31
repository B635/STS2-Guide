"""Value of gaining a relic (TASK-027 ticket 02).

A relic option is scored as the fitness of the deck *after* taking it, with
the same deck-fitness engine and needs as a card reward:

* its pickup effects (``obtain`` in the reviewed fact) change the deck first:
  a chosen removal, upgrade, transform or enchantment takes the card the
  engine values most for it; random results are averaged over samples drawn
  with a seed from the decision, so a recommendation is reproducible;
* its combat mechanics join the relics the engine already scores;
* gold, HP and max HP changes are returned as amounts; the policy converts
  them with the resource exchange parameters (ticket 03).

Anything that cannot be valued (a potion, a random relic, a between-fight
effect) is a disclosed gap and the option's score is a lower bound; it is
never counted as zero without saying so.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from advisor.deck_fitness import _roll_rarity
from advisor.deck_mechanics import CardProfile, RelicProfile, enchant, relic_profile
from storage.assembly_facts import class_to_entity_id

PARAMETERS_KEY = "relic_choice_v1"
EXCHANGE_KEY = "resource_exchange_v1"
# Pickup effects that change the deck or the relics.
DECK_KINDS = frozenset({
    "remove_from_deck", "transform_cards", "upgrade_cards", "add_card", "enchant_cards", "replace_relic",
    "offer_rewards", "gain_relic",
})
RELIC_SOURCES = frozenset({"next_in_pool", "back_of_pool"})
CHARACTERS = ("IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT")
RESOURCE_KINDS = frozenset({
    "gain_gold", "lose_gold", "gain_max_hp", "lose_max_hp", "heal", "hp_loss",
})
CHOICES = frozenset({"choose", "random", "first", "last"})
# Card filters the valuation understands; any other key is a gap, never "all cards".
FILTER_KEYS = frozenset({
    "card_type", "tag", "card", "rarity", "transformable", "removable", "upgradable", "can_enchant",
})


@dataclass
class _Variant:
    """One possible result of taking the relic, with its probability."""

    weight: float
    cards: List[Optional[CardProfile]]
    relics: List[RelicProfile]
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelicOptionValue:
    relic_id: str
    raw: Optional[float]               # expected fitness after taking it (None: not scorable)
    immediate: Optional[float]         # the same without looking ahead
    resources: Mapping[str, float]     # gold / hp / max_hp changes
    notes: tuple[str, ...]             # what taking it does, for the reasons
    gaps: tuple[str, ...]              # effects left out (a lower bound when non-empty)


class RelicChoiceValuer:
    """Scores relic options against one situation (see the module docstring)."""

    def __init__(self, fitness_policy):
        self.policy = fitness_policy
        self.repository = fitness_policy.repository
        # Card pools by (color, rarities, type, tag, upgrades): fixed for one release database.
        self.pools: Dict[tuple, List[CardProfile]] = {}

    def evaluate(
        self,
        request,
        options: Sequence[tuple[str, Mapping]],
    ) -> tuple[Optional[float], Dict[str, RelicOptionValue], tuple[str, ...]]:
        """``(baseline raw, {relic_id: value}, context gaps)``.

        ``options`` pairs each relic ID with the parameters its option fixed
        (``obtain_params``; e.g. the character of a Sea Glass).
        """

        context, context_gaps = self.policy.deck_context(request)
        if context is None:
            return None, {}, tuple(context_gaps)
        params = self.repository.mechanic_constant(PARAMETERS_KEY)
        exchange = self.repository.mechanic_constant(EXCHANGE_KEY)
        if isinstance(params, Mapping) and isinstance(exchange, Mapping):
            # How a run unfolds (floors per act, reaching the next act) has one source.
            params = {**params, **{k: exchange.get(k) for k in ("act_floors", "reach_next_act", "rest_heals_per_act")}}
        parameter_gaps = _parameter_gaps(params)
        if parameter_gaps:
            return None, {}, (*context_gaps, *parameter_gaps)
        cache: Dict[tuple, float] = {}

        def score(cards: Sequence[Optional[CardProfile]], relics: Sequence[RelicProfile]) -> float:
            # The engine's result depends on which cards and relics there are, not their order.
            key = (tuple(sorted(map(id, cards))), tuple(sorted(r.relic_id for r in relics)))
            if key not in cache:
                cache[key] = context.score(cards, relics)[0].raw
            return cache[key]

        run = _Run(self, request, context, params, score)
        # The current deck on the same scale as an option's ``raw``: looking ahead
        # over the act's coming rewards, so "raw - baseline" is what the relic changes.
        relics = list(context.relics)
        ahead, _ = self.policy._lookahead(
            request, {"option": list(context.deck)}, lambda cards: score(cards, relics), context.needs, context.params,
        )
        baseline = ahead.get("option", score(context.deck, relics))
        values = {relic_id: run.value(relic_id, dict(option_params or {})) for relic_id, option_params in options}
        return baseline, values, tuple(context_gaps)


class _Run:
    """One evaluation: the situation, the parameters and the engine cache."""

    def __init__(self, valuer: RelicChoiceValuer, request, context, params, score: Callable):
        self.policy = valuer.policy
        self.pools = valuer.pools
        self.repository = valuer.repository
        self.request = request
        self.context = context
        self.params = params
        self.samples = int(params["samples"])
        self.max_variants = int(params["max_variants"])
        self.score = score
        self.character = str(request.world.character or "").upper()
        self.info: Dict[str, tuple] = {}
        odds = context.params["card_reward_odds"]
        high = int(request.world.ascension or 0) >= int(odds.get("ascension_threshold", 99))
        self.regular_odds = odds["ascension" if high else "base"]["MONSTER"]
        self.rare_offset = float(odds.get("rare_offset_initial", 0.0))

    def value(self, relic_id: str, option_params: Mapping) -> RelicOptionValue:
        fact = self.repository.find_assembly_fact("relics", relic_id)
        profile, profile_gaps = relic_profile(self.repository, relic_id)
        if fact is None or profile is None:
            return RelicOptionValue(relic_id, None, None, {}, (), (f"relic_fact_missing:{relic_id}",))
        gaps: List[str] = list(profile_gaps)
        groups = (fact.get("triage") or {}).get("hook_groups") or {}
        if (groups.get("run") or groups.get("reward")) and "TASK-027" not in str(fact.get("reviewed_by") or ""):
            # Out-of-combat hooks never reviewed for a choice: an empty
            # ``obtain`` would say "nothing", which nobody has checked.
            gaps.append(f"relic_choice:out_of_combat_unreviewed:{relic_id}")
        resources: Dict[str, float] = {}
        self.rng = random.Random(_seed(self.request.decision_id, relic_id))
        self.option_params = option_params
        self.partials: List[str] = []
        variants = [_Variant(1.0, list(self.context.deck), [*self.context.relics, profile])]
        for effect in fact.get("obtain") or ():
            kind = effect.get("kind")
            if kind in RESOURCE_KINDS:
                problem = _add_resource(resources, effect, self.request.world)
                if problem:
                    gaps.append(f"relic_choice:{problem}:{relic_id}")
                continue
            if kind not in DECK_KINDS:
                gaps.append(f"relic_choice:obtain_unmodeled:{relic_id}:{kind}")
                continue
            changed, problem = self.apply(variants, effect)
            if changed is None:
                gaps.append(f"relic_choice:{problem}:{relic_id}")
            else:
                variants = _cap(changed, self.max_variants, self.rng)
            gaps.extend(f"relic_choice:{note}:{relic_id}" for note in self.partials)
            self.partials = []
        recurring = 0.0
        for mechanic in fact.get("between_fights") or ():
            gain, problem = self.between_fights(mechanic, variants, resources)
            if problem:
                gaps.append(f"relic_choice:between_fights_unmodeled:{relic_id}:{problem}")
            recurring += gain
        immediate = sum(v.weight * self.score(v.cards, v.relics) for v in variants) + recurring
        # Looking ahead over the act's coming rewards, as a card reward does:
        # once on the likeliest result, which the others move by their
        # difference in immediate fitness.
        likeliest = max(variants, key=lambda v: v.weight)
        relics = likeliest.relics
        ahead, _ = self.policy._lookahead(
            self.request, {"option": list(likeliest.cards)},
            lambda cards: self.score(cards, relics), self.context.needs, self.context.params,
        )
        raw = immediate
        if "option" in ahead:
            raw = ahead["option"] + immediate - self.score(likeliest.cards, likeliest.relics)
        notes = tuple(dict.fromkeys(note for v in variants for note in v.notes))
        return RelicOptionValue(relic_id, raw, immediate, resources, notes, tuple(dict.fromkeys(gaps)))

    # -- between fights ---------------------------------------------------

    def between_fights(self, mechanic: Mapping, variants: List[_Variant], resources: Dict[str, float]) -> tuple:
        """``(points, gap)`` of one recurring or later out-of-combat effect.

        A deck change made now counts once (as every pickup change does). One
        made later counts by the horizon left after it: the rest of its act
        and the later acts, each weighted by how often a run reaches it,
        divided by the same horizon from now (``horizon_after``). So effects
        that pay off in later acts are counted, not cut at the end of this
        one. Resources go to ``resources`` (the exchange prices their own
        future use; later ones are weighted by reaching their act).
        """

        when, modifier = mechanic.get("when"), mechanic.get("modifier")
        if modifier == "gold_gained" and mechanic.get("multiply") == 0:
            resources["gold_forgone"] = resources.get("gold_forgone", 0.0) + self.income_ahead()
            return 0.0, ""
        act = int(self.act_key())
        if when == "combat_end":
            flt = mechanic.get("filter") or {}
            kind = flt.get("fight_kind")
            if set(flt) - {"fight_kind", "act"}:
                return 0.0, "combat_end:filter"
            # (act, expected count, mean horizon weight of one occurrence)
            here = [k for k in self.context.needs.fight_sequence if kind is None or k == kind]
            events = [(act, float(len(here)), self.horizon_after(act, mid=True))]
            reach = 1.0
            for later in range(act + 1, 4):
                reach *= float(self.params["reach_next_act"][str(later - 1)])
                counts = self.params["fights_per_act"][str(later)]
                n = sum(float(counts[k]) for k in counts if kind is None or k == kind)
                events.append((later, reach * n, self.horizon_after(later, mid=True)))
            if "act" in flt:
                events = [e for e in events if e[0] == int(flt["act"])]
                if kind == "BOSS":     # the act's boss closes it: what follows is the later acts
                    events = [(a, n, self.horizon_after(a, mid=False, end=True)) for a, n, _ in events]
            every = max(1, int(mechanic.get("every") or 1))
            limit = mechanic.get("limit") or {}
            cap = next((float(limit[k]) for k in ("times", "combats") if isinstance(limit.get(k), (int, float))), None)
            occurrences, weighted = 0.0, 0.0
            for _, n, weight in events:
                n = n / every
                if cap is not None:
                    n = max(0.0, min(n, cap - occurrences))
                occurrences += n
                weighted += n * weight
        elif when == "rest_site" and (mechanic.get("filter") or {}).get("option") == "HEAL":
            occurrences = self.share_left() * float(self.params["rest_heals_per_act"][self.act_key()])
            reach = 1.0
            for later in range(act + 1, 4):
                reach *= float(self.params["reach_next_act"][str(later - 1)])
                occurrences += reach * float(self.params["rest_heals_per_act"][str(later)])
            weighted = 0.0
        else:
            return 0.0, str(modifier or when)
        if occurrences <= 0:
            return 0.0, ""
        effect = mechanic.get("effect") or {}
        kind = effect.get("kind")
        if kind in RESOURCE_KINDS and isinstance(effect.get("amount"), (int, float)):
            target = {"gain_gold": "gold", "gain_max_hp": "max_hp", "heal": "hp"}.get(kind)
            if target is None:
                return 0.0, f"{when}:{kind}"
            resources[target] = resources.get(target, 0.0) + effect["amount"] * occurrences
            return 0.0, ""
        likeliest = max(variants, key=lambda v: v.weight)
        base = self.score(likeliest.cards, likeliest.relics)
        if kind == "upgrade_cards" and effect.get("choice") == "random" and isinstance(effect.get("count"), int):
            targets = self.targets(likeliest.cards, effect, kind)
            if not targets:
                return 0.0, ""
            gains = [
                self.score([*likeliest.cards[:i], likeliest.cards[i].upgraded_profile, *likeliest.cards[i + 1:]],
                           likeliest.relics) - base
                for i in targets
            ]
            each = min(effect["count"], len(targets)) * sum(gains) / len(gains)
        elif kind == "offer_rewards" and all(r.get("type") == "relic" for r in effect.get("rewards") or ()):
            drawn = [likeliest]
            for reward in effect.get("rewards") or ():
                drawn = self.random_relic(drawn, reward.get("rarity"), skippable=True)
                if drawn is None:
                    return 0.0, f"{when}:offer_rewards"
            each = sum(v.weight * self.score(v.cards, v.relics) for v in drawn) / likeliest.weight - base
        elif kind == "add_card" and effect.get("card"):
            profile, _ = self.policy._card(_entity_id(effect["card"]), 1 if effect.get("upgraded") else 0)
            if profile is None:
                return 0.0, f"{when}:add_card"
            each = self.score([*likeliest.cards, profile], likeliest.relics) - base
        else:
            return 0.0, f"{when}:{kind}"
        return each * weighted, ""

    def horizon_after(self, act: int, *, mid: bool = True, end: bool = False) -> float:
        """Share of the run's remaining horizon (from now) left after an event in ``act``:
        at the middle of what is left of it (``mid``), or at its end (``end``)."""

        here = int(self.act_key())
        reach_to = {here: 1.0}
        reach = 1.0
        for later in range(here + 1, 4):
            reach *= float(self.params["reach_next_act"][str(later - 1)])
            reach_to[later] = reach
        whole = self.share_left() + sum(reach_to[b] for b in reach_to if b > here)
        own = self.share_left() if act == here else 1.0
        within = 0.0 if end else (own / 2.0 if mid else own)
        after = reach_to[act] * within + sum(reach_to[b] for b in reach_to if b > act)
        return after / whole if whole > 0 else 0.0

    def per(self, variants: Sequence[_Variant]) -> int:
        """Draws per current result at a random step: about ``samples`` in all,
        so a chain of random steps grows by addition, not multiplication."""

        return max(1, self.samples // max(1, len(variants)))

    def common(self, *key) -> random.Random:
        """A generator shared by every option of this decision for the same
        draw (the same pool, the same sample index): options are compared on
        the same random results, so sampling noise does not decide between them."""

        return random.Random(_seed(self.request.decision_id, repr(key)))

    def act_key(self) -> str:
        return str(min(3, max(1, int(self.request.world.act or 1))))

    def share_left(self) -> float:
        """Share of the current act still ahead, by floor."""
        act = int(self.act_key())
        floors = float(self.params["act_floors"])
        into = max(0.0, float(self.request.world.floor or 0) - floors * (act - 1) - 1.0)
        return max(0.0, 1.0 - into / floors)

    def income_ahead(self) -> float:
        """Gold the run is expected to gain from here (later acts weighted by reaching them)."""
        act = int(self.act_key())
        income = self.params["gold_income_per_act"]
        total, reach = self.share_left() * float(income[str(act)]), 1.0
        for later in range(act + 1, 4):
            reach *= float(self.params["reach_next_act"][str(later - 1)])
            total += reach * float(income[str(later)])
        return total

    # -- deck changes -----------------------------------------------------

    def apply(self, variants: List[_Variant], effect: Mapping) -> tuple[Optional[List[_Variant]], str]:
        kind = effect["kind"]
        unknown = set(effect.get("filter") or {}) - FILTER_KEYS
        if unknown:
            return None, f"obtain_unmodeled:{kind}:filter:{sorted(unknown)[0]}"
        if kind == "replace_relic":
            return self.replace_relic(variants, effect)
        if kind == "add_card":
            return self.add_cards(variants, effect)
        if kind == "offer_rewards":
            return self.rewards(variants, effect)
        if kind == "gain_relic":
            if effect.get("source") not in RELIC_SOURCES or not isinstance(effect.get("count", 1), int):
                return None, f"obtain_unmodeled:gain_relic:{effect.get('source')}"
            for _ in range(int(effect.get("count", 1))):
                variants = self.random_relic(variants, effect.get("rarity"), skippable=False)
                if variants is None:
                    return None, "obtain_unmodeled:gain_relic:pool"
            return variants, ""
        if kind == "enchant_cards" and "can_enchant" in (effect.get("filter") or {}):
            # The enchantment's own applicability (its fact's can_enchant; none: any card).
            fact = self.repository.find_assembly_fact(
                "enchantments", str(effect.get("enchantment") or "").upper()) or {}
            flt = {k: v for k, v in effect["filter"].items() if k != "can_enchant"}
            flt.update(fact.get("can_enchant") or {})
            effect = {**effect, "filter": flt}
        if kind == "transform_cards" and (effect.get("scope") or "deck") != "deck":
            return None, "obtain_unmodeled:transform_cards:scope"
        count = effect.get("count", 1)
        choice = effect.get("choice") or "choose"
        if (count == "all" or choice == "all") and kind == "transform_cards":
            # Every matching card becomes a random result: each sample draws them all.
            result = []
            for v in variants:
                positions = self.targets(v.cards, effect, kind)
                for s in range(self.samples if positions else 1):
                    cards = list(v.cards)
                    for rank, i in enumerate(positions):
                        pool = self.transform_pool(effect, cards[i])
                        if not pool:
                            return None, f"obtain_unmodeled:{kind}:result"
                        cards[i] = self.common("transform", _ids(pool), s, rank).choice(pool)
                    result.append(_Variant(v.weight / (self.samples if positions else 1), cards, v.relics,
                                           [*v.notes, _note(kind, None, self.policy)]))
            return result, ""
        if count == "all" or choice == "all":
            if kind not in ("upgrade_cards", "enchant_cards"):
                return None, f"obtain_unmodeled:{kind}:all"
            result = []
            for v in variants:
                cards = list(v.cards)
                for i in self.targets(cards, effect, kind):
                    changed = self.changed(kind, effect, cards[i])
                    if changed is None:
                        return None, f"obtain_unmodeled:{kind}:result"
                    cards[i] = changed
                result.append(_Variant(v.weight, cards, v.relics, [*v.notes, _note(kind, None, self.policy)]))
            return result, ""
        if choice not in CHOICES or not isinstance(count, int) or count < 1:
            return None, f"obtain_unmodeled:{kind}:choice"
        for _ in range(count):
            step: List[_Variant] = []
            self.per_now = self.per(variants)
            for v in variants:
                positions = self.targets(v.cards, effect, kind)
                if not positions:
                    step.append(v)
                    continue
                if choice == "choose":
                    outcomes = self.best_change(v, positions, effect)
                    if (outcomes is not None and effect.get("min_count") == 0
                            and sum(o.weight * self.score(o.cards, o.relics) for o in outcomes)
                            <= v.weight * self.score(v.cards, v.relics)):
                        outcomes = [v]      # optional (e.g. Claws: 0-6 cards): stop when it no longer helps
                elif choice in ("first", "last"):
                    # The game takes the first (or last) match in deck order.
                    outcomes = self.change(v, positions[0 if choice == "first" else -1], effect)
                else:
                    picks = positions if len(positions) <= self.samples else self.rng.sample(positions, self.samples)
                    outcomes = []
                    for i in picks:
                        changed = self.change(v, i, effect)
                        if changed is None:
                            outcomes = None
                            break
                        outcomes.extend(_Variant(o.weight / len(picks), o.cards, o.relics, o.notes) for o in changed)
                if outcomes is None:
                    return None, f"obtain_unmodeled:{kind}:result"
                step.extend(outcomes)
            variants = _cap(step, self.max_variants, self.rng)
        return variants, ""

    def targets(self, cards: Sequence[Optional[CardProfile]], effect: Mapping, kind: str) -> List[int]:
        flt = dict(effect.get("filter") or {})
        # Removal and transform take only removable cards (CardModel.IsRemovable:
        # not Eternal; IsTransformable is the same for a card in the deck).
        if kind in ("remove_from_deck", "transform_cards"):
            flt["removable"] = True
        return [
            i for i, card in enumerate(cards)
            if card is not None and self.matches(card, flt)
            and (kind != "upgrade_cards" or card.upgraded_profile is not None)
        ]

    def card_info(self, card_id: str) -> tuple:
        """(color, rarity, keywords) of a card, from the catalog and its fact."""

        if card_id not in self.info:
            entity = self.repository.find_card(card_id) or {}
            fact = self.repository.find_assembly_fact("cards", card_id) or {}
            self.info[card_id] = (
                str(entity.get("color") or "").lower(),
                str(entity.get("rarity_key") or "").lower(),
                frozenset(str(k) for k in fact.get("keywords") or ()),
            )
        return self.info[card_id]

    def matches(self, card: CardProfile, flt: Mapping) -> bool:
        _, rarity, keywords = self.card_info(card.card_id)
        types = flt.get("card_type")
        if types and card.card_type not in (types if isinstance(types, (list, tuple)) else [types]):
            return False
        tags = flt.get("tag")
        if tags and not any(tag in card.tags for tag in (tags if isinstance(tags, (list, tuple)) else [tags])):
            return False
        if flt.get("card"):
            wanted = flt["card"] if isinstance(flt["card"], (list, tuple)) else [flt["card"]]
            if card.card_id not in {_entity_id(name) for name in wanted}:
                return False
        if flt.get("rarity"):
            wanted = flt["rarity"] if isinstance(flt["rarity"], (list, tuple)) else [flt["rarity"]]
            if rarity not in {str(r).lower() for r in wanted}:
                return False
        if (flt.get("removable") or flt.get("transformable")) and "Eternal" in keywords:
            return False
        if flt.get("upgradable") and card.upgraded_profile is None:
            return False
        return True

    def best_change(self, variant: _Variant, positions: Sequence[int], effect: Mapping) -> Optional[List[_Variant]]:
        """The chosen card: the one whose change leaves the fittest deck (copies are interchangeable)."""

        best_value, best = None, None
        seen = set()
        for i in positions:
            if id(variant.cards[i]) in seen:
                continue
            seen.add(id(variant.cards[i]))
            outcomes = self.change(variant, i, effect)
            if outcomes is None:
                return None
            value = sum(o.weight * self.score(o.cards, o.relics) for o in outcomes)
            if best_value is None or value > best_value:
                best_value, best = value, outcomes
        return best

    def change(self, variant: _Variant, i: int, effect: Mapping) -> Optional[List[_Variant]]:
        """The results of changing the card at position ``i`` (weights sum to the variant's)."""

        kind = effect["kind"]
        cards, card = variant.cards, variant.cards[i]
        note = [*variant.notes, _note(kind, card, self.policy)]
        if kind == "remove_from_deck":
            return [_Variant(variant.weight, cards[:i] + cards[i + 1:], variant.relics, note)]
        if kind == "transform_cards":
            pool = self.transform_pool(effect, card)
            if not pool:
                return None
            draws = getattr(self, "per_now", self.samples)
            picks = pool if len(pool) <= draws else self.common("transform", _ids(pool), len(cards)).sample(pool, draws)
            return [
                _Variant(variant.weight / len(picks), cards[:i] + [result] + cards[i + 1:], variant.relics, note)
                for result in picks
            ]
        changed = self.changed(kind, effect, card)
        if changed is None:
            return None
        return [_Variant(variant.weight, cards[:i] + [changed] + cards[i + 1:], variant.relics, note)]

    def changed(self, kind: str, effect: Mapping, card: CardProfile) -> Optional[CardProfile]:
        if kind == "upgrade_cards":
            return card.upgraded_profile
        if kind == "enchant_cards":
            fact = self.repository.find_assembly_fact("enchantments", str(effect.get("enchantment") or "").upper())
            profile, enchant_gaps = enchant(card, fact, float(effect.get("amount") or 1.0))
            return None if enchant_gaps else profile
        return None

    def transform_pool(self, effect: Mapping, card: CardProfile) -> List[CardProfile]:
        into = effect.get("into")
        upgrades = 1 if effect.get("upgraded") else 0
        if isinstance(into, Mapping) and isinstance(into.get("map"), Mapping):
            mapped = {_entity_id(k): _entity_id(v) for k, v in into["map"].items()}.get(card.card_id)
            if mapped is None:
                return []
            keep = 1 if effect.get("keep_upgrade") and card.upgraded else upgrades
            profile, card_gaps = self.policy._card(mapped, keep)
            self.partials.extend(f"added_card_partial:{g}" for g in card_gaps if f"added_card_partial:{g}" not in self.partials)
            return [profile] if profile is not None else []
        if isinstance(into, str):
            profile, card_gaps = self.policy._card(_entity_id(into), upgrades)
            self.partials.extend(f"added_card_partial:{g}" for g in card_gaps if f"added_card_partial:{g}" not in self.partials)
            return [profile] if profile is not None else []
        if not isinstance(into, Mapping) or into.get("pool") not in ("character", "original"):
            return []
        # "original": the transformed card's own pool (its color).
        color = self.card_info(card.card_id)[0] if into["pool"] == "original" else self.character.lower()
        pool = self.pool(color, into.get("rarity"), into.get("card_type"), upgrades=upgrades)
        if into.get("same_type"):
            pool = [c for c in pool if c.card_type == card.card_type]
        return [c for c in pool if c.card_id != card.card_id]

    def pool(self, color: str, rarity, card_type, *, tag=None, upgrades: int = 0) -> List[CardProfile]:
        """Cards of a color (a character or colorless) at the given rarities, with facts."""

        wanted = None if not rarity else frozenset(
            str(r).lower() for r in (rarity if isinstance(rarity, (list, tuple)) else [rarity])
        )
        key = (color.lower(), wanted, card_type, tag, upgrades)
        if key in self.pools:
            return self.pools[key]
        result = []
        for card_id, card_rarity in self.repository.character_cards(color.lower()):
            if wanted is not None and card_rarity not in wanted:
                continue
            fact = self.repository.find_assembly_fact("cards", card_id)
            if fact is None or fact.get("multiplayer_only"):
                continue
            profile, _ = self.policy._card(card_id, upgrades)
            if profile is None or (card_type and profile.card_type != card_type) or (tag and tag not in profile.tags):
                continue
            result.append(profile)
        self.pools[key] = result
        return result

    def add_cards(self, variants: List[_Variant], effect: Mapping) -> tuple[Optional[List[_Variant]], str]:
        count = effect.get("count", 1)
        if not isinstance(count, int) or count < 1:
            return None, "obtain_unmodeled:add_card:count"
        upgrades = 1 if effect.get("upgraded") else 0
        if effect.get("card"):
            card_id = str(effect["card"])
            profile, card_gaps = self.policy._card(card_id.upper(), upgrades)
            if profile is None:     # written as the class name (CurseOfTheBell)
                profile, card_gaps = self.policy._card(class_to_entity_id(card_id), upgrades)
            if profile is None:
                return None, f"obtain_unmodeled:add_card:{effect['card']}"
            # A part of that card the engine cannot value (e.g. losing max HP
            # on every play) is a part of this option it cannot value.
            self.partials.extend(f"added_card_partial:{g}" for g in card_gaps)
            name = self.policy._card_name(profile.card_id)
            return [
                _Variant(v.weight, [*v.cards, *([profile] * count)], v.relics,
                         [*v.notes, f"加入 {name}" + (f" ×{count}" if count > 1 else "")])
                for v in variants
            ], ""
        spec = effect.get("pool")
        if not isinstance(spec, Mapping) or spec.get("source") not in ("character", "character_of", "colorless"):
            return None, "obtain_unmodeled:add_card:pool"
        character = (
            str(self.option_params.get("character") or "").upper() if spec["source"] == "character_of"
            else "COLORLESS" if spec["source"] == "colorless"
            else self.character
        )
        if isinstance(spec.get("rarity_counts"), Mapping):
            return self.pick_any(variants, effect, character, spec, upgrades)
        excluded = {_entity_id(name) for name in spec.get("exclude") or ()}
        pool = [c for c in self.pool(character, spec.get("rarity"), spec.get("card_type"), tag=spec.get("tag"),
                                     upgrades=upgrades) if c.card_id not in excluded]
        if not pool:
            return None, "obtain_unmodeled:add_card:pool"
        shown = int(effect.get("choose_from") or count)
        skippable = bool(effect.get("skippable"))
        note = (f"从 {shown} 张中选 {count} 张" if shown > count else f"随机加入 {count} 张") + "（按卡池抽样）"
        result = []
        per = self.per(variants)
        for vi, v in enumerate(variants):
            for k in range(per):
                s = vi * per + k
                offer = self.common("add", _ids(pool), shown, s).sample(pool, min(shown, len(pool)))
                cards = list(v.cards)
                if shown <= count:
                    cards.extend(offer)          # every card shown is added
                else:
                    # Pick ``count`` of the shown cards, best first; a
                    # skippable screen takes none unless it helps.
                    for _ in range(count):
                        best, best_value = None, self.score(cards, v.relics) if skippable else None
                        for card in offer:
                            value = self.score([*cards, card], v.relics)
                            if best_value is None or value > best_value:
                                best, best_value = card, value
                        if best is None:
                            break
                        cards.append(best)
                        offer.remove(best)
                result.append(_Variant(v.weight / per, cards, v.relics, [*v.notes, note]))
        return result, ""

    def rewards(self, variants: List[_Variant], effect: Mapping) -> tuple[Optional[List[_Variant]], str]:
        """Card reward screens are valued as picks (best of the offer, or none
        when skippable); relic, potion and other rewards are left out (a gap)."""

        rewards = effect.get("rewards") or ()
        problem = ""
        for reward in rewards:
            if reward.get("type") == "relic" and (reward.get("source") or "next_in_pool") in RELIC_SOURCES:
                drawn = self.random_relic(variants, reward.get("rarity"), skippable=True)
                if drawn is None:
                    problem = "obtain_unmodeled:offer_rewards:relic"
                    continue
                variants = drawn
                continue
            if reward.get("type") != "card":
                problem = f"obtain_unmodeled:offer_rewards:{reward.get('type')}"
                continue
            spec = reward.get("pool") or {}
            shown = int(reward.get("options") or 3)
            offers = self.reward_offers(spec, shown)
            if offers is None:
                problem = "obtain_unmodeled:offer_rewards:card_pool"
                continue
            result = []
            per = self.per(variants)
            for vi, v in enumerate(variants):
                for offer in (offers[(vi * per + k) % len(offers)] for k in range(per)):
                    cards = list(v.cards)
                    best, best_value = None, self.score(cards, v.relics) if reward.get("skippable", True) else None
                    for card in offer:
                        value = self.score([*cards, card], v.relics)
                        if best_value is None or value > best_value:
                            best, best_value = card, value
                    if best is not None:
                        cards.append(best)
                    result.append(_Variant(v.weight / per, cards, v.relics,
                                           [*v.notes, f"一次 {shown} 选 1 的卡牌奖励（按卡池抽样）"]))
            variants = _cap(result, self.max_variants, self.rng)
        if problem and not any(r.get("type") in ("card", "relic") for r in rewards):
            return None, problem
        if problem:
            self.partials.append(problem)
        return variants, ""

    def reward_offers(self, spec: Mapping, shown: int) -> Optional[List[List[CardProfile]]]:
        """Seeded samples of one card reward screen."""

        source = spec.get("source") or "character"
        samples = []
        for s in range(self.samples):
            rng = self.common("reward", repr(sorted(spec.items())), shown, s)
            if source == "other_characters" and spec.get("one_card_per_pool"):
                others = [c for c in CHARACTERS if c != self.character]
                colors = rng.sample(others, min(shown, len(others)))
            elif source in ("character", "colorless"):
                colors = [self.character if source == "character" else "COLORLESS"] * shown
            else:
                return None
            offer = []
            for color in colors:
                rarity = spec.get("rarity") or _roll_rarity(rng, self.regular_odds, self.rare_offset)
                pool = [c for c in self.pool(color, rarity, spec.get("card_type")) if c not in offer]
                if not pool:
                    return None
                offer.append(rng.choice(pool))
            samples.append(offer)
        return samples

    def random_relic(self, variants: List[_Variant], rarity, *, skippable: bool) -> Optional[List[_Variant]]:
        """A relic from the pool (shared and the player's own), rarity rolled by
        the relic odds unless fixed; only its combat mechanics are counted
        (its own pickup effects are a disclosed gap). A skippable reward is
        taken only when it helps."""

        result = []
        per = self.per(variants)
        for vi, v in enumerate(variants):
            owned = {r.relic_id for r in v.relics}
            for k in range(per):
                rng = self.common("relic", str(rarity), vi * per + k, len(v.relics))
                wanted = str(rarity).lower() if rarity and str(rarity).lower() != "rolled" else self.roll_relic_rarity(rng)
                pool = self.relic_pool(wanted, owned)
                if not pool:
                    return None
                relic_id = rng.choice(pool)
                profile, _ = relic_profile(self.repository, relic_id)
                relics = [*v.relics, profile]
                if skippable and self.score(v.cards, relics) <= self.score(v.cards, v.relics):
                    relics = list(v.relics)
                result.append(_Variant(v.weight / per, v.cards, relics,
                                       [*v.notes, f"随机{ {'common': '普通', 'uncommon': '罕见', 'rare': '稀有', 'shop': '商店'}.get(wanted, '')}遗物（按遗物池抽样，只计战斗效果）"]))
        if "random_relic_combat_only" not in self.partials:
            self.partials.append("random_relic_combat_only")
        return _cap(result, self.max_variants, self.rng)

    def roll_relic_rarity(self, rng: random.Random) -> str:
        odds = self.params["relic_rarity_odds"]
        roll = rng.random()
        if roll < float(odds["common"]):
            return "common"
        if roll < float(odds["common"]) + float(odds["uncommon"]):
            return "uncommon"
        return "rare"

    def relic_pool(self, rarity: str, owned) -> List[str]:
        key = ("relics", rarity, self.character)
        if key not in self.pools:
            self.pools[key] = [
                relic_id for relic_id, relic_rarity, pool in self.repository.relic_catalog()
                if relic_rarity == rarity and pool in ("shared", self.character.lower())
                and relic_profile(self.repository, relic_id)[0] is not None
            ]
        return [r for r in self.pools[key] if r not in owned]

    def pick_any(self, variants, effect, character, spec, upgrades) -> tuple[Optional[List[_Variant]], str]:
        """A screen of fixed numbers of cards per rarity, keeping any number of
        them: each sample keeps the cards that improve the deck, best first."""

        pools = {}
        for rarity, count in spec["rarity_counts"].items():
            pool = self.pool(character, rarity, spec.get("card_type"), upgrades=upgrades)
            if len(pool) < int(count):
                return None, "obtain_unmodeled:add_card:pool"
            pools[rarity] = (pool, int(count))
        result = []
        per = self.per(variants)
        for vi, v in enumerate(variants):
            for k in range(per):
                s = vi * per + k
                offer = [c for rarity, (pool, count) in pools.items()
                         for c in self.common("keep_any", rarity, _ids(pool), s).sample(pool, count)]
                cards = list(v.cards)
                current = self.score(cards, v.relics)
                # Each card's own gain, best first; keep adding while the deck improves
                # (linear in the offer: a full greedy search is quadratic).
                ranked = sorted(offer, key=lambda c: -self.score([*v.cards, c], v.relics))
                for card in ranked:
                    if self.score([*v.cards, card], v.relics) <= self.score(v.cards, v.relics):
                        break
                    value = self.score([*cards, card], v.relics)
                    if value <= current:
                        break
                    cards.append(card)
                    current = value
                result.append(_Variant(v.weight / per, cards, v.relics,
                                       [*v.notes, f"从 {sum(c for _, c in pools.values())} 张中保留有帮助的牌（按卡池抽样）"]))
        return result, ""

    def replace_relic(self, variants: List[_Variant], effect: Mapping) -> tuple[Optional[List[_Variant]], str]:
        to = effect.get("to")
        if effect.get("from") == "starter" and isinstance(to, Mapping):
            # The first starter relic the player has, by the fact's map (else its fallback).
            mapping = {_entity_id(k): _entity_id(v) for k, v in (to.get("map") or {}).items()}
            result = []
            for v in variants:
                starter = next((r for r in v.relics if _relic_rarity(self.repository, r.relic_id) == "starter"), None)
                if starter is None:
                    return None, "obtain_unmodeled:replace_relic:no_starter"
                new = mapping.get(starter.relic_id) or _entity_id(str(to.get("fallback") or ""))
                profile, _ = relic_profile(self.repository, new)
                if profile is None:
                    return None, "obtain_unmodeled:replace_relic"
                result.append(_Variant(v.weight, v.cards, [*(r for r in v.relics if r is not starter), profile],
                                       [*v.notes, f"{self.policy._relic_name(starter.relic_id)} 换成 {self.policy._relic_name(new)}"]))
            return result, ""
        old, new = str(effect.get("from") or "").upper(), str(to or "").upper()
        profile, _ = relic_profile(self.repository, new)
        if profile is None or not old:
            return None, "obtain_unmodeled:replace_relic"
        return [
            _Variant(v.weight, v.cards, [*(r for r in v.relics if r.relic_id != old), profile],
                     [*v.notes, f"{self.policy._relic_name(old)} 换成 {self.policy._relic_name(new)}"])
            for v in variants
        ], ""


def _add_resource(resources: Dict[str, float], effect: Mapping, world) -> str:
    """Adds a gold / HP change; returns a gap name when it cannot be read."""

    kind, amount = effect["kind"], effect.get("amount")
    if kind == "lose_gold" and effect.get("amount_from") == "all":
        if world.gold is None:
            return "gold_missing"
        amount = world.gold
    if kind == "heal" and effect.get("amount_from") == "full":
        if world.hp is None or world.max_hp is None:
            return "hp_missing"
        amount = world.max_hp - world.hp
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return f"obtain_unmodeled:{kind}:amount"
    if kind in ("gain_gold", "lose_gold"):
        resources["gold"] = resources.get("gold", 0.0) + (amount if kind == "gain_gold" else -amount)
    elif kind in ("gain_max_hp", "lose_max_hp"):
        resources["max_hp"] = resources.get("max_hp", 0.0) + (amount if kind == "gain_max_hp" else -amount)
    elif kind == "heal":
        if world.hp is None or world.max_hp is None:
            return "hp_missing"
        resources["hp"] = resources.get("hp", 0.0) + min(amount, max(0, world.max_hp - world.hp))
    else:  # hp_loss
        resources["hp"] = resources.get("hp", 0.0) - amount
    return ""


def _entity_id(name) -> str:
    """A card or relic named by its ID (BASH) or its class name (Bash)."""

    text = str(name)
    return text.upper() if text.isupper() or "_" in text else class_to_entity_id(text)


def _relic_rarity(repository, relic_id: str) -> str:
    return str((repository.find_relic(relic_id) or {}).get("rarity_key") or "").lower()


def _note(kind: str, card: Optional[CardProfile], policy) -> str:
    name = policy._card_name(card.card_id) if card is not None else "所有符合条件的牌"
    return {
        "remove_from_deck": f"移除 {name}",
        "transform_cards": f"变化 {name}",
        "upgrade_cards": f"升级 {name}",
        "enchant_cards": f"附魔 {name}",
    }[kind]


def _cap(variants: List[_Variant], limit: int, rng: random.Random) -> List[_Variant]:
    """At most ``limit`` results: a sample of them, reweighted to the same total."""

    if len(variants) <= limit:
        return variants
    kept = rng.sample(variants, limit)
    scale = sum(v.weight for v in variants) / sum(v.weight for v in kept)
    return [_Variant(v.weight * scale, v.cards, v.relics, v.notes) for v in kept]


def _ids(pool) -> tuple:
    return tuple(c.card_id for c in pool)


def _seed(decision_id: str, relic_id: str) -> int:
    return int(hashlib.sha256(f"{decision_id}|{relic_id}".encode("utf-8")).hexdigest()[:16], 16)


def _parameter_gaps(params) -> tuple[str, ...]:
    if not isinstance(params, Mapping):
        return (f"parameter_missing:{PARAMETERS_KEY}",)
    gaps = []
    for name in ("samples", "max_variants"):
        value = params.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.{name}")
    for name in ("act_floors",):
        value = params.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.{name}")
    odds = params.get("relic_rarity_odds")
    if not isinstance(odds, Mapping) or not all(
        isinstance(odds.get(r), (int, float)) and 0 <= odds[r] <= 1 for r in ("common", "uncommon", "rare")
    ):
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.relic_rarity_odds")
    fights = params.get("fights_per_act")
    if not isinstance(fights, Mapping) or not all(
        isinstance(fights.get(a), Mapping) and all(
            isinstance(fights[a].get(k), (int, float)) and fights[a][k] >= 0 for k in ("MONSTER", "ELITE", "BOSS"))
        for a in ("1", "2", "3")
    ):
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.fights_per_act")
    for name, acts in (("rest_heals_per_act", ("1", "2", "3")), ("gold_income_per_act", ("1", "2", "3")),
                       ("reach_next_act", ("1", "2"))):
        table = params.get(name)
        if not isinstance(table, Mapping) or not all(
            isinstance(table.get(a), (int, float)) and not isinstance(table.get(a), bool) and table[a] >= 0
            for a in acts
        ):
            gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.{name}")
    return tuple(gaps)

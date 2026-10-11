"""Expected per-turn behaviour of a monster from its assembly-verified fact.

A monster fact (docs/dev/mechanic-facts.md, section "怪物") records the move
state machine exactly as the game builds it. This module walks that machine
to give, for each turn of a fight, the expected attack damage, number of hits,
block gained, debuffs put on the player and cards added to the player's piles.

Random branches are averaged over seeded walks (reproducible); a conditional
branch follows the branch the fact marks as the default. Anything the walk
cannot decide is reported as a gap, never guessed silently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

WALKS = 48
MAX_HOPS = 16
REPEAT_TYPES = {"CannotRepeat": ("max_repeat", 1), "UseOnlyOnce": ("once", True)}
PLAYER_DEBUFFS = frozenset({"Weak", "Frail", "Vulnerable"})


@dataclass(frozen=True)
class MonsterTurns:
    """Per-turn expectations for one monster over ``len(attack)`` turns."""

    hp: float
    attack: tuple          # total attack damage aimed at the player
    hits: tuple            # attack hits aimed at the player
    block: tuple           # block the monster gains
    debuffs: Mapping[str, tuple]   # stacks of Weak / Frail / Vulnerable applied to the player
    cards: tuple           # cards added to the player's piles
    gaps: tuple = ()


def _value(effect: Mapping, key: str, ascension: int) -> Optional[float]:
    level = effect.get("ascension_level")
    raised = effect.get(f"{key}_ascension")
    if raised is not None and level is not None and ascension >= int(level):
        value = raised
    else:
        value = effect.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _branch_rule(branch: Mapping) -> Mapping:
    rule = dict(branch)
    kind, value = REPEAT_TYPES.get(str(branch.get("repeat_type") or ""), (None, None))
    if kind and rule.get(kind) is None:
        rule[kind] = value
    return rule


class _Walker:
    def __init__(
        self, fact: Mapping, ascension: int, rng: random.Random, gaps: set,
        initial_state: Optional[str] = None,
    ):
        pattern = fact.get("pattern") or {}
        self.states: Dict[str, Mapping] = {
            s["id"]: s for s in pattern.get("states") or () if isinstance(s, Mapping) and s.get("id")
        }
        self.moves: Dict[str, Mapping] = {m["id"]: m for m in fact.get("moves") or () if m.get("id")}
        self.ascension = ascension
        self.rng = rng
        self.gaps = gaps
        # An encounter may fix the opening state (slot or a field it sets).
        self.state = initial_state if initial_state in self.states else pattern.get("initial")
        if self.state not in self.states:
            # The opening move is set outside the state machine (encounter
            # slot, RNG): unknown, so start from the first move state.
            self.gaps.add("monster_opening_unknown")
            self.state = next((sid for sid, s in self.states.items() if s.get("type") == "move"), None)
        self.last: Optional[str] = None
        self.streak = 0
        self.since: Dict[str, int] = {}
        self.used: set = set()

    def _allowed(self, branch: Mapping) -> bool:
        target = branch.get("to")
        if branch.get("once") and target in self.used:
            return False
        limit = branch.get("max_repeat")
        if isinstance(limit, (int, float)) and target == self.last and self.streak >= int(limit):
            return False
        cooldown = branch.get("cooldown")
        if isinstance(cooldown, (int, float)) and target in self.since and self.since[target] < int(cooldown):
            return False
        return True

    def _resolve(self) -> Optional[Mapping]:
        """Follow random/conditional states to the next move state."""

        for _ in range(MAX_HOPS):
            state = self.states.get(self.state)
            if state is None:
                return None
            kind = state.get("type")
            if kind == "move":
                return state
            branches = [_branch_rule(b) for b in state.get("branches") or () if b.get("to") in self.states]
            if not branches:
                return None
            if kind == "conditional":
                chosen = next((b for b in branches if b.get("default") is True), None)
                if chosen is None:
                    # Depends on player actions or RNG: every branch is possible.
                    self.gaps.add(f"monster_branch_undetermined:{state['id']}")
                    chosen = self.rng.choice(branches)
            else:
                allowed = [b for b in branches if self._allowed(b)] or branches[:1]
                weights = [max(0.0, float(b.get("weight") or 1.0)) for b in allowed]
                chosen = self.rng.choices(allowed, weights=weights if sum(weights) > 0 else None)[0]
            self.state = chosen["to"]
        return None

    def step(self) -> Optional[Mapping]:
        state = self._resolve()
        if state is None:
            return None
        sid = state["id"]
        self.streak = self.streak + 1 if sid == self.last else 1
        self.last = sid
        for key in self.since:
            self.since[key] += 1
        self.since[sid] = 0
        self.used.add(sid)
        self.state = state.get("next") if state.get("next") in self.states else sid
        return self.moves.get(state.get("move"))


def monster_turns(
    fact: Mapping, turns: int, ascension: int = 0, seed: int = 0, walks: int = WALKS,
    initial_state: Optional[str] = None,
) -> MonsterTurns:
    """Expected behaviour for each of the first ``turns`` turns."""

    hp = fact.get("hp") or {}
    raised = int(ascension) >= int(hp.get("ascension_level") or 10 ** 6)
    low = hp.get("min_ascension" if raised and hp.get("min_ascension") is not None else "min") or 0
    high = hp.get("max_ascension" if raised and hp.get("max_ascension") is not None else "max") or low
    gaps: set = set()
    attack = [0.0] * turns
    hits = [0.0] * turns
    block = [0.0] * turns
    cards = [0.0] * turns
    debuffs: Dict[str, List[float]] = {name: [0.0] * turns for name in PLAYER_DEBUFFS}
    strength_start = sum(
        _value(e, "amount", ascension) or 0.0 for e in fact.get("innate") or ()
        if e.get("kind") == "apply_power" and e.get("power") == "Strength" and e.get("target") == "self"
    )
    # A machine with no random state and no undetermined branch always
    # walks the same way: once is exact.
    states = (fact.get("pattern") or {}).get("states") or ()
    fixed = not any(
        s.get("type") == "random" or (
            s.get("type") == "conditional"
            and not any(b.get("default") is True for b in s.get("branches") or ())
        )
        for s in states if isinstance(s, Mapping)
    )
    walks = 1 if fixed else max(1, walks)
    for walk in range(walks):
        walker = _Walker(fact, ascension, random.Random(seed * 1009 + walk), gaps, initial_state)
        strength = strength_start
        for turn in range(turns):
            move = walker.step()
            if move is None:
                gaps.add("monster_pattern_unresolved")
                break
            for effect in move.get("effects") or ():
                kind, target = effect.get("kind"), effect.get("target")
                amount = _value(effect, "amount", ascension)
                if kind == "damage" and target == "player":
                    count = _value(effect, "hits", ascension) or 1.0
                    if amount is None:
                        gaps.add(f"monster_damage_unresolved:{move['id']}")
                        continue
                    attack[turn] += max(0.0, amount + strength) * count
                    hits[turn] += count
                elif kind == "block" and target == "self" and amount is not None:
                    block[turn] += amount
                elif kind == "apply_power" and amount is not None:
                    power = effect.get("power")
                    if target == "self" and power == "Strength":
                        strength += amount
                    elif target == "player" and power in PLAYER_DEBUFFS:
                        debuffs[power][turn] += amount
                elif kind == "add_card" and target == "player":
                    count = _value(effect, "count", ascension)
                    if count is not None:
                        cards[turn] += count
    scale = 1.0 / max(1, walks)
    return MonsterTurns(
        hp=(float(low) + float(high)) / 2.0,
        attack=tuple(v * scale for v in attack),
        hits=tuple(v * scale for v in hits),
        block=tuple(v * scale for v in block),
        debuffs={name: tuple(v * scale for v in values) for name, values in debuffs.items()},
        cards=tuple(v * scale for v in cards),
        gaps=tuple(sorted(gaps)),
    )


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

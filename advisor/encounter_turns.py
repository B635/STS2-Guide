"""Expected per-turn pressure of one encounter, from encounter and monster facts.

An encounter fact lists the monsters a fight really spawns (each line-up with
its probability, and the opening state the encounter fixes). Combined with
``monster_turns`` this gives what the player faces each turn: attack damage,
attack hits, the group's HP and size, debuffs and cards pushed on the player.

Monsters do not all live for the whole fight. Without simulating combat, the
group is assumed to be killed one monster at a time at an even pace over the
fight, in spawn order: monster ``k`` of ``n`` acts until turn ``(k+1)·T/n``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

from advisor.monster_turns import PLAYER_DEBUFFS, monster_turns


@dataclass(frozen=True)
class EncounterTurns:
    hp: float                       # total HP of the spawned monsters
    enemies: float                  # monsters spawned
    attack: tuple                   # attack damage aimed at the player, per turn
    hits: tuple                     # attack hits aimed at the player, per turn
    debuffs: Mapping[str, tuple]    # Weak / Frail / Vulnerable stacks applied, per turn
    cards: tuple                    # cards added to the player's piles, per turn
    gaps: tuple = ()


def _alive_share(index: int, count: int, turn: int, turns: int) -> float:
    """Share of ``turn`` (0-based) monster ``index`` of ``count`` still acts."""

    dies_at = (index + 1) * turns / count
    return max(0.0, min(1.0, dies_at - turn))


def encounter_turns(
    encounter: Mapping,
    monster_fact: Callable[[str], Optional[Mapping]],
    turns: int,
    ascension: int = 0,
    seed: int = 0,
) -> Optional[EncounterTurns]:
    """Weighted over the encounter's line-ups; ``None`` without any line-up."""

    variants = [v for v in encounter.get("variants") or () if v.get("spawns")]
    total_weight = sum(float(v.get("weight") or 0.0) for v in variants)
    if not variants or total_weight <= 0:
        return None
    attack = [0.0] * turns
    hits = [0.0] * turns
    cards = [0.0] * turns
    debuffs: Dict[str, List[float]] = {name: [0.0] * turns for name in PLAYER_DEBUFFS}
    hp = enemies = 0.0
    gaps: set = set()
    for variant in variants:
        share = float(variant.get("weight") or 0.0) / total_weight
        spawns = variant["spawns"]
        for index, spawn in enumerate(spawns):
            monster_id = str(spawn.get("monster") or "")
            fact = monster_fact(monster_id)
            if fact is None:
                gaps.add(f"monster_fact_missing:{monster_id}")
                continue
            walked = monster_turns(
                fact, turns, ascension, seed=seed + index, initial_state=spawn.get("initial_state"),
            )
            gaps.update(f"{gap}:{monster_id}" if ":" not in gap else gap for gap in walked.gaps)
            hp += share * walked.hp
            enemies += share
            for turn in range(turns):
                alive = share * _alive_share(index, len(spawns), turn, turns)
                attack[turn] += alive * walked.attack[turn]
                hits[turn] += alive * walked.hits[turn]
                cards[turn] += alive * walked.cards[turn]
                for name in PLAYER_DEBUFFS:
                    debuffs[name][turn] += alive * walked.debuffs[name][turn]
    return EncounterTurns(
        hp=hp,
        enemies=enemies,
        attack=tuple(attack),
        hits=tuple(hits),
        debuffs={name: tuple(values) for name, values in debuffs.items()},
        cards=tuple(cards),
        gaps=tuple(sorted(gaps)),
    )

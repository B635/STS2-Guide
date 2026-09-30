"""Shared, read-only Boss preparation facts; never a predicted turn sequence."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from storage.relational import RelationalRepository


def _positive(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def monster_peak_attack(
    moves: Iterable[Mapping[str, Any]], *, ascension: int,
) -> tuple[float, bool]:
    """Largest explicit attack, including fixed hits, not a sum of alternatives."""
    attacks: list[float] = []
    for move in moves:
        if "ATTACK" not in str(move.get("intent") or "").upper():
            continue
        primary, fallback = (
            ("damage_ascension", "damage_normal") if ascension >= 9
            else ("damage_normal", "damage_ascension")
        )
        damage = _positive(move.get(primary)) or _positive(move.get(fallback))
        if damage:
            attacks.append(damage * max(1.0, _positive(move.get("hit_count"))))
    return (max(attacks), True) if attacks else (0.0, False)


@dataclass(frozen=True)
class BossEncounterFacts:
    encounter_id: str
    total_hp: float
    peak_attack_pressure: float
    member_count: int


@dataclass(frozen=True)
class BossPreparation:
    encounters: tuple[BossEncounterFacts, ...]
    data_gaps: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return bool(self.encounters) and not self.data_gaps

    def route_threat(self, expected: float) -> float:
        # Preserve the existing route risk formula, including its fallback.
        if not self.encounters:
            return expected
        return round(sum(
            expected + min(12.0, row.total_hp / 45.0 + row.peak_attack_pressure / 18.0)
            for row in self.encounters
        ) / len(self.encounters), 3)


def build_boss_preparation(
    repository: RelationalRepository, identifiers: Iterable[str], *, ascension: int,
) -> BossPreparation:
    ids = tuple(dict.fromkeys(identifiers))
    if not ids:
        return BossPreparation((), ("boss_encounter_ids_missing",))
    encounters: list[BossEncounterFacts] = []
    gaps: list[str] = []
    for identifier in ids:
        encounter = repository.encounter_profile(identifier)
        if encounter is None:
            gaps.append(f"unknown_boss_encounter:{identifier}")
            continue
        monsters = encounter.get("_monsters") or []
        if not monsters:
            gaps.append(f"boss_encounter_members_missing:{identifier}")
            continue
        hp, attacks = 0.0, 0.0
        for monster in monsters:
            member = monster.get("id") or "unknown"
            hp_value = (
                _positive(monster.get("max_hp_ascension")) if ascension >= 8
                else _positive(monster.get("max_hp"))
            ) or _positive(monster.get("max_hp")) or _positive(
                monster.get("min_hp_ascension")
            ) or _positive(monster.get("min_hp"))
            if not hp_value:
                gaps.append(f"boss_hp_missing:{identifier}:{member}")
            hp += hp_value
            peak, known = monster_peak_attack(monster.get("moves") or (), ascension=ascension)
            if not known:
                gaps.append(f"boss_attack_moves_missing:{identifier}:{member}")
            attacks += peak
        encounters.append(BossEncounterFacts(identifier, hp, attacks, len(monsters)))
    return BossPreparation(tuple(encounters), tuple(dict.fromkeys(gaps)))

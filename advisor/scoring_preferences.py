"""Bounded local scoring preferences; never game facts or eligibility rules."""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger("sts2-guide")
_DIMENSIONS = ("survival", "immediate_power", "long_term_growth")
# Only explicitly reviewed soft card factors are adjustable.  In particular,
# community priors, mechanic gates, route modes and skip-derived comparisons
# are not classified by substring or weighted a second time.
_CARD_DIMENSIONS = {
    "block_efficiency": "survival",
    "defense_coverage": "survival",
    "critical_hp_defense": "survival",
    "low_hp_defense": "survival",
    "self_harm_cost": "survival",
    "critical_hp_slow_card": "survival",
    "critical_hp_self_harm": "survival",
    "boss_prepare_defense": "survival",
    "damage_efficiency": "immediate_power",
    "attack_coverage": "immediate_power",
    "aoe_value": "immediate_power",
    "aoe_coverage": "immediate_power",
    "early_act_frontload": "immediate_power",
    "boss_prepare_damage": "immediate_power",
    "scaling_value": "long_term_growth",
    "upgrade_value": "long_term_growth",
    "upgraded_offer": "long_term_growth",
    "late_act_scaling": "long_term_growth",
    "boss_prepare_scaling": "long_term_growth",
}


@dataclass(frozen=True)
class ScoringPreferences:
    survival: float = 1.0
    immediate_power: float = 1.0
    long_term_growth: float = 1.0

    def __post_init__(self) -> None:
        for name in _DIMENSIONS:
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not 0.75 <= value <= 1.25
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a number between 0.75 and 1.25")
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoringPreferences":
        if not isinstance(value, Mapping):
            raise ValueError("scoring preferences must be an object")
        if set(value) != {"schema_version", *_DIMENSIONS}:
            raise ValueError("scoring preferences require exactly the documented fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported scoring preferences schema_version")
        return cls(**{name: value[name] for name in _DIMENSIONS})

    def as_dict(self) -> dict[str, float | int]:
        return {"schema_version": 1, **{name: getattr(self, name) for name in _DIMENSIONS}}

    def adjust(self, delta: float, dimension: str | None, *, code: str = "") -> float:
        if dimension not in _DIMENSIONS:
            return delta
        multiplier = getattr(self, dimension)
        # A preference below 1 must never make a life-cost penalty cheaper.
        if delta < 0 and (dimension == "survival" or code == "max_hp_change"):
            multiplier = max(1.0, multiplier)
        return delta if multiplier == 1.0 else round(delta * multiplier, 4)

    def card_factors(self, factors: list[dict]) -> list[dict]:
        return [
            {**item, "delta": self.adjust(float(item["delta"]), _CARD_DIMENSIONS.get(item["code"]), code=item["code"])}
            for item in factors
        ]


DEFAULT_SCORING_PREFERENCES = ScoringPreferences()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate scoring preference field")
        result[key] = value
    return result


def load_scoring_preferences(path: Path) -> ScoringPreferences:
    """Read one whole configuration; an invalid file never partially applies."""
    try:
        raw = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return DEFAULT_SCORING_PREFERENCES
    except (OSError, UnicodeError):
        LOGGER.warning("Scoring preferences unreadable; using all defaults. Check %s", path)
        return DEFAULT_SCORING_PREFERENCES
    try:
        return ScoringPreferences.from_dict(json.loads(raw, object_pairs_hook=_unique_object))
    except (ValueError, TypeError, OverflowError, RecursionError):
        LOGGER.warning("Scoring preferences invalid; using all defaults. Check %s", path)
        return DEFAULT_SCORING_PREFERENCES

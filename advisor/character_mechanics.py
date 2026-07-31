"""Shared character-mechanic signals for card-reward scoring.

Adapters in this module only translate versioned structured catalog facts into
provider/payoff-style signals.  They do not own character-specific weights.
All score deltas are applied by the shared ``assess_card_mechanics`` function.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Protocol

from storage.effect_tags import derive_mechanic_effect_tags


MECHANIC_ROLES = frozenset(
    {"provider", "payoff", "spender", "capacity", "multiplier"}
)
MECHANIC_CONFIDENCE = frozenset({"structured", "derived"})


@dataclass(frozen=True)
class MechanicSignal:
    domain: str
    role: str
    magnitude: float
    source_code: str
    confidence: str

    def __post_init__(self) -> None:
        if not self.domain.strip():
            raise ValueError("mechanic signal domain is required")
        if self.role not in MECHANIC_ROLES:
            raise ValueError(f"unknown mechanic signal role: {self.role}")
        if (
            isinstance(self.magnitude, bool)
            or not isinstance(self.magnitude, (int, float))
            or not math.isfinite(float(self.magnitude))
            or float(self.magnitude) < 0
        ):
            raise ValueError("mechanic signal magnitude must be finite")
        if not self.source_code.strip():
            raise ValueError("mechanic signal source is required")
        if self.confidence not in MECHANIC_CONFIDENCE:
            raise ValueError(
                f"unknown mechanic signal confidence: {self.confidence}"
            )

    @property
    def family(self) -> str:
        return self.domain.split(".", 1)[0]

    def as_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "role": self.role,
            "magnitude": round(float(self.magnitude), 4),
            "source_code": self.source_code,
            "confidence": self.confidence,
        }


class CharacterMechanicsAdapter(Protocol):
    character_id: str
    domains: frozenset[str]

    def signals_for_card(self, card: Mapping) -> tuple[MechanicSignal, ...]:
        ...

    def data_gaps(
        self,
        card: Mapping,
        signals: tuple[MechanicSignal, ...],
    ) -> tuple[str, ...]:
        ...


class _TagBackedAdapter:
    character_id = ""
    domains: frozenset[str] = frozenset()

    def signals_for_card(
        self,
        card: Mapping,
    ) -> tuple[MechanicSignal, ...]:
        signals = []
        for tag, (magnitude, source) in derive_mechanic_effect_tags(
            dict(card)
        ).items():
            match = re.fullmatch(
                r"mechanic:(.+):(provider|payoff|spender|capacity|multiplier)",
                tag,
            )
            if match is None:
                continue
            domain, role = match.groups()
            family = domain.split(".", 1)[0]
            if family not in self.domains:
                continue
            signals.append(
                MechanicSignal(
                    domain=domain,
                    role=role,
                    magnitude=float(magnitude),
                    source_code=source,
                    confidence=(
                        "derived"
                        if source.startswith("description")
                        else "structured"
                    ),
                )
            )
        return tuple(
            sorted(
                signals,
                key=lambda signal: (
                    signal.family,
                    signal.domain,
                    signal.role,
                    signal.source_code,
                ),
            )
        )

    def data_gaps(
        self,
        card: Mapping,
        signals: tuple[MechanicSignal, ...],
    ) -> tuple[str, ...]:
        return ()


class IroncladMechanicsAdapter(_TagBackedAdapter):
    character_id = "IRONCLAD"
    domains = frozenset({"blood", "exhaust", "strength"})


class SilentMechanicsAdapter(_TagBackedAdapter):
    character_id = "SILENT"
    domains = frozenset({"discard", "shiv", "poison"})


class DefectMechanicsAdapter(_TagBackedAdapter):
    character_id = "DEFECT"
    domains = frozenset({"orb"})


class RegentMechanicsAdapter(_TagBackedAdapter):
    character_id = "REGENT"
    domains = frozenset({"star", "forge"})


class NecrobinderMechanicsAdapter(_TagBackedAdapter):
    character_id = "NECROBINDER"
    domains = frozenset({"osty", "doom", "soul"})

    def data_gaps(
        self,
        card: Mapping,
        signals: tuple[MechanicSignal, ...],
    ) -> tuple[str, ...]:
        description = " ".join(
            (
                str(card.get("description") or ""),
                str(card.get("upgrade_description") or ""),
            )
        )
        has_osty_payoff = any(
            signal.family == "osty" and signal.role == "payoff"
            for signal in signals
        )
        if has_osty_payoff and re.search(
            r"奥斯提.{0,18}当前生命值|"
            r"osty.{0,18}current.{0,8}(hp|health)",
            description,
            flags=re.IGNORECASE,
        ):
            return ("character_state:osty_current_hp",)
        return ()


_ADAPTERS: Dict[str, CharacterMechanicsAdapter] = {
    adapter.character_id: adapter
    for adapter in (
        IroncladMechanicsAdapter(),
        SilentMechanicsAdapter(),
        DefectMechanicsAdapter(),
        RegentMechanicsAdapter(),
        NecrobinderMechanicsAdapter(),
    )
}

_DOMAIN_LABELS = {
    "blood": "自身失血",
    "exhaust": "耗竭",
    "strength": "力量",
    "discard": "弃牌/奇巧",
    "shiv": "小刀",
    "poison": "中毒",
    "orb": "充能球",
    "star": "星数",
    "forge": "铸造/君王之剑",
    "osty": "召唤/奥斯提",
    "doom": "灾厄",
    "soul": "灵魂",
}


def adapter_for(
    character_id: str,
) -> CharacterMechanicsAdapter | None:
    return _ADAPTERS.get(str(character_id or "").strip().upper())


def supports_character(character_id: str) -> bool:
    return adapter_for(character_id) is not None


def signals_for_card(
    character_id: str,
    card: Mapping | None,
) -> tuple[MechanicSignal, ...]:
    if card is None:
        return ()
    adapter = adapter_for(character_id)
    if adapter is None:
        return ()
    return adapter.signals_for_card(card)


def build_mechanic_context(
    character_id: str,
    resolved_deck: Iterable[Mapping],
) -> Dict:
    """Aggregate deck evidence without applying any score weights."""
    adapter = adapter_for(character_id)
    if adapter is None:
        return {
            "known": False,
            "character": str(character_id or "").strip().upper(),
            "evidence": {},
            "exact_domains": {},
            "data_gaps": [
                "character_mechanics:unsupported:"
                + str(character_id or "").strip().upper(),
            ],
        }

    evidence = defaultdict(lambda: defaultdict(
        lambda: {"count": 0, "magnitude": 0.0}
    ))
    exact_domains = defaultdict(int)
    for row in resolved_deck:
        card = row.get("card")
        if not card:
            continue
        for signal in adapter.signals_for_card(card):
            slot = evidence[signal.family][signal.role]
            slot["count"] += 1
            slot["magnitude"] += float(signal.magnitude)
            exact_domains[signal.domain] += 1

    return {
        "known": True,
        "character": adapter.character_id,
        "evidence": {
            family: {
                role: {
                    "count": int(values["count"]),
                    "magnitude": round(float(values["magnitude"]), 4),
                }
                for role, values in sorted(roles.items())
            }
            for family, roles in sorted(evidence.items())
        },
        "exact_domains": dict(sorted(exact_domains.items())),
        "data_gaps": [],
    }


def _factor(code: str, delta: float, message: str) -> Dict:
    return {
        "code": code,
        "delta": round(float(delta), 4),
        "message": message,
    }


def _role_evidence(context: Mapping, family: str, role: str) -> Dict:
    return (
        context.get("evidence", {})
        .get(family, {})
        .get(role, {"count": 0, "magnitude": 0.0})
    )


def _complement_evidence(context: Mapping, family: str) -> Dict:
    count = 0
    magnitude = 0.0
    for role in ("payoff", "spender", "capacity", "multiplier"):
        evidence = _role_evidence(context, family, role)
        count += int(evidence.get("count") or 0)
        magnitude += float(evidence.get("magnitude") or 0.0)
    return {"count": count, "magnitude": magnitude}


def assess_card_mechanics(
    character_id: str,
    card: Mapping | None,
    context: Mapping,
) -> Dict:
    """Apply one shared provider/payoff scoring rule to all five adapters."""
    adapter = adapter_for(character_id)
    if adapter is None or card is None:
        return {
            "signals": (),
            "factors": (),
            "data_gaps": tuple(context.get("data_gaps") or ()),
        }

    signals = adapter.signals_for_card(card)
    grouped: Dict[tuple[str, str], float] = {}
    for signal in signals:
        key = (signal.family, signal.role)
        grouped[key] = max(grouped.get(key, 0.0), signal.magnitude)

    factors = []
    for (family, role), magnitude in sorted(grouped.items()):
        label = _DOMAIN_LABELS.get(family, family)
        provider = _role_evidence(context, family, "provider")
        provider_count = int(provider.get("count") or 0)
        provider_magnitude = float(provider.get("magnitude") or 0.0)

        if role == "provider":
            complement = _complement_evidence(context, family)
            if int(complement["count"]) > 0:
                delta = min(
                    5.0,
                    1.5 + 0.6 * int(complement["count"]),
                )
                factors.append(_factor(
                    f"mechanic_synergy_provider:{family}",
                    delta,
                    f"牌组已有{label}收益方，这张牌能提供对应资源。",
                ))
            continue

        if role == "spender":
            if provider_count <= 0:
                delta = -min(8.0, 1.5 + magnitude)
                factors.append(_factor(
                    f"mechanic_resource_unsupported:{family}",
                    delta,
                    f"牌组没有明确的{label}来源，当前资源消费者难以稳定使用。",
                ))
            elif provider_magnitude + 1e-9 < magnitude:
                shortfall = magnitude - provider_magnitude
                delta = -min(4.0, 0.75 + 0.5 * shortfall)
                factors.append(_factor(
                    f"mechanic_resource_shortfall:{family}",
                    delta,
                    f"牌组已有{label}来源，但供给仍低于这张牌的资源需求。",
                ))
            else:
                delta = min(
                    5.0,
                    2.0 + 0.5 * provider_count,
                )
                factors.append(_factor(
                    f"mechanic_resource_supported:{family}",
                    delta,
                    f"牌组已有足够的{label}来源，可支持这张资源消费者。",
                ))
            continue

        if provider_count <= 0:
            continue
        if role == "payoff":
            code = f"mechanic_synergy_payoff:{family}"
            role_label = "收益"
        elif role == "multiplier":
            code = f"mechanic_synergy_multiplier:{family}"
            role_label = "放大"
        else:
            code = f"mechanic_synergy_capacity:{family}"
            role_label = "容量"
        delta = min(6.0, 2.0 + 0.65 * provider_count)
        factors.append(_factor(
            code,
            delta,
            f"牌组已有{label}提供者，这张牌能形成{role_label}协同。",
        ))

    data_gaps = tuple(dict.fromkeys(
        (
            *tuple(context.get("data_gaps") or ()),
            *adapter.data_gaps(card, signals),
        )
    ))
    return {
        "signals": signals,
        "factors": tuple(factors),
        "data_gaps": data_gaps,
    }

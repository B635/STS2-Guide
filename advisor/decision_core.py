"""Policy-neutral contracts for every realtime STS2 decision.

Protocol models are adapted into these immutable domain objects at the
realtime boundary.  Policies therefore do not depend on Pydantic, the file
bridge, SQLite lifecycle tables, or game node classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Protocol


CARD_REWARD = "card_reward"
ROUTE_CHOICE = "route_choice"
ROUTE_MODE_BALANCED = "balanced"
ROUTE_MODE_SURVIVAL = "survival"
ROUTE_MODE_GROWTH = "growth"
ROUTE_MODES = frozenset({
    ROUTE_MODE_BALANCED,
    ROUTE_MODE_SURVIVAL,
    ROUTE_MODE_GROWTH,
})
RECOMMENDATION_CONTRACT_VERSION = 1
ROUTE_RECOMMENDATION_CONTRACT_VERSION = 2
RECOMMENDATION_DIMENSIONS = (
    "immediate_power",
    "survival",
    "long_term_growth",
    "resource_efficiency",
    "deck_burden",
    "synergy",
    "route_fit",
    "data_completeness",
)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _deep_freeze(child)
            for key, child in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(child) for child in value)
    return value


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _deep_freeze(dict(value or {}))


def _plain_value(value: Any) -> Any:
    """Return a JSON/checkpoint-safe copy of a deeply frozen value."""
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, frozenset)):
        return [_plain_value(child) for child in value]
    return value


@dataclass(frozen=True)
class WorldDeckCard:
    card: str
    count: int = 1
    upgrades: int = 0
    enchantment: str | None = None
    enchantment_amount: int | None = None
    affliction: str | None = None
    affliction_amount: int | None = None

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "WorldDeckCard":
        return cls(
            card=str(value.get("card") or ""),
            count=int(value.get("count", 1)),
            upgrades=int(value.get("upgrades", 0)),
            enchantment=value.get("enchantment"),
            enchantment_amount=value.get("enchantment_amount"),
            affliction=value.get("affliction"),
            affliction_amount=value.get("affliction_amount"),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "card": self.card,
            "count": self.count,
            "upgrades": self.upgrades,
            "enchantment": self.enchantment,
            "enchantment_amount": self.enchantment_amount,
            "affliction": self.affliction,
            "affliction_amount": self.affliction_amount,
        }


@dataclass(frozen=True)
class WorldRelic:
    relic: str
    display_amount: int | None = None
    stack_count: int = 1
    status: str | None = None

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "WorldRelic":
        return cls(
            relic=str(value.get("relic") or ""),
            display_amount=value.get("display_amount"),
            stack_count=int(value.get("stack_count", 1)),
            status=value.get("status"),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "relic": self.relic,
            "display_amount": self.display_amount,
            "stack_count": self.stack_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class WorldPotion:
    potion: str
    slot: int

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "WorldPotion":
        return cls(
            potion=str(value.get("potion") or ""),
            slot=int(value.get("slot", 0)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {"potion": self.potion, "slot": self.slot}


@dataclass(frozen=True)
class WorldMapNode:
    node_id: str
    kind: str
    row: int
    col: int
    edges: tuple[str, ...] = ()
    label: str | None = None

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "WorldMapNode":
        return cls(
            node_id=str(value.get("node_id") or ""),
            kind=str(value.get("kind") or "UNKNOWN"),
            row=int(value.get("row", 0)),
            col=int(value.get("col", 0)),
            edges=tuple(str(edge) for edge in value.get("edges") or ()),
            label=value.get("label"),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "row": self.row,
            "col": self.col,
            "edges": list(self.edges),
            "label": self.label,
        }


@dataclass(frozen=True)
class WorldMap:
    map_name: str | None = None
    player_row: int | None = None
    nodes: tuple[WorldMapNode, ...] = ()
    node_count: int | None = None
    current_node_id: str | None = None
    origin_node_id: str | None = None
    available_next_node_ids: tuple[str, ...] = ()
    boss_node_ids: tuple[str, ...] = ()
    boss_encounter_ids: tuple[str, ...] = ()

    @classmethod
    def create(cls, value: Mapping[str, Any]) -> "WorldMap":
        return cls(
            map_name=value.get("map_name"),
            player_row=value.get("player_row"),
            nodes=tuple(
                WorldMapNode.create(node)
                for node in value.get("nodes") or ()
            ),
            node_count=value.get("node_count"),
            current_node_id=value.get("current_node_id"),
            origin_node_id=value.get("origin_node_id") or value.get("current_node_id"),
            available_next_node_ids=tuple(
                str(node_id)
                for node_id in value.get("available_next_node_ids") or ()
            ),
            boss_node_ids=tuple(
                str(node_id)
                for node_id in value.get("boss_node_ids") or ()
            ),
            boss_encounter_ids=tuple(
                str(encounter_id)
                for encounter_id in value.get("boss_encounter_ids") or ()
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "map_name": self.map_name,
            "player_row": self.player_row,
            "nodes": [node.as_dict() for node in self.nodes],
            "node_count": self.node_count if self.node_count is not None else len(self.nodes),
            "current_node_id": self.current_node_id,
            "origin_node_id": self.origin_node_id,
            "available_next_node_ids": list(self.available_next_node_ids),
            "boss_node_ids": list(self.boss_node_ids),
            "boss_encounter_ids": list(self.boss_encounter_ids),
        }


@dataclass(frozen=True)
class WorldState:
    """One typed, immutable current-run observation shared by all policies."""

    run_id: str
    sequence: int
    route_mode: str
    game_version: str | None
    scene: str
    character: str
    ascension: int
    act: int
    floor: int
    hp: int | None
    max_hp: int | None
    gold: int | None
    energy: int
    deck: tuple[WorldDeckCard, ...]
    relic_ids: tuple[str, ...]
    relics: tuple[WorldRelic, ...]
    potions: tuple[WorldPotion, ...]
    max_potion_slots: int | None
    modifiers: tuple[str, ...]
    capture_warnings: tuple[str, ...]
    map: WorldMap | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        state: Mapping[str, Any],
        map_context: Mapping[str, Any] | None = None,
        route_mode: str = ROUTE_MODE_BALANCED,
        game_version: str | None = None,
        scene: str = "unknown",
    ) -> "WorldState":
        run_id = str(run_id).strip()
        character = str(state.get("character") or "").strip()
        if not run_id or not character or sequence < 1:
            raise ValueError("world requires run, sequence, and character")
        route_mode = str(route_mode).strip()
        if route_mode not in ROUTE_MODES:
            raise ValueError("world requires a supported route mode")
        relic_ids = tuple(str(value) for value in state.get("relics") or ())
        relic_states = tuple(
            WorldRelic.create(value)
            for value in state.get("relic_states") or ()
        )
        if not relic_states:
            relic_states = tuple(WorldRelic(relic) for relic in relic_ids)
        if not relic_ids:
            relic_ids = tuple(relic.relic for relic in relic_states)
        return cls(
            run_id=run_id,
            sequence=int(sequence),
            route_mode=route_mode,
            game_version=(game_version or state.get("game_version")),
            scene=str(scene or "unknown"),
            character=character,
            ascension=int(state.get("ascension", 0)),
            act=int(state.get("act", 1)),
            floor=int(state.get("floor", 0)),
            hp=state.get("hp"),
            max_hp=state.get("max_hp"),
            gold=state.get("gold"),
            energy=int(state.get("energy", 3)),
            deck=tuple(
                WorldDeckCard.create(value)
                for value in state.get("deck") or ()
            ),
            relic_ids=relic_ids,
            relics=relic_states,
            potions=tuple(
                WorldPotion.create(value)
                for value in state.get("potions") or ()
            ),
            max_potion_slots=state.get("max_potion_slots"),
            modifiers=tuple(
                str(value) for value in state.get("modifiers") or ()
            ),
            capture_warnings=tuple(
                str(value)
                for value in state.get("capture_warnings") or ()
            ),
            map=(
                WorldMap.create(map_context)
                if map_context is not None
                else None
            ),
        )

    def scoring_state(self) -> Dict[str, Any]:
        """Compatibility adapter for the already-validated P0 scorer."""
        payload: Dict[str, Any] = {
            "character": self.character,
            "ascension": self.ascension,
            "act": self.act,
            "floor": self.floor,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "gold": self.gold,
            "energy": self.energy,
            "deck": [card.as_dict() for card in self.deck],
            "relics": list(self.relic_ids),
            "relic_states": [relic.as_dict() for relic in self.relics],
            "potions": [potion.as_dict() for potion in self.potions],
            "max_potion_slots": self.max_potion_slots,
            "modifiers": list(self.modifiers),
            "capture_warnings": list(self.capture_warnings),
            "game_version": self.game_version,
            "guide_preferences": {
                "route_mode": self.route_mode,
            },
        }
        if self.map is not None:
            payload["map_context"] = self.map.as_dict()
        return payload


@dataclass(frozen=True)
class DecisionCandidate:
    """A stable candidate identity plus policy-specific structured fields."""

    candidate_id: str
    label: str = ""
    display_index: int | None = None
    eligible: bool = True
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        candidate_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
        display_index: int | None = None,
        eligible: bool = True,
    ) -> "DecisionCandidate":
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must not be empty")
        if display_index is not None and display_index < 0:
            raise ValueError("candidate display_index must not be negative")
        return cls(
            candidate_id=candidate_id,
            label=str(label or candidate_id),
            display_index=display_index,
            eligible=bool(eligible),
            payload=_frozen_mapping(payload),
        )


@dataclass(frozen=True)
class DecisionRequest:
    decision_id: str
    decision_type: str
    world: WorldState
    candidates: tuple[DecisionCandidate, ...]
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        decision_id: str,
        decision_type: str,
        world: WorldState,
        candidates: Iterable[DecisionCandidate],
        constraints: Mapping[str, Any] | None = None,
    ) -> "DecisionRequest":
        decision_id = str(decision_id).strip()
        decision_type = str(decision_type).strip()
        candidate_tuple = tuple(
            candidate
            if candidate.display_index is not None
            else replace(candidate, display_index=index)
            for index, candidate in enumerate(candidates)
        )
        if not decision_id or not decision_type:
            raise ValueError("decision identity and type are required")
        if not candidate_tuple:
            raise ValueError("a decision requires at least one candidate")
        candidate_ids = [candidate.candidate_id for candidate in candidate_tuple]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a decision")
        display_indices = [
            candidate.display_index for candidate in candidate_tuple
        ]
        if display_indices != list(range(len(candidate_tuple))):
            raise ValueError(
                "candidate display indices must match request order"
            )
        if not any(candidate.eligible for candidate in candidate_tuple):
            raise ValueError("a decision requires an eligible candidate")
        return cls(
            decision_id=decision_id,
            decision_type=decision_type,
            world=world,
            candidates=candidate_tuple,
            constraints=_frozen_mapping(constraints),
        )


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    label: str
    display_index: int
    eligible: bool
    score: float | None
    rank: int | None
    factors: tuple[Mapping[str, Any], ...] = ()
    dimensions: Mapping[str, float | None] = field(default_factory=dict)
    data_gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("assessment candidate_id must not be empty")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("assessment label must not be empty")
        if (
            isinstance(self.display_index, bool)
            or not isinstance(self.display_index, int)
            or self.display_index < 0
        ):
            raise ValueError("assessment display_index must not be negative")
        if self.score is not None:
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (int, float))
                or not math.isfinite(float(self.score))
                or not 0 <= float(self.score) <= 100
            ):
                raise ValueError("assessment score must be between 0 and 100")
        if not isinstance(self.eligible, bool):
            raise ValueError("assessment eligible must be a boolean")
        if not self.eligible and (
            self.score is not None or self.rank is not None
        ):
            raise ValueError(
                "ineligible assessment cannot have a score or rank"
            )
        if self.score is None and self.rank is not None:
            raise ValueError("unscored assessment cannot have a rank")
        if self.rank is not None and (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ValueError("assessment rank must be a positive integer")
        if any(not isinstance(factor, Mapping) for factor in self.factors):
            raise ValueError("assessment factors must be objects")
        unknown = set(self.dimensions) - set(RECOMMENDATION_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown recommendation dimensions: {unknown}")
        for name, value in self.dimensions.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"assessment dimension {name} must be finite or null"
                )
        if any(
            not isinstance(gap, str) or not gap.strip()
            for gap in self.data_gaps
        ) or len(self.data_gaps) != len(set(self.data_gaps)):
            raise ValueError(
                "assessment data gaps must be unique non-empty strings"
            )
        object.__setattr__(
            self,
            "factors",
            tuple(_frozen_mapping(factor) for factor in self.factors),
        )
        object.__setattr__(self, "dimensions", _frozen_mapping(self.dimensions))
        object.__setattr__(self, "data_gaps", tuple(self.data_gaps))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "display_index": self.display_index,
            "eligible": self.eligible,
            "score": self.score,
            "rank": self.rank,
            "factors": [dict(factor) for factor in self.factors],
            "dimensions": dict(self.dimensions),
            "data_gaps": list(self.data_gaps),
        }


@dataclass(frozen=True)
class Recommendation:
    """Policy-neutral envelope plus a temporary legacy P0 payload."""

    decision_id: str
    decision_type: str
    payload: Mapping[str, Any]
    world_sequence: int
    policy_version: str
    candidates: tuple[CandidateAssessment, ...] = ()
    recommended_candidate_id: str | None = None
    status: str = "recommend"
    confidence: str = "low"
    data_gaps: tuple[str, ...] = ()
    contract_version: int = RECOMMENDATION_CONTRACT_VERSION
    presentation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError("recommendation decision_id must not be empty")
        if (
            not isinstance(self.decision_type, str)
            or not self.decision_type.strip()
        ):
            raise ValueError("recommendation decision_type must not be empty")
        if self.status not in {"recommend", "skip", "uncertain"}:
            raise ValueError("unsupported recommendation status")
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("unsupported recommendation confidence")
        if self.status == "uncertain" and self.recommended_candidate_id is not None:
            raise ValueError(
                "uncertain recommendation cannot select a candidate"
            )
        if self.status == "recommend" and (
            self.recommended_candidate_id is None
            or self.recommended_candidate_id == "skip"
        ):
            raise ValueError(
                "recommend status requires a non-skip candidate"
            )
        if self.status == "skip" and (
            self.decision_type != CARD_REWARD
            or self.recommended_candidate_id != "skip"
        ):
            raise ValueError(
                "skip status requires the Card Reward skip candidate"
            )
        if self.contract_version not in (
            RECOMMENDATION_CONTRACT_VERSION,
            ROUTE_RECOMMENDATION_CONTRACT_VERSION,
        ):
            raise ValueError("unsupported recommendation contract version")
        if self.decision_type == ROUTE_CHOICE:
            if self.contract_version != ROUTE_RECOMMENDATION_CONTRACT_VERSION:
                raise ValueError("route_choice requires recommendation contract v2")
            presentation = self.presentation or {}
            if not isinstance(presentation, Mapping):
                raise ValueError("route presentation must be an object")
            if str(presentation.get("kind") or "") != "route_paths":
                raise ValueError("route recommendation requires route_paths presentation")
            required = {
                "origin_node_id",
                "primary_path_node_ids",
                "backup_path_node_ids",
                "paths",
            }
            if set(presentation) != {"kind", *required}:
                raise ValueError("route presentation has invalid fields")
            primary = presentation["primary_path_node_ids"]
            backup = presentation["backup_path_node_ids"]
            paths = presentation["paths"]
            if not isinstance(primary, (list, tuple)) or not isinstance(
                backup, (list, tuple)
            ) or not isinstance(paths, (list, tuple)):
                raise ValueError("route presentation paths must be arrays")
            if any(
                not isinstance(node_id, str) or not node_id.strip()
                for node_id in (*primary, *backup)
            ):
                raise ValueError("route presentation node IDs must be non-empty strings")
            if backup:
                raise ValueError(
                    "route presentation backup path must remain empty"
                )
            origin = presentation["origin_node_id"]
            if origin is not None and (
                not isinstance(origin, str) or not origin.strip()
            ):
                raise ValueError("route presentation origin_node_id is invalid")
            expected_ids = [candidate.candidate_id for candidate in self.candidates]
            if len(expected_ids) != len(set(expected_ids)):
                raise ValueError("route recommendation has duplicate assessment IDs")
            paths_by_candidate: dict[str, tuple[str, ...]] = {}
            for path in paths:
                if not isinstance(path, Mapping):
                    raise ValueError("route path row must be an object")
                if set(path) != {"candidate_id", "node_ids", "score"}:
                    raise ValueError("route path row has invalid fields")
                candidate_id = path.get("candidate_id")
                node_ids = path.get("node_ids")
                score = path.get("score")
                if (
                    not isinstance(candidate_id, str)
                    or not candidate_id.strip()
                    or candidate_id in paths_by_candidate
                    or candidate_id not in expected_ids
                    or not isinstance(node_ids, (list, tuple))
                    or not node_ids
                    or any(
                        not isinstance(node_id, str) or not node_id.strip()
                        for node_id in node_ids
                    )
                    or node_ids[0] != candidate_id
                ):
                    raise ValueError("route path row has invalid candidate or node IDs")
                if score is not None:
                    if isinstance(score, bool) or not isinstance(score, (int, float)):
                        raise ValueError("route path score must be a JSON number or null")
                    numeric_score = float(score)
                    if not math.isfinite(numeric_score) or not 0 <= numeric_score <= 100:
                        raise ValueError("route path score must be between 0 and 100")
                paths_by_candidate[candidate_id] = tuple(node_ids)
            if self.recommended_candidate_id is None:
                if primary or backup or paths:
                    raise ValueError("uncertain route recommendation cannot expose paths")
            else:
                if not isinstance(origin, str) or not origin.strip():
                    raise ValueError("recommended route requires a real origin")
                if not primary or primary[0] != self.recommended_candidate_id:
                    raise ValueError(
                        "route primary path must start at the recommended candidate"
                    )
                if backup and backup[0] == primary[0]:
                    raise ValueError(
                        "route backup path must start at a different candidate"
                    )
                if set(paths_by_candidate) != set(expected_ids):
                    raise ValueError(
                        "recommended route must expose one typed path for every candidate"
                    )
                if tuple(primary) != paths_by_candidate.get(
                    self.recommended_candidate_id
                ):
                    raise ValueError(
                        "route primary path must match the recommended path row"
                    )
                if backup and tuple(backup) != paths_by_candidate.get(backup[0]):
                    raise ValueError("route backup path must match its path row")
        elif self.contract_version == RECOMMENDATION_CONTRACT_VERSION:
            if self.presentation:
                raise ValueError("recommendation contract v1 cannot carry presentation")
        elif self.contract_version == ROUTE_RECOMMENDATION_CONTRACT_VERSION:
            # Contract v2 is deliberately reusable by existing consumers
            # during migration.  Card Reward keeps producing v1, but its
            # consumer must be able to read a valid presentation-free v2.
            if self.presentation:
                raise ValueError(
                    "non-route recommendation contract v2 cannot carry route presentation"
                )
        else:
            raise ValueError("route_choice requires recommendation contract v2")
        if (
            isinstance(self.world_sequence, bool)
            or not isinstance(self.world_sequence, int)
            or self.world_sequence < 1
        ):
            raise ValueError("recommendation world_sequence must be positive")
        if (
            not isinstance(self.policy_version, str)
            or not self.policy_version.strip()
        ):
            raise ValueError("recommendation policy_version must not be empty")
        if not self.candidates:
            raise ValueError("recommendation must assess at least one candidate")
        candidate_ids = [
            candidate.candidate_id for candidate in self.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("recommendation has duplicate assessment IDs")
        if (
            self.recommended_candidate_id is not None
            and self.recommended_candidate_id not in candidate_ids
        ):
            raise ValueError("recommended candidate has no assessment")
        if self.recommended_candidate_id is not None:
            recommended = next(
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == self.recommended_candidate_id
            )
            if not recommended.eligible or recommended.score is None:
                raise ValueError(
                    "recommended candidate is ineligible or unscored"
                )
        if any(
            not isinstance(gap, str) or not gap.strip()
            for gap in self.data_gaps
        ) or len(self.data_gaps) != len(set(self.data_gaps)):
            raise ValueError(
                "recommendation data gaps must be unique non-empty strings"
            )
        object.__setattr__(self, "payload", _frozen_mapping(self.payload))
        object.__setattr__(self, "presentation", _frozen_mapping(self.presentation))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "data_gaps", tuple(self.data_gaps))

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "contract_version": self.contract_version,
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "world_sequence": self.world_sequence,
            "policy_version": self.policy_version,
            "status": self.status,
            "confidence": self.confidence,
            "recommended_candidate_id": self.recommended_candidate_id,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "data_gaps": list(self.data_gaps),
        }
        if self.presentation:
            result["presentation"] = _plain_value(self.presentation)
        return result

    def payload_dict(self) -> Dict[str, Any]:
        """Materialize the temporary legacy payload without mutable aliases."""
        return _plain_value(self.payload)


class DecisionPolicy(Protocol):
    decision_type: str

    def recommend(self, request: DecisionRequest) -> Recommendation:
        ...


class PolicyRegistry:
    """Explicit policy lookup; unsupported decisions fail closed."""

    def __init__(self, policies: Iterable[DecisionPolicy] = ()):
        self._policies: Dict[str, DecisionPolicy] = {}
        for policy in policies:
            self.register(policy)

    def register(self, policy: DecisionPolicy) -> None:
        decision_type = str(policy.decision_type).strip()
        if not decision_type:
            raise ValueError("policy decision_type must not be empty")
        if decision_type in self._policies:
            raise ValueError(f"duplicate policy for {decision_type}")
        self._policies[decision_type] = policy

    def supports(self, decision_type: str) -> bool:
        return decision_type in self._policies

    def recommend(self, request: DecisionRequest) -> Recommendation:
        policy = self._policies.get(request.decision_type)
        if policy is None:
            raise LookupError(f"no policy registered for {request.decision_type}")
        result = policy.recommend(request)
        if (
            result.decision_id != request.decision_id
            or result.decision_type != request.decision_type
        ):
            raise ValueError("policy returned a mismatched recommendation")
        if result.world_sequence != request.world.sequence:
            raise ValueError("policy returned a mismatched world sequence")
        expected_order = [
            candidate.candidate_id for candidate in request.candidates
        ]
        expected = set(expected_order)
        returned_ids = [candidate.candidate_id for candidate in result.candidates]
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("policy returned duplicate candidate assessments")
        if set(returned_ids) != expected:
            raise ValueError("policy must assess every requested candidate exactly once")
        if returned_ids != expected_order:
            raise ValueError("policy changed candidate assessment order")
        requested_by_id = {
            candidate.candidate_id: candidate
            for candidate in request.candidates
        }
        for assessment in result.candidates:
            candidate = requested_by_id[assessment.candidate_id]
            if (
                assessment.display_index != candidate.display_index
                or assessment.eligible != candidate.eligible
            ):
                raise ValueError("policy changed candidate presentation identity")
        ranks = [
            candidate.rank
            for candidate in result.candidates
            if candidate.rank is not None
        ]
        if len(ranks) != len(set(ranks)):
            raise ValueError("policy returned duplicate candidate ranks")
        if (
            result.recommended_candidate_id is not None
            and result.recommended_candidate_id not in returned_ids
        ):
            raise ValueError("recommended candidate has no assessment")
        if result.recommended_candidate_id is not None:
            recommended = next(
                candidate
                for candidate in result.candidates
                if candidate.candidate_id == result.recommended_candidate_id
            )
            if not recommended.eligible:
                raise ValueError("recommended candidate is ineligible")
        return result


class DecisionPhase(str, Enum):
    OPENED = "opened"
    UPDATED = "updated"
    CLOSED = "closed"


@dataclass(frozen=True)
class DecisionTransition:
    run_id: str
    decision_id: str
    decision_type: str
    phase: DecisionPhase
    sequence: int
    outcome: str | None = None
    observation_event_id: str | None = None


class DecisionLifecycleManager:
    """Run-scoped lifecycle state; never persisted as decision history."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._active: Dict[str, DecisionTransition] = {}
        self._closed: Dict[str, DecisionTransition] = {}

    def restore_active(
        self,
        *,
        run_id: str,
        decision_id: str,
        decision_type: str,
        sequence: int,
        observation_event_id: str,
    ) -> DecisionTransition:
        """Hydrate the one recoverable decision from checkpoint v2."""
        with self._lock:
            existing = self._active.get(run_id)
            if existing is not None:
                if (
                    existing.decision_id != decision_id
                    or existing.decision_type != decision_type
                    or existing.sequence != sequence
                    or existing.observation_event_id != observation_event_id
                ):
                    raise ValueError(
                        "checkpoint decision conflicts with active lifecycle"
                    )
                return existing
            transition = DecisionTransition(
                run_id=run_id,
                decision_id=decision_id,
                decision_type=decision_type,
                phase=DecisionPhase.OPENED,
                sequence=sequence,
                observation_event_id=observation_event_id,
            )
            self._active[run_id] = transition
            self._closed.pop(run_id, None)
            return transition

    def restore_closed(
        self,
        *,
        run_id: str,
        decision_id: str,
        decision_type: str,
        sequence: int,
        outcome: str | None = None,
        observation_event_id: str | None = None,
    ) -> DecisionTransition:
        """Hydrate the one closed-decision tombstone from checkpoint v2."""
        with self._lock:
            if self._active.get(run_id) is not None:
                raise ValueError(
                    "checkpoint closed decision conflicts with active lifecycle"
                )
            existing = self._closed.get(run_id)
            if existing is not None:
                if (
                    existing.decision_id != decision_id
                    or existing.decision_type != decision_type
                    or existing.sequence != sequence
                    or existing.outcome != outcome
                ):
                    raise ValueError(
                        "checkpoint closed decision conflicts with lifecycle"
                    )
                return existing
            transition = DecisionTransition(
                run_id=run_id,
                decision_id=decision_id,
                decision_type=decision_type,
                phase=DecisionPhase.CLOSED,
                sequence=sequence,
                outcome=outcome,
                observation_event_id=observation_event_id,
            )
            self._closed[run_id] = transition
            return transition

    def open(
        self,
        request: DecisionRequest,
        *,
        observation_event_id: str | None = None,
    ) -> DecisionTransition:
        with self._lock:
            existing = self._active.get(request.world.run_id)
            closed = self._closed.get(request.world.run_id)
            if closed is not None and closed.decision_id == request.decision_id:
                raise ValueError("closed decision cannot be reopened")
            if (
                closed is not None
                and request.world.sequence <= closed.sequence
            ):
                raise ValueError("decision sequence did not advance")
            if (
                existing is not None
                and existing.decision_id == request.decision_id
                and existing.decision_type != request.decision_type
            ):
                raise ValueError(
                    "stable decision ID cannot change decision type"
                )
            if existing is not None and request.world.sequence <= existing.sequence:
                raise ValueError("decision sequence did not advance")
            if existing is not None and existing.decision_id != request.decision_id:
                raise ValueError("another decision is still active for this run")
            phase = (
                DecisionPhase.UPDATED
                if existing is not None
                and existing.decision_id == request.decision_id
                else DecisionPhase.OPENED
            )
            transition = DecisionTransition(
                run_id=request.world.run_id,
                decision_id=request.decision_id,
                decision_type=request.decision_type,
                phase=phase,
                sequence=request.world.sequence,
                observation_event_id=observation_event_id,
            )
            self._active[request.world.run_id] = transition
            if phase == DecisionPhase.OPENED:
                self._closed.pop(request.world.run_id, None)
            return transition

    def close(
        self,
        *,
        run_id: str,
        decision_id: str,
        decision_type: str,
        sequence: int,
        outcome: str,
        allow_recovered: bool = False,
        parent_event_id: str | None = None,
    ) -> DecisionTransition:
        with self._lock:
            existing = self._active.get(run_id)
            closed = self._closed.get(run_id)
            if closed is not None and closed.decision_id == decision_id:
                if closed.outcome != outcome:
                    raise ValueError("decision outcome conflicts with its final result")
                return closed
            if existing is None and not allow_recovered:
                raise ValueError("no active decision to close")
            if existing is not None:
                if existing.decision_id != decision_id:
                    raise ValueError("close event targets another decision")
                if sequence <= existing.sequence:
                    raise ValueError("decision close sequence did not advance")
                if (
                    existing.observation_event_id is not None
                    and parent_event_id != existing.observation_event_id
                ):
                    raise ValueError(
                        "decision close parent is not the latest observation"
                    )
                decision_type = existing.decision_type
            transition = DecisionTransition(
                run_id=run_id,
                decision_id=decision_id,
                decision_type=decision_type,
                phase=DecisionPhase.CLOSED,
                sequence=sequence,
                outcome=outcome,
                observation_event_id=parent_event_id,
            )
            self._active.pop(run_id, None)
            self._closed[run_id] = transition
            return transition

    def end_run(self, run_id: str) -> None:
        with self._lock:
            self._active.pop(run_id, None)
            self._closed.pop(run_id, None)

    def active(self, run_id: str) -> DecisionTransition | None:
        with self._lock:
            return self._active.get(run_id)

    def closed(self, run_id: str) -> DecisionTransition | None:
        with self._lock:
            return self._closed.get(run_id)

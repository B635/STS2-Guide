"""Shared state and policy contracts for realtime decision advisors.

This module is deliberately independent from the game protocol and storage
implementations.  Protocol events are adapted at the realtime boundary, while
individual policies receive one stable request shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Protocol


CARD_REWARD = "card_reward"


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class WorldState:
    """One current-run observation shared by every decision policy."""

    run_id: str
    sequence: int
    character: str
    state: Mapping[str, Any]
    map_context: Mapping[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        sequence: int,
        state: Mapping[str, Any],
        map_context: Mapping[str, Any] | None = None,
    ) -> "WorldState":
        return cls(
            run_id=run_id,
            sequence=sequence,
            character=str(state.get("character") or ""),
            state=_frozen_mapping(state),
            map_context=(
                _frozen_mapping(map_context)
                if map_context is not None
                else None
            ),
        )

    def scoring_state(self) -> Dict[str, Any]:
        payload = dict(self.state)
        if self.map_context is not None:
            payload["map_context"] = dict(self.map_context)
        return payload


@dataclass(frozen=True)
class DecisionCandidate:
    """A stable candidate identity plus policy-specific structured fields."""

    candidate_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        candidate_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> "DecisionCandidate":
        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must not be empty")
        return cls(candidate_id, _frozen_mapping(payload))


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
        candidate_tuple = tuple(candidates)
        if not decision_id or not decision_type:
            raise ValueError("decision identity and type are required")
        if not candidate_tuple:
            raise ValueError("a decision requires at least one candidate")
        candidate_ids = [candidate.candidate_id for candidate in candidate_tuple]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a decision")
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
    score: float | None
    rank: int | None
    factors: tuple[Mapping[str, Any], ...] = ()
    dimensions: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Recommendation:
    """Policy-neutral envelope; payload remains backwards compatible for P0."""

    decision_id: str
    decision_type: str
    payload: Mapping[str, Any]
    candidates: tuple[CandidateAssessment, ...] = ()
    recommended_candidate_id: str | None = None
    status: str = "recommend"
    confidence: str = "low"
    data_gaps: tuple[str, ...] = ()


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
            raise LookupError(
                f"no policy registered for {request.decision_type}"
            )
        result = policy.recommend(request)
        if (
            result.decision_id != request.decision_id
            or result.decision_type != request.decision_type
        ):
            raise ValueError("policy returned a mismatched recommendation")
        allowed_ids = {
            candidate.candidate_id for candidate in request.candidates
        } | {"skip"}
        returned_ids = [candidate.candidate_id for candidate in result.candidates]
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("policy returned duplicate candidate assessments")
        if any(candidate_id not in allowed_ids for candidate_id in returned_ids):
            raise ValueError("policy returned an unknown candidate assessment")
        if (
            result.recommended_candidate_id is not None
            and result.recommended_candidate_id not in allowed_ids
        ):
            raise ValueError("policy recommended an unknown candidate")
        return result

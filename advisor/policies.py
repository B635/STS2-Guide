"""Adapters from the shared decision core to concrete local policies."""
from __future__ import annotations

from types import MappingProxyType

from advisor.card_reward import recommend_card_reward
from advisor.data_sources import LocalCardTierSource
from advisor.decision_core import (
    CARD_REWARD,
    CandidateAssessment,
    DecisionRequest,
    Recommendation,
)
from storage.relational import RelationalRepository


class CardRewardPolicy:
    decision_type = CARD_REWARD

    def __init__(
        self,
        repository: RelationalRepository,
        local_tiers: LocalCardTierSource | None = None,
    ):
        self.repository = repository
        self.local_tiers = local_tiers

    def recommend(self, request: DecisionRequest) -> Recommendation:
        options = [dict(candidate.payload) for candidate in request.candidates]
        advice = recommend_card_reward(
            request.world.scoring_state(),
            options,
            self.repository,
            local_tiers=self.local_tiers,
            can_skip=bool(request.constraints.get("can_skip", True)),
        )
        assessments = []
        for row in advice.get("recommendations", []):
            option_index = int(row["option_index"])
            assessments.append(CandidateAssessment(
                candidate_id=request.candidates[option_index].candidate_id,
                score=float(row["score"]),
                rank=int(row["rank"]),
                factors=tuple(row.get("factors") or []),
            ))
        skip = advice.get("skip_candidate") or {}
        assessments.append(CandidateAssessment(
            candidate_id="skip",
            score=float(skip["score"]),
            rank=int(skip["rank"]),
            factors=tuple(skip.get("factors") or []),
        ))
        recommended_index = int(advice["recommended_option_index"])
        recommended_candidate_id = (
            "skip"
            if recommended_index == len(request.candidates)
            else request.candidates[recommended_index].candidate_id
        )
        data_gaps = tuple(
            f"unknown_card:{row['card']}"
            for row in advice.get("recommendations", [])
            if not row.get("known", False)
        )
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload=MappingProxyType(advice),
            candidates=tuple(assessments),
            recommended_candidate_id=recommended_candidate_id,
            status=str(advice.get("decision_status") or "uncertain"),
            confidence=str(advice.get("confidence") or "low"),
            data_gaps=data_gaps,
        )

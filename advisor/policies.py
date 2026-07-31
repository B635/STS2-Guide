"""Adapters from the shared decision core to concrete local policies."""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from advisor.card_reward import recommend_card_reward
from advisor.data_sources import LocalCardTierSource
from advisor.versions import CARD_REWARD_POLICY_VERSION
from advisor.decision_core import (
    CARD_REWARD,
    CandidateAssessment,
    DecisionRequest,
    Recommendation,
)
from storage.relational import RelationalRepository


def _factor_dimension(code: str) -> str | None:
    """Classify existing score factors without changing their weights."""
    code = code.lower()
    if code.startswith("route_"):
        return "route_fit"
    if code.startswith("mechanic_resource_"):
        return "resource_efficiency"
    if any(token in code for token in (
        "block", "defense", "hp", "healing", "self_harm",
    )):
        return "survival"
    if any(token in code for token in (
        "scaling", "late_act", "upgrade", "growth",
    )):
        return "long_term_growth"
    if any(token in code for token in (
        "energy", "cost", "draw", "curve",
    )):
        return "resource_efficiency"
    if any(token in code for token in (
        "duplicate", "affliction", "ethereal", "deck_size",
    )):
        return "deck_burden"
    if any(token in code for token in (
        "synergy", "continuity", "engine", "relic",
        "exhaust_interaction", "vulnerable", "poison",
    )):
        return "synergy"
    if any(token in code for token in (
        "damage", "attack", "aoe", "frontload", "strong_candidate",
    )):
        return "immediate_power"
    return None


def _dimensions(
    factors: list[dict],
    *,
    known: bool,
) -> dict[str, float]:
    dimensions: dict[str, float] = {}
    for factor in factors:
        dimension = _factor_dimension(str(factor.get("code") or ""))
        if dimension is None:
            continue
        dimensions[dimension] = dimensions.get(dimension, 0.0) + float(
            factor.get("delta") or 0.0
        )
    dimensions = {
        key: round(value, 4) for key, value in dimensions.items()
    }
    dimensions["data_completeness"] = 1.0 if known else 0.0
    return dimensions


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
        card_candidates = [
            candidate
            for candidate in request.candidates
            if candidate.candidate_id != "skip"
        ]
        skip_candidate = next(
            (
                candidate
                for candidate in request.candidates
                if candidate.candidate_id == "skip"
            ),
            None,
        )
        options = [dict(candidate.payload) for candidate in card_candidates]
        advice = recommend_card_reward(
            request.world.scoring_state(),
            options,
            self.repository,
            local_tiers=self.local_tiers,
            can_skip=bool(request.constraints.get("can_skip", True)),
        )
        rows_by_index = {
            int(row["option_index"]): row
            for row in advice.get("recommendations", [])
        }
        assessments = []
        for option_index, candidate in enumerate(card_candidates):
            row = rows_by_index[option_index]
            known = bool(row.get("known", False))
            factors = list(row.get("factors") or [])
            row_gaps = tuple(
                dict.fromkeys(
                    str(gap)
                    for gap in row.get("data_gaps") or ()
                    if str(gap).strip()
                )
            )
            if not known:
                row_gaps = tuple(
                    dict.fromkeys(
                        (*row_gaps, f"unknown_card:{row['card']}")
                    )
                )
            assessments.append(CandidateAssessment(
                candidate_id=candidate.candidate_id,
                label=str(row.get("card") or candidate.label),
                display_index=int(candidate.display_index),
                eligible=candidate.eligible,
                score=(float(row["score"]) if known else None),
                rank=int(row["rank"]),
                factors=tuple(factors),
                dimensions=_dimensions(
                    factors,
                    known=known and not row_gaps,
                ),
                data_gaps=row_gaps,
            ))
        skip = advice.get("skip_candidate") or {}
        if skip_candidate is not None:
            skip_factors = list(skip.get("factors") or [])
            any_known = any(
                assessment.score is not None for assessment in assessments
            )
            assessments.append(CandidateAssessment(
                candidate_id="skip",
                label="跳过",
                display_index=int(skip_candidate.display_index),
                eligible=skip_candidate.eligible,
                score=(float(skip["score"]) if any_known else None),
                rank=(int(skip["rank"]) if skip_candidate.eligible else None),
                factors=tuple(skip_factors),
                dimensions=_dimensions(skip_factors, known=True),
            ))
        eligible_ranked = sorted(
            (assessment for assessment in assessments if assessment.eligible),
            key=lambda assessment: (
                -float(assessment.score or 0.0),
                assessment.display_index,
            ),
        )
        rank_by_id = {
            assessment.candidate_id: rank
            for rank, assessment in enumerate(eligible_ranked, start=1)
        }
        assessments = [
            replace(
                assessment,
                rank=rank_by_id.get(assessment.candidate_id),
            )
            for assessment in assessments
        ]
        recommended_index = int(advice["recommended_option_index"])
        recommended_candidate_id = (
            "skip"
            if recommended_index == len(card_candidates)
            else card_candidates[recommended_index].candidate_id
        )
        recommended_assessment = next(
            assessment
            for assessment in assessments
            if assessment.candidate_id == recommended_candidate_id
        )
        if recommended_assessment.score is None:
            recommended_candidate_id = None
        recommendation_status = str(
            advice.get("decision_status") or "uncertain"
        )
        if (
            recommended_candidate_id is None
            or recommendation_status == "uncertain"
        ):
            recommended_candidate_id = None
            recommendation_status = "uncertain"
        data_gaps = tuple(
            dict.fromkeys(
                (
                    *(
                        str(gap)
                        for gap in advice.get("data_gaps") or ()
                        if str(gap).strip()
                    ),
                    *(
                        gap
                        for assessment in assessments
                        for gap in assessment.data_gaps
                    ),
                    *(
                        f"capture_warning:{warning}"
                        for warning in request.world.capture_warnings
                    ),
                )
            )
        )
        return Recommendation(
            decision_id=request.decision_id,
            decision_type=request.decision_type,
            payload=MappingProxyType(advice),
            candidates=tuple(assessments),
            recommended_candidate_id=recommended_candidate_id,
            status=recommendation_status,
            confidence=str(advice.get("confidence") or "low"),
            data_gaps=data_gaps,
            policy_version=CARD_REWARD_POLICY_VERSION,
            world_sequence=request.world.sequence,
        )

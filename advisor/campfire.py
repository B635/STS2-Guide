"""Deterministic campfire action policy using structured effects only."""
from __future__ import annotations

from advisor.choice_effects import (
    AssessmentDraft,
    bundle_with_costs,
    finalize_recommendation,
)
from advisor.decision_core import DecisionRequest, Recommendation
from advisor.resource_budget import (
    build_resource_budget,
    score_effect_bundle,
)
from storage.relational import RelationalRepository
from advisor.versions import CAMPFIRE_POLICY_VERSION


CAMPFIRE_ACTION = "campfire_action"
class CampfirePolicy:
    decision_type = CAMPFIRE_ACTION

    def __init__(self, repository: RelationalRepository):
        self.repository = repository

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("campfire policy received another decision type")
        budget = build_resource_budget(request.world)
        drafts = []
        for candidate in request.candidates:
            draft = AssessmentDraft(candidate)
            drafts.append(draft)
            if not candidate.eligible:
                continue
            kind = str(
                candidate.payload.get("candidate_kind") or ""
            ).strip()
            if kind == "leave":
                draft.add(
                    "campfire_leave",
                    -2.0,
                    "离开会放弃本次篝火机会。",
                    "long_term_growth",
                )
                continue
            if kind != "rest_action":
                draft.gap(
                    f"campfire:unknown_candidate_kind:{kind or 'missing'}"
                )
                continue
            if not str(candidate.payload.get("action_id") or "").strip():
                draft.gap("campfire:action_id_missing")
                continue
            self._score_effect_action(
                draft,
                request,
                budget,
            )
        return finalize_recommendation(
            request,
            drafts,
            policy_version=CAMPFIRE_POLICY_VERSION,
        )

    def _score_effect_action(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
        budget,
    ) -> None:
        try:
            bundle = bundle_with_costs(draft.candidate.payload)
        except ValueError as exc:
            draft.gap(f"campfire:effects_invalid:{_gap_token(exc)}")
            return
        score_effect_bundle(
            draft,
            budget,
            bundle,
            request.world,
            self.repository,
            replacement_supported=(
                draft.candidate.payload.get("replacement_supported") is True
            ),
        )


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]

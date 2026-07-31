"""Structured Neow blessing policy with no narrative-text inference."""
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
from advisor.versions import NEOW_POLICY_VERSION


NEOW_BLESSING = "neow_blessing"


class NeowPolicy:
    decision_type = NEOW_BLESSING

    def __init__(self, repository: RelationalRepository):
        self.repository = repository

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("Neow policy received another decision type")
        budget = build_resource_budget(request.world)
        stages = {
            str(candidate.payload.get("stage_id") or "").strip()
            for candidate in request.candidates
            if candidate.eligible
        }
        stage_mismatch = len(stages) > 1
        drafts = []
        for candidate in request.candidates:
            draft = AssessmentDraft(candidate)
            drafts.append(draft)
            if not candidate.eligible:
                continue
            candidate_kind = str(
                candidate.payload.get("candidate_kind") or ""
            ).strip()
            if candidate_kind != "neow_blessing":
                draft.gap(
                    "neow:unknown_candidate_kind:"
                    + (candidate_kind or "missing")
                )
                continue
            if stage_mismatch:
                draft.gap("neow:candidate_stage_mismatch")
                continue
            blessing_id = str(
                candidate.payload.get("blessing_id") or ""
            ).strip()
            stage_id = str(candidate.payload.get("stage_id") or "").strip()
            if not blessing_id:
                draft.gap("neow:blessing_id_missing")
                continue
            if not stage_id:
                draft.gap("neow:stage_id_missing")
                continue
            try:
                bundle = bundle_with_costs(candidate.payload)
            except ValueError as exc:
                draft.gap(f"neow:effects_invalid:{_gap_token(exc)}")
                continue
            score_effect_bundle(
                draft,
                budget,
                bundle,
                request.world,
                self.repository,
                replacement_supported=(
                    candidate.payload.get("replacement_supported") is True
                ),
            )
        return finalize_recommendation(
            request,
            drafts,
            policy_version=NEOW_POLICY_VERSION,
        )


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]

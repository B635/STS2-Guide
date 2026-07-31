"""Deterministic second-stage card upgrade/removal policy."""
from __future__ import annotations

from advisor.choice_effects import (
    AssessmentDraft,
    ChoiceEffect,
    finalize_recommendation,
)
from advisor.deck_edit_value import score_deck_effect
from advisor.decision_core import DecisionRequest, Recommendation
from storage.relational import RelationalRepository
from advisor.versions import DECK_EDIT_POLICY_VERSION


DECK_EDIT = "deck_edit"
_OPERATIONS = frozenset({"upgrade", "remove", "transform"})


class DeckEditPolicy:
    decision_type = DECK_EDIT

    def __init__(self, repository: RelationalRepository):
        self.repository = repository

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("deck edit policy received another decision type")
        operations = {
            str(candidate.payload.get("operation") or "").strip()
            for candidate in request.candidates
            if candidate.eligible
            and str(candidate.payload.get("candidate_kind") or "").strip()
            == "deck_edit"
        }
        operation_mismatch = len(operations) > 1
        drafts: list[AssessmentDraft] = []
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
                    "deck_edit_leave",
                    0.0,
                    "保留当前牌组，不执行牌组编辑。",
                    "deck_burden",
                )
                continue
            if kind != "deck_edit":
                draft.gap(
                    f"deck_edit:unknown_candidate_kind:{kind or 'missing'}"
                )
                continue
            if operation_mismatch:
                draft.gap("deck_edit:operation_mismatch")
                continue
            operation = str(
                candidate.payload.get("operation") or ""
            ).strip()
            entity_id = str(
                candidate.payload.get("entity_id") or ""
            ).strip()
            if operation not in _OPERATIONS:
                draft.gap(
                    f"deck_edit:unknown_operation:{operation or 'missing'}"
                )
                continue
            if not entity_id:
                draft.gap("deck_edit:entity_id_missing")
                continue
            score_deck_effect(
                draft,
                ChoiceEffect(
                    kind=f"{operation}_card",
                    entity_type="cards",
                    entity_id=entity_id,
                    target_mode="specific",
                    certainty="exact",
                    source_code="decision_candidate:deck_edit",
                ),
                request.world,
                self.repository,
            )
        return finalize_recommendation(
            request,
            drafts,
            policy_version=DECK_EDIT_POLICY_VERSION,
        )

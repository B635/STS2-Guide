"""Allowlisted structured event-option policy.

Event descriptions are presentation data only.  This module intentionally
never reads a description/title field and cannot derive effects from prose.
"""
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
from advisor.versions import EVENT_POLICY_VERSION


EVENT_OPTION = "event_option"


class EventPolicy:
    decision_type = EVENT_OPTION

    def __init__(self, repository: RelationalRepository):
        self.repository = repository

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("event policy received another decision type")
        budget = build_resource_budget(request.world)
        scopes = {
            (
                str(candidate.payload.get("event_id") or "").strip(),
                str(candidate.payload.get("page_id") or "").strip(),
            )
            for candidate in request.candidates
            if candidate.eligible
            and str(
                candidate.payload.get("candidate_kind") or ""
            ).strip() == "event_option"
        }
        scope_mismatch = len(scopes) > 1
        drafts = []
        for candidate in request.candidates:
            draft = AssessmentDraft(candidate)
            drafts.append(draft)
            if not candidate.eligible:
                continue
            candidate_kind = str(
                candidate.payload.get("candidate_kind") or ""
            ).strip()
            if candidate_kind == "leave":
                draft.add(
                    "event_leave",
                    0.0,
                    "离开会保留当前已观察到的资源。",
                    "resource_efficiency",
                )
                continue
            if candidate_kind != "event_option":
                draft.gap(
                    "event:unknown_candidate_kind:"
                    + (candidate_kind or "missing")
                )
                continue
            if scope_mismatch:
                draft.gap("event:candidate_scope_mismatch")
                continue
            event_id = str(candidate.payload.get("event_id") or "").strip()
            page_id = str(candidate.payload.get("page_id") or "").strip()
            option_id = str(candidate.payload.get("option_id") or "").strip()
            if not event_id:
                draft.gap("event:event_id_missing")
                continue
            if not page_id:
                draft.gap("event:page_id_missing")
                continue
            if not option_id:
                draft.gap("event:option_id_missing")
                continue
            if not self._known_option(event_id, page_id, option_id):
                draft.gap(
                    f"event:unknown_option:{event_id}:{page_id}:{option_id}"
                )
                continue
            try:
                bundle = bundle_with_costs(candidate.payload)
            except ValueError as exc:
                draft.gap(f"event:effects_invalid:{_gap_token(exc)}")
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
            policy_version=EVENT_POLICY_VERSION,
        )

    def _known_option(
        self,
        event_id: str,
        page_id: str,
        option_id: str,
    ) -> bool:
        event = self.repository.event_tree(event_id)
        if event is None:
            return False
        page = next(
            (
                row
                for row in event.get("_pages") or ()
                if str(row.get("id") or "") == page_id
            ),
            None,
        )
        if page is not None:
            return any(
                str(option.get("id") or "") == option_id
                for option in page.get("options") or ()
            )
        # The public runtime API exposes stable EventOption.TextKey values
        # but not the author's page key.  A Mod page fingerprint may
        # therefore differ from the static catalog ID.  Resolve only when
        # the structured option identity occurs on exactly one catalog
        # page; reused options remain unknown instead of parsing prose.
        matching_pages = [
            row
            for row in event.get("_pages") or ()
            if any(
                str(option.get("id") or "") == option_id
                for option in row.get("options") or ()
            )
        ]
        return len(matching_pages) == 1


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]

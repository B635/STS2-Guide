"""Allowlisted structured event-option policy.

Event descriptions are presentation data only.  This module intentionally
never reads a description/title field and cannot derive effects from prose.
"""
from __future__ import annotations

import re

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
from advisor.scoring_preferences import DEFAULT_SCORING_PREFERENCES, ScoringPreferences
from storage.relational import RelationalRepository
from advisor.versions import EVENT_POLICY_VERSION


EVENT_OPTION = "event_option"


class EventPolicy:
    decision_type = EVENT_OPTION

    def __init__(self, repository: RelationalRepository,
                 preferences: ScoringPreferences = DEFAULT_SCORING_PREFERENCES):
        self.repository = repository
        self.preferences = preferences

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
            draft = AssessmentDraft(candidate, preferences=self.preferences)
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
        pages = event.get("_pages") or ()
        fingerprint = re.fullmatch(r"PAGE_[0-9A-F]{16}", page_id) is not None
        # The runtime key is an API identity, not prose: preserve it in the
        # candidate and resolve only this event's exact author page/option.
        # Never strip an arbitrary suffix or silently retry another page.
        if "." in option_id:
            match = re.fullmatch(
                re.escape(event_id) + r"\.pages\.([A-Z0-9_]+)\.options\.([A-Z0-9_]+)",
                option_id,
            )
            if match is None:
                return False
            catalog_page, catalog_option = match.groups()
            if not fingerprint and page_id != catalog_page:
                return False
            matching_pages = [row for row in pages if str(row.get("id") or "") == catalog_page]
            return len(matching_pages) == 1 and sum(
                str(option.get("id") or "") == catalog_option
                for option in matching_pages[0].get("options") or ()
            ) == 1
        page = next(
            (
                row
                for row in pages
                if str(row.get("id") or "") == page_id
            ),
            None,
        )
        if page is not None:
            return any(
                str(option.get("id") or "") == option_id
                for option in page.get("options") or ()
            )
        # Older replay fixtures can have a short option ID and an opaque
        # runtime page fingerprint. Require a unique catalog occurrence;
        # an explicit but unknown page is not such a fingerprint.
        if not fingerprint:
            return False
        matching_pages = [
            row
            for row in pages
            if any(
                str(option.get("id") or "") == option_id
                for option in row.get("options") or ()
            )
        ]
        return len(matching_pages) == 1


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]

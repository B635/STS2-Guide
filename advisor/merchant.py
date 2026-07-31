"""Deterministic merchant policy over real inventory candidates."""
from __future__ import annotations

from typing import Any, Mapping

from advisor.card_reward import recommend_card_reward
from advisor.choice_effects import (
    AssessmentDraft,
    actual_gold_cost,
    finalize_recommendation,
)
from advisor.deck_edit_value import best_remove_opportunity
from advisor.decision_core import DecisionRequest, Recommendation
from advisor.resource_budget import build_resource_budget, score_price
from advisor.versions import MERCHANT_POLICY_VERSION
from storage.relational import RelationalRepository


MERCHANT_CHOICE = "merchant_choice"
_OFFER_KINDS = frozenset({
    "card",
    "relic",
    "potion",
    "card_removal",
})


class MerchantPolicy:
    decision_type = MERCHANT_CHOICE

    def __init__(self, repository: RelationalRepository):
        self.repository = repository

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("merchant policy received another decision type")
        budget = build_resource_budget(request.world)
        drafts = []
        for candidate in request.candidates:
            draft = AssessmentDraft(candidate)
            drafts.append(draft)
            if not candidate.eligible:
                continue
            payload = candidate.payload
            candidate_kind = str(
                payload.get("candidate_kind") or ""
            ).strip()
            if candidate_kind == "leave":
                draft.add(
                    "merchant_leave_flexibility",
                    0.0,
                    "离开商店会保留当前资源。",
                    "resource_efficiency",
                )
                continue
            if candidate_kind != "merchant_offer":
                draft.gap(
                    "merchant:unknown_candidate_kind:"
                    + (candidate_kind or "missing")
                )
                continue
            kind = str(payload.get("offer_kind") or "").strip()
            if kind not in _OFFER_KINDS:
                draft.gap(
                    f"merchant:unknown_offer_kind:{kind or 'missing'}"
                )
                continue
            if payload.get("is_stocked") is not True:
                draft.gap("merchant:eligible_offer_not_stocked")
                continue

            entity_id = str(payload.get("entity_id") or "").strip()
            if kind != "card_removal" and not entity_id:
                draft.gap("merchant:entity_id_missing")
                continue
            if kind == "card":
                self._score_card(draft, request, entity_id)
            elif kind == "relic":
                self._score_relic(draft, entity_id)
            elif kind == "potion":
                self._score_potion(draft, request, payload, entity_id)
            else:
                self._score_removal(draft, request)
            try:
                price = actual_gold_cost(payload)
            except ValueError as exc:
                draft.gap(f"merchant:costs_invalid:{_gap_token(exc)}")
                continue
            score_price(draft, budget, price)
        return finalize_recommendation(
            request,
            drafts,
            policy_version=MERCHANT_POLICY_VERSION,
        )

    def _score_card(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
        entity_id: str,
    ) -> None:
        if self.repository.find_card(entity_id) is None:
            draft.gap(f"unknown_card:{entity_id}")
            return
        advice = recommend_card_reward(
            request.world.scoring_state(),
            [{"card": entity_id}],
            self.repository,
            can_skip=False,
        )
        row = advice["recommendations"][0]
        if not row.get("known"):
            draft.gap(f"unknown_card:{entity_id}")
            return
        contextual = max(
            -10.0,
            min(14.0, (float(row["state_score"]) - 50.0) * 0.55),
        )
        draft.add(
            "merchant_card_access",
            10.0,
            "商店卡牌是一次确定的牌组补强机会。",
            "long_term_growth",
        )
        draft.add(
            "merchant_card_context_fit",
            contextual,
            "按当前牌组、角色机制和真实路线计算卡牌适配。",
            "synergy",
        )
        for gap in row.get("data_gaps") or ():
            draft.gap(str(gap))

    def _score_relic(
        self,
        draft: AssessmentDraft,
        entity_id: str,
    ) -> None:
        relic = self.repository.find_relic(entity_id)
        if relic is None:
            draft.gap(f"unknown_relic:{entity_id}")
            return
        effects = relic.get("_effect_tags") or {}
        draft.add(
            "merchant_relic_permanence",
            14.0,
            "遗物是持续生效的长期资源。",
            "long_term_growth",
        )
        if any(tag in effects for tag in ("block", "healing")):
            draft.add(
                "merchant_relic_survival",
                3.0,
                "该遗物提供结构化生存效果。",
                "survival",
            )

    def _score_potion(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
        payload: Mapping[str, Any],
        entity_id: str,
    ) -> None:
        potion = self.repository.find_entity("potions", entity_id)
        if potion is None:
            draft.gap(f"unknown_potion:{entity_id}")
            return
        budget = build_resource_budget(request.world)
        if not budget.potions.known:
            draft.gap("resource:potion_slots_missing")
            return
        if int(budget.potions.free_slots or 0) <= 0:
            if payload.get("replacement_supported") is not True:
                draft.gap("resource:potion_replacement_unresolved")
                return
            draft.add(
                "merchant_potion_replacement",
                -3.0,
                "药水槽已满，计入替换机会成本。",
                "resource_efficiency",
            )
        draft.add(
            "merchant_potion_access",
            8.0,
            "药水提供一次性战斗资源。",
            "resource_efficiency",
        )

    def _score_removal(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
    ) -> None:
        opportunity = best_remove_opportunity(
            request.world,
            self.repository,
        )
        if opportunity.value is None:
            for gap in opportunity.data_gaps:
                draft.gap(gap)
            return
        draft.add(
            "merchant_remove_opportunity",
            opportunity.value,
            f"当前最佳移除目标为 {opportunity.target_id}。",
            "deck_burden",
        )


def _gap_token(error: Exception) -> str:
    return "_".join(str(error).lower().split())[:80]

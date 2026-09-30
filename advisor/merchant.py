"""Deterministic merchant policy over real inventory candidates."""
from __future__ import annotations

from typing import Any, Mapping

from advisor.card_reward import recommend_card_reward
from advisor.choice_effects import (
    AssessmentDraft,
    ChoiceEffect,
    EffectBundle,
    actual_gold_cost,
    finalize_recommendation,
)
from advisor.deck_edit_value import best_remove_opportunity
from advisor.decision_core import DecisionRequest, Recommendation
from advisor.resource_budget import (
    build_resource_budget,
    score_price,
    score_resource_effects,
)
from advisor.scoring_preferences import DEFAULT_SCORING_PREFERENCES, ScoringPreferences
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

    def __init__(
        self, repository: RelationalRepository,
        preferences: ScoringPreferences = DEFAULT_SCORING_PREFERENCES,
    ):
        self.repository = repository
        self.preferences = preferences

    def recommend(self, request: DecisionRequest) -> Recommendation:
        if request.decision_type != self.decision_type:
            raise ValueError("merchant policy received another decision type")
        budget = build_resource_budget(request.world)
        drafts = []
        for candidate in request.candidates:
            draft = AssessmentDraft(candidate, preferences=self.preferences)
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
                self._score_relic(draft, request, entity_id)
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
        upgrades = draft.candidate.payload.get("upgrades")
        if type(upgrades) is not int or not 0 <= upgrades <= 9:
            draft.gap("merchant:card_upgrades_unknown")
            return
        if upgrades > 1:
            draft.gap("merchant:repeat_upgrade_unmodeled")
            return
        advice = recommend_card_reward(
            request.world.scoring_state(),
            [{"card": entity_id, "upgrades": upgrades}],
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
        # The inner score already includes numerical card factors. Forward
        # only explicit zero-contribution scope/coverage disclosures.
        for factor in row.get("factors") or ():
            if (
                factor.get("code") in {"estimate_scope", "unmodeled_mechanic"}
                and factor.get("delta") == 0.0
                and str(factor.get("message") or "").strip()
            ):
                draft.add(
                    factor["code"], 0.0, factor["message"], "data_completeness",
                )
        for gap in row.get("data_gaps") or ():
            draft.gap(str(gap))

    def _score_relic(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
        entity_id: str,
    ) -> None:
        relic = self.repository.find_relic(entity_id)
        if relic is None:
            draft.gap(f"unknown_relic:{entity_id}")
            return
        effects = relic.get("_effect_tags") or {}
        if not effects:
            draft.gap(f"merchant:relic_mechanism_unmodeled:{entity_id}")
            return
        draft.add(
            "merchant_relic_permanence",
            14.0,
            "遗物是持续生效的长期资源。",
            "long_term_growth",
        )
        if relic.get("id") == "TOUGH_BANDAGES":
            # Verified 0.111.0 AfterCardDiscarded requires the owner's card
            # and the owner's side to be active. Existing discard tags mark
            # active providers, not discard-pile nouns or Sly payoffs.
            if not request.world.deck or request.world.capture_warnings:
                draft.gap("merchant:relic_trigger_deck_unknown:TOUGH_BANDAGES")
                return
            has_discard_source = False
            for entry in request.world.deck:
                if (entry.count < 1 or entry.upgrades not in (0, 1)
                        or entry.enchantment or entry.affliction):
                    draft.gap("merchant:relic_trigger_deck_unmodeled:TOUGH_BANDAGES")
                    continue
                card = self.repository.find_card(entry.card)
                if card is None:
                    draft.gap(f"unknown_card:{entry.card}")
                    continue
                if (card.get("_effect_tags") or {}).get("discard", 0) > 0:
                    has_discard_source = True
            if draft.blocking_unknown:
                return
            draft.add(
                "estimate_scope",
                0.0,
                (
                    "已有主动弃牌来源，仅表示潜在配合；还需弃掉自己的牌且轮到己方，"
                    "未预测触发频率、每回合格挡或完整遗物价值。"
                    if has_discard_source else
                    "当前牌组未观察到主动弃牌来源，本次不给予生存适配分；"
                    "未预测未来来源、触发频率或完整遗物价值。"
                ),
                "data_completeness",
            )
            if not has_discard_source:
                return
        # Relic tags indicate qualitative support, not guaranteed per-turn
        # output.  Do not invent a numeric block/heal amount or activation rate.
        if any(tag in effects for tag in ("supports_block", "healing")):
            draft.add(
                "merchant_relic_survival",
                3.0,
                "该遗物具有生存相关标签，实际触发仍取决于遗物条件。",
                "survival",
            )
            budget = build_resource_budget(request.world)
            if not budget.health.known:
                draft.gap("resource:hp_missing")
                return
            missing_ratio = 1.0 - int(budget.health.hp or 0) / max(
                1, int(budget.health.max_hp or 1)
            )
            if missing_ratio > 0:
                draft.add(
                    "merchant_relic_survival_urgency",
                    round(min(3.0, missing_ratio * 4.0), 4),
                    "当前生命缺口提高生存型遗物的相对价值，并非承诺立即恢复生命。",
                    "survival",
                )
            if "supports_block" in effects:
                total_cards = 0
                block_cards = 0
                for row in request.world.deck:
                    card = self.repository.find_card(row.card)
                    if card is None:
                        draft.gap(f"unknown_card:{row.card}")
                        continue
                    total_cards += row.count
                    if (card.get("_effect_tags") or {}).get("block", 0) > 0:
                        block_cards += row.count
                if total_cards and block_cards / total_cards < 0.3:
                    draft.add(
                        "merchant_relic_defense_gap",
                        round(2.0 * (1.0 - block_cards / total_cards / 0.3), 4),
                        "牌组中直接格挡牌占比较低，有限提高防御相关遗物的价值。",
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
        # Only this exact mechanic is verified in the release-bound assembly.
        # Do not infer other potion effects from their names or prose.
        if potion.get("id") != "BLOOD_POTION":
            draft.gap(f"merchant:potion_mechanism_unmodeled:{entity_id}")
            return
        tag = "potion_heal_max_hp_percent"
        percent = (potion.get("_effect_tags") or {}).get(tag)
        source = (potion.get("_effect_tag_sources") or {}).get(tag)
        if (
            type(percent) not in (int, float)
            or percent != 20
            or source != "assembly:0.111.0:BloodPotion.CanonicalVars.HealPercent"
        ):
            draft.gap(f"merchant:potion_mechanism_unmodeled:{entity_id}")
            return
        if not budget.health.known:
            draft.gap("resource:hp_missing")
            return
        # Creature.SetCurrentHpInternal caps at MaxHp then casts decimal to
        # int (truncation). Current HP is integral, so truncate this gain first.
        amount = int(budget.health.max_hp or 0) * int(percent) // 100
        recoverable = min(
            amount,
            int(budget.health.max_hp or 0) - int(budget.health.hp or 0),
        )
        factor_start = len(draft.factors)
        score_resource_effects(
            draft,
            budget,
            EffectBundle((ChoiceEffect(
                kind="hp_delta", amount=amount, source_code=source,
            ),)),
        )
        for factor in draft.factors[factor_start:]:
            factor["message"] = (
                f"按当前生命计算，鲜血药水当前可恢复{recoverable}点生命，"
                "恢复价值沿用近期真实路线压力。"
            )
        draft.add(
            "estimate_scope",
            0.0,
            f"购买不等于饮用；仅评估当前可恢复{recoverable}点生命的机会，"
            "未计入未来保留价值或战斗用药联动。",
            "data_completeness",
        )

    def _score_removal(
        self,
        draft: AssessmentDraft,
        request: DecisionRequest,
    ) -> None:
        opportunity = best_remove_opportunity(
            request.world,
            self.repository,
            draft.candidate.payload.get("deck_edit_targets"),
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
